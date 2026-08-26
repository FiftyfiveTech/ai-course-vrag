"""Answer with citations — VRAG-019.

Question in, `schemas.answer.Answer` out: the answer text, the timestamps it came from, or an
abstention. This is the module the Phase 2 gate (VRAG-021) scores; retrieval only has to put
the right moment in the top 5, and this has to notice that it is there.

    make answer Q="how old was Bernini when he met the Pope"
    make answer-dev                 # every dev pair, one line each
    make gate-phase2a               # the VRAG-019 gate

    from src.answer import answer
    run = answer("what two tools do I need to cut paper?", cfg, meter)
    print(run.answer.answer, [str(c) for c in run.answer.citations])

Four steps, and each one is somewhere a wrong answer can come from:

    retrieve  ->  render_context  ->  the model  ->  validate  ->  ground

`retrieve` is VRAG-016 and unchanged. `render_context` prints each passage with the exact
`video_id`/`t_start`/`t_end` the model is told to copy. The model is constrained at generation
time by `schemas.answer.json_schema()`, then its output is validated against the same
declaration — that is the "schema-valid" number the gate reports, and it is measured on what
the model produced, not on a repaired copy. `ground` runs last and is the subject of the long
comment on it: a well-formed citation can still point nowhere.

Arms
----
`answer.arm` in config.toml, the same shape `transcript.arm` uses:

    arm = "groq"     hosted, free tier, strict JSON schema at generation time
    arm = "ollama"   local, no key, no network; strict JSON schema too

The model is named by its HF repo id in both cases, and the arm translates. For Groq that
translation is the identity — `openai/gpt-oss-120b` is both the HF repo id and Groq's wire id,
which is the first model in this pipeline where those two agree (whisper's does not: see
`transcript._groq_model_name`). It is spelled out anyway rather than assumed, because the
whisper id and the nomic id have both cost a session to a name that was almost right.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from schemas.answer import Answer, json_schema
from src.config import Config
from src.config import load as load_config
from src.retrieve import RetrievedChunk, retrieve
from src.telemetry import Meter

DEV_DIR = Path("evals/dev")


class AnswerError(Exception):
    """The answer path failed — message says which step and why."""


class _GroqRateLimitError(AnswerError):
    """Groq returned 429 — caller can fall back to the local arm."""


@dataclass(frozen=True)
class AnswerRun:
    """One question, and everything needed to explain what came back.

    `answer` is None exactly when the model's JSON did not validate; `error` says why. The
    gate needs both — a schema failure is only actionable next to the text that caused it.
    """

    question: str
    hits: list[RetrievedChunk]
    raw: str
    answer: Answer | None
    error: str | None = None
    repairs: list[str] = field(default_factory=list)
    tokens: int = 0

    @property
    def valid(self) -> bool:
        return self.answer is not None

    @property
    def abstained(self) -> bool:
        return self.answer is not None and self.answer.abstain


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def answer(question: str, cfg: Config, meter: Meter) -> AnswerRun:
    """Retrieve, ask, validate, ground. Never raises on a bad model reply.

    A model that returns unparseable JSON is a result the gate has to be able to count, not
    an exception that stops the run halfway through the dev set — so the failure is carried
    in `AnswerRun.error` and only genuine infrastructure faults (no key, no index, no model)
    raise `AnswerError`.
    """
    hits = retrieve(question, cfg, meter)
    system, user = build_messages(question, hits, cfg)

    raw, tokens = _ask(system, user, cfg, meter)

    try:
        parsed = Answer.model_validate_json(raw)
    except ValidationError as exc:
        return AnswerRun(
            question=question,
            hits=hits,
            raw=raw,
            answer=None,
            error=_terse(exc),
            tokens=tokens,
        )
    except ValueError as exc:
        # Not JSON at all. Same class of result as a validation failure: countable, not fatal.
        return AnswerRun(
            question=question,
            hits=hits,
            raw=raw,
            answer=None,
            error=f"not valid JSON: {exc}",
            tokens=tokens,
        )

    grounded, repairs = ground(parsed, hits)
    return AnswerRun(
        question=question,
        hits=hits,
        raw=raw,
        answer=grounded,
        repairs=repairs,
        tokens=tokens,
    )


def build_messages(
    question: str, hits: list[RetrievedChunk], cfg: Config
) -> tuple[str, str]:
    """The two messages, from the prompt file named in config. No prompt text lives here."""
    system, template = load_prompt(Path(cfg.get("answer.prompt")))
    user = template.replace("{{context}}", render_context(hits)).replace(
        "{{question}}", question
    )
    return system, user


def render_context(hits: list[RetrievedChunk]) -> str:
    """The retrieved passages, with the three numbers the model is told to copy.

    The header carries `video_id`, `t_start` and `t_end` at one decimal place, which is the
    precision `ground` matches on and the precision a citation is scored at. Printing more
    would invite the model to reproduce a float it cannot; printing fewer would make two
    adjacent chunks indistinguishable.
    """
    if not hits:
        return "(no passages were retrieved for this question)"
    blocks = []
    for n, hit in enumerate(hits, start=1):
        text = " ".join(hit.text.split())
        blocks.append(
            f"[{n}] video_id={hit.video_id}  t_start={hit.t_start:.1f}  "
            f"t_end={hit.t_end:.1f}\n{text}"
        )
    return "\n\n".join(blocks)


def load_prompt(path: Path) -> tuple[str, str]:
    """Split a prompt file into its `## System` and `## User` sections.

    Everything outside those two H2 sections is commentary for a human — the file explains
    why it is written the way it is, and that explanation must not be sent to the model.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AnswerError(f"cannot read the prompt file {path}: {exc}") from exc

    sections = _h2_sections(text)
    missing = [name for name in ("system", "user") if name not in sections]
    if missing:
        raise AnswerError(
            f"{path}: no {' or '.join('## ' + m.title() for m in missing)} section. A prompt "
            f"file needs both; see prompts/answer_v1.md."
        )

    user = sections["user"]
    for token in ("{{context}}", "{{question}}"):
        if token not in user:
            raise AnswerError(
                f"{path}: the ## User section never uses {token}, so the "
                f"{token.strip('{}')} would never reach the model."
            )
    return sections["system"], user


def ground(ans: Answer, hits: list[RetrievedChunk]) -> tuple[Answer, list[str]]:
    """Make every citation point at a passage that was actually retrieved.

    Validation proves a citation is well formed. It cannot prove the citation is real: the
    model is free to emit `{"video_id": "611", "t_start": 412.0}` for a question whose
    retrieved passages were all from video 701, and that object is perfectly valid.

    So each citation is matched back to the retrieved set:

    * no retrieved passage from that `video_id` — the citation was invented. Dropped.
    * otherwise it is snapped onto the retrieved passage it is nearest to, and carries that
      passage's exact range. A model that shifts a timestamp by a few seconds is not making a
      new claim, it is copying badly, and the passage's own range is the truthful version of
      the claim it is making.

    If nothing survives and the response was not already an abstention, the response becomes
    one. That direction is safe under QA_SPEC §5 and it is worth being explicit about why: §2
    requires at least one citation on the ground-truth video for an answerable question to
    count, so a response whose every citation was invented is already scored incorrect and
    abstaining cannot lose a point that was available. On an unanswerable question it turns an
    incorrect hallucination into a correct abstention. And to a user, an answer whose only
    citation goes nowhere is worse than no answer — which is the reason that is not about the
    scoring rule.

    Returns the repaired answer and a line per repair, so `make answer-dev` shows how often
    the prompt is being taken literally.
    """
    repairs: list[str] = []
    if ans.abstain:
        return ans, repairs

    by_video: dict[str, list[RetrievedChunk]] = {}
    for hit in hits:
        by_video.setdefault(hit.video_id, []).append(hit)

    kept: list[dict] = []
    for cite in ans.citations:
        candidates = by_video.get(cite.video_id)
        if not candidates:
            seen = ", ".join(sorted(by_video)) or "nothing"
            repairs.append(
                f"dropped {cite} - no retrieved passage from video {cite.video_id} "
                f"(retrieved: {seen})"
            )
            continue
        near = min(candidates, key=lambda h: abs(h.t_start - cite.t_start))
        if abs(near.t_start - cite.t_start) > 0.05 or abs(near.t_end - cite.t_end) > 0.05:
            repairs.append(
                f"snapped {cite} onto video {near.video_id} "
                f"{near.t_start:.1f}s-{near.t_end:.1f}s"
            )
        entry = {"video_id": near.video_id, "t_start": near.t_start, "t_end": near.t_end}
        if entry not in kept:
            kept.append(entry)

    if not kept:
        repairs.append("no citation survived grounding - forced to abstain")
        return Answer.abstention(), repairs

    return Answer(answer=ans.answer, citations=kept, abstain=False), repairs


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def _ask(system: str, user: str, cfg: Config, meter: Meter) -> tuple[str, int]:
    """Route to the configured arm. Returns the raw reply text and the tokens it cost."""
    arm = str(cfg.get("answer.arm")).strip().lower()
    model = cfg.get("answer.model")
    temperature = float(cfg.get("answer.temperature"))
    max_tokens = int(cfg.get("answer.max_tokens"))

    if arm == "groq":
        ollama_model = cfg.get("answer.ollama_model")
        try:
            return _groq_arm(system, user, model, temperature, max_tokens, meter)
        except _GroqRateLimitError as exc:
            print(
                f"WARNING: Groq rate limit hit — falling back to ollama ({ollama_model})",
                file=__import__("sys").stderr,
            )
            return _ollama_arm(system, user, ollama_model, temperature, max_tokens, meter)
    if arm == "ollama":
        ollama_model = cfg.get("answer.ollama_model")
        return _ollama_arm(system, user, ollama_model, temperature, max_tokens, meter)
    raise AnswerError(
        f"config.toml: answer.arm is {arm!r}, which is not an arm. Use 'groq' or 'ollama'."
    )


def _groq_wire_name(hf_repo_id: str) -> str:
    """HF repo id -> the id Groq's chat endpoint wants.

    For `openai/gpt-oss-120b` these are the same string, and this function exists to say so
    rather than to leave it looking like an oversight. Whisper's are not the same
    (`openai/whisper-large-v3-turbo` 404s; Groq wants `whisper-large-v3-turbo`), and the
    embedding model needed a `-GGUF:F16` suffix before Ollama would load it. Two model names
    in this pipeline have already been almost right, so the third one gets a named function
    and a test rather than a `.split("/")` inline somewhere.
    """
    return hf_repo_id


def _groq_client(api_key: str):
    """The Groq client. One seam, so a test can drive the arm without a network call."""
    from groq import Groq

    return Groq(api_key=api_key)


def _groq_arm(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
    meter: Meter,
) -> tuple[str, int]:
    try:
        import groq  # noqa: F401
    except ImportError as exc:
        raise AnswerError("groq package not installed — run `uv sync`") from exc

    api_key = _env_key("GROQ_API_KEY")
    if not api_key:
        raise AnswerError(
            "GROQ_API_KEY is not set. Add it to ~/.config/ai-course-vrag.env or the process "
            'environment, or run the local arm instead: answer.arm = "ollama".'
        )

    client = _groq_client(api_key)
    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=_groq_wire_name(model),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Strict mode constrains generation to schemas.answer.json_schema(), so a
            # reply that fails validation downstream means the schema handed to the model
            # and the validator disagree — which is why both come off one declaration.
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": json_schema(),
                },
            },
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
    except Exception as exc:
        msg = str(exc)
        if "429" in msg or "rate_limit" in msg.lower() or "rate limit" in msg.lower():
            raise _GroqRateLimitError(f"groq rate limit ({model}): {exc}") from exc
        raise AnswerError(f"groq arm failed ({model}): {exc}") from exc

    text = (response.choices[0].message.content or "").strip()
    usage = getattr(response, "usage", None)
    tokens = int(getattr(usage, "total_tokens", 0) or 0)
    meter.log(model, time.perf_counter() - t0, tokens=tokens)
    return text, tokens


def _ollama_arm(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
    meter: Meter,
) -> tuple[str, int]:
    try:
        import ollama
    except ImportError as exc:
        raise AnswerError("ollama package not installed — run `uv sync`") from exc

    from src.embed import _hf_to_ollama_tag

    tag = _hf_to_ollama_tag(model)
    t0 = time.perf_counter()
    try:
        response = ollama.chat(
            model=tag,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=json_schema(),
            options={"temperature": temperature, "num_predict": max_tokens},
        )
    except Exception as exc:
        raise AnswerError(
            f"ollama arm failed ({model}): {exc}\nMake sure it is pulled: ollama pull {tag}"
        ) from exc

    message = response["message"] if "message" in response else {}
    text = str((message or {}).get("content", "")).strip()
    tokens = int(response.get("prompt_eval_count") or 0) + int(
        response.get("eval_count") or 0
    )
    meter.log(model, time.perf_counter() - t0, tokens=tokens)
    return text, tokens


# Both arms time the call themselves and log it with `Meter.log` rather than wrapping it in
# `meter.span`. The reason is the token count: it exists only once the reply is back, and
# `span` fixes a call's units when it exits, so using it would mean either logging the latency
# with tokens=0 and the volume as a second zero-latency call - 15 questions then read as 30
# calls in the gate's cost line - or dropping the volume. `Meter.log` is documented for
# exactly this: "a call whose latency was measured externally". RATES prices this model at
# $0.00 on Groq's free tier; the volume is recorded anyway so that a move to a paid tier shows
# up in the $/video-hour line rather than nowhere.


def _env_key(key: str) -> str:
    from src.env import load_env

    if os.environ.get(key):
        return os.environ[key]
    value, _ = load_env().get(key, ("", ""))
    return value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_H2 = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def _h2_sections(text: str) -> dict[str, str]:
    """Map lowercased H2 heading -> the text under it, up to the next H2."""
    out: dict[str, str] = {}
    matches = list(_H2.finditer(text))
    for n, match in enumerate(matches):
        end = matches[n + 1].start() if n + 1 < len(matches) else len(text)
        out[match.group(1).strip().lower()] = text[match.end() : end].strip()
    return out


def _terse(exc: ValidationError) -> str:
    """A ValidationError as one readable line — the field, and what was wrong with it."""
    parts = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        parts.append(f"{where}: {err.get('msg', '')}")
    return "; ".join(parts) or str(exc)


def load_dev_pairs(dev_dir: Path = DEV_DIR) -> list[dict]:
    """Every dev pair, answerable and not. Abstention is half of what this module does."""
    pairs = []
    for path in sorted(dev_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def report(run: AnswerRun, out=None) -> None:
    out = out or sys.stdout
    print(f"\nQ: {run.question}", file=out)
    if run.answer is None:
        print(f"   SCHEMA-INVALID  {run.error}", file=out)
        print(f"   raw: {run.raw[:300]}", file=out)
        return
    if run.answer.abstain:
        print(f"   ABSTAIN  {run.answer.answer}", file=out)
    else:
        print(f"   A: {run.answer.answer}", file=out)
        for cite in run.answer.citations:
            print(f"      cite {cite}", file=out)
    for line in run.repairs:
        print(f"      grounding: {line}", file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="*", help="the question to answer")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="answer every pair in evals/dev instead, and print the schema-valid tally",
    )
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args(argv)

    # A citation line rendered on a cp1252 console must not exit non-zero for an encoding
    # accident — src/leakage.py hit exactly that and it read as a failure.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    cfg = load_config(args.config)
    meter = Meter()

    if not args.dev and not args.question:
        parser.error('give a question, or --dev. e.g. make answer Q="how old was Bernini?"')

    try:
        if args.dev:
            pairs = load_dev_pairs()
            if not pairs:
                print(f"FAIL - {DEV_DIR} holds no pairs", file=sys.stderr)
                return 1
            runs = []
            for pair in pairs:
                run = answer(pair["question"], cfg, meter)
                mark = "u" if pair.get("unanswerable") else " "
                print(f"\n[{pair.get('id', '?')}{mark}]", end="")
                report(run)
                runs.append((pair, run))
            _dev_summary(runs, cfg, meter)
        else:
            report(answer(" ".join(args.question), cfg, meter))
            _spend(cfg, meter)
    except Exception as exc:
        print(f"FAIL - {exc}", file=sys.stderr)
        return 1

    return 0


def _dev_summary(runs: list[tuple[dict, AnswerRun]], cfg: Config, meter: Meter) -> None:
    total = len(runs)
    valid = sum(1 for _, r in runs if r.valid)
    unanswerable = [(p, r) for p, r in runs if p.get("unanswerable")]
    answerable = [(p, r) for p, r in runs if not p.get("unanswerable")]
    abstained = sum(1 for _, r in unanswerable if r.abstained)
    false_abstain = sum(1 for _, r in answerable if r.abstained)
    repaired = sum(1 for _, r in runs if r.repairs)

    print(f"\nschema-valid                  {valid}/{total}")
    print(f"abstained, and should have    {abstained}/{len(unanswerable)} unanswerable pairs")
    print(f"abstained, and should not     {false_abstain}/{len(answerable)} answerable pairs")
    print(f"citations repaired by ground()  {repaired}/{total} pairs")
    _spend(cfg, meter)
    # Said every run, for the same reason src/probe.py says its version: the tally above is
    # not the Phase 2 result, and the temptation to read one off it is why the line is here.
    print("No accuracy here by design - QA_SPEC section 5 is scored on evals/heldout (VRAG-021).")


def _spend(cfg: Config, meter: Meter) -> None:
    calls = meter._calls
    print(
        f"\n{len(calls)} model call(s), {sum(c.latency_s for c in calls):.2f}s, "
        f"${sum(c.cost_usd for c in calls):.4f}  "
        f"(answer.arm={cfg.get('answer.arm')} {cfg.get('answer.model')})"
    )


if __name__ == "__main__":
    raise SystemExit(main())
