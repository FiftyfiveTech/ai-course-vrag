"""Answer with citations — VRAG-019.

Question in, `schemas.answer.Answer` out: the answer text, the timestamps it came from, or an
abstention. This is the module the Phase 2 gate (VRAG-021) scores; retrieval only has to put
the right moment in the top 5, and this has to notice that it is there.

    make answer Q="how old was Bernini when he met the Pope"
    make answer Q="@611 how old was Bernini?"   # that video only - src/mention.py
    make answer-dev                 # every dev pair, one line each
    make gate-phase2a               # the VRAG-019 gate

    from src.answer import answer
    run = answer("what two tools do I need to cut paper?", cfg, meter)
    print(run.answer.answer, [str(c) for c in run.answer.citations])

Five steps, and each one is somewhere a wrong answer can come from:

    scope  ->  retrieve  ->  render_context  ->  the model  ->  validate  ->  ground

`scope` reads the `@source` tags out of the question and turns them into a store-level
filter (`src.mention`); with no tag it is the identity and retrieval sees the whole
index. `retrieve` is VRAG-016 and takes that filter. `render_context` prints each passage with the exact
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
from src.mention import scope as parse_scope
from src.retrieve import RetrievedChunk, retrieve
from src.telemetry import Meter

DEV_DIR = Path("evals/dev")

# The two questions this module can be asked, named once so that the API's enum, the CLI's
# flag and the branch in `answer()` cannot drift apart into three spellings of the same word.
EXTRACTIVE = "extractive"  # five retrieved passages; declines what they do not state
OVERVIEW = "overview"  # the whole video, from the document src.overview built
MODES = (EXTRACTIVE, OVERVIEW)


class AnswerError(Exception):
    """The answer path failed — message says which step and why."""


class _GroqRateLimitError(AnswerError):
    """Groq returned 429 — the per-minute bucket is empty and refills.

    Transient. This is the one the local-arm fallback exists for: the same request, sent a
    minute later or sent to Ollama now, completes.
    """


class _GroqRequestTooLargeError(AnswerError):
    """Groq returned 413 — the request is bigger than the whole per-minute budget.

    Permanent, and deliberately *not* the same class as a 429, because the correct handling
    is the opposite one. Falling back here does not recover the work; it silently produces
    the run on a different, much smaller model and labels it as this one.

    Measured, on the free `on_demand` tier, building an overview for video 611:

        413  {'code': 'rate_limit_exceeded', 'type': 'tokens'}
        "Request too large for model `openai/gpt-oss-120b` ... on tokens per minute (TPM):
         Limit 8000, Requested 17152"
        x-ratelimit-remaining-tokens: 8000     <- a full bucket
        x-should-retry: false                  <- the provider's own verdict

    A full bucket is the whole point. Nothing was consumed by earlier traffic; a 13k-token
    prompt asking for 4k completion simply cannot fit an 8k budget, so waiting changes
    nothing. Groq reports both failures under `rate_limit_exceeded`, which is why the old
    string match on "rate_limit" read this as throttling and fell back — and why every
    overview build ran on a 3B local model until someone read the head of stderr.
    """


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

    # The model that actually produced `raw`, as an HF repo id. Not read off config at
    # report time, because config says which arm was *asked* and this says which one
    # *answered* — they differ whenever the groq arm hits a 429 and falls back to the
    # local one mid-call. Empty only for an AnswerRun built by a test fake.
    model: str = ""

    # Which videos retrieval was allowed to see, out of the `@source` tags in the question
    # (src.mention). Empty is the whole index and is the default. Recorded rather than
    # re-derived downstream: `question` still carries the tags a person typed, and a reader
    # of a run has to be able to see the scope that was actually *applied* — a tag that
    # resolved to nothing is a refusal, but a tag someone edited out of the text by hand
    # would otherwise leave the two disagreeing with no way to tell which won.
    scope: tuple[str, ...] = ()

    # The text that was really embedded and really shown to the model: `question` with the
    # tags removed. `@611` is not a word the corpus ever says, so leaving it in moves the
    # query vector for nothing.
    query: str = ""

    # Which of the two questions was asked — `extractive` or `overview`. Recorded because it
    # decides what `hits` even are: retrieved passages in one mode, the spans of a stored
    # overview in the other. A reader of a run that could not tell the two apart would be
    # comparing a 1.5 %-of-the-video answer with a whole-video one and calling them the same
    # measurement.
    mode: str = EXTRACTIVE

    @property
    def scoped(self) -> bool:
        return bool(self.scope)

    @property
    def valid(self) -> bool:
        return self.answer is not None

    @property
    def abstained(self) -> bool:
        return self.answer is not None and self.answer.abstain


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def answer(
    question: str, cfg: Config, meter: Meter, *, mode: str = EXTRACTIVE
) -> AnswerRun:
    """Scope, retrieve, ask, validate, ground. Never raises on a bad model reply.

    A model that returns unparseable JSON is a result the gate has to be able to count, not
    an exception that stops the run halfway through the dev set — so the failure is carried
    in `AnswerRun.error` and only genuine infrastructure faults (no key, no index, no model)
    raise `AnswerError`.

    A question may tag its sources — `"@611 what two tools do I need?"` — and then retrieval
    sees video 611 and nothing else. That is parsed here, once, rather than at each entry
    point, so `make answer`, `make ask`, `make api` and the frontend all behave the same way
    because they run the same code. An unresolvable tag raises `MentionError`: the caller's
    input is what is wrong, not the pipeline, and quietly widening the search back to the
    whole index would answer a question nobody asked.

    `mode` picks which of the two questions is being asked, and it is a parameter rather than
    something inferred from the wording. `"extractive"` is retrieval: five passages, and a
    prompt that declines anything they do not state. `"overview"` is the whole video: the
    document `src.overview` built at index time, and a prompt that is allowed to synthesise.
    Nothing here guesses between them — a question is routed by what the caller asked for, so
    the extractive path and the numbers measured on it cannot move because someone phrased a
    question differently.
    """
    scope = parse_scope(question, cfg)
    if mode == OVERVIEW:
        return _overview_run(question, scope, cfg, meter)
    if mode != EXTRACTIVE:
        raise AnswerError(
            f"{mode!r} is not an answering mode. Use {EXTRACTIVE!r} or {OVERVIEW!r}."
        )

    hits = retrieve(scope.text, cfg, meter, video_ids=scope.video_ids)
    with meter.stage("answer.prompt"):
        system, user = build_messages(scope.text, hits, cfg)

    raw, tokens, used = _ask(system, user, cfg, meter)

    def run(**kw) -> AnswerRun:
        return AnswerRun(
            question=question,
            hits=hits,
            raw=raw,
            tokens=tokens,
            model=used,
            scope=scope.video_ids,
            query=scope.text,
            **kw,
        )

    try:
        parsed = Answer.model_validate_json(raw)
    except ValidationError as exc:
        return run(answer=None, error=_terse(exc))
    except ValueError as exc:
        # Not JSON at all. Same class of result as a validation failure: countable, not fatal.
        return run(answer=None, error=f"not valid JSON: {exc}")

    with meter.stage("answer.ground"):
        grounded, repairs = ground(parsed, hits)
    return run(answer=grounded, repairs=repairs)


def _overview_run(question, scope, cfg: Config, meter: Meter) -> AnswerRun:
    """Answer a question about a whole video, from the overview built at index time.

    The shape is deliberately the same as the extractive path below the model call: validate
    against `Answer`, then `ground`. What changes is where the evidence comes from — the
    spans of a stored overview instead of five retrieved chunks — and that is the only
    difference a reader has to hold. Grounding, `to_citation`, `stream_url` and the player's
    seek are all reached unchanged, so an overview citation is as clickable as any other.

    One video, required. "What is this about?" with no `@` tag is not a question with a
    missing answer, it is a question with a missing subject, and picking a video for the user
    would answer something they did not ask. Two tags are the same problem twice.
    """
    from src.overview import as_chunks, load, render_overview

    if len(scope.video_ids) != 1:
        raise AnswerError(
            "an overview answers one video, and this question is scoped to "
            f"{scope.describe()}. Tag exactly one source — `@611 what is this about?` — "
            "and ask again."
        )
    video_id = scope.video_ids[0]

    stored = load(video_id)
    if stored is None:
        raise AnswerError(
            f"video {video_id} has no overview on this host, so there is nothing to answer "
            f"a whole-video question from. Build it: make overview VIDEO=<the file in "
            f"samples/>."
        )

    hits = as_chunks(stored)
    with meter.stage("answer.prompt"):
        system, user = build_messages(
            scope.text,
            [],
            cfg,
            prompt=Path(cfg.get("overview.answer_prompt")),
            context=render_overview(stored),
        )

    raw, tokens, used = _ask(system, user, cfg, meter)

    def run(**kw) -> AnswerRun:
        return AnswerRun(
            question=question,
            hits=hits,
            raw=raw,
            tokens=tokens,
            model=used,
            scope=scope.video_ids,
            query=scope.text,
            mode=OVERVIEW,
            **kw,
        )

    try:
        parsed = Answer.model_validate_json(raw)
    except ValidationError as exc:
        return run(answer=None, error=_terse(exc))
    except ValueError as exc:
        return run(answer=None, error=f"not valid JSON: {exc}")

    with meter.stage("answer.ground"):
        grounded, repairs = ground(parsed, hits)
    return run(answer=grounded, repairs=repairs)


def build_messages(
    question: str,
    hits: list[RetrievedChunk],
    cfg: Config,
    *,
    prompt: Path | None = None,
    context: str | None = None,
) -> tuple[str, str]:
    """The two messages, from the prompt file named in config. No prompt text lives here.

    `prompt` and `context` are the two seams the overview path needs (`src.overview`): a
    different prompt file, and a context that is a whole video rather than a list of
    retrieved passages. Both default to the extractive behaviour, so every existing caller
    is unchanged and there is still one function that turns a prompt file into two messages.
    """
    system, template = load_prompt(Path(prompt or cfg.get("answer.prompt")))
    body = render_context(hits) if context is None else context
    user = template.replace("{{context}}", body).replace("{{question}}", question)
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


def effective_model(cfg: Config) -> str:
    """The HF repo id the configured arm will actually run.

    `answer.model` is the hosted id and `answer.ollama_model` the GGUF one; they are two
    different models with two different measured numbers (see the table in the README), so
    reporting the first while running the second is not a cosmetic slip — it labels a run
    with a number that was never measured on it. Every place that prints or serves the
    answer model goes through here rather than reading `answer.model` directly.

    This is what the arm is *configured* to run. What actually ran is `AnswerRun.model`,
    which differs on a 429 fallback.
    """
    arm = str(cfg.get("answer.arm")).strip().lower()
    if arm == "ollama":
        return str(cfg.get("answer.ollama_model"))
    return str(cfg.get("answer.model"))


def _ask(
    system: str,
    user: str,
    cfg: Config,
    meter: Meter,
    *,
    schema: dict | None = None,
    schema_name: str = "answer",
    max_tokens: int | None = None,
) -> tuple[str, int, str]:
    """Route to the configured arm.

    Returns the raw reply text, the tokens it cost, and the HF repo id of the model that
    produced it — the third is not always the configured one, because the groq arm falls
    back to the local model on a 429.

    `schema` is what constrains generation, and it is a parameter rather than a constant so
    that `src.overview` can ask for a `schemas.overview.Overview` through this same door.
    Everything a caller cares about on the way — the arm choice, the 429 fallback to the
    local model, the metering — is written once and applies to both shapes. Default is the
    `Answer` schema, so every existing caller is unchanged.
    """
    arm = str(cfg.get("answer.arm")).strip().lower()
    model = cfg.get("answer.model")
    temperature = float(cfg.get("answer.temperature"))
    cap = int(cfg.get("answer.max_tokens")) if max_tokens is None else int(max_tokens)
    shape = json_schema() if schema is None else schema

    if arm == "groq":
        ollama_model = str(cfg.get("answer.ollama_model"))
        try:
            text, tokens = _groq_arm(
                system, user, model, temperature, cap, meter, shape, schema_name,
                str(cfg.get("answer.reasoning_effort")),
            )
            return text, tokens, str(model)
        except _GroqRequestTooLargeError:
            # Never falls back, whatever answer.fallback says. The local arm would happily
            # answer, and the caller would record that answer as this run's — which is
            # exactly how a 3B model's output came to be read as an overview prompt that
            # did not work. A request that cannot fit is a bug in what was sent, and it has
            # to reach a human rather than be papered over with a smaller model.
            raise
        except _GroqRateLimitError:
            if not bool(cfg.get("answer.fallback")):
                raise
            print(
                f"WARNING: Groq rate limit hit — falling back to ollama ({ollama_model})",
                file=sys.stderr,
            )
            text, tokens = _ollama_arm(
                system, user, ollama_model, temperature, cap, meter, shape
            )
            return text, tokens, ollama_model
    if arm == "ollama":
        ollama_model = str(cfg.get("answer.ollama_model"))
        text, tokens = _ollama_arm(
            system, user, ollama_model, temperature, cap, meter, shape
        )
        return text, tokens, ollama_model
    raise AnswerError(
        f"config.toml: answer.arm is {arm!r}, which is not an arm. Use 'groq' or 'ollama'."
    )


# How many times _groq_arm will send the same request before giving up on a 429. Three, not
# more: this is a per-minute bucket, so two waits cover the worst honest case (a full minute
# plus a partially-consumed one), and anything past that is a stuck job rather than a busy
# provider. A 413 does not retry at all — see _GroqRequestTooLargeError.
GROQ_ATTEMPTS = 3

# Cap on a single wait, whatever the provider asks for. Groq's retry-after on a 429 has been
# seen at 69 s; a value far past that means the tier is not going to serve this run and
# sleeping on it just hides that behind a hang.
GROQ_MAX_WAIT_S = 120

# What to wait when the response carries no usable retry-after. One minute, because the
# limit being ridden out is per-minute.
GROQ_DEFAULT_WAIT_S = 60


def _groq_retry_after_s(exc: Exception) -> int:
    """Seconds to wait before resending, from the provider's own `retry-after` header.

    Preferred over a fixed backoff because Groq states the refill time exactly, and guessing
    either wastes wall clock or retries into the same wall. Falls back to one minute when the
    header is absent or unparseable, and is capped either way.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    raw = None
    if headers is not None:
        try:
            raw = headers.get("retry-after")
        except Exception:  # noqa: BLE001 - a header bag that will not be read is just absent
            raw = None
    try:
        wait = int(float(raw))
    except (TypeError, ValueError):
        wait = GROQ_DEFAULT_WAIT_S
    return max(1, min(wait, GROQ_MAX_WAIT_S))


def _classify_groq_error(exc: Exception, model: str) -> AnswerError:
    """Sort a provider exception into the three outcomes a caller can act on differently.

    Groq reports *both* of its capacity failures with `code: rate_limit_exceeded`, and they
    want opposite handling — see `_GroqRequestTooLargeError` for the measurement. So the
    HTTP status is read first and the message text only as a fallback, because the status is
    the field that actually separates them:

        413  too large for the tier's per-minute budget. Permanent; do not fall back.
        429  throttled. Transient; the local arm can carry this one call.

    The status comes off the exception rather than out of the string because
    `groq.APIStatusError` carries it, and `"413" in str(exc)` would also match a token count
    that happened to contain those digits.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)

    msg = str(exc)
    lowered = msg.lower()

    if status == 413 or "request too large" in lowered:
        return _GroqRequestTooLargeError(
            f"groq request too large ({model}): {msg}\n"
            f"  This is not throttling — the request exceeds the tier's whole per-minute "
            f"token budget, so retrying and waiting both fail. Send less context (for an "
            f"overview, lower overview.max_context_chars or fold the transcript in "
            f"windows), ask for fewer completion tokens, or run a model with a larger "
            f"allowance."
        )
    if status == 429 or "rate_limit" in lowered or "rate limit" in lowered:
        return _GroqRateLimitError(f"groq rate limit ({model}): {msg}")
    return AnswerError(f"groq arm failed ({model}): {msg}")


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
    schema: dict | None = None,
    schema_name: str = "answer",
    reasoning_effort: str = "",
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

    # Only sent when the config asks for it, because it is not a universal parameter: the
    # gpt-oss models take reasoning_effort, and a model that does not would 400 on it. An
    # empty lever therefore means "send nothing" rather than "send a default".
    #
    # What it is for: gpt-oss thinks before it emits, and those reasoning tokens are charged
    # against max_completion_tokens. On a fold window that produced a perfectly good abstract
    # and six people, the JSON was still cut off before `topics` with the cap at 4000 -
    # strict mode then rejected the whole reply for a missing property. The same window at
    # reasoning_effort="low" finished in 1539 completion tokens. This is the lever that makes
    # the completion cap affordable, and the completion cap is what the TPM budget is spent on.
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    t0 = time.perf_counter()
    for attempt in range(GROQ_ATTEMPTS):
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
                        "name": schema_name,
                        "strict": True,
                        "schema": json_schema() if schema is None else schema,
                    },
                },
                temperature=temperature,
                max_completion_tokens=max_tokens,
                **extra,
            )
            break
        except Exception as exc:
            error = _classify_groq_error(exc, model)
            # Only 429 is worth waiting out, and only while attempts remain. A 413 is
            # permanent (the request cannot fit the budget at all) and anything else is not
            # a capacity problem, so both go straight up.
            if not isinstance(error, _GroqRateLimitError) or attempt == GROQ_ATTEMPTS - 1:
                raise error from exc
            wait = _groq_retry_after_s(exc)
            print(
                f"WARNING: Groq throttled ({model}), waiting {wait}s "
                f"[attempt {attempt + 1}/{GROQ_ATTEMPTS}]",
                file=sys.stderr,
            )
            time.sleep(wait)

    text = (response.choices[0].message.content or "").strip()
    usage = getattr(response, "usage", None)
    tokens = int(getattr(usage, "total_tokens", 0) or 0)
    meter.log(model, time.perf_counter() - t0, tokens=tokens, phase="answer.generate")
    return text, tokens


def _ollama_arm(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
    meter: Meter,
    schema: dict | None = None,
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
            format=json_schema() if schema is None else schema,
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
    meter.log(model, time.perf_counter() - t0, tokens=tokens, phase="answer.generate")
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
    if run.scoped:
        # Printed whenever it applies, because the alternative is an answer that cites
        # one video with no sign on screen that the others were never eligible.
        print(f"   scope: {', '.join('video ' + v for v in run.scope)} only", file=out)
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
            run = answer(" ".join(args.question), cfg, meter)
            report(run)
            _spend(cfg, meter, run.model)
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


def _spend(cfg: Config, meter: Meter, used: str = "") -> None:
    # `used` is the model that actually answered, which the caller can name only when there
    # was a single run. The dev sweep has fifteen, so it falls back to the model the
    # configured arm runs — the same string unless a 429 sent one of them elsewhere.
    calls = meter._calls
    print(
        f"\n{len(calls)} model call(s), {sum(c.latency_s for c in calls):.2f}s, "
        f"${sum(c.cost_usd for c in calls):.4f}  "
        f"(answer.arm={cfg.get('answer.arm')} {used or effective_model(cfg)})"
    )


if __name__ == "__main__":
    raise SystemExit(main())
