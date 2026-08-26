"""Probe the index with a file of plain questions - VRAG-017 follow-on.

    make probe                              # the starter set
    make probe QUESTIONS=my_questions.txt   # your own
    echo "what tools do I need" | uv run python -m src.probe -

Throw unlabelled questions at the index and look at what comes back. One question per
line in a .txt (blank lines and `#` comments skipped), or a .jsonl whose objects carry a
`question` field - the same shape `evals/dev` uses, so a labelled file can be probed
without stripping it first.

This is NOT a gate and NOT a score.
-----------------------------------
It prints no aggregate and computes no recall, deliberately. Scoring needs a ground-truth
`video_id` and `t_ref` per question, which is exactly what this path does not have; the two
things that do score are `src.retrieve.recall_at_k` and `tests/gates/gate_phase1.py`, and
they read `evals/dev`. What this is for is the judgement a number cannot give you: reading
thirty results and noticing that the right passage keeps landing at rank 4, or that a
question about something the corpus never mentions still comes back looking confident.

Each hit prints its `video_id`, time range, cosine distance and a link to the source video
at the cited second, so a claim can be checked by eye rather than by trusting the distance.

Costs nothing and calls nothing hosted: the questions are embedded locally through Ollama
and the index is already on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import Config
from src.config import load as load_config
from src.retrieve import RetrievedChunk, retrieve
from src.telemetry import Meter

MANIFEST = Path("data/corpus/manifest.json")
DEFAULT_QUESTIONS = Path("evals/probe_questions.txt")


class ProbeError(Exception):
    """The questions could not be read - message says which file and why."""


# --------------------------------------------------------------------------- input


def read_questions(path: Path) -> list[str]:
    """Questions from a .txt (one per line) or a .jsonl (a `question` field per object).

    `-` reads stdin, so this composes with anything that can print a question.
    """
    if str(path) == "-":
        text, name = sys.stdin.read(), "<stdin>"
    else:
        if not path.is_file():
            raise ProbeError(
                f"{path}: not a file. One question per line, or a .jsonl with a "
                f"'question' field. `-` reads stdin."
            )
        text, name = path.read_text(encoding="utf-8"), str(path)

    questions = (
        _from_jsonl(text, name) if str(path).endswith(".jsonl") else _from_lines(text)
    )
    if not questions:
        raise ProbeError(f"{name}: no questions in it")
    return questions


def _from_lines(text: str) -> list[str]:
    """Plain text. Blank lines and `#` comments are skipped, so a file can explain itself."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _from_jsonl(text: str, name: str) -> list[str]:
    out = []
    for n, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProbeError(f"{name}:{n}: not valid JSON - {exc}") from exc
        question = (row or {}).get("question")
        if question:
            out.append(str(question))
    return out


def video_urls(manifest: Path = MANIFEST) -> dict[str, str]:
    """video_id -> source url, for links that can be opened and checked.

    Missing or unreadable manifest is not fatal: the hits still print, without links.
    """
    try:
        videos = json.loads(manifest.read_text(encoding="utf-8"))["videos"]
    except (OSError, KeyError, json.JSONDecodeError):
        return {}
    return {str(v["video_id"]): str(v.get("url", "")) for v in videos}


# --------------------------------------------------------------------------- output


def cite(hit: RetrievedChunk, urls: dict[str, str]) -> str:
    """A hit as a line you can act on: where it is, and a link to that second."""
    url = urls.get(hit.video_id, "")
    at = f"  {url}&t={int(hit.t_start)}s" if "youtube.com/watch?v=" in url else ""
    return (
        f"video {hit.video_id:>4}  {hit.t_start:8.1f}s-{hit.t_end:8.1f}s  "
        f"dist={hit.score:.3f}{at}"
    )


def report(question: str, hits: list[RetrievedChunk], urls: dict[str, str], out=None) -> None:
    out = out or sys.stdout
    print(f"\nQ: {question}", file=out)
    if not hits:
        print("   (nothing in the index - run `make index-dev`)", file=out)
        return
    for rank, hit in enumerate(hits, start=1):
        print(f"  {rank}. {cite(hit, urls)}", file=out)
        print(f"     {' '.join(hit.text.split())[:100]}", file=out)


def probe(questions: list[str], cfg: Config, meter: Meter, out=None) -> list[list]:
    """Retrieve for each question and print the hits. Returns the hits per question."""
    urls = video_urls()
    results = []
    for question in questions:
        hits = retrieve(question, cfg, meter)
        report(question, hits, urls, out)
        results.append(hits)
    return results


# --------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "questions",
        nargs="?",
        default=str(DEFAULT_QUESTIONS),
        help="a .txt (one question per line) or .jsonl (a 'question' field); - is stdin",
    )
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args(argv)

    # A distance line rendered on a cp1252 console must not exit non-zero for an encoding
    # accident - src/leakage.py hit exactly that and it read as a failure.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    cfg = load_config(args.config)
    meter = Meter()

    try:
        questions = read_questions(Path(args.questions))
        probe(questions, cfg, meter, sys.stdout)
    except Exception as exc:
        print(f"FAIL - {exc}", file=sys.stderr)
        return 1

    k = int(cfg.get("retrieve.top_k"))
    print(
        f"\n{len(questions)} question(s), top-{k} each, "
        f"{cfg.get('embed.model')}, {sum(c.latency_s for c in meter._calls):.2f}s, "
        f"${sum(c.cost_usd for c in meter._calls):.4f}"
    )
    # Said every run, because the temptation to read a tally off this output is the whole
    # reason the module refuses to print one.
    print("No score here by design - recall is scored on evals/dev by `make gate-phase1`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
