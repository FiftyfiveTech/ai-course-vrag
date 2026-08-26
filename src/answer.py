"""Answer with citations — VRAG-019.

Given a question and retrieved chunks, produce a structured answer with citations,
or abstain when the context does not support one.

Usage:

    from src.answer import answer_question
    from src.config import load
    from src.telemetry import Meter
    from src.retrieve import retrieve

    cfg = load()
    meter = Meter()
    chunks = retrieve("What does the performer place on the table?", cfg, meter)
    ans = answer_question("What does the performer place on the table?", chunks, cfg, meter)
    if ans.abstain:
        print("no answer found")
    else:
        print(ans.answer)
        for c in ans.citations:
            print(f"  video {c.video_id} @ {c.t_start:.1f}s–{c.t_end:.1f}s")

Output schema (QA_SPEC §2):
    {
      "answer":    "<answer text, or empty string when abstaining>",
      "citations": [{"video_id": "<id>", "t_start": <s>, "t_end": <s>}],
      "abstain":   <true|false>
    }

A response is correct when abstain=false AND at least one citation has the right
video_id AND |citation.t_start - t_ref| ≤ 30 s (CITATION_TOLERANCE_S in retrieve.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.config import Config
from src.retrieve import RetrievedChunk
from src.telemetry import Meter

PROMPT_PATH = Path("prompts/answer_v1.md")


class AnswerError(Exception):
    """Answering failed — message says which step and why."""


@dataclass(frozen=True)
class Citation:
    """One chunk cited in an answer."""

    video_id: str
    t_start: float
    t_end: float


@dataclass(frozen=True)
class Answer:
    """The pipeline's response to one question."""

    answer: str
    citations: list  # list[Citation]
    abstain: bool


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def answer_question(
    question: str,
    chunks: Sequence[RetrievedChunk],
    cfg: Config,
    meter: Meter,
) -> Answer:
    """Answer a question using retrieved chunks as context.

    Dispatches on answer.arm in config.  Returns an Answer; raises AnswerError on failure.
    """
    arm = cfg.get("answer.arm")
    model = cfg.get("answer.model")
    system_prompt = _load_prompt()

    if arm == "ollama":
        return _ollama_arm(question, list(chunks), model, system_prompt, meter)
    raise AnswerError(f"unknown answer arm: {arm!r}  — set answer.arm in config.toml")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _load_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise AnswerError(f"cannot read prompt {PROMPT_PATH}: {exc}") from exc


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as numbered blocks for the LLM."""
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] video_id={c.video_id}  t_start={c.t_start:.1f}s  t_end={c.t_end:.1f}s\n"
            f"{c.text}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Ollama arm
# ---------------------------------------------------------------------------


def _ollama_arm(
    question: str,
    chunks: list[RetrievedChunk],
    model: str,
    system_prompt: str,
    meter: Meter,
) -> Answer:
    try:
        import ollama
    except ImportError as exc:
        raise AnswerError("ollama package not installed — run `uv sync`") from exc

    from src.embed import _hf_to_ollama_tag

    ollama_model = _hf_to_ollama_tag(model)
    context = _format_context(chunks)
    user_msg = f"Chunks:\n\n{context}\n\nQuestion: {question}"

    try:
        with meter.span(model, tokens=len(user_msg.split())):
            response = ollama.chat(
                model=ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                options={"temperature": 0.0},
            )
    except Exception as exc:
        raise AnswerError(
            f"ollama chat failed for model {ollama_model!r}: {exc}\n"
            f"Make sure the model is pulled: ollama pull {ollama_model}"
        ) from exc

    content = _extract_content(response)
    return _parse_response(content)


def _extract_content(response) -> str:
    """Pull the text content out of an Ollama chat response (object or dict)."""
    if isinstance(response, dict):
        return response.get("message", {}).get("content", "")
    msg = getattr(response, "message", None)
    if msg is None:
        return ""
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "") or ""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response(text: str) -> Answer:
    """Parse the LLM's JSON output into an Answer.

    Handles markdown code fences the model may wrap around the JSON, and falls
    back to extracting the first {...} block when the model adds commentary.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        end = -1 if lines[-1].strip() == "```" else len(lines)
        cleaned = "\n".join(lines[1:end])

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(cleaned[start:end])
            except json.JSONDecodeError as exc:
                raise AnswerError(
                    f"LLM returned unparseable JSON: {exc}\n{text[:300]}"
                ) from exc
        else:
            raise AnswerError(f"LLM returned no JSON object:\n{text[:300]}")

    abstain = bool(data.get("abstain", False))
    answer_text = str(data.get("answer", ""))
    citations = []
    for c in data.get("citations", []):
        try:
            citations.append(
                Citation(
                    video_id=str(c["video_id"]),
                    t_start=float(c["t_start"]),
                    t_end=float(c["t_end"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue  # skip malformed citations silently

    return Answer(answer=answer_text, citations=citations, abstain=abstain)
