"""Tests for src/answer.py — VRAG-019.

No real Ollama calls.  Internal helpers are tested directly; answer_question()
is tested via patching.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.answer import (
    Answer,
    AnswerError,
    Citation,
    _extract_content,
    _format_context,
    _parse_response,
    answer_question,
)
from src.retrieve import RetrievedChunk
from src.telemetry import Meter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def meter():
    return Meter()


@pytest.fixture()
def cfg(tmp_path):
    from src.config import load

    p = tmp_path / "config.toml"
    p.write_text(
        "[answer]\n"
        'arm = "ollama"\n'
        'model = "meta-llama/Llama-3.2-3B-Instruct"\n'
        "[retrieve]\ntop_k = 5\n"
        "[embed]\n"
        'model = "nomic-ai/nomic-embed-text-v1.5"\n'
        'chroma_path = "./chroma_test"\n'
        'collection = "vrag_test"\n'
        "batch_size = 32\n",
        encoding="utf-8",
    )
    return load(p)


@pytest.fixture()
def prompt_file(tmp_path, monkeypatch):
    """Write a minimal prompt file and patch PROMPT_PATH to point to it."""
    import src.answer as mod

    p = tmp_path / "answer_v1.md"
    p.write_text("You are a helpful assistant. Respond with JSON.", encoding="utf-8")
    monkeypatch.setattr(mod, "PROMPT_PATH", p)
    return p


def _make_chunk(video_id="181", t_start=0.0, t_end=30.0, text="hello", score=0.1):
    return RetrievedChunk(
        video_id=video_id, t_start=t_start, t_end=t_end, text=text, score=score
    )


def _json_response(answer="The answer", citations=None, abstain=False) -> str:
    if citations is None:
        citations = [{"video_id": "181", "t_start": 0.0, "t_end": 30.0}]
    return json.dumps({"answer": answer, "citations": citations, "abstain": abstain})


# ---------------------------------------------------------------------------
# _format_context
# ---------------------------------------------------------------------------


def test_format_context_single_chunk():
    chunks = [_make_chunk(video_id="181", t_start=0.0, t_end=30.0, text="hello world")]
    ctx = _format_context(chunks)
    assert "video_id=181" in ctx
    assert "t_start=0.0s" in ctx
    assert "hello world" in ctx


def test_format_context_multiple_chunks():
    chunks = [
        _make_chunk(video_id="181", t_start=0.0, text="first"),
        _make_chunk(video_id="521", t_start=30.0, text="second"),
    ]
    ctx = _format_context(chunks)
    assert "[1]" in ctx
    assert "[2]" in ctx
    assert "first" in ctx
    assert "second" in ctx


def test_format_context_empty():
    assert _format_context([]) == ""


# ---------------------------------------------------------------------------
# _extract_content
# ---------------------------------------------------------------------------


def test_extract_content_dict_style():
    response = {"message": {"content": "hello"}}
    assert _extract_content(response) == "hello"


def test_extract_content_object_style():
    response = SimpleNamespace(message=SimpleNamespace(content="world"))
    assert _extract_content(response) == "world"


def test_extract_content_object_with_dict_message():
    response = SimpleNamespace(message={"content": "test"})
    assert _extract_content(response) == "test"


def test_extract_content_missing_returns_empty():
    assert _extract_content({}) == ""
    assert _extract_content(SimpleNamespace()) == ""


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


def test_parse_response_basic():
    raw = json.dumps(
        {"answer": "42", "citations": [{"video_id": "181", "t_start": 1.0, "t_end": 5.0}], "abstain": False}
    )
    ans = _parse_response(raw)
    assert ans.answer == "42"
    assert not ans.abstain
    assert len(ans.citations) == 1
    assert ans.citations[0].video_id == "181"
    assert ans.citations[0].t_start == 1.0


def test_parse_response_abstain():
    raw = json.dumps({"answer": "", "citations": [], "abstain": True})
    ans = _parse_response(raw)
    assert ans.abstain is True
    assert ans.citations == []
    assert ans.answer == ""


def test_parse_response_strips_markdown_fence():
    raw = "```json\n" + json.dumps({"answer": "x", "citations": [], "abstain": False}) + "\n```"
    ans = _parse_response(raw)
    assert ans.answer == "x"


def test_parse_response_strips_fence_no_language_tag():
    raw = "```\n" + json.dumps({"answer": "y", "citations": [], "abstain": False}) + "\n```"
    ans = _parse_response(raw)
    assert ans.answer == "y"


def test_parse_response_extracts_json_from_prose():
    payload = {"answer": "z", "citations": [], "abstain": False}
    raw = f"Here is the answer:\n{json.dumps(payload)}\nDone."
    ans = _parse_response(raw)
    assert ans.answer == "z"


def test_parse_response_skips_malformed_citations():
    raw = json.dumps(
        {
            "answer": "test",
            "citations": [
                {"video_id": "181", "t_start": 1.0, "t_end": 5.0},  # good
                {"video_id": "bad"},  # missing t_start/t_end — skipped
            ],
            "abstain": False,
        }
    )
    ans = _parse_response(raw)
    assert len(ans.citations) == 1


def test_parse_response_no_json_raises():
    with pytest.raises(AnswerError, match="no JSON"):
        _parse_response("This is not JSON at all.")


def test_parse_response_bad_json_raises():
    with pytest.raises(AnswerError):
        _parse_response("{broken json")


def test_parse_response_multiple_citations():
    raw = json.dumps(
        {
            "answer": "many",
            "citations": [
                {"video_id": "181", "t_start": 0.0, "t_end": 10.0},
                {"video_id": "521", "t_start": 20.0, "t_end": 30.0},
            ],
            "abstain": False,
        }
    )
    ans = _parse_response(raw)
    assert len(ans.citations) == 2
    assert ans.citations[1].video_id == "521"


# ---------------------------------------------------------------------------
# answer_question — dispatch
# ---------------------------------------------------------------------------


def test_answer_question_returns_answer(cfg, meter, prompt_file):
    fake_response = SimpleNamespace(
        message=SimpleNamespace(content=_json_response("The answer is 42."))
    )
    chunks = [_make_chunk()]
    with patch("src.answer._ollama_arm") as mock_arm:
        mock_arm.return_value = Answer(
            answer="The answer is 42.", citations=[], abstain=False
        )
        result = answer_question("What is the answer?", chunks, cfg, meter)
    assert result.answer == "The answer is 42."
    assert not result.abstain


def test_answer_question_unknown_arm_raises(meter, prompt_file, tmp_path):
    from src.config import load

    # Build a config with an unknown arm.
    p = tmp_path / "config.toml"
    p.write_text(
        "[answer]\narm = \"unknown_arm\"\nmodel = \"fake/model\"\n"
        "[retrieve]\ntop_k = 5\n"
        "[embed]\nmodel = \"nomic-ai/nomic-embed-text-v1.5\"\n"
        "chroma_path = \"./chroma_test\"\ncollection = \"vrag_test\"\nbatch_size = 32\n",
        encoding="utf-8",
    )
    bad_cfg = load(p)

    with pytest.raises(AnswerError, match="unknown answer arm"):
        answer_question("Q", [_make_chunk()], bad_cfg, meter)


def test_answer_question_abstain(cfg, meter, prompt_file):
    chunks = [_make_chunk()]
    with patch("src.answer._ollama_arm") as mock_arm:
        mock_arm.return_value = Answer(answer="", citations=[], abstain=True)
        result = answer_question("Unanswerable?", chunks, cfg, meter)
    assert result.abstain is True
    assert result.citations == []


def test_answer_question_passes_chunks(cfg, meter, prompt_file):
    chunks = [_make_chunk(video_id="521")]
    captured = {}

    def fake_ollama_arm(question, ch, model, prompt, m):
        captured["chunks"] = ch
        return Answer(answer="ok", citations=[], abstain=False)

    with patch("src.answer._ollama_arm", side_effect=fake_ollama_arm):
        answer_question("Q", chunks, cfg, meter)

    assert len(captured["chunks"]) == 1
    assert captured["chunks"][0].video_id == "521"
