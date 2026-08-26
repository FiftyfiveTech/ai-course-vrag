"""src/answer.py — VRAG-019. No network, no model, no index.

Everything that talks to a model is behind a seam: the Groq client comes from
`_groq_client`, the arm is chosen by config, and `answer()` calls `retrieve` and `_ask` by
name. So the whole path can be driven with fakes, and what is left to test with a live model
is the one thing a fake cannot tell you — whether the model follows the prompt. That is the
gate's job (`tests/gates/gate_phase2a.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from schemas.answer import ABSTAIN_TEXT, Answer
from src import answer as mod
from src.answer import (
    AnswerError,
    AnswerRun,
    answer,
    build_messages,
    ground,
    load_prompt,
    render_context,
)
from src.config import Config
from src.config import load as load_config
from src.retrieve import RetrievedChunk
from src.telemetry import Meter

PROMPT = Path("prompts/answer_v1.md")


def chunk(video_id="611", t_start=20.0, t_end=45.0, text="Bernini was eight.", score=0.3):
    return RetrievedChunk(
        video_id=video_id, t_start=t_start, t_end=t_end, text=text, score=score
    )


def cfg_from(text: str, tmp_path: Path) -> Config:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


# --------------------------------------------------------------------- the prompt file


def test_the_shipped_prompt_loads():
    system, user = load_prompt(PROMPT)
    assert system.strip()
    assert "{{context}}" in user and "{{question}}" in user


def test_the_shipped_prompt_does_not_leak_its_own_commentary_into_the_messages():
    """prompts/answer_v1.md explains itself to a human. That must not be sent."""
    system, user = load_prompt(PROMPT)
    for message in (system, user):
        assert "## Why it is written this way" not in message
        assert "VRAG-019" not in message


def test_a_prompt_with_no_user_section_is_refused(tmp_path):
    path = tmp_path / "p.md"
    path.write_text("## System\nbe good\n", encoding="utf-8")
    with pytest.raises(AnswerError, match="## User"):
        load_prompt(path)


def test_a_prompt_whose_user_section_forgets_the_question_is_refused(tmp_path):
    """A template missing {{question}} would send context and no question, silently."""
    path = tmp_path / "p.md"
    path.write_text("## System\ns\n\n## User\n{{context}}\n", encoding="utf-8")
    with pytest.raises(AnswerError, match=r"\{\{question\}\}"):
        load_prompt(path)


def test_a_missing_prompt_file_names_itself(tmp_path):
    with pytest.raises(AnswerError, match="cannot read the prompt file"):
        load_prompt(tmp_path / "nope.md")


def test_build_messages_substitutes_both_placeholders(tmp_path):
    prompt = tmp_path / "p.md"
    prompt.write_text("## System\ns\n\n## User\nC:{{context}} Q:{{question}}\n", encoding="utf-8")
    cfg = cfg_from(f'[answer]\nprompt = "{prompt.as_posix()}"\n', tmp_path)
    _, user = build_messages("how old?", [chunk()], cfg)
    assert "Q:how old?" in user
    assert "video_id=611" in user
    assert "{{" not in user


# --------------------------------------------------------------------- rendering context


def test_render_context_prints_the_three_numbers_the_model_is_told_to_copy():
    text = render_context([chunk(t_start=21.84, t_end=47.83)])
    assert "video_id=611" in text
    assert "t_start=21.8" in text
    assert "t_end=47.8" in text


def test_render_context_numbers_the_passages_and_collapses_whitespace():
    text = render_context([chunk(text="a\n\n  b"), chunk(video_id="701")])
    assert text.startswith("[1] ")
    assert "[2] " in text
    assert "a b" in text


def test_render_context_says_so_when_nothing_was_retrieved():
    """An empty context block that just looks empty invites the model to fill it in."""
    assert "no passages" in render_context([])


# --------------------------------------------------------------------- grounding


def test_grounding_leaves_a_faithful_citation_alone():
    hits = [chunk(t_start=21.8, t_end=47.8)]
    ans = Answer(
        answer="Eight.",
        citations=[{"video_id": "611", "t_start": 21.8, "t_end": 47.8}],
        abstain=False,
    )
    out, repairs = ground(ans, hits)
    assert repairs == []
    assert out.citations[0].t_start == 21.8


def test_grounding_snaps_a_drifted_timestamp_onto_the_passage_it_came_from():
    """A shifted timestamp is bad copying, not a new claim — the passage's range is truth."""
    hits = [chunk(t_start=21.8, t_end=47.8)]
    ans = Answer(
        answer="Eight.",
        citations=[{"video_id": "611", "t_start": 24.0, "t_end": 30.0}],
        abstain=False,
    )
    out, repairs = ground(ans, hits)
    assert (out.citations[0].t_start, out.citations[0].t_end) == (21.8, 47.8)
    assert any("snapped" in r for r in repairs)


def test_grounding_picks_the_nearest_passage_from_the_right_video():
    hits = [chunk(t_start=10.0, t_end=35.0), chunk(t_start=600.0, t_end=625.0)]
    ans = Answer(
        answer="x",
        citations=[{"video_id": "611", "t_start": 590.0, "t_end": 599.0}],
        abstain=False,
    )
    out, _ = ground(ans, hits)
    assert out.citations[0].t_start == 600.0


def test_grounding_drops_a_citation_for_a_video_that_was_never_retrieved():
    """The hallucination that validation cannot catch: a well-formed pointer to nowhere."""
    hits = [chunk(video_id="701", t_start=100.0, t_end=125.0)]
    ans = Answer(
        answer="Eight.",
        citations=[
            {"video_id": "611", "t_start": 21.8, "t_end": 47.8},
            {"video_id": "701", "t_start": 100.0, "t_end": 125.0},
        ],
        abstain=False,
    )
    out, repairs = ground(ans, hits)
    assert [c.video_id for c in out.citations] == ["701"]
    assert any("no retrieved passage from video 611" in r for r in repairs)


def test_grounding_forces_an_abstention_when_no_citation_survives():
    hits = [chunk(video_id="701")]
    ans = Answer(
        answer="David Chen.",
        citations=[{"video_id": "791", "t_start": 10.0, "t_end": 12.0}],
        abstain=False,
    )
    out, repairs = ground(ans, hits)
    assert out.abstain is True
    assert out.citations == []
    assert out.answer == ABSTAIN_TEXT
    assert any("forced to abstain" in r for r in repairs)


def test_grounding_forces_an_abstention_when_nothing_was_retrieved_at_all():
    ans = Answer(
        answer="Eight.",
        citations=[{"video_id": "611", "t_start": 21.8, "t_end": 47.8}],
        abstain=False,
    )
    out, repairs = ground(ans, [])
    assert out.abstain is True
    assert any("retrieved: nothing" in r for r in repairs)


def test_grounding_deduplicates_citations_that_snap_onto_the_same_passage():
    hits = [chunk(t_start=21.8, t_end=47.8)]
    ans = Answer(
        answer="x",
        citations=[
            {"video_id": "611", "t_start": 21.8, "t_end": 47.8},
            {"video_id": "611", "t_start": 23.0, "t_end": 40.0},
        ],
        abstain=False,
    )
    out, _ = ground(ans, hits)
    assert len(out.citations) == 1


def test_grounding_leaves_an_abstention_untouched():
    out, repairs = ground(Answer.abstention(), [chunk()])
    assert out.abstain is True and repairs == []


# --------------------------------------------------------------------- the arms


def test_the_groq_wire_name_is_the_hf_repo_id_unchanged():
    """Unlike whisper's. Asserted so a future 'helpful' .split('/') has to break a test."""
    assert mod._groq_wire_name("openai/gpt-oss-120b") == "openai/gpt-oss-120b"


def test_an_unknown_arm_is_refused_by_name(tmp_path):
    cfg = cfg_from(
        '[answer]\narm = "vllm"\nmodel = "m"\nprompt = "p"\ntemperature = 0.0\n'
        "max_tokens = 10\n",
        tmp_path,
    )
    with pytest.raises(AnswerError, match="not an arm"):
        mod._ask("s", "u", cfg, Meter())


def test_a_missing_lever_raises_rather_than_defaulting(tmp_path):
    """config.py's rule: a lever has no default. answer.temperature is a lever."""
    from src.config import ConfigError

    cfg = cfg_from('[answer]\narm = "groq"\nmodel = "m"\nprompt = "p"\n', tmp_path)
    with pytest.raises(ConfigError, match="answer.temperature"):
        mod._ask("s", "u", cfg, Meter())


def test_the_groq_arm_sends_the_strict_schema_and_logs_the_tokens(monkeypatch):
    """Drives the real arm against a fake client — no key, no network."""
    sent = {}

    class FakeCompletions:
        def create(self, **kwargs):
            sent.update(kwargs)
            payload = json.dumps(
                {"answer": "Eight.", "citations": [], "abstain": True}
            )
            return type(
                "R",
                (),
                {
                    "choices": [
                        type("C", (), {"message": type("M", (), {"content": payload})()})()
                    ],
                    "usage": type("U", (), {"total_tokens": 512})(),
                },
            )()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(mod, "_groq_client", lambda key: FakeClient())
    monkeypatch.setattr(mod, "_env_key", lambda key: "gsk_fake")

    meter = Meter()
    text, tokens = mod._groq_arm("sys", "usr", "openai/gpt-oss-120b", 0.0, 1200, meter)

    assert json.loads(text)["abstain"] is True
    assert tokens == 512
    assert sent["model"] == "openai/gpt-oss-120b"
    assert sent["temperature"] == 0.0
    assert sent["max_completion_tokens"] == 1200
    schema = sent["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert set(schema["schema"]["properties"]) == {"answer", "citations", "abstain"}
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    # One entry, carrying both the latency and the volume. CLAUDE.md - every model call goes
    # through the shared logger; two entries per call would read as double the calls in the
    # gate's cost line, which is what the comment on Meter.log in src/answer.py is about.
    assert [c.model for c in meter._calls] == ["openai/gpt-oss-120b"]
    assert meter._calls[0].latency_s >= 0.0


def test_the_groq_arm_says_which_key_is_missing(monkeypatch):
    monkeypatch.setattr(mod, "_env_key", lambda key: "")
    with pytest.raises(AnswerError, match="GROQ_API_KEY is not set"):
        mod._groq_arm("s", "u", "openai/gpt-oss-120b", 0.0, 10, Meter())


def test_a_groq_failure_names_the_model(monkeypatch):
    class Boom:
        chat = type(
            "Chat",
            (),
            {
                "completions": type(
                    "C",
                    (),
                    {"create": lambda self, **kw: (_ for _ in ()).throw(RuntimeError("413"))},
                )()
            },
        )()

    monkeypatch.setattr(mod, "_groq_client", lambda key: Boom())
    monkeypatch.setattr(mod, "_env_key", lambda key: "gsk_fake")
    with pytest.raises(AnswerError, match=r"groq arm failed \(openai/gpt-oss-120b\)"):
        mod._groq_arm("s", "u", "openai/gpt-oss-120b", 0.0, 10, Meter())


# --------------------------------------------------------------------- answer()


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """`answer()` with retrieval and the model replaced. Returns a setter for the reply."""
    prompt = tmp_path / "p.md"
    prompt.write_text(
        "## System\nrules\n\n## User\n{{context}}\n{{question}}\n", encoding="utf-8"
    )
    cfg = cfg_from(
        f'[answer]\narm = "groq"\nmodel = "openai/gpt-oss-120b"\n'
        f'prompt = "{prompt.as_posix()}"\ntemperature = 0.0\nmax_tokens = 100\n',
        tmp_path,
    )
    state = {"hits": [chunk()], "reply": "{}"}

    monkeypatch.setattr(mod, "retrieve", lambda q, c, m: state["hits"])
    monkeypatch.setattr(mod, "_ask", lambda s, u, c, m: (state["reply"], 7))
    return cfg, state


def test_answer_returns_a_grounded_answer(wired):
    cfg, state = wired
    state["reply"] = json.dumps(
        {
            "answer": "Eight.",
            "citations": [{"video_id": "611", "t_start": 22.5, "t_end": 44.0}],
            "abstain": False,
        }
    )
    run = answer("how old?", cfg, Meter())
    assert isinstance(run, AnswerRun)
    assert run.valid and not run.abstained
    assert (run.answer.citations[0].t_start, run.answer.citations[0].t_end) == (20.0, 45.0)
    assert run.tokens == 7


def test_answer_carries_a_schema_failure_instead_of_raising(wired):
    """The gate has to count a bad reply, not be stopped halfway through the dev set."""
    cfg, state = wired
    state["reply"] = json.dumps({"answer": "Eight.", "abstain": False, "confidence": 0.9})
    run = answer("how old?", cfg, Meter())
    assert not run.valid
    assert run.answer is None
    assert "confidence" in run.error
    assert run.raw == state["reply"]


def test_answer_carries_a_non_json_reply_instead_of_raising(wired):
    cfg, state = wired
    state["reply"] = "I think he was eight."
    run = answer("how old?", cfg, Meter())
    assert not run.valid
    assert run.error


def test_answer_reports_an_abstention_as_such(wired):
    cfg, state = wired
    state["reply"] = json.dumps(
        {"answer": "Not covered.", "citations": [], "abstain": True}
    )
    run = answer("what is her name?", cfg, Meter())
    assert run.valid and run.abstained
    assert run.answer.citations == []


def test_answer_turns_a_hallucinated_citation_into_an_abstention(wired):
    """End to end: validation passes, grounding finds the pointer goes nowhere."""
    cfg, state = wired
    state["hits"] = [chunk(video_id="701", t_start=100.0, t_end=125.0)]
    state["reply"] = json.dumps(
        {
            "answer": "David Chen.",
            "citations": [{"video_id": "791", "t_start": 10.0, "t_end": 12.0}],
            "abstain": False,
        }
    )
    run = answer("what is her name?", cfg, Meter())
    assert run.valid and run.abstained
    assert any("forced to abstain" in r for r in run.repairs)


# --------------------------------------------------------------------- config wiring


def test_the_shipped_config_names_an_arm_a_model_and_a_prompt_that_exist():
    cfg = load_config()
    assert cfg.get("answer.arm") in {"groq", "ollama"}
    assert "/" in cfg.get("answer.model"), "models are named by HF repo id (CLAUDE.md)"
    assert Path(cfg.get("answer.prompt")).is_file()
    assert float(cfg.get("answer.temperature")) == 0.0


def test_the_answer_model_has_a_rate_in_the_shared_logger():
    """A model absent from RATES is priced $0.00 silently — same as a free one."""
    from src.telemetry import RATES

    assert load_config().get("answer.model") in RATES
