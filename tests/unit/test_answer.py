"""src/answer.py — VRAG-019. No network, no model, no index.

Everything that talks to a model is behind a seam: the Groq client comes from
`_groq_client`, the arm is chosen by config, and `answer()` calls `retrieve` and `_ask` by
name. So the whole path can be driven with fakes, and what is left to test with a live model
is the one thing a fake cannot tell you — whether the model follows the prompt. That is the
gate's job (`tests/gates/gate_phase2a.py`).
"""

from __future__ import annotations

import io
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


# ---------------------------------------------------------------------------
# Which model is reported — effective_model()
# ---------------------------------------------------------------------------
#
# answer.model is the hosted id and answer.ollama_model the GGUF one. They are two different
# models with two different measured numbers (README: 15/15 vs 12/15 schema-valid on dev), so
# reporting the first while running the second labels a run with a number that was never
# measured on it. That is not cosmetic in a repo whose rule is that a number travels with the
# command that produced it, and it is what these three pin.

_ARMS = (
    '[answer]\narm = "{arm}"\nmodel = "openai/gpt-oss-120b"\n'
    'ollama_model = "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"\n'
    'prompt = "p"\ntemperature = 0.0\nmax_tokens = 10\nfallback = true\n'
    'reasoning_effort = ""\n'
)
# The same config with the 429 fallback off - what a gate or a deployed container runs
# with, because a silent substitution attributes one model's number to another.
_ARMS_NO_FALLBACK = _ARMS.replace("fallback = true", "fallback = false")
_LOCAL = "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"


def test_the_reported_model_follows_the_arm(tmp_path):
    other = tmp_path / "b"
    other.mkdir()
    hosted = cfg_from(_ARMS.format(arm="groq"), tmp_path)
    local = cfg_from(_ARMS.format(arm="ollama"), other)
    assert mod.effective_model(hosted) == "openai/gpt-oss-120b"
    assert mod.effective_model(local) == _LOCAL


def test_the_local_arm_reports_the_model_that_actually_answered(tmp_path, monkeypatch):
    """The bug this exists for: the local arm reported the Groq id it never called."""
    cfg = cfg_from(_ARMS.format(arm="ollama"), tmp_path)
    seen = {}

    # `schema` arrived with the overview work: _ask passes the shape it wants generation
    # constrained to, so both the Answer and the Overview contracts go through one door.
    def fake_ollama(system, user, model, temperature, max_tokens, meter, schema=None):
        seen["model"] = model
        return "{}", 3

    monkeypatch.setattr(mod, "_ollama_arm", fake_ollama)
    _, _, used = mod._ask("s", "u", cfg, Meter())

    assert seen["model"] == _LOCAL
    assert used == seen["model"], "reported an arm it did not call"


def test_a_rate_limit_fallback_reports_the_model_that_answered(tmp_path, monkeypatch):
    """The hosted arm plus a 429 runs locally. Naming gpt-oss-120b there would describe a
    call that never completed, and the two models do not score the same."""
    cfg = cfg_from(_ARMS.format(arm="groq"), tmp_path)

    def boom(*a, **k):
        raise mod._GroqRateLimitError("groq rate limit (openai/gpt-oss-120b): 429")

    monkeypatch.setattr(mod, "_groq_arm", boom)
    monkeypatch.setattr(mod, "_ollama_arm", lambda *a, **k: ("{}", 3))
    _, _, used = mod._ask("s", "u", cfg, Meter())

    assert used == _LOCAL


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
    state = {"hits": [chunk()], "reply": "{}", "asked": None, "video_ids": ()}

    # Records the scope it was handed, so a test can assert that an `@` tag reached the
    # store as a filter rather than being left in the question text.
    def fake_retrieve(question, cfg_, meter, video_ids=None):
        state["asked"] = question
        state["video_ids"] = tuple(video_ids or ())
        return state["hits"]

    monkeypatch.setattr(mod, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        mod, "_ask", lambda s, u, c, m: (state["reply"], 7, "openai/gpt-oss-120b")
    )
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


# ---------------------------------------------------------------------------
# `@source` — the tag is parsed here, once, for every entry point
# ---------------------------------------------------------------------------


def _sources():
    from src.mention import Source

    return [
        Source(video_id="611", label="Knowledge / Literature & Art", indexed=True, split="dev"),
        Source(video_id="181", label="Artistic Performance / Stage Play", indexed=True, split="dev"),
    ]


@pytest.fixture
def tagged(wired, monkeypatch):
    """`wired`, with the source catalogue stubbed so no manifest or index is read."""
    monkeypatch.setattr(mod, "parse_scope", _scoped)
    return wired


def _scoped(question, cfg):
    from src.mention import resolve

    return resolve(question, _sources())


def test_a_tagged_question_reaches_retrieval_as_a_filter(tagged):
    cfg, state = tagged
    state["reply"] = json.dumps({"answer": "Eight.", "citations": [], "abstain": True})
    mod.answer("@611 how old?", cfg, Meter())
    assert state["video_ids"] == ("611",)


def test_the_tag_does_not_reach_the_embedder(tagged):
    # `@611` is not a word the corpus ever says, so leaving it in the text moves the query
    # vector for nothing.
    cfg, state = tagged
    state["reply"] = json.dumps({"answer": "x", "citations": [], "abstain": True})
    mod.answer("@611 how old?", cfg, Meter())
    assert state["asked"] == "how old?"


def test_an_untagged_question_is_unfiltered(tagged):
    cfg, state = tagged
    state["reply"] = json.dumps({"answer": "x", "citations": [], "abstain": True})
    mod.answer("how old?", cfg, Meter())
    assert state["video_ids"] == ()
    assert state["asked"] == "how old?"


def test_the_run_records_the_scope_that_was_applied(tagged):
    cfg, state = tagged
    state["reply"] = json.dumps({"answer": "x", "citations": [], "abstain": True})
    run = mod.answer("@611 how old?", cfg, Meter())
    assert run.scope == ("611",)
    assert run.scoped is True
    # The question keeps the tag the person typed; `query` is what the model actually saw.
    assert run.question == "@611 how old?"
    assert run.query == "how old?"


def test_two_tags_scope_to_both(tagged):
    cfg, state = tagged
    state["reply"] = json.dumps({"answer": "x", "citations": [], "abstain": True})
    run = mod.answer("@611 @181 how old?", cfg, Meter())
    assert run.scope == ("611", "181")


def test_a_schema_invalid_reply_still_records_the_scope(tagged):
    # The scope is a property of the request, not of the reply — a run that carried it on
    # the happy path only would lose it in exactly the cases worth reading.
    cfg, state = tagged
    state["reply"] = "not json at all"
    run = mod.answer("@611 how old?", cfg, Meter())
    assert run.valid is False
    assert run.scope == ("611",)


def test_an_unresolvable_tag_raises_rather_than_answering_unscoped(tagged):
    from src.mention import MentionError

    cfg, state = tagged
    with pytest.raises(MentionError):
        mod.answer("@bernini how old?", cfg, Meter())
    # And nothing was asked of the model: the refusal happens before any spend.
    assert state["asked"] is None


def test_the_report_says_what_the_answer_was_scoped_to(tagged):
    cfg, state = tagged
    state["reply"] = json.dumps(
        {
            "answer": "Eight.",
            "citations": [{"video_id": "611", "t_start": 20.0, "t_end": 45.0}],
            "abstain": False,
        }
    )
    run = mod.answer("@611 how old?", cfg, Meter())
    out = io.StringIO()
    mod.report(run, out)
    assert "scope: video 611 only" in out.getvalue()


def test_the_report_says_nothing_about_scope_when_there_was_none(tagged):
    cfg, state = tagged
    state["reply"] = json.dumps({"answer": "x", "citations": [], "abstain": True})
    run = mod.answer("how old?", cfg, Meter())
    out = io.StringIO()
    mod.report(run, out)
    assert "scope:" not in out.getvalue()


# --------------------------------------------------------------- 413 is not 429
#
# Groq reports both capacity failures with code `rate_limit_exceeded`, and the old string
# match on "rate_limit" treated them alike. The cost was specific and silent: every overview
# build ran on the 3B local model, and its output was read as a prompt that did not work.
#
# The real response, free `on_demand` tier, building an overview for video 611:
#   413  "Request too large ... on tokens per minute (TPM): Limit 8000, Requested 17152"
#   x-ratelimit-remaining-tokens: 8000     (a FULL bucket - nothing was throttling it)
#   x-should-retry: false                  (the provider's own verdict)


class _FakeStatusError(Exception):
    """Stands in for groq.APIStatusError, which carries the status as an attribute."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


_TOO_LARGE = (
    "Error code: 413 - {'error': {'message': 'Request too large for model "
    "`openai/gpt-oss-120b` ... on tokens per minute (TPM): Limit 8000, Requested 17152', "
    "'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)
_THROTTLED = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached', "
    "'code': 'rate_limit_exceeded'}}"
)


def test_a_413_is_classified_as_too_large_not_as_throttling():
    """Both carry code `rate_limit_exceeded`; only the status separates them."""
    err = mod._classify_groq_error(_FakeStatusError(413, _TOO_LARGE), "openai/gpt-oss-120b")
    assert isinstance(err, mod._GroqRequestTooLargeError)
    assert not isinstance(err, mod._GroqRateLimitError), "413 must not be retried as a 429"


def test_a_429_is_still_a_rate_limit():
    err = mod._classify_groq_error(_FakeStatusError(429, _THROTTLED), "openai/gpt-oss-120b")
    assert isinstance(err, mod._GroqRateLimitError)


def test_any_other_status_is_a_plain_answer_error():
    err = mod._classify_groq_error(_FakeStatusError(500, "boom"), "m")
    assert type(err) is AnswerError


def test_too_large_never_falls_back_even_with_fallback_enabled(tmp_path, monkeypatch):
    """The whole point. The local arm would answer, and that answer would be recorded as
    this run's — which is how a 3B model's output was read as an overview prompt failure."""
    cfg = cfg_from(_ARMS.format(arm="groq"), tmp_path)  # fallback = true

    def boom(*a, **k):
        raise mod._GroqRequestTooLargeError("groq request too large (m): 413 ...")

    called = []
    monkeypatch.setattr(mod, "_groq_arm", boom)
    monkeypatch.setattr(mod, "_ollama_arm", lambda *a, **k: (called.append(1), ("{}", 3))[1])
    with pytest.raises(mod._GroqRequestTooLargeError):
        mod._ask("s", "u", cfg, Meter())
    assert called == [], "fell back to the local arm on a request that can never fit"


def test_the_message_says_it_is_not_throttling_and_what_to_do():
    """A 413 that reads as 'rate limit' sends the reader off to wait for a bucket that is
    already full. The message has to say so and name the levers that change it."""
    err = mod._classify_groq_error(_FakeStatusError(413, _TOO_LARGE), "openai/gpt-oss-120b")
    text = str(err)
    assert "not throttling" in text
    assert "overview.max_context_chars" in text


def test_fallback_false_re_raises_the_rate_limit_instead_of_substituting(tmp_path, monkeypatch):
    """What a gate or a container runs with: finishing on a different model would attribute
    one model's number to the other."""
    cfg = cfg_from(_ARMS_NO_FALLBACK.format(arm="groq"), tmp_path)

    def boom(*a, **k):
        raise mod._GroqRateLimitError("groq rate limit (openai/gpt-oss-120b): 429")

    called = []
    monkeypatch.setattr(mod, "_groq_arm", boom)
    monkeypatch.setattr(mod, "_ollama_arm", lambda *a, **k: (called.append(1), ("{}", 3))[1])
    with pytest.raises(mod._GroqRateLimitError):
        mod._ask("s", "u", cfg, Meter())
    assert called == []


def test_fallback_true_still_substitutes_on_a_429(tmp_path, monkeypatch):
    """The laptop case, unchanged: the bucket refills, so finishing beats stopping."""
    cfg = cfg_from(_ARMS.format(arm="groq"), tmp_path)

    def boom(*a, **k):
        raise mod._GroqRateLimitError("groq rate limit (openai/gpt-oss-120b): 429")

    monkeypatch.setattr(mod, "_groq_arm", boom)
    monkeypatch.setattr(mod, "_ollama_arm", lambda *a, **k: ("{}", 3))
    _, _, used = mod._ask("s", "u", cfg, Meter())
    assert used == _LOCAL
