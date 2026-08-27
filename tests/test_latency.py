"""src/latency.py and the session log — offline, no model, no index, no network.

Two things are under test and they fail in different ways.

**The recorder** (`src/telemetry.py`). It runs on the hot path of every model call in the
pipeline, so the bar is that it cannot break a run and cannot change a number anything else
reports. Both of those were violated by the first version and both have a test here:

* a latency-only `stage()` span landed in `Meter._calls`, and `make ask` then printed
  "7 model call(s), 3.12s" for a run that made two. Four gates and three cost lines read
  `_calls`; widening what "a call" means silently rewrites all of them.
* the session's zero point was taken at the first *write*, which happens at the end of the
  first span — so span one reported `t_offset` 0.0 for a 0.686s call and span two claimed to
  start 2 ms in. The timeline showed an overlap that never happened.

**The reporter** (`src/latency.py`). Its failure mode is a plausible table that misattributes
time, so the tests pin the arithmetic: percentages are of wall time and not of the span total
(otherwise they sum to 100% and the unattributed remainder — the whole reason this exists —
is invisible), and `unattributed` never goes negative just because a process embedded
something at start-up before any request arrived.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest

from src import telemetry
from src.latency import (
    LatencyError,
    PhaseStat,
    Session,
    Span,
    Request,
    _short_model,
    find_session,
    latest_session,
    list_sessions,
    read_session,
    report,
    session_paths,
)
from src.telemetry import Meter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def logging_to(tmp_path, monkeypatch):
    """Turn the session log back on, pointed at tmp_path. Returns the directory."""
    session_dir = tmp_path / "telemetry"
    monkeypatch.setenv(telemetry.DISABLE_ENV, "1")
    monkeypatch.setattr(telemetry, "SESSION_DIR", session_dir)
    telemetry.reset_sink_for_tests()
    yield session_dir
    telemetry.reset_sink_for_tests()


def write_log(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return path


def a_session(
    spans: list[tuple[str, float, float]] | None = None,
    requests: list[tuple[float, float]] | None = None,
    model: str = "",
) -> Session:
    """A Session built by hand: spans as (phase, t, seconds)."""
    return Session(
        id="20260827-000000-1",
        path=Path("x.jsonl"),
        spans=[Span(phase=p, model=model, t=t, s=s) for p, t, s in (spans or [])],
        requests=[Request(what="POST /ask", t=t, s=s) for t, s in (requests or [])],
    )


def rendered(session: Session) -> str:
    buf = io.StringIO()
    report(session, out=buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# The recorder must not change what anything else counts
# ---------------------------------------------------------------------------


def test_a_stage_is_not_a_model_call():
    # The regression: stages in _calls made `make ask` report 7 model calls for a run that
    # made 2, and every gate that prints len(meter._calls) would have inherited it.
    meter = Meter()
    with meter.stage("retrieve.query"):
        pass
    meter.log("openai/gpt-oss-120b", 1.0, tokens=10, phase="answer.generate")
    assert len(meter._calls) == 1
    assert len(meter._stages) == 1
    assert len(meter.spans()) == 2


def test_a_stage_costs_nothing_and_does_not_move_the_bill():
    meter = Meter()
    meter.log("openai/whisper-large-v3-turbo", 1.0, audio_s=3600.0)
    before = meter.total_cost_usd()
    with meter.stage("retrieve.query"):
        pass
    assert meter.total_cost_usd() == before
    assert before > 0


def test_a_stage_records_even_when_the_body_raises():
    meter = Meter()
    with pytest.raises(ValueError):
        with meter.stage("answer.ground"):
            raise ValueError("boom")
    assert [c.phase for c in meter._stages] == ["answer.ground"]


def test_the_phase_label_reaches_the_call():
    meter = Meter()
    with meter.span("m", tokens=1, phase="retrieve.embed"):
        pass
    call = meter.log("m2", 0.5, tokens=1, phase="answer.generate")
    assert meter._calls[0].phase == "retrieve.embed"
    assert call.phase == "answer.generate"


def test_an_unlabelled_call_still_works():
    # Every pre-existing call site was unlabelled before this change; none may break.
    meter = Meter()
    with meter.span("m", tokens=1):
        pass
    assert meter._calls[0].phase == ""
    assert "unlabelled" in meter.by_phase()


def test_spans_do_not_report_a_false_overlap():
    # The t_offset bug: the session clock started at the first write, which is *after* the
    # first span finished, so span 1 got offset 0.0 and span 2 appeared to start before
    # span 1 had ended.
    meter = Meter()
    with meter.stage("first"):
        time.sleep(0.05)
    with meter.stage("second"):
        time.sleep(0.01)
    first, second = meter.spans()
    assert first.phase == "first" and second.phase == "second"
    assert second.t_offset >= first.t_offset + first.latency_s - 0.005


def test_by_phase_groups_both_kinds_of_span():
    meter = Meter()
    with meter.stage("retrieve.query"):
        pass
    meter.log("m", 1.0, tokens=1, phase="answer.generate")
    meter.log("m2", 2.0, tokens=1, phase="answer.generate")
    grouped = meter.by_phase()
    assert set(grouped) == {"retrieve.query", "answer.generate"}
    assert len(grouped["answer.generate"]) == 2


# ---------------------------------------------------------------------------
# The session log
# ---------------------------------------------------------------------------


def test_the_suite_itself_writes_no_session_log():
    """`tests/conftest.py` must have turned the sink off for the whole run.

    Without this, a `make gate` leaves session files behind and `make latency` reports on
    the last pytest instead of the last real run. It regressed once already: the first
    version disabled the sink in an autouse *fixture*, and gate_phase2a's session-scoped
    fixture had already run — and logged — all fifteen questions before any function-scoped
    fixture got a turn.
    """
    import os

    assert os.environ.get(telemetry.DISABLE_ENV) == "0"
    meter = Meter()
    with meter.stage("retrieve.query"):
        pass
    assert not telemetry._sink().enabled
    assert meter._stages, "still recorded in memory — only the file is off"


def test_spans_are_appended_to_a_session_file(logging_to):
    meter = Meter()
    with meter.stage("retrieve.query"):
        pass
    meter.log("openai/gpt-oss-120b", 1.5, tokens=99, phase="answer.generate")

    files = list(logging_to.glob("*.jsonl"))
    assert len(files) == 1
    records = [json.loads(l) for l in files[0].read_text(encoding="utf-8").splitlines()]
    assert records[0]["kind"] == "session"
    assert [r["phase"] for r in records[1:]] == ["retrieve.query", "answer.generate"]


def test_the_log_is_written_as_it_goes_not_at_exit(logging_to):
    # `make api` is ended with Ctrl+C and a killed process runs no atexit hook, so a
    # buffered log would be empty for the one session most worth reading. Proven by reading
    # the file while the "process" is still going.
    meter = Meter()
    with meter.stage("first"):
        pass
    mid = list(logging_to.glob("*.jsonl"))[0].read_text(encoding="utf-8")
    assert "first" in mid
    with meter.stage("second"):
        pass
    assert "second" not in mid  # the earlier read really was mid-session


def test_a_request_bracket_is_recorded_but_is_not_a_phase(logging_to):
    meter = Meter()
    with meter.request("POST /ask"):
        with meter.stage("retrieve.query"):
            pass
    # It overlaps everything inside it, so counting it as a phase would put a
    # 100%-of-everything row in the report.
    assert "request" not in meter.by_phase()
    assert meter._stages and len(meter._calls) == 0
    kinds = [
        json.loads(l)["kind"]
        for l in list(logging_to.glob("*.jsonl"))[0].read_text(encoding="utf-8").splitlines()
    ]
    assert kinds.count("request") == 1


def test_the_env_switch_turns_the_log_off(tmp_path, monkeypatch):
    session_dir = tmp_path / "telemetry"
    monkeypatch.setenv(telemetry.DISABLE_ENV, "0")
    monkeypatch.setattr(telemetry, "SESSION_DIR", session_dir)
    telemetry.reset_sink_for_tests()
    with Meter().stage("retrieve.query"):
        pass
    assert not session_dir.exists()


def test_a_log_that_cannot_be_written_does_not_break_the_run(tmp_path, monkeypatch, capsys):
    # A telemetry write that breaks an answer is strictly worse than no telemetry. The sink
    # disables itself and warns once; the pipeline carries on.
    blocker = tmp_path / "blocked"
    blocker.write_text("I am a file, not a directory", encoding="utf-8")
    monkeypatch.setenv(telemetry.DISABLE_ENV, "1")
    monkeypatch.setattr(telemetry, "SESSION_DIR", blocker / "telemetry")
    telemetry.reset_sink_for_tests()

    meter = Meter()
    with meter.stage("retrieve.query"):  # must not raise
        pass
    meter.log("m", 1.0, tokens=1, phase="answer.generate")

    assert len(meter.spans()) == 2, "the meter still recorded in memory"
    assert "session log disabled" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Reading a session back
# ---------------------------------------------------------------------------


def test_a_round_trip_through_the_log(logging_to):
    meter = Meter()
    with meter.request("POST /ask"):
        with meter.stage("retrieve.query"):
            time.sleep(0.01)
        meter.log("openai/gpt-oss-120b", 0.5, tokens=10, phase="answer.generate")

    session = read_session(list(logging_to.glob("*.jsonl"))[0])
    assert session.requests and session.requests[0].what == "POST /ask"
    phases = {p.phase: p for p in session.phases()}
    assert phases["answer.generate"].total_s == pytest.approx(0.5, abs=0.01)
    assert phases["answer.generate"].model_label == "gpt-oss-120b"
    assert phases["retrieve.query"].model_label == "-"


def test_a_truncated_last_line_is_counted_not_fatal(tmp_path):
    # The log is appended to while a server runs, so reading it live can catch a half-written
    # line. Losing one span is not a reason to refuse to report on the rest.
    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps({"kind": "session", "id": "s", "started_at": "now"})
        + "\n"
        + json.dumps({"kind": "span", "phase": "a", "model": "", "t": 0, "s": 1})
        + '\n{"kind": "span", "phase": "b"',
        encoding="utf-8",
    )
    session = read_session(path)
    assert len(session.spans) == 1
    assert session.bad_lines == 1


def test_sessions_are_ordered_by_start_time_not_by_mtime(tmp_path):
    # A server that ran for an hour has a newer mtime than a `make ask` that started after
    # it. Ordering by mtime would call the server "the last session" for the rest of the day.
    old = write_log(tmp_path / "20260827-090000-1.jsonl", [{"kind": "session", "id": "a"}])
    new = write_log(tmp_path / "20260827-100000-2.jsonl", [{"kind": "session", "id": "b"}])
    old.touch()  # the OLD session now has the NEWEST mtime
    assert [p.name for p in session_paths(tmp_path)] == [new.name, old.name]


def test_the_latest_session_skips_one_that_recorded_nothing(tmp_path):
    # Importing the pipeline is enough to create a sink, so a process that exited before
    # doing work leaves a header and nothing else. "The last session" must not mean that.
    write_log(
        tmp_path / "20260827-090000-1.jsonl",
        [
            {"kind": "session", "id": "a"},
            {"kind": "span", "phase": "answer.generate", "model": "m", "t": 0, "s": 1.0},
        ],
    )
    write_log(tmp_path / "20260827-100000-2.jsonl", [{"kind": "session", "id": "b"}])
    assert latest_session(tmp_path).id == "20260827-090000-1"


def test_no_logs_at_all_says_what_to_run(tmp_path):
    with pytest.raises(LatencyError, match="make ask"):
        latest_session(tmp_path / "nothing")


def test_only_empty_logs_says_so_rather_than_reporting_zero(tmp_path):
    write_log(tmp_path / "20260827-100000-2.jsonl", [{"kind": "session", "id": "b"}])
    with pytest.raises(LatencyError, match="none recorded a span"):
        latest_session(tmp_path)


def test_a_missing_session_id_lists_the_ones_that_exist(tmp_path):
    write_log(tmp_path / "20260827-090000-1.jsonl", [{"kind": "session", "id": "a"}])
    with pytest.raises(LatencyError, match="20260827-090000-1"):
        find_session("nope", tmp_path)


# ---------------------------------------------------------------------------
# The arithmetic — where a plausible table would misattribute time
# ---------------------------------------------------------------------------


def test_phases_are_ranked_slowest_first():
    session = a_session([("fast", 0.0, 0.1), ("slow", 0.2, 2.0), ("mid", 2.3, 0.5)])
    assert [p.phase for p in session.phases()] == ["slow", "mid", "fast"]


def test_repeated_spans_under_one_phase_are_summed_and_averaged():
    session = a_session([("gen", 0.0, 1.0), ("gen", 1.0, 3.0)])
    stat = session.phases()[0]
    assert stat.calls == 2
    assert stat.total_s == pytest.approx(4.0)
    assert stat.mean_s == pytest.approx(2.0)
    assert stat.max_s == pytest.approx(3.0)


def test_wall_time_comes_from_the_request_brackets_when_there_are_any():
    # Two 1s requests with a 10s idle gap between them is 2s of work, not 12s. A server
    # sitting idle overnight must not read as twelve hours of latency.
    session = a_session(
        [("gen", 0.0, 0.9), ("gen", 11.0, 0.9)], requests=[(0.0, 1.0), (11.0, 1.0)]
    )
    assert session.measured_wall_s == pytest.approx(2.0)


def test_without_request_brackets_the_span_timeline_is_the_wall():
    # What a gate run or an ingest has — neither goes through an /ask.
    session = a_session([("asr", 1.0, 2.0), ("asr", 5.0, 1.0)])
    assert session.measured_wall_s == pytest.approx(5.0)  # 1.0 -> 6.0


def test_unattributed_is_wall_minus_the_spans_inside_requests():
    session = a_session([("gen", 0.1, 0.5)], requests=[(0.0, 2.0)])
    assert session.unattributed_s == pytest.approx(1.5)


def test_startup_work_before_any_request_does_not_go_negative():
    # A `make api` embeds at import time and serves its first question later. Charging that
    # warm-up against a request that had not started yet showed a negative remainder.
    session = a_session([("index.embed", 0.0, 5.0), ("gen", 10.1, 0.5)], requests=[(10.0, 1.0)])
    assert session.unattributed_s == pytest.approx(0.5)
    assert session.unattributed_s >= 0.0


def test_percentages_are_of_wall_time_so_the_remainder_stays_visible():
    # Against the span total they would sum to 100% and hide the unattributed row, which is
    # the one thing this report exists to surface.
    session = a_session([("gen", 0.0, 1.0)], requests=[(0.0, 4.0)])
    text = rendered(session)
    assert "25.0%" in text
    assert "unattributed" in text
    assert "75.0%" in text


def test_the_report_names_the_slowest_phase(tmp_path):
    session = a_session([("retrieve.query", 0.0, 1.4), ("answer.generate", 1.4, 1.0)])
    text = rendered(session)
    assert "slowest phase: retrieve.query" in text
    assert text.index("retrieve.query") < text.index("answer.generate")


def test_a_session_with_no_spans_reports_that_rather_than_dividing_by_zero():
    assert "no spans recorded" in rendered(a_session([]))


def test_two_models_under_one_phase_show_as_a_fallback():
    # What a Groq 429 followed by the Ollama retry looks like: one phase, two models.
    stat = PhaseStat.of(
        "answer.generate",
        [
            Span(phase="answer.generate", model="openai/gpt-oss-120b", t=0, s=0.2),
            Span(phase="answer.generate", model="bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M", t=0.2, s=9.0),
        ],
    )
    assert stat.calls == 2
    assert "+" in stat.model_label
    assert "gpt-oss-120b" in stat.model_label


def test_cost_is_carried_through_to_the_report():
    session = Session(
        id="s",
        path=Path("x"),
        spans=[Span(phase="transcript.asr", model="openai/whisper-large-v3-turbo", t=0, s=1, cost=0.04)],
    )
    assert "$0.040000" in rendered(session)


@pytest.mark.parametrize(
    "model, expected",
    [
        ("nomic-ai/nomic-embed-text-v1.5-GGUF:F16", "nomic-embed-text-v1.5-GGUF:F16"),
        ("openai/gpt-oss-120b", "gpt-oss-120b"),
        ("whisper-large-v3-turbo", "whisper-large-v3-turbo"),
        ("", ""),
    ],
)
def test_the_org_prefix_is_dropped_from_the_model_column(model, expected):
    assert _short_model(model) == expected


def test_listing_shows_the_slowest_phase_per_session(tmp_path):
    write_log(
        tmp_path / "20260827-090000-1.jsonl",
        [
            {"kind": "session", "id": "a", "started_at": "now"},
            {"kind": "span", "phase": "retrieve.query", "model": "", "t": 0, "s": 3.0},
            {"kind": "span", "phase": "answer.generate", "model": "m", "t": 3, "s": 1.0},
            {"kind": "request", "what": "POST /ask", "t": 0, "s": 4.5},
        ],
    )
    buf = io.StringIO()
    list_sessions(tmp_path, out=buf)
    text = buf.getvalue()
    assert "20260827-090000-1" in text
    assert "retrieve.query" in text


def test_listing_an_empty_directory_says_so(tmp_path):
    buf = io.StringIO()
    list_sessions(tmp_path / "nothing", out=buf)
    assert "no session logs" in buf.getvalue()


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def test_the_cli_reports_the_latest_session(tmp_path, capsys):
    from src.latency import main

    write_log(
        tmp_path / "20260827-090000-1.jsonl",
        [
            {"kind": "session", "id": "a", "started_at": "now", "argv": ["src/ask.py", "q"]},
            {"kind": "span", "phase": "answer.generate", "model": "m", "t": 0, "s": 2.0},
        ],
    )
    assert main(["--dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "answer.generate" in out
    assert "ask.py q" in out


def test_the_cli_fails_loudly_with_no_logs(tmp_path, capsys):
    from src.latency import main

    assert main(["--dir", str(tmp_path / "nothing")]) == 1
    assert "FAIL" in capsys.readouterr().err
