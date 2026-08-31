"""Telemetry — VRAG-006.

Shared cost/latency logger.  Every model call in the pipeline goes through here.

Two numbers this module is responsible for:

  $/video-hour   total API cost normalised to one hour of source video
  ×realtime      pipeline wall-time relative to video duration

Usage (one Meter per run, created at pipeline start):

    meter = Meter()
    with meter.span("openai/whisper-large-v3-turbo", audio_s=clip_s):
        result = groq_client.audio.transcriptions.create(...)
    ...
    print(meter.summary_line(video_duration_s=clip_s, wall_s=total_wall_s))
    # → $0.0400/video-hour  48.3×realtime

For a phase that has no model calls (e.g. ingest, which is pure ffmpeg), create
the meter, skip the span() calls, and call summary_line() at the end — cost will
be $0.0000 and ×realtime reflects the ffmpeg wall time.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Cost rates, keyed by HF repo id.
#
# Unit is either "audio_s" (seconds of audio input) or "tokens" (total tokens).
# Rate is USD per unit.  Free-tier and local models carry a rate of 0.0 —
# they are still logged so the $/video-hour line prints consistently and a
# provider swap immediately shows up in the number.
# ---------------------------------------------------------------------------
_UNIT_AUDIO = "audio_s"
_UNIT_TOKENS = "tokens"

RATES: dict[str, tuple[str, float]] = {
    # Groq: billed per second of audio.  $0.04 / audio-hour.
    "openai/whisper-large-v3-turbo": (_UNIT_AUDIO, 0.04 / 3600),
    # Ollama local embedding — no charge. Both ids: config.toml names the -GGUF variant
    # (the only one Ollama can pull, VRAG-017) and the base id stays so a run recorded
    # under the old name still resolves to a rate rather than falling through to the
    # default. RATES.get() defaults to $0.00, so a missing key here is silent — which for
    # a paid model would understate the bill.
    "nomic-ai/nomic-embed-text-v1.5": (_UNIT_TOKENS, 0.0),
    "nomic-ai/nomic-embed-text-v1.5-GGUF": (_UNIT_TOKENS, 0.0),
    "nomic-ai/nomic-embed-text-v1.5-GGUF:F16": (_UNIT_TOKENS, 0.0),
    # Answer generation, VRAG-019. Groq free tier: $0.00 as we run it. The entry exists at
    # 0.0 rather than being left out because RATES.get() defaults to $0.00 silently, and a
    # model that is absent from this table is indistinguishable from a model that is free —
    # so the day this moves to a paid tier, the fix is a number in this row instead of a
    # hunt for which module makes the call.
    "openai/gpt-oss-120b": (_UNIT_TOKENS, 0.0),
    "openai/gpt-oss-20b": (_UNIT_TOKENS, 0.0),
    # The local arm's generator (arm = "ollama"). Local, so $0.00 is the real rate.
    "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M": (_UNIT_TOKENS, 0.0),
    # Keyframe captions, VRAG-023. The hosted arm runs on NVIDIA NIM's free tier and the local
    # arm runs on this laptop, so $0.00 is the real rate for both *as we run them* — the same
    # situation as the two gpt-oss rows above, and the entries exist for the same reason: a
    # model absent from this table is indistinguishable from a model that is free, because
    # RATES.get() defaults to zero silently.
    #
    # Deliberately NOT carrying a modelled paid rate, unlike the whisper row. The whisper
    # figure is $0.04/audio-hour off Groq's published price list; no equivalent per-token price
    # for these two was in front of us when this landed, and inventing one would put a number
    # in a cost table that no command produced. The two-arm table therefore reports tokens,
    # latency and calls — which are measured — and $0.00, which is true.
    "meta-llama/Llama-3.2-11B-Vision-Instruct": (_UNIT_TOKENS, 0.0),
    "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M": (_UNIT_TOKENS, 0.0),
}


@dataclass(frozen=True)
class Call:
    """One logged span: a model invocation, or a stage of the pipeline that made none.

    `model` and `cost_usd` are empty and zero for a `Meter.stage(...)` span — the Chroma
    query, prompt assembly, grounding. Those cost nothing and are why the two new fields
    exist: before them the meter could say a request took 1.44s of model time and could not
    say that the request itself took 3.59s, because 60% of it was in code no span covered.

    `phase` is the attribution key. It is deliberately not the model name: one model can
    serve two phases (the Groq→Ollama fallback runs `answer.generate` twice, on different
    models) and one phase can have no model at all.

    `t_offset` is seconds from the start of the *session*, not of this Meter. The API builds
    one Meter per request, so a Meter-relative offset could not order two requests against
    each other, and ordering is what makes a gap in the timeline visible as a gap.
    """

    model: str
    latency_s: float
    cost_usd: float
    phase: str = ""
    t_offset: float = 0.0


@dataclass
class Meter:
    """Accumulate cost and latency for one pipeline run.

    Create once at the start of a run.  Pass it into every module that makes
    a model call.  Call summary_line() at the end of the run.

    Every span is also appended to the process's session log under `runs/telemetry/`, so
    "which part was slowest last session" is answerable after the process is gone —
    `uv run python -m src.latency`. That is a side effect on purpose: a Meter is created in
    a dozen places and none of them should have to opt in to being measured.
    """

    # _calls stays what it has always been: model invocations, the things that cost money.
    # Every consumer reads it — the four gates, `make ask`'s "N model call(s)" line,
    # summary_line's $/video-hour — so latency-only stages go in their own list rather than
    # widening what "a call" means. Putting them in _calls made `make ask` report "7 model
    # call(s), 3.12s" for a run that made two, which is the sort of quietly wrong number
    # this repo keeps writing rules about.
    _calls: list[Call] = field(default_factory=list)
    _stages: list[Call] = field(default_factory=list)
    _started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def span(
        self,
        model: str,
        *,
        audio_s: float = 0.0,
        tokens: int = 0,
        phase: str = "",
    ) -> Iterator[None]:
        """Time a model call and record its cost.

        with meter.span("openai/whisper-large-v3-turbo", audio_s=clip_s):
            response = groq_client.audio.transcriptions.create(...)
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            latency = time.perf_counter() - t0
            self._record(
                _make_call(
                    model,
                    latency,
                    audio_s=audio_s,
                    tokens=tokens,
                    phase=phase,
                    t_offset=_offset(t0),
                )
            )

    @contextmanager
    def stage(self, phase: str) -> Iterator[None]:
        """Time a stage that makes no model call, so it can be blamed for its own latency.

        with meter.stage("retrieve.query"):
            hits = _query(vector, k, path, name)

        The cost is always $0.00 and that is the point: this is latency-only instrumentation,
        and it exists because the un-instrumented stages were 60% of the first request's wall
        time — 1.35s of Chroma client construction that no number in this repo could see.

        Spans are meant to be flat and non-overlapping. Nesting one stage inside another
        double-counts in the report's percentages, so wrap leaves, not whole pipelines; the
        thing that measures a whole request is `request()`, which is recorded separately for
        exactly that reason.
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            call = Call(
                model="",
                latency_s=time.perf_counter() - t0,
                cost_usd=0.0,
                phase=phase,
                t_offset=_offset(t0),
            )
            self._stages.append(call)
            _write_span(call)

    @contextmanager
    def request(self, what: str) -> Iterator[None]:
        """Bracket one end-to-end unit of work — an HTTP request, one `make ask`.

        Recorded to the session log but NOT into `_calls`: it overlaps every span inside it,
        so counting it as a phase would put a 100%-of-everything row in the report and break
        every percentage under it. What it is for is the difference — request wall time minus
        the spans inside it is the unattributed remainder, which is the only honest way to
        say "and this much of it is in code nothing measures".
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sink().write(
                {
                    "kind": "request",
                    "what": what,
                    "t": round(_offset(t0), 6),
                    "s": round(time.perf_counter() - t0, 6),
                }
            )

    def log(
        self,
        model: str,
        latency_s: float,
        *,
        audio_s: float = 0.0,
        tokens: int = 0,
        phase: str = "",
    ) -> Call:
        """Record a call whose latency was measured externally."""
        call = _make_call(
            model,
            latency_s,
            audio_s=audio_s,
            tokens=tokens,
            phase=phase,
            # The call has already finished, so its start was latency_s ago.
            t_offset=_offset(time.perf_counter() - latency_s),
        )
        self._record(call)
        return call

    def _record(self, call: Call) -> Call:
        self._calls.append(call)
        _write_span(call)
        return call

    def spans(self) -> list[Call]:
        """Every span this meter timed — model calls and stages — in the order they ran."""
        return sorted(self._calls + self._stages, key=lambda c: c.t_offset)

    def by_phase(self) -> dict[str, list[Call]]:
        """This meter's spans grouped by phase, for a caller that wants the in-process split.

        `src.api` uses this to put a per-phase breakdown in its `/ask` response, so a client
        sees where its own second went without reading a log file.
        """
        out: dict[str, list[Call]] = {}
        for call in self.spans():
            out.setdefault(call.phase or "unlabelled", []).append(call)
        return out

    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self._calls)

    @property
    def elapsed_s(self) -> float:
        """Wall time since this Meter was made.

        For the API, where a Meter is built per request, this is the request's duration —
        which is not the same number as the sum of its spans, and the difference is the
        point. `Spend.latency_s` had been serving as both and was understating a 3.59s
        request as 1.44s.
        """
        return time.perf_counter() - self._started

    def summary_line(self, video_duration_s: float, wall_s: float | None = None) -> str:
        """The one line every run must end with.

        Returns a string of the form:
            $0.0400/video-hour  48.3×realtime

        wall_s defaults to elapsed time since the Meter was created.
        """
        if wall_s is None:
            wall_s = time.perf_counter() - self._started

        cost = self.total_cost_usd()
        if video_duration_s > 0:
            cost_per_video_hour = cost / (video_duration_s / 3600)
        else:
            cost_per_video_hour = 0.0

        x_rt = (video_duration_s / wall_s) if wall_s > 0 else 0.0

        return f"${cost_per_video_hour:.4f}/video-hour  {x_rt:.1f}×realtime"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_call(
    model: str,
    latency_s: float,
    *,
    audio_s: float,
    tokens: int,
    phase: str = "",
    t_offset: float = 0.0,
) -> Call:
    unit, rate = RATES.get(model, (_UNIT_AUDIO, 0.0))
    units = audio_s if unit == _UNIT_AUDIO else float(tokens)
    return Call(
        model=model,
        latency_s=latency_s,
        cost_usd=units * rate,
        phase=phase,
        t_offset=t_offset,
    )


# ---------------------------------------------------------------------------
# The session log
#
# One file per process under runs/telemetry/, one JSON object per line. It is what makes
# "which part was slowest last session" answerable *after* the process has exited, which is
# the only time anyone actually asks: `make api` is ended with Ctrl+C and a `make ask` is
# gone the moment it prints.
#
# Written append-as-you-go rather than flushed at exit, and that is the whole design
# decision. An atexit hook does not run when uvicorn is killed, and Ctrl+C is exactly how a
# dev server ends — so a buffered log would be empty for the one session most worth reading.
# A line per span is a handful of syscalls per request against a request that takes seconds.
#
# Not a config.toml lever, deliberately. config.toml holds knobs that change what a run costs
# or how good it is (src/config.py), and a debug log changes neither. It is also read by a
# Meter, which is constructed in a dozen places with no Config in scope — threading one
# through all of them to carry a directory name would be a larger change than this feature.
# The escape hatch is the environment: VRAG_TELEMETRY_LOG=0.
# ---------------------------------------------------------------------------

SESSION_DIR = Path("runs/telemetry")
DISABLE_ENV = "VRAG_TELEMETRY_LOG"

# Captured at import, not when the sink is first written to. The sink is created lazily so
# that importing this module touches no disk — but taking the session's zero point from that
# first write puts it *after* the first span had already finished, so span one reported
# t_offset 0.0 for a 0.686s call and span two claimed to start 2ms in. The timeline then
# showed two overlapping spans that never overlapped. Importing is free; the file is still
# only created on the first write.
_T0 = time.perf_counter()


class _Sink:
    """Appends span records to this process's session file. Never raises at a caller.

    A telemetry write that breaks an answer is strictly worse than no telemetry, so the
    first OSError disables the sink for the rest of the process and says so once on stderr.
    A read-only checkout or a full disk should cost a warning, not a request.
    """

    def __init__(self) -> None:
        self.t0 = _T0
        self.enabled = os.environ.get(DISABLE_ENV, "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        self.session_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        self.path = SESSION_DIR / f"{self.session_id}.jsonl"
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._header_written = False

    def write(self, record: dict) -> None:
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                if not self._header_written:
                    # First line identifies the session, so the reader does not have to
                    # parse the filename to know when it ran or what invoked it.
                    self._header_written = True
                    fh.write(
                        json.dumps(
                            {
                                "kind": "session",
                                "id": self.session_id,
                                "started_at": self.started_at,
                                "argv": sys.argv[:8],
                            }
                        )
                        + "\n"
                    )
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            self.enabled = False
            print(
                f"telemetry: session log disabled ({exc}). Set {DISABLE_ENV}=0 to silence.",
                file=sys.stderr,
            )


_SINK: _Sink | None = None


def _sink() -> _Sink:
    """The one sink for this process. Lazy so that importing telemetry writes no file."""
    global _SINK
    if _SINK is None:
        _SINK = _Sink()
    return _SINK


def _offset(t: float) -> float:
    """A perf_counter reading as seconds since the session started."""
    return max(0.0, t - _T0)


def _write_span(call: Call) -> None:
    """Append one span to the session log. The only shape `src.latency` reads."""
    _sink().write(
        {
            "kind": "span",
            "phase": call.phase,
            "model": call.model,
            "t": round(call.t_offset, 6),
            "s": round(call.latency_s, 6),
            "cost": call.cost_usd,
        }
    )


def reset_sink_for_tests() -> None:
    """Drop the process sink so a test can point SESSION_DIR somewhere else.

    Named for what it is. The sink is process-global on purpose — the API's per-request
    Meters have to land in one session file — and process-global state is exactly what a
    test needs a documented way to clear.
    """
    global _SINK
    _SINK = None
