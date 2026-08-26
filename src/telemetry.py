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

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
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
}


@dataclass(frozen=True)
class Call:
    """One logged model invocation."""

    model: str
    latency_s: float
    cost_usd: float


@dataclass
class Meter:
    """Accumulate cost and latency for one pipeline run.

    Create once at the start of a run.  Pass it into every module that makes
    a model call.  Call summary_line() at the end of the run.
    """

    _calls: list[Call] = field(default_factory=list)
    _started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def span(
        self,
        model: str,
        *,
        audio_s: float = 0.0,
        tokens: int = 0,
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
            self._calls.append(_make_call(model, latency, audio_s=audio_s, tokens=tokens))

    def log(
        self,
        model: str,
        latency_s: float,
        *,
        audio_s: float = 0.0,
        tokens: int = 0,
    ) -> Call:
        """Record a call whose latency was measured externally."""
        call = _make_call(model, latency_s, audio_s=audio_s, tokens=tokens)
        self._calls.append(call)
        return call

    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self._calls)

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
    model: str, latency_s: float, *, audio_s: float, tokens: int
) -> Call:
    unit, rate = RATES.get(model, (_UNIT_AUDIO, 0.0))
    units = audio_s if unit == _UNIT_AUDIO else float(tokens)
    return Call(model=model, latency_s=latency_s, cost_usd=units * rate)
