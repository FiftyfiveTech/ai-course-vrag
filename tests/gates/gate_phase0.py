"""GATE Phase 0 — VRAG-010.

One command produces: transcript, frames, metadata, latency, $/video-hour.

    pytest tests/gates/gate_phase0.py -v
    # or
    make gate

The gate generates a synthetic sample video, runs the full Phase 0 pipeline
(ingest → transcript), then asserts all required artefacts exist and prints
the two Phase 0 numbers.

Pass criteria
-------------
- media.json exists and contains duration, frames, audio, timing
- frames/ directory contains at least one frame
- audio.wav exists
- transcript returns without error (segment count ≥ 0 — synthetic audio has no
  real speech so we do not assert segment content)
- x_realtime > 1.0  (pipeline faster than real-time)
- $/video-hour is computed and printed (no upper bound at Phase 0)

The supervisor re-runs this gate after reviewing the PR.  The numbers in the
output — not the ones written in a comment — are what count.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Project root so imports resolve without an editable install.
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load as load_config
from src.ingest import ingest
from src.sample import synthetic as generate_sample
from src.telemetry import Meter
from src.transcript import TranscriptError, transcribe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "config.toml")


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory, cfg):
    """Run ingest + transcript once for the whole module."""
    tmp = tmp_path_factory.mktemp("gate_phase0")
    video = tmp / "gate.mp4"

    # Generate a synthetic test video (same as `make sample`).
    generate_sample(video, cfg)

    meter = Meter()
    result = ingest(video, cfg, out_root=tmp / "runs")

    # Transcript — may return empty list on synthetic audio; that is fine.
    wav = Path(result["audio"]["path"])
    try:
        segments = transcribe(wav, cfg, meter)
    except TranscriptError as exc:
        segments = []
        result["_transcript_error"] = str(exc)

    result["_segments"] = segments
    result["_meter"] = meter
    return result


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------


def test_media_json_exists(pipeline_result):
    manifest = Path(pipeline_result["manifest"])
    assert manifest.is_file(), f"media.json not found at {manifest}"


def test_media_json_has_duration(pipeline_result):
    duration = pipeline_result["media"]["duration_s"]
    assert isinstance(duration, float) and duration > 0, (
        f"duration_s must be a positive float, got {duration!r}"
    )


def test_frames_directory_has_frames(pipeline_result):
    frames_dir = Path(pipeline_result["frames"]["dir"])
    fmt = pipeline_result["frames"]["format"]
    files = list(frames_dir.glob(f"frame_*.{fmt}"))
    assert len(files) > 0, f"no frames found in {frames_dir}"
    print(f"\nframes: {len(files)} × {fmt} in {frames_dir}/")


def test_audio_wav_exists(pipeline_result):
    wav = Path(pipeline_result["audio"]["path"])
    assert wav.is_file(), f"audio.wav not found at {wav}"
    size_mb = wav.stat().st_size / 1e6
    print(f"\naudio: {wav} ({size_mb:.2f} MB)")


def test_transcript_ran_without_crash(pipeline_result):
    err = pipeline_result.get("_transcript_error")
    assert err is None, f"transcript raised TranscriptError: {err}"


def test_transcript_returns_list(pipeline_result):
    segments = pipeline_result["_segments"]
    assert isinstance(segments, list)
    print(f"\ntranscript: {len(segments)} segment(s)")
    for s in segments:
        print(f"  {s.t_start:.1f}s–{s.t_end:.1f}s  {s.text!r}")


def test_x_realtime_above_one(pipeline_result):
    """Pipeline must be faster than real-time."""
    x_rt = pipeline_result["timing"]["x_realtime"]
    print(f"\nx_realtime: {x_rt}")
    assert x_rt is not None and x_rt > 1.0, (
        f"x_realtime={x_rt} — pipeline slower than real-time"
    )


def test_cost_meter_summary_printed(pipeline_result):
    """$/video-hour must be computed and printed — that is the Phase 0 number."""
    meter: Meter = pipeline_result["_meter"]
    duration_s = pipeline_result["media"]["duration_s"]
    wall_s = pipeline_result["timing"]["total_s"]

    # The ingest wall time covers ffmpeg only; add transcript latency.
    transcript_latency = sum(c.latency_s for c in meter._calls)
    total_wall_s = wall_s + transcript_latency

    line = meter.summary_line(duration_s, wall_s=total_wall_s)

    # THE PHASE 0 NUMBER — must appear in output.
    print(f"\n{line}")

    assert "/video-hour" in line
    assert "×realtime" in line


def test_telemetry_line_in_ingest_output(pipeline_result):
    """ingest result must carry the telemetry summary line (VRAG-006 contract)."""
    line = pipeline_result.get("telemetry", "")
    print(f"\ningest telemetry: {line}")
    assert "/video-hour" in line and "×realtime" in line
