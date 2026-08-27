"""Tests for src/telemetry.py — VRAG-006."""

import time

import pytest

from src.telemetry import Call, Meter, RATES, _make_call


# ---------------------------------------------------------------------------
# RATES sanity
# ---------------------------------------------------------------------------


def test_whisper_rate_is_positive():
    unit, rate = RATES["openai/whisper-large-v3-turbo"]
    assert unit == "audio_s"
    assert rate > 0


def test_nomic_rate_is_zero():
    unit, rate = RATES["nomic-ai/nomic-embed-text-v1.5"]
    assert unit == "tokens"
    assert rate == 0.0


# ---------------------------------------------------------------------------
# _make_call
# ---------------------------------------------------------------------------


def test_make_call_whisper_cost():
    # 3600 s of audio at $0.04/hour should cost exactly $0.04
    call = _make_call("openai/whisper-large-v3-turbo", 1.0, audio_s=3600.0, tokens=0)
    assert abs(call.cost_usd - 0.04) < 1e-9


def test_make_call_unknown_model_zero_cost():
    call = _make_call("unknown/model", 0.5, audio_s=100.0, tokens=0)
    assert call.cost_usd == 0.0


def test_make_call_local_embed_zero_cost():
    call = _make_call("nomic-ai/nomic-embed-text-v1.5", 0.2, audio_s=0.0, tokens=512)
    assert call.cost_usd == 0.0


# ---------------------------------------------------------------------------
# Meter.log
# ---------------------------------------------------------------------------


def test_meter_log_accumulates():
    meter = Meter()
    meter.log("openai/whisper-large-v3-turbo", 1.0, audio_s=3600.0)
    meter.log("openai/whisper-large-v3-turbo", 0.5, audio_s=3600.0)
    assert abs(meter.total_cost_usd() - 0.08) < 1e-9


def test_meter_starts_empty():
    meter = Meter()
    assert meter.total_cost_usd() == 0.0


# ---------------------------------------------------------------------------
# Meter.span
# ---------------------------------------------------------------------------


def test_span_records_call():
    meter = Meter()
    with meter.span("openai/whisper-large-v3-turbo", audio_s=1800.0):
        pass  # no actual API call in tests
    assert len(meter._calls) == 1
    assert abs(meter._calls[0].cost_usd - 0.02) < 1e-9


def test_span_records_latency():
    meter = Meter()
    with meter.span("openai/whisper-large-v3-turbo", audio_s=0.0):
        time.sleep(0.01)
    assert meter._calls[0].latency_s >= 0.01


def test_span_records_call_on_exception():
    meter = Meter()
    with pytest.raises(ValueError):
        with meter.span("openai/whisper-large-v3-turbo", audio_s=60.0):
            raise ValueError("simulated failure")
    # call was still recorded
    assert len(meter._calls) == 1


# ---------------------------------------------------------------------------
# Meter.summary_line
# ---------------------------------------------------------------------------


def test_summary_line_format_zero_cost():
    meter = Meter()
    line = meter.summary_line(video_duration_s=3600.0, wall_s=60.0)
    assert line.startswith("$0.0000/video-hour")
    assert "×realtime" in line


def test_summary_line_xrealtime_value():
    meter = Meter()
    # 60 s of video processed in 2 s → 30×realtime
    line = meter.summary_line(video_duration_s=60.0, wall_s=2.0)
    assert "30.0×realtime" in line


def test_summary_line_cost_per_video_hour():
    meter = Meter()
    # 1 hour of audio → $0.04 cost; 1 hour of video → $0.04/video-hour
    meter.log("openai/whisper-large-v3-turbo", 1.0, audio_s=3600.0)
    line = meter.summary_line(video_duration_s=3600.0, wall_s=10.0)
    assert line.startswith("$0.0400/video-hour")


def test_summary_line_uses_elapsed_when_wall_s_omitted():
    meter = Meter()
    time.sleep(0.02)
    line = meter.summary_line(video_duration_s=1.0)
    # Just check it doesn't blow up and contains the expected tokens
    assert "/video-hour" in line
    assert "×realtime" in line


def test_summary_line_zero_duration_no_crash():
    meter = Meter()
    line = meter.summary_line(video_duration_s=0.0, wall_s=1.0)
    assert "/video-hour" in line


def test_summary_line_zero_wall_no_crash():
    meter = Meter()
    line = meter.summary_line(video_duration_s=30.0, wall_s=0.0)
    assert "/video-hour" in line
