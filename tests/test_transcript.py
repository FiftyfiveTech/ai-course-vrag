"""Tests for src/transcript.py — VRAG-008.

No real API calls.  The arm functions are tested by injecting fake responses
through the parse helpers, and dispatch is tested by patching the private arm
functions.
"""

from __future__ import annotations

import wave
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.transcript import (
    Segment,
    TranscriptError,
    _groq_model_name,
    _hf_to_ollama_tag,
    _parse_groq_segments,
    _parse_ollama_segments,
    _wav_duration_s,
    transcribe,
)
from src.telemetry import Meter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def meter():
    return Meter()


@pytest.fixture()
def wav_file(tmp_path) -> Path:
    """A minimal valid 16 kHz mono WAV file with 1 second of silence."""
    path = tmp_path / "audio.wav"
    sample_rate = 16000
    n_frames = sample_rate  # 1 second
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return path


@pytest.fixture()
def cfg_groq(tmp_path):
    """Config pointing to groq arm."""
    from src.config import load
    p = tmp_path / "config.toml"
    p.write_text(
        '[transcript]\narm = "groq"\nmodel = "openai/whisper-large-v3-turbo"\nlanguage = "en"\n'
    )
    return load(p)


@pytest.fixture()
def cfg_ollama(tmp_path):
    """Config pointing to ollama arm."""
    from src.config import load
    p = tmp_path / "config.toml"
    p.write_text(
        '[transcript]\narm = "ollama"\nmodel = "openai/whisper-large-v3-turbo"\nlanguage = "en"\n'
    )
    return load(p)


@pytest.fixture()
def cfg_bad_arm(tmp_path):
    """Config with an unknown arm."""
    from src.config import load
    p = tmp_path / "config.toml"
    p.write_text('[transcript]\narm = "unknown"\nmodel = "openai/whisper-large-v3-turbo"\nlanguage = ""\n')
    return load(p)


# ---------------------------------------------------------------------------
# Model name helpers
# ---------------------------------------------------------------------------


def test_groq_model_name_strips_owner():
    assert _groq_model_name("openai/whisper-large-v3-turbo") == "whisper-large-v3-turbo"


def test_groq_model_name_no_owner():
    assert _groq_model_name("whisper-large-v3-turbo") == "whisper-large-v3-turbo"


def test_hf_to_ollama_tag_adds_prefix():
    assert _hf_to_ollama_tag("openai/whisper-large-v3-turbo") == "hf.co/openai/whisper-large-v3-turbo"


def test_hf_to_ollama_tag_already_prefixed():
    assert _hf_to_ollama_tag("hf.co/openai/whisper-large-v3-turbo") == "hf.co/openai/whisper-large-v3-turbo"


# ---------------------------------------------------------------------------
# _parse_groq_segments
# ---------------------------------------------------------------------------


def test_parse_groq_segments_object_style():
    seg = SimpleNamespace(start=0.0, end=2.5, text=" Hello world")
    response = SimpleNamespace(segments=[seg])
    result = _parse_groq_segments(response)
    assert len(result) == 1
    assert result[0] == Segment(t_start=0.0, t_end=2.5, text="Hello world")


def test_parse_groq_segments_dict_style():
    response = SimpleNamespace(segments=[{"start": 1.0, "end": 3.0, "text": "Hi"}])
    result = _parse_groq_segments(response)
    assert result[0].t_start == 1.0
    assert result[0].text == "Hi"


def test_parse_groq_segments_strips_whitespace():
    response = SimpleNamespace(segments=[{"start": 0.0, "end": 1.0, "text": "  spaced  "}])
    assert _parse_groq_segments(response)[0].text == "spaced"


def test_parse_groq_segments_skips_empty_text():
    response = SimpleNamespace(segments=[
        {"start": 0.0, "end": 1.0, "text": "   "},
        {"start": 1.0, "end": 2.0, "text": "real"},
    ])
    result = _parse_groq_segments(response)
    assert len(result) == 1
    assert result[0].text == "real"


def test_parse_groq_segments_no_segments_attr():
    response = SimpleNamespace()  # no .segments
    assert _parse_groq_segments(response) == []


def test_parse_groq_segments_empty_list():
    response = SimpleNamespace(segments=[])
    assert _parse_groq_segments(response) == []


# ---------------------------------------------------------------------------
# _parse_ollama_segments
# ---------------------------------------------------------------------------


def test_parse_ollama_segments_dict_response():
    response = {"segments": [{"start": 0.0, "end": 1.5, "text": "hello"}]}
    result = _parse_ollama_segments(response)
    assert len(result) == 1
    assert result[0].t_end == 1.5


def test_parse_ollama_segments_object_response():
    seg = SimpleNamespace(start=2.0, end=4.0, text="world")
    response = SimpleNamespace(segments=[seg])
    result = _parse_ollama_segments(response)
    assert result[0].t_start == 2.0


def test_parse_ollama_segments_empty():
    assert _parse_ollama_segments({"segments": []}) == []


def test_parse_ollama_segments_no_key():
    assert _parse_ollama_segments({}) == []


# ---------------------------------------------------------------------------
# _wav_duration_s
# ---------------------------------------------------------------------------


def test_wav_duration_s(wav_file):
    duration = _wav_duration_s(wav_file)
    assert abs(duration - 1.0) < 0.01


# ---------------------------------------------------------------------------
# transcribe() dispatch
# ---------------------------------------------------------------------------


def test_transcribe_raises_on_missing_file(cfg_groq, meter, tmp_path):
    with pytest.raises(TranscriptError, match="not a file"):
        transcribe(tmp_path / "nonexistent.wav", cfg_groq, meter)


def test_transcribe_raises_on_unknown_arm(cfg_bad_arm, meter, wav_file):
    from src.config import ConfigError
    with pytest.raises(ConfigError, match="transcript.arm"):
        transcribe(wav_file, cfg_bad_arm, meter)


def test_transcribe_dispatches_to_groq(cfg_groq, meter, wav_file):
    expected = [Segment(t_start=0.0, t_end=1.0, text="test")]
    with patch("src.transcript._groq_arm", return_value=expected) as mock_arm:
        result = transcribe(wav_file, cfg_groq, meter)
    mock_arm.assert_called_once()
    assert result == expected


def test_transcribe_dispatches_to_ollama(cfg_ollama, meter, wav_file):
    expected = [Segment(t_start=0.0, t_end=1.0, text="local")]
    with patch("src.transcript._ollama_arm", return_value=expected) as mock_arm:
        result = transcribe(wav_file, cfg_ollama, meter)
    mock_arm.assert_called_once()
    assert result == expected


def test_transcribe_passes_model_and_language_to_groq_arm(cfg_groq, meter, wav_file):
    with patch("src.transcript._groq_arm", return_value=[]) as mock_arm:
        transcribe(wav_file, cfg_groq, meter)
    _, model, language, _ = mock_arm.call_args[0]
    assert model == "openai/whisper-large-v3-turbo"
    assert language == "en"


def test_transcribe_converts_empty_language_to_none(tmp_path, meter, wav_file):
    from src.config import load
    p = tmp_path / "c.toml"
    p.write_text('[transcript]\narm = "groq"\nmodel = "openai/whisper-large-v3-turbo"\nlanguage = ""\n')
    cfg = load(p)
    with patch("src.transcript._groq_arm", return_value=[]) as mock_arm:
        transcribe(wav_file, cfg, meter)
    _, _, language, _ = mock_arm.call_args[0]
    assert language is None
