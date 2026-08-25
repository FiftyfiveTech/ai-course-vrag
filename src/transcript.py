"""Transcript — VRAG-008.

Timestamped transcript, two arms behind one interface.

Both arms accept a 16 kHz mono WAV file (as written by ingest) and return a
list of Segment objects.  Each Segment carries t_start, t_end, and text — the
same fields the chunker (VRAG-014) and retriever (VRAG-016) will key on.

Select the arm in config.toml:

    [transcript]
    arm      = "groq"                        # or "ollama"
    model    = "openai/whisper-large-v3-turbo"
    language = "en"                          # "" to auto-detect

Arm notes
---------
groq:   Uses the Groq Python SDK.  Requires GROQ_API_KEY.  The model name is
        the HF repo id with the owner stripped ("whisper-large-v3-turbo").
        Groq returns verbose_json with per-segment timestamps automatically.

ollama: Uses the Ollama Python SDK.  No API key required.  The model must be
        pulled first:
            ollama pull hf.co/openai/whisper-large-v3-turbo
        Ollama's transcribe() endpoint returns segments with start/end times.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.config import Config, ConfigError
from src.telemetry import Meter


class TranscriptError(Exception):
    """Transcription failed — message says which arm and why."""


@dataclass(frozen=True)
class Segment:
    """One timestamped unit of speech."""

    t_start: float  # seconds from video start
    t_end: float
    text: str


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def transcribe(wav: Path, cfg: Config, meter: Meter) -> list[Segment]:
    """Transcribe wav using the arm configured in config.toml.

    Returns segments ordered by t_start.  Empty list if the audio has no
    speech.  Raises TranscriptError on any failure so the caller can log and
    decide whether to retry or skip.
    """
    wav = Path(wav)
    if not wav.is_file():
        raise TranscriptError(f"{wav}: not a file")

    arm = cfg.get("transcript.arm")
    model = cfg.get("transcript.model")
    language = cfg.get("transcript.language") or None  # "" → None (auto-detect)

    if arm == "groq":
        return _groq_arm(wav, model, language, meter)
    if arm == "ollama":
        return _ollama_arm(wav, model, language, meter)
    raise ConfigError(
        f"config.toml: transcript.arm={arm!r} is not recognised. "
        f"Use 'groq' or 'ollama'."
    )


# ---------------------------------------------------------------------------
# Groq arm
# ---------------------------------------------------------------------------


def _groq_model_name(hf_repo_id: str) -> str:
    """Strip the owner prefix: 'openai/whisper-large-v3-turbo' → 'whisper-large-v3-turbo'."""
    return hf_repo_id.split("/")[-1]


def _parse_groq_segments(response) -> list[Segment]:
    """Turn a Groq verbose_json transcription response into Segments.

    The Groq SDK returns an object whose .segments attribute is a list of
    objects with .start, .end, and .text.  Handles both object-style and
    dict-style responses so tests can pass plain dicts.
    """
    raw = getattr(response, "segments", None)
    if raw is None:
        return []
    segments = []
    for s in raw:
        if isinstance(s, dict):
            start, end, text = s.get("start", 0.0), s.get("end", 0.0), s.get("text", "")
        else:
            start, end, text = s.start, s.end, s.text
        text = text.strip()
        if text:
            segments.append(Segment(t_start=float(start), t_end=float(end), text=text))
    return segments


def _groq_arm(
    wav: Path, model: str, language: str | None, meter: Meter
) -> list[Segment]:
    try:
        from groq import Groq
    except ImportError as exc:
        raise TranscriptError(
            "groq package not installed — run `uv sync` to install it"
        ) from exc

    api_key = os.environ.get("GROQ_API_KEY") or _read_env_key("GROQ_API_KEY")
    if not api_key:
        raise TranscriptError(
            "GROQ_API_KEY is not set. Add it to ~/.config/ai-course-vrag.env "
            "or the process environment."
        )

    client = Groq(api_key=api_key)
    groq_model = _groq_model_name(model)

    try:
        with wav.open("rb") as fh:
            audio_s = _wav_duration_s(wav)
            with meter.span(model, audio_s=audio_s):
                kwargs: dict = {
                    "file": (wav.name, fh, "audio/wav"),
                    "model": groq_model,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment"],
                }
                if language:
                    kwargs["language"] = language
                response = client.audio.transcriptions.create(**kwargs)
    except Exception as exc:
        raise TranscriptError(f"groq arm failed for {wav}: {exc}") from exc

    return _parse_groq_segments(response)


# ---------------------------------------------------------------------------
# Ollama arm
# ---------------------------------------------------------------------------


def _hf_to_ollama_tag(hf_repo_id: str) -> str:
    """Convert HF repo id to the Ollama tag used after `ollama pull hf.co/<repo>`.

    'openai/whisper-large-v3-turbo' → 'hf.co/openai/whisper-large-v3-turbo'
    """
    if hf_repo_id.startswith("hf.co/"):
        return hf_repo_id
    return f"hf.co/{hf_repo_id}"


def _parse_ollama_segments(response) -> list[Segment]:
    """Turn an Ollama transcription response into Segments.

    Ollama returns a dict (or object) whose 'segments' key holds a list of
    dicts with 'start', 'end', and 'text'.
    """
    if isinstance(response, dict):
        raw = response.get("segments", [])
    else:
        raw = getattr(response, "segments", []) or []

    segments = []
    for s in raw:
        if isinstance(s, dict):
            start, end, text = s.get("start", 0.0), s.get("end", 0.0), s.get("text", "")
        else:
            start, end, text = s.start, s.end, s.text
        text = text.strip()
        if text:
            segments.append(Segment(t_start=float(start), t_end=float(end), text=text))
    return segments


def _ollama_arm(
    wav: Path, model: str, language: str | None, meter: Meter
) -> list[Segment]:
    try:
        import ollama
    except ImportError as exc:
        raise TranscriptError(
            "ollama package not installed — run `uv sync` to install it"
        ) from exc

    tag = _hf_to_ollama_tag(model)
    audio_s = _wav_duration_s(wav)

    try:
        with meter.span(model, audio_s=audio_s):
            kwargs: dict = {"model": tag, "file": str(wav)}
            if language:
                kwargs["language"] = language
            response = ollama.transcribe(**kwargs)
    except Exception as exc:
        raise TranscriptError(
            f"ollama arm failed for {wav} with model {tag!r}: {exc}\n"
            f"Make sure the model is pulled: ollama pull {tag}"
        ) from exc

    return _parse_ollama_segments(response)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wav_duration_s(wav: Path) -> float:
    """Duration of a WAV file in seconds from its header — no ffprobe needed."""
    import struct
    import wave

    try:
        with wave.open(str(wav)) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate if rate else 0.0
    except Exception:
        # Fall back to file size heuristic for 16-bit mono 16 kHz (ingest standard).
        size = wav.stat().st_size - 44  # subtract WAV header
        return max(0.0, size / (16000 * 1 * 2))


def _read_env_key(key: str) -> str:
    """Read a key from the project env files (same search order as src.env)."""
    from src.env import load_env
    env = load_env()
    value, _ = env.get(key, ("", ""))
    return value
