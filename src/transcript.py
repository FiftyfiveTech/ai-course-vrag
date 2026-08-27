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
import tempfile
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
        max_bytes = int(float(cfg.get("transcript.max_upload_mb")) * 1_000_000)
        search_s = float(cfg.get("transcript.split_search_s"))
        return _groq_arm(wav, model, language, meter, max_bytes, search_s)
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


def split_points(
    n_frames: int, frame_rate: int, max_frames: int, search_frames: int, read_window
) -> list[int]:
    """Frame offsets to cut a too-long recording at, first cut first.

    Cuts land no more than `max_frames` apart, so every piece is under the upload cap. Each
    nominal cut is then moved to the quietest 200 ms within +/- `search_frames` of it, because
    a cut through the middle of a word costs that word in both pieces: whisper hears half a
    syllable at the end of one and half at the start of the next, and the segment timings
    either side drift to cover it.

    `read_window(start, count)` returns that many frames of mono 16-bit PCM as a bytes-like
    object. Passed in rather than reading the file here so this is testable without a wav.
    """
    if n_frames <= max_frames:
        return []

    quiet_frames = max(1, int(0.2 * frame_rate))
    cuts: list[int] = []
    pos = 0
    while n_frames - pos > max_frames:
        nominal = pos + max_frames
        # Never search past the cap, or the piece before the cut would exceed it.
        lo = max(pos + quiet_frames, nominal - search_frames)
        hi = min(n_frames - quiet_frames, nominal)
        cut = _quietest_frame(lo, hi, quiet_frames, read_window) if hi > lo else nominal
        # Monotonic and strictly forward, whatever the audio looks like: a cut at or before
        # the previous one would emit an empty piece and loop forever.
        cut = max(cut, pos + 1)
        cuts.append(cut)
        pos = cut
    return cuts


def _quietest_frame(lo: int, hi: int, window: int, read_window) -> int:
    """Start frame of the quietest `window` frames in [lo, hi). Mean |amplitude|.

    One pass: a prefix sum over the search region, then the minimum window sum over it.
    """
    import array

    count = hi - lo + window
    raw = read_window(lo, count)
    samples = array.array("h")
    samples.frombytes(raw[: (len(raw) // 2) * 2])
    if len(samples) < window + 1:
        return lo

    prefix = [0] * (len(samples) + 1)
    for i, v in enumerate(samples):
        prefix[i + 1] = prefix[i] + (v if v >= 0 else -v)

    best_sum, best_at = None, lo
    last = len(samples) - window
    for i in range(last + 1):
        total = prefix[i + window] - prefix[i]
        if best_sum is None or total < best_sum:
            best_sum, best_at = total, lo + i
    return best_at


def split_wav(
    wav: Path, max_bytes: int, search_s: float, out_dir: Path
) -> list[tuple[float, Path]]:
    """Cut wav into pieces under max_bytes. Returns (offset seconds, piece path) pairs.

    Returns a single pair pointing at the original file when it already fits, so the caller
    has one code path either way. The offset is what puts each piece's segment times back on
    the video clock - a piece transcribed on its own reports times from its own zero.
    """
    import wave

    with wave.open(str(wav), "rb") as src:
        params = src.getparams()
        n_frames = src.getnframes()
        frame_rate = src.getframerate()
        frame_bytes = src.getsampwidth() * src.getnchannels()

        header_slack = 4096  # wav header plus multipart overhead on the request
        max_frames = max(1, (max_bytes - header_slack) // frame_bytes)
        if n_frames <= max_frames:
            return [(0.0, wav)]

        def read_window(start: int, count: int) -> bytes:
            src.setpos(min(start, n_frames))
            return src.readframes(min(count, n_frames - min(start, n_frames)))

        searchable = src.getsampwidth() == 2 and src.getnchannels() == 1
        cuts = split_points(
            n_frames,
            frame_rate,
            max_frames,
            int(search_s * frame_rate) if searchable else 0,
            read_window,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        pieces: list[tuple[float, Path]] = []
        bounds = [0, *cuts, n_frames]
        for i in range(len(bounds) - 1):
            start, end = bounds[i], bounds[i + 1]
            src.setpos(start)
            data = src.readframes(end - start)
            path = out_dir / f"{wav.stem}.part{i:03d}.wav"
            with wave.open(str(path), "wb") as dst:
                dst.setnchannels(params.nchannels)
                dst.setsampwidth(params.sampwidth)
                dst.setframerate(params.framerate)
                dst.writeframes(data)
            pieces.append((start / frame_rate, path))
    return pieces


def offset_segments(segments: list[Segment], offset_s: float) -> list[Segment]:
    """Shift a piece's segment times onto the video clock."""
    if not offset_s:
        return segments
    return [
        Segment(t_start=s.t_start + offset_s, t_end=s.t_end + offset_s, text=s.text)
        for s in segments
    ]


def _groq_arm(
    wav: Path,
    model: str,
    language: str | None,
    meter: Meter,
    max_bytes: int,
    search_s: float,
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

    client = _groq_client(api_key)
    groq_model = _groq_model_name(model)

    with tempfile.TemporaryDirectory(prefix="vrag-asr-") as tmp:
        try:
            pieces = split_wav(wav, max_bytes, search_s, Path(tmp))
        except Exception as exc:
            raise TranscriptError(
                f"groq arm could not split {wav} under {max_bytes} bytes: {exc}"
            ) from exc

        if len(pieces) > 1:
            total_s = _wav_duration_s(wav)
            print(
                f"  splitting {wav.name} ({wav.stat().st_size / 1e6:.1f} MB, "
                f"{total_s / 60:.1f} min) into {len(pieces)} piece(s) under "
                f"{max_bytes / 1e6:.0f} MB"
            )

        segments: list[Segment] = []
        for offset_s, piece in pieces:
            raw = _groq_one(client, piece, groq_model, language, model, meter)
            # Bound each piece against its own length, before it is shifted onto the video
            # clock. Per piece and not once at the end, because every piece has a padded
            # final window and only the last one's overrun would show up as running past
            # the video - a middle piece's would land silently on top of the next piece.
            bounded = bound_to_audio(raw, _wav_duration_s(piece), piece.name)
            segments.extend(offset_segments(bounded, offset_s))

    # Pieces are transcribed in order and each is shifted onto the video clock, but the sort
    # is what the contract promises and it costs nothing to keep it true here.
    return sorted(drop_impossible(segments, wav), key=lambda s: s.t_start)


def drop_impossible(segments: list[Segment], wav: Path | None = None) -> list[Segment]:
    """Discard segments whose range does not run forward, and say how many.

    Whisper occasionally reports t_end before t_start, and it does it at the start of an
    audio file - a piece that opens mid-utterance gets a segment starting at 6.2 s and
    ending at 0.0. Splitting a long video makes several such openings instead of one, which
    is how this surfaced: dev video 701 produced

        segment has an impossible time range (t_start=2952.97, t_end=2946.75): 'Eric I don think'

    and the chunker refused the whole video rather than index one bad range. Repairing the
    duration would be inventing it. A segment whose timestamp does not run forward cannot be
    cited, and a citation is the only thing this pipeline produces, so it is dropped and
    counted where somebody can see it.
    """
    good = [s for s in segments if s.t_end >= s.t_start and s.t_start >= 0]
    dropped = len(segments) - len(good)
    if dropped:
        where = f" in {wav.name}" if wav is not None else ""
        print(
            f"  dropped {dropped} of {len(segments)} segment(s){where} whose time range "
            f"does not run forward - unciteable, see transcript.drop_impossible"
        )
    return good


def bound_to_audio(
    segments: list[Segment], duration_s: float, where: str = ""
) -> list[Segment]:
    """Hold segments inside the audio they were transcribed from.

    Whisper pads its last analysis window to a full 30 s and will caption the padding, so
    the final segment of a recording routinely ends after the recording does. On the 90.9
    min client meeting it was:

        5448.136 -> 5478.116  'Thank you.'     29.98 s, 23.5 s of it past the audio

    29.98 s is the tell - one whisper window, near enough exactly - and the audio is
    5454.656 s long. The chunker then refused the whole video, correctly: a chunk inherits
    its range from the segments inside it, so that one segment made a chunk that claimed to
    end 23 s after the video did, and a citation into it would seek a player past the end.

    Clamping is not the same as repairing a duration, which `drop_impossible` refuses to do.
    The end of the audio is a measured fact - it is in media.json, off the wav header - and
    a segment cannot describe sound that was never recorded. So t_end is held to it and the
    text is kept, because whether those words were spoken at 5448 s is not something this
    function knows. A segment that starts at or after the end is a different case: there is
    no audio under any of it, so it is dropped rather than flattened onto the last instant.
    """
    if duration_s <= 0:
        return segments

    kept: list[Segment] = []
    clamped = 0
    dropped = 0
    worst = 0.0
    for s in segments:
        if s.t_start >= duration_s:
            dropped += 1
            worst = max(worst, s.t_end - duration_s)
            continue
        if s.t_end > duration_s:
            clamped += 1
            worst = max(worst, s.t_end - duration_s)
            kept.append(Segment(t_start=s.t_start, t_end=duration_s, text=s.text))
            continue
        kept.append(s)

    # 0.02 s is whisper's timestamp quantum, so a segment landing a tick past the end is
    # rounding and not a hallucination. Clamp it silently; say something about the rest.
    if worst > 0.02 and (clamped or dropped):
        place = f" in {where}" if where else ""
        print(
            f"  bounded {clamped + dropped} segment(s){place} to the {duration_s:.3f}s of "
            f"audio ({clamped} clamped, {dropped} dropped, worst {worst:.2f}s past the end) "
            f"- see transcript.bound_to_audio"
        )
    return kept


def _groq_client(api_key: str):
    """The Groq client. One seam, so a test can drive the arm without a network call."""
    from groq import Groq

    return Groq(api_key=api_key)


def _groq_one(
    client,
    wav: Path,
    groq_model: str,
    language: str | None,
    model: str,
    meter: Meter,
) -> list[Segment]:
    """One upload. Times come back relative to this file's own zero."""
    try:
        with wav.open("rb") as fh:
            audio_s = _wav_duration_s(wav)
            with meter.span(model, audio_s=audio_s, phase="transcript.asr"):
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
        with meter.span(model, audio_s=audio_s, phase="transcript.asr"):
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
