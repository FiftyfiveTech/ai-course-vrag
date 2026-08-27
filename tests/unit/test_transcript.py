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
    _groq_arm,
    _groq_model_name,
    _hf_to_ollama_tag,
    _parse_groq_segments,
    _parse_ollama_segments,
    _wav_duration_s,
    bound_to_audio,
    drop_impossible,
    offset_segments,
    split_points,
    split_wav,
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
        "max_upload_mb = 20.0\nsplit_search_s = 5.0\n"
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
    _, model, language, _, max_bytes, search_s = mock_arm.call_args[0]
    assert model == "openai/whisper-large-v3-turbo"
    assert language == "en"
    # The two levers reach the arm as bytes and seconds, not as the MB in the file.
    assert max_bytes == 20_000_000
    assert search_s == 5.0


def test_transcribe_converts_empty_language_to_none(tmp_path, meter, wav_file):
    from src.config import load
    p = tmp_path / "c.toml"
    p.write_text(
        '[transcript]\narm = "groq"\nmodel = "openai/whisper-large-v3-turbo"\nlanguage = ""\n'
        "max_upload_mb = 20.0\nsplit_search_s = 5.0\n"
    )
    cfg = load(p)
    with patch("src.transcript._groq_arm", return_value=[]) as mock_arm:
        transcribe(wav_file, cfg, meter)
    language = mock_arm.call_args[0][2]
    assert language is None


# ---------------------------------------------------------------------------
# Splitting oversized audio — VRAG-017
#
# Groq returns 413 above its upload cap, and ingest writes 16 kHz mono s16le at 32 kB/s, so
# the arm tops out near 13 min of video per request. Two of the four dev videos are past it.
# ---------------------------------------------------------------------------


@pytest.fixture()
def groq_key(monkeypatch):
    """The arm checks for a key before it uploads anything. Never the real one in a test."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")


def _wav(path: Path, samples: list[int], rate: int = 16000) -> Path:
    """A mono 16-bit wav holding exactly these samples."""
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path


def _reader(samples: list[int]):
    """read_window(start, count) over an in-memory sample list."""

    def read(start: int, count: int) -> bytes:
        chunk = samples[start : start + count]
        return struct.pack(f"<{len(chunk)}h", *chunk)

    return read


# --- split_points ---------------------------------------------------------------------


def test_split_points_is_empty_when_the_file_already_fits():
    assert split_points(1000, 16000, 2000, 1600, _reader([0] * 1000)) == []


def test_split_points_never_exceeds_the_cap():
    """Every piece must be under max_frames, or the split was pointless."""
    n, cap = 10_000, 3_000
    cuts = split_points(n, 100, cap, 0, _reader([0] * n))
    bounds = [0, *cuts, n]
    sizes = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    assert cuts, "a file 3x the cap has to be cut"
    assert all(s <= cap for s in sizes), sizes


def test_split_points_are_strictly_increasing():
    n = 50_000
    cuts = split_points(n, 100, 7_000, 0, _reader([0] * n))
    assert cuts == sorted(set(cuts))
    assert all(0 < c < n for c in cuts)


def test_split_points_cuts_at_the_quiet_stretch_not_the_nominal_boundary():
    """The whole point of the search window: land the cut in a pause, not mid-word.

    Loud everywhere except a 0.2 s trough placed before the nominal cut. The cut should
    move back onto the trough.
    """
    rate = 100  # 100 frames per "second" keeps the fixture small
    # n under 2x the cap, so exactly one cut is needed and the assertion is unambiguous.
    n = 900
    cap = 500
    quiet_at = 460  # inside the search window, before the nominal cut at 500
    samples = [8000 if i % 2 == 0 else -8000 for i in range(n)]
    for i in range(quiet_at, quiet_at + int(0.2 * rate)):
        samples[i] = 0

    (cut,) = split_points(n, rate, cap, int(1.0 * rate), _reader(samples))
    assert cut == quiet_at, f"cut landed at {cut}, not on the silence at {quiet_at}"


def test_split_points_falls_back_to_the_nominal_cut_without_a_search_window():
    """search_frames=0 (non-mono or non-16-bit audio) must still produce valid cuts."""
    n, cap = 900, 400
    cuts = split_points(n, 100, cap, 0, _reader([5000] * n))
    assert cuts == [400, 800]


def test_split_points_terminates_on_uniform_silence():
    """All-silent audio makes every window equally quiet. The loop still has to end."""
    n, cap = 5_000, 1_000
    cuts = split_points(n, 100, cap, 100, _reader([0] * n))
    bounds = [0, *cuts, n]
    assert all(bounds[i + 1] > bounds[i] for i in range(len(bounds) - 1))
    assert bounds[-1] == n


# --- split_wav -----------------------------------------------------------------------


def test_split_wav_returns_the_original_when_it_fits(tmp_path):
    """One code path for both cases: a small file comes back as a single piece at offset 0."""
    src = _wav(tmp_path / "a.wav", [0] * 16000)
    pieces = split_wav(src, 10_000_000, 5.0, tmp_path / "out")
    assert pieces == [(0.0, src)]


def test_split_wav_pieces_are_each_under_the_cap(tmp_path):
    rate = 16000
    src = _wav(tmp_path / "long.wav", [100] * (rate * 10), rate=rate)  # 10 s, 320 kB
    cap = 100_000
    pieces = split_wav(src, cap, 0.5, tmp_path / "out")
    assert len(pieces) > 1
    for _, path in pieces:
        assert path.stat().st_size <= cap, f"{path.name} is {path.stat().st_size} bytes"


def test_split_wav_offsets_are_the_piece_start_on_the_video_clock(tmp_path):
    """The offset is what puts a piece's segment times back on the video clock."""
    rate = 16000
    src = _wav(tmp_path / "long.wav", [100] * (rate * 10), rate=rate)
    pieces = split_wav(src, 100_000, 0.5, tmp_path / "out")

    assert pieces[0][0] == 0.0
    offsets = [o for o, _ in pieces]
    assert offsets == sorted(offsets)
    # Each offset must equal the total duration of everything before it, or the transcript
    # develops a gap or an overlap at every boundary.
    running = 0.0
    for offset, path in pieces:
        assert abs(offset - running) < 1e-6, f"offset {offset} != {running}"
        with wave.open(str(path), "rb") as wf:
            running += wf.getnframes() / wf.getframerate()


def test_split_wav_loses_no_audio(tmp_path):
    """Total frames across the pieces must equal the source. A dropped piece is dropped
    speech, and nothing downstream would notice."""
    rate = 16000
    n = rate * 10
    src = _wav(tmp_path / "long.wav", [i % 1000 for i in range(n)], rate=rate)
    pieces = split_wav(src, 100_000, 0.5, tmp_path / "out")

    total = 0
    for _, path in pieces:
        with wave.open(str(path), "rb") as wf:
            total += wf.getnframes()
    assert total == n


def test_split_wav_preserves_the_audio_format(tmp_path):
    """Whisper wants 16 kHz mono s16le. A piece written at another rate would transcribe
    at the wrong speed and every timestamp in it would be wrong."""
    rate = 16000
    src = _wav(tmp_path / "long.wav", [100] * (rate * 10), rate=rate)
    for _, path in split_wav(src, 100_000, 0.5, tmp_path / "out"):
        with wave.open(str(path), "rb") as wf:
            assert (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) == (1, 2, rate)


# --- offset_segments -----------------------------------------------------------------


def test_offset_segments_shifts_both_ends():
    got = offset_segments([Segment(t_start=1.0, t_end=2.5, text="x")], 600.0)
    assert got == [Segment(t_start=601.0, t_end=602.5, text="x")]


def test_offset_segments_returns_the_input_unchanged_at_zero():
    segs = [Segment(t_start=1.0, t_end=2.0, text="x")]
    assert offset_segments(segs, 0.0) is segs


def test_offset_segments_of_nothing_is_nothing():
    assert offset_segments([], 42.0) == []


# --- the arm end to end, with a fake Groq client --------------------------------------


def test_groq_arm_stitches_pieces_onto_one_timeline(tmp_path, meter, groq_key):
    """The bug this is here to catch: each piece reports times from its own zero, so
    without the offset every piece's speech lands in the first few minutes of the video."""
    rate = 16000
    src = _wav(tmp_path / "long.wav", [100] * (rate * 10), rate=rate)

    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs["file"][0])
        # Every piece claims speech at 1.0-2.0 s, relative to itself.
        return SimpleNamespace(
            segments=[{"start": 1.0, "end": 2.0, "text": f"piece{len(calls)}"}]
        )

    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.side_effect = fake_create

    with patch("src.transcript._groq_client", return_value=fake_client):
        segments = _groq_arm(
            src, "openai/whisper-large-v3-turbo", "en", meter, 100_000, 0.5
        )

    assert len(calls) > 1, "a 320 kB wav against a 100 kB cap has to be split"
    assert len(segments) == len(calls)
    # Distinct, increasing start times — not every piece stacked at 1.0 s.
    starts = [s.t_start for s in segments]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts), starts
    assert starts[0] == 1.0


def test_groq_arm_meters_every_piece(tmp_path, meter, groq_key):
    """Cost is per second of audio uploaded. Metering only the first piece would report a
    fraction of the real spend on exactly the videos that cost the most."""
    rate = 16000
    src = _wav(tmp_path / "long.wav", [100] * (rate * 10), rate=rate)

    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = SimpleNamespace(segments=[])

    with patch("src.transcript._groq_client", return_value=fake_client):
        _groq_arm(src, "openai/whisper-large-v3-turbo", "en", meter, 100_000, 0.5)

    n_pieces = fake_client.audio.transcriptions.create.call_count
    assert n_pieces > 1
    assert len(meter._calls) == n_pieces
    # 10 s of audio billed, whatever the piece count.
    assert abs(sum(c.cost_usd for c in meter._calls) - 10.0 * (0.04 / 3600)) < 1e-9


def test_groq_arm_does_not_split_a_file_that_fits(tmp_path, meter, groq_key):
    rate = 16000
    src = _wav(tmp_path / "short.wav", [100] * rate, rate=rate)

    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = SimpleNamespace(
        segments=[{"start": 0.0, "end": 1.0, "text": "hi"}]
    )

    with patch("src.transcript._groq_client", return_value=fake_client):
        segments = _groq_arm(
            src, "openai/whisper-large-v3-turbo", "en", meter, 10_000_000, 5.0
        )

    assert fake_client.audio.transcriptions.create.call_count == 1
    assert segments == [Segment(t_start=0.0, t_end=1.0, text="hi")]


# --- bound_to_audio ------------------------------------------------------------------
#
# Whisper pads its last analysis window to a full 30 s and captions the padding, so the
# final segment of a recording routinely ends after the recording does. That is not a
# cosmetic wrong number: a chunk inherits its range from the segments inside it, so one
# such segment made a chunk claiming to end 23 s after the video, chunk.verify() refused
# it, and a 90.9 min video could not be indexed at all.


def test_bound_to_audio_clamps_a_segment_that_outruns_the_recording():
    """The real one: 29.98 s of 'Thank you.' on a 5454.656 s meeting, one whisper window."""
    segs = [
        Segment(t_start=5443.876, t_end=5444.236, text="Bye."),
        Segment(t_start=5448.136, t_end=5478.116, text="Thank you."),
    ]
    out = bound_to_audio(segs, 5454.658)
    assert [s.t_end for s in out] == [5444.236, 5454.658]
    assert out[1].text == "Thank you.", "the text is kept — only the range was impossible"


def test_bound_to_audio_drops_a_segment_that_starts_past_the_end():
    """No audio under any of it. Flattening it onto the last instant would invent a moment."""
    segs = [Segment(t_start=61.0, t_end=90.0, text="hallucinated")]
    assert bound_to_audio(segs, 60.0) == []


def test_bound_to_audio_leaves_segments_inside_the_audio_alone():
    segs = [Segment(t_start=1.0, t_end=2.0, text="a"), Segment(t_start=2.0, t_end=60.0, text="b")]
    assert bound_to_audio(segs, 60.0) == segs


def test_bound_to_audio_is_silent_about_a_rounding_tick(capsys):
    """0.02 s is whisper's timestamp quantum. Clamp it, but do not call it a hallucination."""
    bound_to_audio([Segment(t_start=1.0, t_end=60.01, text="a")], 60.0)
    assert capsys.readouterr().out == ""


def test_bound_to_audio_reports_what_it_changed(capsys):
    bound_to_audio([Segment(t_start=1.0, t_end=90.0, text="a")], 60.0, "piece-9.wav")
    out = capsys.readouterr().out
    assert "piece-9.wav" in out and "30.00s past the end" in out


def test_bound_to_audio_does_nothing_without_a_duration():
    """0.0 means the wav header could not be read — refuse to bound rather than drop it all."""
    segs = [Segment(t_start=1.0, t_end=90.0, text="a")]
    assert bound_to_audio(segs, 0.0) == segs


# --- drop_impossible -----------------------------------------------------------------
#
# Found by running the split arm on dev video 701, not by reading it: whisper reported
# t_end before t_start on a piece that opens mid-utterance, and the chunker refused the
# whole 54-minute video over one segment.


def test_drop_impossible_keeps_forward_ranges():
    segs = [Segment(t_start=0.0, t_end=1.0, text="a"), Segment(t_start=1.0, t_end=3.0, text="b")]
    assert drop_impossible(segs) == segs


def test_drop_impossible_drops_a_backwards_range():
    """The exact shape 701 produced: a segment ending before it starts."""
    bad = Segment(t_start=2952.97, t_end=2946.75, text="Eric I don think")
    good = Segment(t_start=2953.0, t_end=2955.0, text="real speech")
    assert drop_impossible([bad, good]) == [good]


def test_drop_impossible_keeps_a_zero_length_range():
    """t_end == t_start is degenerate but forward, and the chunker accepts it."""
    seg = Segment(t_start=5.0, t_end=5.0, text="x")
    assert drop_impossible([seg]) == [seg]


def test_drop_impossible_drops_a_negative_start():
    seg = Segment(t_start=-1.0, t_end=2.0, text="x")
    assert drop_impossible([seg]) == []


def test_drop_impossible_reports_what_it_dropped(capsys, tmp_path):
    bad = Segment(t_start=10.0, t_end=2.0, text="x")
    drop_impossible([bad], tmp_path / "audio.wav")
    out = capsys.readouterr().out
    assert "dropped 1 of 1" in out and "audio.wav" in out


def test_drop_impossible_is_silent_when_nothing_is_wrong(capsys):
    drop_impossible([Segment(t_start=0.0, t_end=1.0, text="a")])
    assert capsys.readouterr().out == ""


def test_groq_arm_drops_an_impossible_segment_rather_than_failing(tmp_path, meter, groq_key):
    """The arm has to hand the chunker something it will accept, or a whole video is lost."""
    rate = 16000
    # 10 s of audio, because the segments below run to 8 s and the arm now holds segments
    # inside the recording (transcript.bound_to_audio). A 1 s fixture would have them
    # dropped for being past the end, which is a different rule than the one under test.
    src = _wav(tmp_path / "short.wav", [100] * rate * 10, rate=rate)

    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = SimpleNamespace(
        segments=[
            {"start": 6.2, "end": 0.0, "text": "opens mid-utterance"},
            {"start": 6.2, "end": 8.0, "text": "fine"},
        ]
    )

    with patch("src.transcript._groq_client", return_value=fake_client):
        segments = _groq_arm(
            src, "openai/whisper-large-v3-turbo", "en", meter, 10_000_000, 5.0
        )

    assert segments == [Segment(t_start=6.2, t_end=8.0, text="fine")]


def test_groq_arm_returns_segments_in_time_order(tmp_path, meter, groq_key):
    rate = 16000
    # 10 s, for the reason above: the segments below run to 6 s.
    src = _wav(tmp_path / "short.wav", [100] * rate * 10, rate=rate)

    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = SimpleNamespace(
        segments=[
            {"start": 5.0, "end": 6.0, "text": "second"},
            {"start": 1.0, "end": 2.0, "text": "first"},
        ]
    )

    with patch("src.transcript._groq_client", return_value=fake_client):
        segments = _groq_arm(
            src, "openai/whisper-large-v3-turbo", "en", meter, 10_000_000, 5.0
        )

    assert [s.text for s in segments] == ["first", "second"]
