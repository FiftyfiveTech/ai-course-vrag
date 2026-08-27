"""Failure paths through ingest — VRAG-009.

`no audio track`, `zero-length`, `unreadable codec`. src/ingest.py has raised on all three
since VRAG-005 and said so in its docstring; what it did not have was a file to raise on.

This file is the **end-to-end** half of the failure coverage, not a replacement for what is
already in test_ingest.py. Those tests — `test_zero_length_media_is_rejected`,
`test_unreadable_container_is_rejected`, `test_extract_audio_refuses_a_silent_file` — drive a
hand-written probe dict, so they pin the branch logic and keep working with no ffmpeg
installed. They cannot tell you that ffprobe actually fails the way those branches assume.
That is what a real broken file is for, and it caught two things a fake dict could not:

* a corrupt input named no file — `probe: ffprobe exited 1 — moov atom not found`. The path
  was in the message only because ffprobe happens to prefix its own stderr with the filename.
* a 0-byte file came back as "moov atom not found", which reads as a corrupt container and
  sends you looking in the wrong place.

Four things are asserted per fixture, because "fails loudly with a useful message" is all
four and any one of them alone is not enough:

1. it raises `IngestError`, not a `CalledProcessError` or a `KeyError` three frames deep;
2. the message contains the phrase that identifies *this* failure, so the reader knows which;
3. the message names the file, which is what matters when nine of ten videos were fine;
4. no partial output survives — no media.json, no frames. A run that half-succeeded is the
   one whose failure gets discovered in Phase 2.

The five fixtures come from `src.sample.BROKEN_KINDS`. Three need no ffmpeg, so most of this
file runs anywhere; the two encoded ones skip without it, and a skip here is not a pass —
`pytest tests/unit -rs` prints the reason and the count is part of the card's evidence.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src import ingest as ing
from src import sample as smp
from src.config import Config
from src.config import load as load_config

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")

# Short: the no-audio fixture is the only one that encodes anything, and 2 s of testsrc2 is
# as good a proof of a missing audio track as 30 s.
FIXTURE = """
[ingest.audio]
sample_rate_hz = 16000
channels = 1
codec = "pcm_s16le"

[ingest.frames]
fps = 0.5
max_frames = 1200
width = 320
format = "jpg"
jpeg_quality = 4

[sample.synthetic]
duration_s = 2
width = 320
height = 240
fps = 25
tone_hz = 440

[sample.real]
max_height = 720
"""

# The phrase that has to be in the message, per kind. Not the whole message — that would pin
# ffprobe's wording, which is not ours to freeze — but enough that a reader can tell the five
# failures apart. `truncated` and `garbage` share ffprobe's error because the .mp4 extension
# makes it choose the mov demuxer for both; they are still different files on disk, and the
# point of having both is that neither reaches media_metadata().
EXPECTED = {
    "no-audio": "no audio stream",
    "zero-duration": "zero-length",
    "empty": "0 bytes",
    "truncated": "moov atom not found",
    "garbage": "moov atom not found",
}

# Which stage is supposed to catch each one. A fixture failing at the wrong stage still
# "fails", and would still pass a test that only checked for IngestError.
#
# `zero-duration` is the one worth reading twice, and this table had it wrong until the test
# below failed: ffprobe *succeeds* on that file — exit 0, probe_score 100, a parseable mov
# container — and reports nb_streams 0 with no duration key at all. So the rejection is
# media_metadata's, not probe's, and the fixture proves something the other four do not: a
# file can get through ffprobe clean and still have nothing in it to ingest.
EXPECTED_STAGE = {
    "no-audio": "audio",
    "zero-duration": "metadata",
    "empty": "preflight",
    "truncated": "probe",
    "garbage": "probe",
}


def make_config(tmp_path: Path) -> Config:
    path = tmp_path / "config.toml"
    path.write_text(FIXTURE, encoding="utf-8")
    return load_config(path)


# --- offline: the fixture builders, with no ffmpeg and no files -----------------------


def test_every_kind_has_a_description():
    """BROKEN_KINDS is the catalogue `make sample-broken` prints. An unnamed fixture is noise."""
    assert set(smp.BROKEN_KINDS) == set(EXPECTED)
    assert all(smp.BROKEN_KINDS[k].strip() for k in smp.BROKEN_KINDS)


def test_every_kind_is_covered_here():
    """Adding a kind without a case in this file has to fail, or the catalogue outgrows the tests."""
    assert set(EXPECTED) == set(smp.BROKEN_KINDS)
    assert set(EXPECTED_STAGE) == set(smp.BROKEN_KINDS)


def test_the_no_audio_argv_differs_from_a_good_clip_only_by_the_audio():
    """The fixture proves nothing if it also changed the picture."""
    spec = {"duration_s": 2, "width": 320, "height": 240, "fps": 25, "tone_hz": 440}
    good = " ".join(smp.synthetic_args(Path("v.mp4"), spec))
    broken = " ".join(smp.broken_args("no-audio", Path("v.mp4"), spec))
    assert "sine" in good and "sine" not in broken
    assert "-c:a" in good and "-c:a" not in broken
    # the video half is untouched
    assert "testsrc2=size=320x240:rate=25" in broken
    assert "libx264" in broken and "yuv420p" in broken


def test_the_zero_duration_argv_asks_for_no_frames():
    spec = {"duration_s": 2, "width": 320, "height": 240, "fps": 25}
    argv = smp.broken_args("zero-duration", Path("v.mp4"), spec)
    assert argv[argv.index("-t") + 1] == "0"


def test_the_byte_written_kinds_are_not_built_by_ffmpeg():
    """Three of five need no encoder, which is why they still run on a machine without one."""
    assert smp.NEEDS_FFMPEG == {"no-audio", "zero-duration"}
    for kind in set(smp.BROKEN_KINDS) - smp.NEEDS_FFMPEG:
        with pytest.raises(smp.SampleError) as exc:
            smp.broken_args(kind, Path("v.mp4"), {})
        assert "byte by byte" in str(exc.value)


def test_an_unknown_kind_lists_the_known_ones(tmp_path):
    with pytest.raises(smp.SampleError) as exc:
        smp.broken("no-such-breakage", tmp_path / "x.mp4", make_config(tmp_path))
    for kind in smp.BROKEN_KINDS:
        assert kind in str(exc.value)


def test_the_truncated_fixture_is_an_mp4_that_stops_early():
    """A real ftyp box so ffprobe commits to the mov demuxer, and no moov anywhere."""
    raw = smp.truncated_bytes()
    assert raw[4:12] == b"ftypisom"
    assert b"moov" not in raw
    assert b"mdat" in raw


def test_the_fixtures_are_deterministic():
    """A fixture that differs between runs makes the test asserting on its message flaky."""
    assert smp.truncated_bytes() == smp.truncated_bytes()

# --- end to end: a real broken file through the real pipeline ------------------------

ALL_KINDS = sorted(smp.BROKEN_KINDS)

# `empty` is caught by ingest()'s own pre-flight before ffprobe is ever invoked, so it is the
# one end-to-end case that runs with no ffmpeg installed. Every other kind reaches ffprobe.
RUNS_WITHOUT_FFMPEG = {"empty"}


@pytest.fixture(scope="module")
def broken(tmp_path_factory) -> tuple[Config, dict[str, Path]]:
    """Build every fixture this machine can build, once."""
    root = tmp_path_factory.mktemp("vrag009")
    cfg = make_config(root)
    built = {}
    for kind in smp.BROKEN_KINDS:
        if kind in smp.NEEDS_FFMPEG and not HAS_FFMPEG:
            continue
        built[kind] = smp.broken(kind, smp.broken_path(kind, root / "broken"), cfg)
    return cfg, built


def _fixture(built: dict[str, Path], kind: str) -> Path:
    if kind not in RUNS_WITHOUT_FFMPEG and not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    return built[kind]


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_ingest_refuses_the_fixture_and_says_which_failure_it_is(kind, broken, tmp_path):
    """(1) IngestError, and (2) the phrase that tells this failure from the other four."""
    cfg, built = broken
    video = _fixture(built, kind)
    with pytest.raises(ing.IngestError) as exc:
        ing.ingest(video, cfg, tmp_path / "runs")
    assert EXPECTED[kind] in str(exc.value), str(exc.value)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_the_message_names_the_file(kind, broken, tmp_path):
    """(3) Nine of ten videos being fine is the normal case; the message has to say which one."""
    cfg, built = broken
    video = _fixture(built, kind)
    with pytest.raises(ing.IngestError) as exc:
        ing.ingest(video, cfg, tmp_path / "runs")
    assert video.name in str(exc.value), str(exc.value)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_the_message_says_which_stage_stopped(kind, broken, tmp_path):
    """A fixture that fails at the wrong stage still 'fails', and still raises IngestError."""
    cfg, built = broken
    video = _fixture(built, kind)
    with pytest.raises(ing.IngestError) as exc:
        ing.ingest(video, cfg, tmp_path / "runs")
    msg = str(exc.value)
    if EXPECTED_STAGE[kind] == "probe":
        assert msg.startswith("probe: "), msg
    else:
        # the stages that reject on a fact we established ourselves lead with the file
        assert msg.startswith(str(video)), msg


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_no_partial_output_survives_the_failure(kind, broken, tmp_path):
    """(4) A half-written run is what makes a failure quiet until Phase 2 reads it."""
    cfg, built = broken
    video = _fixture(built, kind)
    runs = tmp_path / "runs"
    with pytest.raises(ing.IngestError):
        ing.ingest(video, cfg, runs)
    out_dir = runs / video.stem
    assert not (out_dir / "media.json").exists(), "a failed run wrote a manifest"
    assert not (out_dir / "audio.wav").exists(), "a failed run wrote a wav"
    assert list(out_dir.glob("frames/frame_*.jpg")) == [], "a failed run wrote frames"


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_the_cli_exits_non_zero_and_prints_the_reason(kind, broken, tmp_path, capsys):
    """The human-visible half of 'loudly': exit 1 and a FAIL line on stderr, not a traceback."""
    cfg, built = broken
    video = _fixture(built, kind)
    code = ing.main([str(video), "--config", str(cfg.path), "--out", str(tmp_path / "runs")])
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("FAIL — "), err
    assert EXPECTED[kind] in err, err


# --- which stage knows what, per fixture --------------------------------------------


@needs_ffmpeg
def test_a_missing_audio_track_is_diagnosed_not_crashed(broken, tmp_path):
    """The pipeline knows the track is absent before it refuses. That is the difference
    between a diagnosis and a crash: probe and metadata both succeed, the video half is
    intact, and the stop is a decision made on a known fact."""
    cfg, built = broken
    video = built["no-audio"]
    raw, _ = ing.probe(video)
    media = ing.media_metadata(raw, video)
    assert media["audio"] is None
    assert media["video"] is not None and media["duration_s"] > 0
    with pytest.raises(ing.IngestError) as exc:
        ing.extract_audio(video, tmp_path / "a.wav", cfg, media)
    assert "no audio stream" in str(exc.value)
    assert not (tmp_path / "a.wav").exists(), "refused, but wrote an empty wav anyway"


@needs_ffmpeg
def test_a_zero_duration_file_parses_as_a_container_and_still_has_nothing_in_it(broken):
    """ffprobe is happy with this file. Nothing downstream can be."""
    _, built = broken
    raw, _ = ing.probe(built["zero-duration"])
    assert raw["format"]["nb_streams"] == 0
    assert raw["streams"] == []
    assert "duration" not in raw["format"]
    with pytest.raises(ing.IngestError) as exc:
        ing.media_metadata(raw, built["zero-duration"])
    assert "zero-length" in str(exc.value)


@needs_ffmpeg
@pytest.mark.parametrize("kind", ["truncated", "garbage"])
def test_an_unreadable_container_is_caught_by_probe_itself(kind, broken):
    """No metadata to reason about, so this one has to fail at the first command."""
    _, built = broken
    with pytest.raises(ing.IngestError) as exc:
        ing.probe(built[kind])
    msg = str(exc.value)
    assert msg.startswith("probe: ffprobe exited ")
    # ffprobe's own stderr is carried through rather than swallowed
    assert "moov atom not found" in msg


def test_an_empty_file_is_rejected_before_ffprobe_is_ever_run(broken, tmp_path):
    """Deliberately not skipped without ffmpeg: the pre-flight is why this case does not need
    it. ffprobe calls a 0-byte file "moov atom not found / Invalid data found", which reads as
    a corrupt container and sends the reader looking for the wrong problem."""
    cfg, built = broken
    with pytest.raises(ing.IngestError) as exc:
        ing.ingest(built["empty"], cfg, tmp_path / "runs")
    msg = str(exc.value)
    assert "0 bytes" in msg and "empty" in msg
    assert "moov" not in msg and "ffprobe" not in msg
