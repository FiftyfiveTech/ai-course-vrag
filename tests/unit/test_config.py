"""Tests for the config reader — VRAG-005.

The reader has one job the pipeline depends on: a lever that is missing from config.toml
must raise, not quietly become a default. A default is a hardcoded value wearing a hat, and
"sampling rate read from config, not hardcoded" is the acceptance criterion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config as cfgmod

COMMITTED = Path("config.toml")


@pytest.fixture(scope="module")
def committed() -> cfgmod.Config:
    if not COMMITTED.exists():
        pytest.fail(f"{COMMITTED} is missing — the pipeline levers live there")
    return cfgmod.load(COMMITTED)


def write(tmp_path: Path, text: str) -> cfgmod.Config:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return cfgmod.load(path)


# --- the committed file actually carries the levers ---------------------------------


@pytest.mark.parametrize(
    "lever",
    [
        "ingest.audio.sample_rate_hz",
        "ingest.audio.channels",
        "ingest.audio.codec",
        "ingest.frames.fps",
        "ingest.frames.max_frames",
        "ingest.frames.width",
        "ingest.frames.format",
        "ingest.frames.jpeg_quality",
    ],
)
def test_committed_config_defines_every_ingest_lever(committed, lever):
    assert committed.get(lever) is not None


def test_frame_rate_is_a_positive_number(committed):
    fps = committed.get("ingest.frames.fps")
    assert isinstance(fps, (int, float)) and fps > 0


def test_audio_target_is_what_the_asr_arms_want(committed):
    assert committed.get("ingest.audio.sample_rate_hz") == 16000
    assert committed.get("ingest.audio.channels") == 1


# --- no silent defaults -------------------------------------------------------------


def test_missing_lever_raises_instead_of_defaulting(tmp_path):
    cfg = write(tmp_path, "[ingest.frames]\nwidth = 768\n")
    with pytest.raises(cfgmod.ConfigError) as exc:
        cfg.get("ingest.frames.fps")
    assert "fps" in str(exc.value)


def test_the_error_names_the_file_to_fix(tmp_path):
    cfg = write(tmp_path, "[ingest]\n")
    with pytest.raises(cfgmod.ConfigError) as exc:
        cfg.get("ingest.frames.fps")
    assert str(cfg.path) in str(exc.value)


def test_missing_file_raises(tmp_path):
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.load(tmp_path / "nope.toml")


def test_unparseable_file_raises(tmp_path):
    with pytest.raises(cfgmod.ConfigError):
        write(tmp_path, "this is not = = toml\n")


def test_section_rejects_a_scalar(tmp_path):
    cfg = write(tmp_path, "[ingest]\nframes = 3\n")
    with pytest.raises(cfgmod.ConfigError):
        cfg.section("ingest.frames")


# --- a run can be traced back to the bytes that produced it -------------------------


def test_fingerprint_changes_with_the_file(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    one = write(tmp_path / "a", "[ingest.frames]\nfps = 0.2\n")
    two = write(tmp_path / "b", "[ingest.frames]\nfps = 1.0\n")
    assert one.fingerprint()["sha256"] != two.fingerprint()["sha256"]


def test_fingerprint_is_stable_for_the_same_bytes(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    one = write(tmp_path / "a", "[ingest.frames]\nfps = 0.2\n")
    two = write(tmp_path / "b", "[ingest.frames]\nfps = 0.2\n")
    assert one.fingerprint()["sha256"] == two.fingerprint()["sha256"]


def test_fingerprint_never_carries_the_contents(committed):
    printed = committed.fingerprint()
    assert set(printed) == {"path", "sha256"}
    assert len(printed["sha256"]) == 64
