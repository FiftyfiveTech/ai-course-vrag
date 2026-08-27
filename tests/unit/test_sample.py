"""Tests for the sample fixtures — VRAG-005.

Offline. The synthetic clip's encoding is covered end-to-end in test_ingest.py; what matters
here is the two rules around the *real* fetch — it comes from the manifest's recorded url,
and it will not hand you a held-out video by accident.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src import sample as smp
from src.config import load as load_config

CORPUS = [
    {"video_id": "181", "youtube_id": "aaa", "url": "https://youtu.be/aaa", "split": "dev",
     "duration": "short", "domain": "Knowledge"},
    {"video_id": "204", "youtube_id": "bbb", "url": "https://youtu.be/bbb", "split": "heldout",
     "duration": "long", "domain": "Sports"},
]


def test_a_dev_video_is_allowed():
    assert smp.pick(CORPUS, "181", allow_heldout=False)["youtube_id"] == "aaa"


def test_a_heldout_video_is_refused_by_default():
    with pytest.raises(smp.SampleError) as exc:
        smp.pick(CORPUS, "204", allow_heldout=False)
    assert "held-out" in str(exc.value)


def test_a_heldout_video_needs_an_explicit_opt_in():
    assert smp.pick(CORPUS, "204", allow_heldout=True)["youtube_id"] == "bbb"


def test_an_unknown_id_lists_what_is_available():
    with pytest.raises(smp.SampleError) as exc:
        smp.pick(CORPUS, "999", allow_heldout=True)
    assert "181" in str(exc.value)


def test_the_fetch_uses_the_url_recorded_in_the_manifest():
    argv = smp.fetch_args(CORPUS[0], Path("samples"), height=720)
    assert argv[-1] == "https://youtu.be/aaa"
    assert "720" in " ".join(argv)


def test_the_fetch_names_the_file_after_the_corpus_id():
    argv = smp.fetch_args(CORPUS[0], Path("samples"), height=720)
    out = argv[argv.index("-o") + 1]
    assert "181_aaa" in out


def test_the_synthetic_clip_is_generated_not_downloaded():
    """No url anywhere in the argv: the gate fixture must not need the network."""
    cfg = load_config("config.toml")
    spec = {
        "duration_s": cfg.get("sample.synthetic.duration_s"),
        "width": cfg.get("sample.synthetic.width"),
        "height": cfg.get("sample.synthetic.height"),
        "fps": cfg.get("sample.synthetic.fps"),
        "tone_hz": cfg.get("sample.synthetic.tone_hz"),
    }
    argv = smp.synthetic_args(Path("samples/one.mp4"), spec)
    assert "http" not in " ".join(argv)
    assert any("testsrc2" in a for a in argv)


def test_the_synthetic_audio_is_not_already_what_ingest_wants():
    """If the fixture were born 16 kHz mono, the resample-and-downmix path would be untested."""
    cfg = load_config("config.toml")
    spec = {"duration_s": 5, "width": 320, "height": 240, "fps": 25, "tone_hz": 440}
    joined = " ".join(smp.synthetic_args(Path("x.mp4"), spec))
    assert "sample_rate=44100" in joined
    assert cfg.get("ingest.audio.sample_rate_hz") != 44100


def test_no_sample_video_is_committed():
    """The licence reason from VRAG-004, enforced the same way: ask git, not the filesystem."""
    listed = subprocess.run(
        ["git", "ls-files", "samples"], capture_output=True, text=True
    ).stdout.strip()
    assert listed == "", f"samples/ must stay out of git (licence): {listed}"
