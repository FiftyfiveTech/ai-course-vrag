"""Tests for ingest — VRAG-005.

Two layers, on purpose.

Most of this file is offline: the argv builders and the ffprobe/showinfo parsers are pure
functions, so what the pipeline *asks ffmpeg for* can be asserted without encoding anything.
That is where the acceptance criterion lives — the sampling rate has to arrive from
config.toml, and the cheapest proof is that two different config files produce two different
ffmpeg command lines from the same code.

The rest runs ffmpeg for real and is skipped when it is not installed, because a test that
silently passes on a machine without ffmpeg is worse than no test.
"""

from __future__ import annotations

import ast
import json
import shutil
import wave
from pathlib import Path

import pytest

from src import ingest as ing
from src import sample as smp
from src.config import Config, ConfigError
from src.config import load as load_config

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")

COMMITTED = Path("config.toml")

# A short clip keeps the end-to-end tests quick. 12 s at fps=0.5 is 6 frames, spaced 2 s.
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
duration_s = 12
width = 640
height = 360
fps = 25
tone_hz = 440

[sample.real]
max_height = 720
"""


def make_config(tmp_path: Path, text: str = FIXTURE) -> Config:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


@pytest.fixture(scope="module")
def committed() -> Config:
    return load_config(COMMITTED)


# --- the acceptance criterion: the sampling rate comes from config ------------------


def test_frame_rate_reaches_ffmpeg_from_the_config_file(tmp_path):
    cfg = make_config(tmp_path, FIXTURE.replace("fps = 0.5", "fps = 0.25"))
    argv = ing.frame_args(Path("v.mp4"), tmp_path / "f_%05d.jpg", ing.frames_config(cfg))
    # 1/0.25 = one frame every 4 s.
    assert "gte(t-prev_selected_t\\,4)" in " ".join(argv)


def test_two_configs_give_two_different_commands(tmp_path):
    """Same code, different file, different sampling. This is what 'not hardcoded' means."""
    (tmp_path / "slow").mkdir()
    (tmp_path / "fast").mkdir()
    slow = make_config(tmp_path / "slow", FIXTURE.replace("fps = 0.5", "fps = 0.1"))
    fast = make_config(tmp_path / "fast", FIXTURE.replace("fps = 0.5", "fps = 2.0"))
    pattern = tmp_path / "f_%05d.jpg"
    a = ing.frame_args(Path("v.mp4"), pattern, ing.frames_config(slow))
    b = ing.frame_args(Path("v.mp4"), pattern, ing.frames_config(fast))
    assert a != b
    assert "\\,10)" in " ".join(a)
    assert "\\,0.5)" in " ".join(b)


def test_no_sampling_rate_is_hardcoded_in_the_module():
    """The committed fps must not also exist as a literal in the code that uses it.

    Walking the AST rather than grepping the text, so the prose in the docstrings — which
    quotes the rate on purpose, to explain the cost lever — is not mistaken for a default.
    """
    tree = ast.parse(Path("src/ingest.py").read_text(encoding="utf-8"))
    numbers = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    }
    assert load_config(COMMITTED).get("ingest.frames.fps") not in numbers


def test_frame_rate_of_zero_is_rejected(tmp_path):
    cfg = make_config(tmp_path, FIXTURE.replace("fps = 0.5", "fps = 0"))
    with pytest.raises(ConfigError):
        ing.frames_config(cfg)


def test_audio_target_reaches_ffmpeg_from_the_config_file(tmp_path):
    cfg = make_config(tmp_path, FIXTURE.replace("sample_rate_hz = 16000", "sample_rate_hz = 8000"))
    argv = ing.audio_args(Path("v.mp4"), Path("out.wav"), ing.audio_config(cfg))
    assert argv[argv.index("-ar") + 1] == "8000"
    assert argv[argv.index("-ac") + 1] == "1"
    assert argv[argv.index("-acodec") + 1] == "pcm_s16le"


# --- frame selection: the timestamps have to be the real ones -----------------------


def test_frames_are_selected_by_time_not_by_the_fps_filter():
    """`fps=N` relabels each kept frame with the start of its interval; we need its own PTS."""
    chain = ing.frame_filter(0.2, 768)
    assert chain.startswith("select=")
    assert "fps=0.2" not in chain
    assert chain.endswith("showinfo")


def test_frame_filter_preserves_source_timestamps():
    argv = ing.frame_args(Path("v.mp4"), Path("f_%05d.jpg"), {
        "fps": 0.2, "max_frames": 10, "width": 768, "format": "jpg", "jpeg_quality": 4,
    })
    assert argv[argv.index("-fps_mode") + 1] == "passthrough"
    assert argv[argv.index("-frames:v") + 1] == "10"
    # showinfo only speaks at info level, and it is where the timestamps come from.
    assert argv[argv.index("-loglevel") + 1] == "info"


def test_scale_is_skipped_when_width_is_negative():
    assert "scale" not in ing.frame_filter(0.2, -1)
    assert "scale=768:-2" in ing.frame_filter(0.2, 768)


def test_timestamps_are_read_from_showinfo():
    stderr = (
        "[Parsed_showinfo_2 @ 0x1] n:0 pts:0 pts_time:0 duration:1\n"
        "[Parsed_showinfo_2 @ 0x1] n:1 pts:512 pts_time:5.5 duration:1\n"
        "frame=    2 fps=0.0 q=4.0 Lsize=N/A\n"
    )
    assert ing.frame_timestamps(stderr) == [0.0, 5.5]


def test_no_timestamps_parsed_from_unrelated_output():
    assert ing.frame_timestamps("frame= 6 fps=0.0 q=4.0 Lsize=N/A\n") == []


# --- ffprobe -> media metadata ------------------------------------------------------


def probe_json(**over) -> dict:
    raw = {
        "format": {"format_name": "mov,mp4", "duration": "30.0", "size": "100", "bit_rate": "800"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "profile": "High",
                "width": 1280,
                "height": 720,
                "avg_frame_rate": "30000/1001",
                "pix_fmt": "yuv420p",
                "nb_frames": "900",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "44100",
                "channels": 2,
                "channel_layout": "stereo",
                "bit_rate": "128000",
            },
        ],
    }
    raw.update(over)
    return raw


def test_media_metadata_flattens_the_streams():
    m = ing.media_metadata(probe_json(), Path("v.mp4"))
    assert m["duration_s"] == 30.0
    assert m["video"]["codec"] == "h264"
    assert m["video"]["width"] == 1280
    assert m["audio"]["sample_rate_hz"] == 44100
    assert m["audio"]["channels"] == 2
    assert m["streams"] == 2


def test_fractional_frame_rate_becomes_a_number():
    assert ing.media_metadata(probe_json(), Path("v.mp4"))["video"]["fps"] == pytest.approx(29.97, abs=0.01)


def test_zero_length_media_is_rejected():
    raw = probe_json(format={"format_name": "mov,mp4", "duration": "0"})
    with pytest.raises(ing.IngestError) as exc:
        ing.media_metadata(raw, Path("v.mp4"))
    assert "zero-length" in str(exc.value)


def test_unreadable_container_is_rejected():
    with pytest.raises(ing.IngestError):
        ing.media_metadata({"format": {}, "streams": []}, Path("v.mp4"))


def test_a_video_with_no_audio_stream_is_reported_not_guessed():
    raw = probe_json()
    raw["streams"] = [s for s in raw["streams"] if s["codec_type"] == "video"]
    assert ing.media_metadata(raw, Path("v.mp4"))["audio"] is None


def test_extract_audio_refuses_a_silent_file(tmp_path):
    cfg = make_config(tmp_path)
    media = {"audio": None, "video": {}, "duration_s": 10.0}
    with pytest.raises(ing.IngestError) as exc:
        ing.extract_audio(Path("v.mp4"), tmp_path / "a.wav", cfg, media)
    assert "no audio stream" in str(exc.value)


def test_sample_frames_refuses_an_audio_only_file(tmp_path):
    cfg = make_config(tmp_path)
    media = {"audio": {}, "video": None, "duration_s": 10.0}
    with pytest.raises(ing.IngestError):
        ing.sample_frames(Path("v.m4a"), tmp_path / "frames", cfg, media)


def test_a_missing_file_fails_before_ffmpeg_is_called(tmp_path):
    cfg = make_config(tmp_path)
    with pytest.raises(ing.IngestError) as exc:
        ing.ingest(tmp_path / "nope.mp4", cfg, tmp_path / "runs")
    assert "not a file" in str(exc.value)


# --- end to end, with a real ffmpeg -------------------------------------------------


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> tuple[Path, Config]:
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    root = tmp_path_factory.mktemp("clip")
    (root / "config.toml").write_text(FIXTURE, encoding="utf-8")
    cfg = load_config(root / "config.toml")
    return smp.synthetic(root / "one.mp4", cfg), cfg


@pytest.fixture(scope="module")
def run(clip, tmp_path_factory) -> dict:
    video, cfg = clip
    return ing.ingest(video, cfg, tmp_path_factory.mktemp("runs"))


@needs_ffmpeg
def test_the_wav_is_at_the_rate_the_config_asked_for(run):
    with wave.open(run["audio"]["path"], "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2


@needs_ffmpeg
def test_the_wav_covers_the_whole_video(run):
    assert run["audio"]["duration_s"] == pytest.approx(run["media"]["duration_s"], abs=0.2)


@needs_ffmpeg
def test_the_frame_count_follows_the_configured_rate(run):
    # 12 s at one frame every 2 s: t = 0, 2, 4, 6, 8, 10.
    assert run["frames"]["count"] == 6
    assert len(list(Path(run["frames"]["dir"]).glob("frame_*.jpg"))) == 6


@needs_ffmpeg
def test_frame_timestamps_are_spaced_by_the_configured_interval(run):
    times = [f["t_s"] for f in run["frames"]["frames"]]
    assert times == sorted(times)
    step = 1.0 / run["frames"]["fps"]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert all(g == pytest.approx(step, abs=0.05) for g in gaps)


@needs_ffmpeg
def test_every_frame_timestamp_lands_inside_the_video(run):
    for f in run["frames"]["frames"]:
        assert 0 <= f["t_s"] < run["media"]["duration_s"]


@needs_ffmpeg
def test_frames_are_scaled_to_the_configured_width(run, clip):
    _, cfg = clip
    first = Path(run["frames"]["dir"]) / run["frames"]["frames"][0]["file"]
    raw, _ = ing.probe(first)
    assert raw["streams"][0]["width"] == cfg.get("ingest.frames.width")


@needs_ffmpeg
def test_the_run_records_what_produced_it(run):
    assert run["source"]["sha256"] and len(run["source"]["sha256"]) == 64
    assert run["config"]["path"].endswith("config.toml")
    assert len(run["config"]["sha256"]) == 64
    assert {c["stage"] for c in run["commands"]} == {"probe", "audio", "frames"}


@needs_ffmpeg
def test_media_json_is_written_and_is_the_result(run):
    on_disk = json.loads(Path(run["manifest"]).read_text(encoding="utf-8"))
    assert on_disk["frames"]["count"] == run["frames"]["count"]
    assert on_disk["source"]["sha256"] == run["source"]["sha256"]


@needs_ffmpeg
def test_timing_reports_a_realtime_multiple(run):
    t = run["timing"]
    assert t["total_s"] > 0
    assert t["x_realtime"] == pytest.approx(run["media"]["duration_s"] / t["total_s"], rel=0.01)


@needs_ffmpeg
def test_changing_only_the_config_changes_the_output(clip, tmp_path):
    """The end-to-end version of the acceptance criterion."""
    video, _ = clip
    (tmp_path / "cfg").mkdir()
    slower = make_config(tmp_path / "cfg", FIXTURE.replace("fps = 0.5", "fps = 0.25"))
    out = ing.ingest(video, slower, tmp_path / "runs")
    # Same 12 s clip, one frame every 4 s instead of every 2: t = 0, 4, 8.
    assert out["frames"]["count"] == 3
    assert [f["t_s"] for f in out["frames"]["frames"]] == [0.0, 4.0, 8.0]


@needs_ffmpeg
def test_a_rerun_at_a_lower_rate_leaves_no_stale_frames(clip, tmp_path):
    video, cfg = clip
    runs = tmp_path / "runs"
    ing.ingest(video, cfg, runs)
    (tmp_path / "cfg").mkdir()
    slower = make_config(tmp_path / "cfg", FIXTURE.replace("fps = 0.5", "fps = 0.25"))
    out = ing.ingest(video, slower, runs)
    assert len(list(Path(out["frames"]["dir"]).glob("frame_*.jpg"))) == out["frames"]["count"] == 3


@needs_ffmpeg
def test_the_max_frames_ceiling_is_reported_when_it_bites(clip, tmp_path):
    video, _ = clip
    (tmp_path / "cfg").mkdir()
    capped = make_config(tmp_path / "cfg", FIXTURE.replace("max_frames = 1200", "max_frames = 2"))
    out = ing.ingest(video, capped, tmp_path / "runs")
    assert out["frames"]["count"] == 2
    assert out["frames"]["truncated"] is True
    assert out["frames"]["covers_s"] < out["media"]["duration_s"]


@needs_ffmpeg
def test_a_full_run_is_not_marked_truncated(run):
    assert run["frames"]["truncated"] is False
