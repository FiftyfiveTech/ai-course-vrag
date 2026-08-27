"""Ingest — VRAG-005.

One video in; a 16 kHz mono wav, a directory of sampled frames, and a media.json out.

    make demo VIDEO=samples/one.mp4

Three things this module is careful about.

**The sampling rate is not in here.** It is `ingest.frames.fps` in `config.toml`, and
`src.config` refuses to default it. Frame rate is the cost lever of the whole phase — at
fps=1.0 a one-hour video is 3600 vision calls, at fps=0.2 it is 720 — so it has to be a
value you can change and a value a run records, not a number buried in an argument list.

**Every run writes down what produced it.** media.json carries the source file's sha256,
the config path and the sha256 of its bytes, the exact ffmpeg argv, and the wall time of
each stage. That is what makes two runs comparable later.

**Timestamps are the product, not a side effect.** A frame is only useful to VRAG-020 if
we can say when it came from, so every frame's source timestamp is *measured* — ffmpeg's
`showinfo` reports it and we record what it said — rather than computed from a formula.

That is not fussiness. The obvious implementation, `-vf fps=0.2`, is wrong here: the fps
filter picks the frame at the *middle* of each interval and then relabels it with the
interval's *start*. Sampling the 30 s fixture that way writes `t=25.0` next to a picture
whose burnt-in timecode reads 27.48 — a silent 2.5 s lie in the one field the citation
gate is built on. `select` plus `-fps_mode passthrough` keeps each frame's own PTS, and
showinfo then tells us what it is.

Failure paths (no audio track, zero-length, unreadable codec) raise `IngestError` with the
ffmpeg stderr attached, and none of them writes a partial media.json — a half-finished run is
what makes a failure quiet three phases later. VRAG-009 gave them their fixtures
(`src.sample.BROKEN_KINDS`, `make sample-broken`) and their tests
(`tests/unit/test_ingest_failures.py`), which put a real broken file through this pipeline
rather than a hand-written probe dict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import Config, ConfigError
from src.config import load as load_config
from src.telemetry import Meter

OUT_ROOT = Path("runs")

# ffprobe/ffmpeg banners are noise on a gate's terminal; errors are not.
QUIET = ["-hide_banner", "-loglevel", "error"]

PROBE_TIMEOUT_S = 60
FFMPEG_TIMEOUT_S = 3600


class IngestError(Exception):
    """A video could not be ingested. The message says which stage and why."""


@dataclass(frozen=True)
class Stage:
    """One external command: what was run, how long it took."""

    name: str
    argv: list[str]
    seconds: float


def _binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise IngestError(
            f"{name} is not on PATH. Ingest is an ffmpeg pipeline; install it and re-run "
            f"`make doctor` — that check exists for exactly this."
        )
    return path


def _run(
    name: str, argv: list[str], timeout: int, subject: Path | None = None
) -> tuple[subprocess.CompletedProcess, Stage]:
    """Run a command, or raise IngestError carrying its stderr.

    argv[0] is a bare binary name and is resolved here rather than by the callers, so the
    argv builders stay pure — a test can read the sampling rate off a command line on a
    machine with no ffmpeg — and what lands in media.json stays readable.

    `subject` is the file being worked on, and it is in the message because it was not:
    a corrupt input reported itself as `probe: ffprobe exited 1 — moov atom not found` with
    no path in it. ffprobe happens to prefix its own stderr with the filename, so the path
    was there by luck on one code path and absent everywhere else. VRAG-009 made that a
    guarantee rather than a coincidence — a batch of ten videos where one is broken has to
    say which one.
    """
    resolved = [_binary(argv[0]), *argv[1:]]
    started = time.perf_counter()
    try:
        proc = subprocess.run(resolved, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise IngestError(f"{name}: timed out after {timeout}s") from exc
    except OSError as exc:
        raise IngestError(f"{name}: could not run {argv[0]!r} ({exc})") from exc
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        # The frames pass runs at -loglevel info so showinfo is readable; that buries the
        # actual error under per-frame chatter, so drop the chatter before reporting.
        tail = [ln for ln in (proc.stderr or "").strip().splitlines() if "showinfo" not in ln]
        detail = " / ".join(tail[-3:]) if tail else "no stderr"
        where = f" on {subject}" if subject is not None else ""
        raise IngestError(f"{name}: {argv[0]} exited {proc.returncode}{where} — {detail}")
    return proc, Stage(name, argv, elapsed)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Content identity of the source file. Which bytes produced this run."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- probe


def probe_args(video: Path) -> list[str]:
    return [
        "ffprobe",
        *QUIET,
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video),
    ]


def probe(video: Path) -> tuple[dict[str, Any], Stage]:
    proc, stage = _run("probe", probe_args(video), PROBE_TIMEOUT_S, subject=video)
    try:
        return json.loads(proc.stdout), stage
    except json.JSONDecodeError as exc:
        raise IngestError(f"probe: ffprobe returned no usable JSON for {video} ({exc})") from exc


def _num(value: Any, cast=float):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def _fps(rate: str | None) -> float | None:
    """ffprobe reports frame rate as the string "30000/1001". Keep it a number."""
    if not rate or "/" not in rate:
        return _num(rate)
    num, _, den = rate.partition("/")
    n, d = _num(num), _num(den)
    if n is None or not d:
        return None
    return round(n / d, 6)


def media_metadata(raw: dict[str, Any], video: Path) -> dict[str, Any]:
    """Flatten ffprobe's output into the facts the rest of the pipeline actually uses."""
    fmt = raw.get("format", {}) or {}
    streams = raw.get("streams", []) or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = _num(fmt.get("duration"))
    if duration is None and v is not None:
        duration = _num(v.get("duration"))
    if not duration or duration <= 0:
        raise IngestError(
            f"{video}: ffprobe reports no positive duration ({fmt.get('duration')!r}) — "
            f"the file is zero-length or its container is unreadable."
        )

    return {
        "container": fmt.get("format_name"),
        "duration_s": round(duration, 3),
        "bytes": _num(fmt.get("size"), int),
        "bitrate_bps": _num(fmt.get("bit_rate"), int),
        "streams": len(streams),
        "video": None
        if v is None
        else {
            "codec": v.get("codec_name"),
            "profile": v.get("profile"),
            "width": _num(v.get("width"), int),
            "height": _num(v.get("height"), int),
            "fps": _fps(v.get("avg_frame_rate")),
            "pix_fmt": v.get("pix_fmt"),
            "frames": _num(v.get("nb_frames"), int),
        },
        "audio": None
        if a is None
        else {
            "codec": a.get("codec_name"),
            "sample_rate_hz": _num(a.get("sample_rate"), int),
            "channels": _num(a.get("channels"), int),
            "channel_layout": a.get("channel_layout"),
            "bitrate_bps": _num(a.get("bit_rate"), int),
        },
    }


# --------------------------------------------------------------------------- audio


def audio_args(video: Path, out_wav: Path, audio_cfg: dict[str, Any]) -> list[str]:
    """argv for the wav extraction. Pure, so a test can read the rate straight off it."""
    return [
        "ffmpeg",
        "-nostdin",
        *QUIET,
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        str(audio_cfg["channels"]),
        "-ar",
        str(audio_cfg["sample_rate_hz"]),
        "-acodec",
        str(audio_cfg["codec"]),
        str(out_wav),
    ]


def audio_config(cfg: Config) -> dict[str, Any]:
    return {
        "channels": cfg.get("ingest.audio.channels"),
        "sample_rate_hz": cfg.get("ingest.audio.sample_rate_hz"),
        "codec": cfg.get("ingest.audio.codec"),
    }


def extract_audio(
    video: Path, out_wav: Path, cfg: Config, media: dict[str, Any]
) -> tuple[dict[str, Any], Stage]:
    if media["audio"] is None:
        raise IngestError(
            f"{video}: no audio stream. Nothing to transcribe, so ingest stops here rather "
            f"than writing an empty wav. (VRAG-009 covers this as a tested failure path.)"
        )
    acfg = audio_config(cfg)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _, stage = _run("audio", audio_args(video, out_wav, acfg), FFMPEG_TIMEOUT_S, subject=video)
    written = out_wav.stat().st_size
    return {
        "path": out_wav.as_posix(),
        **acfg,
        "bytes": written,
        # 16-bit mono PCM is 2 bytes a sample, so the duration comes off the file we just
        # wrote rather than being copied from the source.
        "duration_s": round(written / (acfg["sample_rate_hz"] * acfg["channels"] * 2), 3),
    }, stage


# --------------------------------------------------------------------------- frames


def frame_filter(fps: float, width: int) -> str:
    """Take the first frame, then the first frame at least 1/fps later, and so on.

    Deliberately not `fps={fps}`: that filter rewrites each kept frame's PTS to the start of
    its interval while the picture it kept comes from the middle, so the timestamps we would
    record are half an interval early. `select` keeps every frame's own PTS, and `showinfo`
    then reports it. `scale` is skipped entirely when width is -1.
    """
    step = 1.0 / fps
    # The comma inside gte() has to be escaped or ffmpeg reads it as the next filter.
    chain = [f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{step:g})'"]
    if width and width > 0:
        # -2 keeps the aspect ratio and lands on an even height, which yuv420p requires.
        chain.append(f"scale={width}:-2")
    chain.append("showinfo")
    return ",".join(chain)


def frame_args(video: Path, pattern: Path, frames_cfg: dict[str, Any]) -> list[str]:
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        # showinfo talks at info level, and its output is where the timestamps come from.
        "-loglevel",
        "info",
        "-y",
        "-i",
        str(video),
        "-vf",
        frame_filter(frames_cfg["fps"], frames_cfg["width"]),
        # Without this ffmpeg re-times the selected frames to a constant rate and the PTS
        # showinfo reports stops being the source position.
        "-fps_mode",
        "passthrough",
        "-frames:v",
        str(frames_cfg["max_frames"]),
        "-q:v",
        str(frames_cfg["jpeg_quality"]),
        str(pattern),
    ]


PTS_RE = re.compile(r"\bpts_time:([0-9]+(?:\.[0-9]+)?)")


def frame_timestamps(stderr: str) -> list[float]:
    """The source timestamp of each kept frame, as showinfo reported it."""
    return [float(m) for m in PTS_RE.findall(stderr)]


def frames_config(cfg: Config) -> dict[str, Any]:
    frames_cfg = {
        "fps": cfg.get("ingest.frames.fps"),
        "max_frames": cfg.get("ingest.frames.max_frames"),
        "width": cfg.get("ingest.frames.width"),
        "format": cfg.get("ingest.frames.format"),
        "jpeg_quality": cfg.get("ingest.frames.jpeg_quality"),
    }
    if not isinstance(frames_cfg["fps"], (int, float)) or frames_cfg["fps"] <= 0:
        raise ConfigError(
            f"{cfg.path}: ingest.frames.fps must be a number > 0, got {frames_cfg['fps']!r}"
        )
    return frames_cfg


def sample_frames(
    video: Path, out_dir: Path, cfg: Config, media: dict[str, Any]
) -> tuple[dict[str, Any], Stage]:
    if media["video"] is None:
        raise IngestError(f"{video}: no video stream, so there is nothing to sample frames from.")

    frames_cfg = frames_config(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    # A re-run at a lower fps must not leave the previous run's extra frames lying around
    # to be counted as this run's output.
    for stale in out_dir.glob(f"frame_*.{frames_cfg['format']}"):
        stale.unlink()

    pattern = out_dir / f"frame_%05d.{frames_cfg['format']}"
    proc, stage = _run(
        "frames", frame_args(video, pattern, frames_cfg), FFMPEG_TIMEOUT_S, subject=video
    )

    files = sorted(out_dir.glob(f"frame_*.{frames_cfg['format']}"))
    times = frame_timestamps(proc.stderr or "")
    if len(times) < len(files):
        # A frame we cannot timestamp is a citation we cannot make, so this is fatal rather
        # than something to paper over with a computed fallback.
        raise IngestError(
            f"{video}: ffmpeg wrote {len(files)} frames but showinfo reported only "
            f"{len(times)} timestamps, so the frames cannot be placed in the video."
        )

    step = 1.0 / frames_cfg["fps"]
    frames = [
        {"file": p.name, "t_s": round(times[i], 3), "bytes": p.stat().st_size}
        for i, p in enumerate(files)
    ]
    covered = (frames[-1]["t_s"] + step) if frames else 0.0
    truncated = len(files) >= frames_cfg["max_frames"] and covered < media["duration_s"]

    return {
        "dir": out_dir.as_posix(),
        "count": len(files),
        **frames_cfg,
        "covers_s": round(min(covered, media["duration_s"]), 3),
        "truncated": truncated,
        "frames": frames,
    }, stage


# --------------------------------------------------------------------------- pipeline


def ingest(video: Path, cfg: Config, out_root: Path = OUT_ROOT) -> dict[str, Any]:
    video = Path(video)
    if not video.is_file():
        raise IngestError(f"{video}: not a file. `make sample` writes samples/one.mp4.")
    # Checked here rather than left to ffprobe. An empty file is the most common broken input
    # there is — an interrupted download, a full disk, a job that wrote nothing — and ffprobe
    # reports it as "moov atom not found / Invalid data found when processing input", which
    # sends the reader looking for a corrupt container instead of an empty file.
    if video.stat().st_size == 0:
        raise IngestError(
            f"{video}: 0 bytes. The file is empty — nothing was written to it, so there is "
            f"no container to read. Check whatever produced it."
        )

    meter = Meter()
    started = time.perf_counter()
    out_dir = out_root / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, probe_stage = probe(video)
    media = media_metadata(raw, video)

    hash_started = time.perf_counter()
    digest = sha256_file(video)
    hash_s = time.perf_counter() - hash_started

    audio, audio_stage = extract_audio(video, out_dir / "audio.wav", cfg, media)
    frames, frames_stage = sample_frames(video, out_dir / "frames", cfg, media)

    total = time.perf_counter() - started
    stages = [probe_stage, audio_stage, frames_stage]

    result = {
        "task": "VRAG-005",
        "source": {
            "path": video.as_posix(),
            "name": video.name,
            "bytes": video.stat().st_size,
            "sha256": digest,
        },
        "media": media,
        "audio": audio,
        "frames": frames,
        "config": cfg.fingerprint(),
        "timing": {
            "probe_s": round(probe_stage.seconds, 3),
            "hash_s": round(hash_s, 3),
            "audio_s": round(audio_stage.seconds, 3),
            "frames_s": round(frames_stage.seconds, 3),
            "total_s": round(total, 3),
            # Seconds of video processed per second of wall clock. VRAG-006 turns this into
            # $/video-hour; on its own it is the latency half of the Phase 0 gate.
            "x_realtime": round(media["duration_s"] / total, 2) if total else None,
        },
        "commands": [
            {"stage": s.name, "argv": s.argv, "seconds": round(s.seconds, 3)} for s in stages
        ],
        "telemetry": meter.summary_line(media["duration_s"], wall_s=total),
    }

    manifest = out_dir / "media.json"
    manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result["manifest"] = manifest.as_posix()
    return result


def report(result: dict[str, Any], out=sys.stdout) -> None:
    """Print the numbers. A number not printed is not a number."""
    m, a, f, t = result["media"], result["audio"], result["frames"], result["timing"]
    v, src_audio = m["video"] or {}, m["audio"] or {}
    src = result["source"]

    print(f"ingest — {src['path']}", file=out)
    print(f"  sha256    {src['sha256'][:16]}…  {src['bytes'] / 1e6:.1f} MB", file=out)
    print(
        f"  media     {m['container']} · {m['duration_s']:.2f} s · "
        f"{v.get('codec')} {v.get('width')}x{v.get('height')} @ {v.get('fps')} fps · "
        f"{src_audio.get('codec')} {src_audio.get('sample_rate_hz')} Hz "
        f"{src_audio.get('channels')} ch",
        file=out,
    )
    print(
        f"  wav       {a['path']} · {a['sample_rate_hz']} Hz {a['channels']} ch "
        f"{a['codec']} · {a['bytes'] / 1e6:.2f} MB · {a['duration_s']:.2f} s",
        file=out,
    )
    print(
        f"  frames    {f['count']} × {f['format']} in {f['dir']}/ · "
        f"fps={f['fps']} (from {result['config']['path']}) · width={f['width']} · "
        f"covers {f['covers_s']:.1f} s of {m['duration_s']:.2f} s",
        file=out,
    )
    if f["frames"]:
        first, last = f["frames"][0], f["frames"][-1]
        print(
            f"            {first['file']} t={first['t_s']}s … {last['file']} t={last['t_s']}s",
            file=out,
        )
    if f["truncated"]:
        print(
            f"  WARN      hit ingest.frames.max_frames={f['max_frames']}; frames stop at "
            f"{f['covers_s']:.1f} s. Raise the ceiling or lower fps in "
            f"{result['config']['path']}.",
            file=out,
        )
    print(
        f"  timing    probe {t['probe_s']}s · sha256 {t['hash_s']}s · wav {t['audio_s']}s · "
        f"frames {t['frames_s']}s · total {t['total_s']}s · {t['x_realtime']}× realtime",
        file=out,
    )
    print(
        f"  config    {result['config']['path']} sha256:{result['config']['sha256'][:16]}…",
        file=out,
    )
    print(f"\nwrote {result['manifest']}", file=out)
    print(result["telemetry"], file=out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Ingest one video: wav + sampled frames + media metadata (VRAG-005)."
    )
    ap.add_argument("video", help="path to the video file")
    ap.add_argument(
        "--config", default="config.toml", help="the file holding the sampling levers"
    )
    ap.add_argument("--out", default=str(OUT_ROOT), help=f"output root (default {OUT_ROOT})")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
        result = ingest(Path(args.video), cfg, Path(args.out))
    except (IngestError, ConfigError) as exc:
        print(f"FAIL — {exc}", file=sys.stderr)
        return 1
    report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
