"""Sample videos for the ingest gate — VRAG-005.

`make demo VIDEO=samples/one.mp4` has to work from a clean clone, but no video file can be
committed: Video-MME's terms forbid redistributing the benchmark "in whole or in part", and
`.gitignore` blocks `*.mp4` so a working copy cannot land by accident (see
data/corpus/PROVENANCE.md). So the sample is *made*, not shipped, and there are two of them
because they answer different questions.

**Synthetic** (`make sample`) — ffmpeg's own `testsrc2` plus a sine tone. Offline,
deterministic, licence-free, and the picture carries a running frame counter, so you can
open `frame_00007.jpg` and see with your eyes whether the t=30.0 s we wrote next to it is
true. This is what the gate runs on: a gate that needs the network is a gate that fails for
reasons that are not about the code.

**Real** (`make sample-real VIDEO_ID=…`) — fetches one video from the url recorded in
data/corpus/manifest.json, which is exactly what PROVENANCE says a reproducer does. It has
speech, so it is what VRAG-008's transcript arms need. It stays out of git.

The real fetch refuses a held-out video id unless you pass `--allow-heldout`. The dev vs
held-out *video* split is public — the Builder has to know which six to avoid — and the
cheapest way to keep avoiding them is to make the tool say no.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.config import Config, ConfigError
from src.config import load as load_config

MANIFEST = Path("data/corpus/manifest.json")
SAMPLES = Path("samples")


class SampleError(Exception):
    """The sample could not be produced."""


def _ffmpeg() -> str:
    """Resolved at run time, not while building argv, so the builders stay testable."""
    path = shutil.which("ffmpeg")
    if not path:
        raise SampleError("ffmpeg is not on PATH — run `make doctor`.")
    return path


def synthetic_args(out: Path, spec: dict[str, Any]) -> list[str]:
    """argv for the generated clip. Pure, so a test can assert the shape without encoding."""
    duration = spec["duration_s"]
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        # testsrc2 burns the elapsed time and frame number into the picture, which is what
        # makes a sampled frame's recorded timestamp checkable by eye.
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={spec['width']}x{spec['height']}:rate={spec['fps']}:duration={duration}",
        # Deliberately 44.1 kHz stereo: the source is not already what the ASR arm wants, so
        # ingest's resample-and-downmix to 16 kHz mono is actually exercised.
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={spec['tone_hz']}:sample_rate=44100:duration={duration}",
        "-ac",
        "2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(out),
    ]


def synthetic(out: Path, cfg: Config) -> Path:
    spec = {
        "duration_s": cfg.get("sample.synthetic.duration_s"),
        "width": cfg.get("sample.synthetic.width"),
        "height": cfg.get("sample.synthetic.height"),
        "fps": cfg.get("sample.synthetic.fps"),
        "tone_hz": cfg.get("sample.synthetic.tone_hz"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = synthetic_args(out, spec)
    proc = subprocess.run([_ffmpeg(), *argv[1:]], capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not out.is_file():
        raise SampleError(f"ffmpeg exited {proc.returncode} — {(proc.stderr or '').strip()[-300:]}")
    print(
        f"sample — {out} · {spec['duration_s']} s · {spec['width']}x{spec['height']} @ "
        f"{spec['fps']} fps · {spec['tone_hz']} Hz tone · {out.stat().st_size / 1e6:.2f} MB"
    )
    print("  source    ffmpeg lavfi testsrc2 + sine — generated, not downloaded")
    return out


def load_manifest(path: Path = MANIFEST) -> list[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))["videos"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise SampleError(f"cannot read the corpus manifest {path} ({exc}) — run `make corpus`") from exc


def pick(videos: list[dict[str, Any]], video_id: str, allow_heldout: bool) -> dict[str, Any]:
    hit = next((v for v in videos if str(v["video_id"]) == str(video_id)), None)
    if hit is None:
        known = ", ".join(f"{v['video_id']}({v['split']})" for v in videos)
        raise SampleError(f"video_id {video_id!r} is not in the corpus. Known: {known}")
    if hit["split"] != "dev" and not allow_heldout:
        raise SampleError(
            f"video_id {video_id} is in the held-out split. The Builder tunes on dev only; "
            f"pass --allow-heldout if you are the Evaluator and mean it."
        )
    return hit


def fetch_args(video: dict[str, Any], out_dir: Path, height: int) -> list[str]:
    """argv for yt-dlp. Run via `uv run --with yt-dlp`, so it is not a project dependency."""
    return [
        "yt-dlp",
        "--no-playlist",
        "-f",
        f"bv*[height<={height}]+ba/b[height<={height}]",
        "--merge-output-format",
        "mp4",
        "-o",
        str(out_dir / f"{video['video_id']}_{video['youtube_id']}.%(ext)s"),
        video["url"],
    ]


def fetch_real(video_id: str, out_dir: Path, cfg: Config, allow_heldout: bool) -> Path:
    video = pick(load_manifest(), video_id, allow_heldout)
    height = cfg.get("sample.real.max_height")
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = ["uv", "run", "--with", "yt-dlp", *fetch_args(video, out_dir, height)]

    print(
        f"fetching video_id {video['video_id']} ({video['split']}, {video['duration']}, "
        f"{video['domain']}) from {video['url']}"
    )
    proc = subprocess.run(argv, text=True)
    if proc.returncode != 0:
        raise SampleError(
            f"yt-dlp exited {proc.returncode}. Pointers can die — `make corpus-pointers` "
            f"checks whether this one still resolves."
        )
    landed = sorted(out_dir.glob(f"{video['video_id']}_{video['youtube_id']}.*"))
    if not landed:
        raise SampleError(f"yt-dlp reported success but wrote nothing into {out_dir}")
    got = landed[0]
    print(f"\nsample — {got} · {got.stat().st_size / 1e6:.1f} MB (gitignored, as the licence requires)")
    print(f"  next      make demo VIDEO={got.as_posix()}")
    return got


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Produce a sample video for the ingest gate (VRAG-005).")
    ap.add_argument("--out", default=str(SAMPLES / "one.mp4"), help="where to write the synthetic clip")
    ap.add_argument("--config", default="config.toml", help="the file holding the sample spec")
    ap.add_argument(
        "--real",
        metavar="VIDEO_ID",
        help="fetch this corpus video from its recorded url instead of generating one",
    )
    ap.add_argument(
        "--allow-heldout",
        action="store_true",
        help="permit fetching a held-out video (Evaluator only)",
    )
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
        if args.real:
            # --out names the synthetic clip; a real fetch lands beside it, named after
            # its corpus id, because yt-dlp picks the extension.
            fetch_real(args.real, Path(args.out).parent, cfg, args.allow_heldout)
        else:
            synthetic(Path(args.out), cfg)
    except (SampleError, ConfigError) as exc:
        print(f"FAIL — {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
