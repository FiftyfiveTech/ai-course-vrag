"""The VRAG-023 deliverable — keyframe captioning, hosted arm vs local arm, cost measured.

    make caption-arms VIDEO=samples/vector7-21aug-client-meeting.mp4
    make caption-arms VIDEO=611 ARMS_FLAGS="--limit 5"

VRAG-023's acceptance criterion is "2-arm cost table hosted vs local; the gate is untouched".
This produces the table. Nothing here writes to `./chroma`, edits `config.toml`, or touches
`runs/<stem>/captions.json` — see "What it does not do" below.

The arm is the only variable
----------------------------
Keyframes are selected **once**, before either arm runs, and the same list is handed to both.
That matters more than it looks: the selection is `src/keyframes.py`, which is deterministic,
but `caption.max_keyframes` and `--limit` both truncate it, and two arms given two different
truncations of the same video would produce a table whose rows differ by their input. So
selection happens here and `src.caption.caption_video` is called with `keyframes=` for each
arm.

The prompt is shared for the same reason (`src.caption.CAPTION_TASK`,
`prompts/caption_v1.md`), and neither arm is given a JSON schema — the two providers do not
offer the same structured-output guarantees, so constraining generation would put "how it was
asked" into a table meant to isolate "what ran". `prompts/caption_v1.md` has the long version.

What the money column means
---------------------------
$0.00, on both rows, and that is the true number rather than a placeholder: the hosted arm
runs on NVIDIA NIM's free tier and the local arm runs on this laptop. `src/telemetry.py`
carries both models at rate 0.0 with a comment saying so, and deliberately does **not** carry
an invented paid rate — a modelled price nobody looked up is a number in a cost table that no
command produced.

So what the two arms actually cost is spent in the other columns: **tokens** and
**seconds**, per call and projected per video-hour. Those are measured, and they are what
differs — a hosted 11B model and a local 3B model do not cost the same thing, they cost
different things.

Run the local arm twice before believing its seconds. Measured on the client meeting, same code
and same frames: 87.65 s/call on the first run and 19.58 s/call on the second. The first paid a
cold load of 2.8 GB of weights into RAM and the second found the model resident in Ollama. A
single-run local benchmark on a freshly pulled model measures the disk, not the model.

What it does not do
-------------------
It does not write `captions.json`. Two arms cannot both own one document, and a table run that
overwrote whichever arm ran last would make `make captions` output depend on the last
experiment rather than on the configured arm. Both arms run with `write=False`; the table and
its JSON are the output.

It also does not index anything. `caption.index` is false and no caption reaches the `vrag`
collection — that is the second half of the acceptance criterion.

And only **counts** leave this script. `data/corpus/PROVENANCE.md` forbids redistributing
Video-MME content, and a caption is that content transcribed — so the JSON records how many
characters a caption had and whether it had any, never the text itself. Same rule
`tools/sweep_chunking.py` follows for chunk text. The captions themselves stay in
`runs/<stem>/captions.json`, which is gitignored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.caption import NIM, OLLAMA, CaptionError, caption_video, frames_record
from src.config import load as load_config
from src.ingest import IngestError
from src.keyframes import select
from src.telemetry import Meter

DEFAULT_OUT = ROOT / "docs" / "learning" / "data" / "caption_arms.json"

# Both arms, hosted first. Ordered rather than a set because the hosted arm is the shipped
# default and a table reads better with the shipped row on top — the same convention
# tools/sweep_chunking.py follows by marking its shipped point.
ARM_ORDER = (NIM, OLLAMA)
ARM_LABELS = {NIM: "hosted", OLLAMA: "local"}

# How many keyframes each arm captions by default. Deliberately small: the point of the table
# is a per-call number, and the client meeting selects 64 stretches, so an unlimited run would
# be 128 vision calls across two arms to learn what 20 teaches. The projection to a whole
# video is done arithmetically from the selection ratio and labelled as a projection.
DEFAULT_LIMIT = 10


def run_arm(
    video_id: str,
    arm: str,
    keyframes: list[dict],
    cfg,
    *,
    runs_found: int,
) -> dict:
    """One arm over the shared keyframes. Never raises — a dead arm is a row, not a crash.

    An arm that cannot run is a real and reportable outcome: the hosted one needs a key and a
    free tier that has not been spent, the local one needs a multi-GB pull. Losing the arm that
    *did* work because the other one did not would be the worst possible failure mode for a
    comparison, so the error lands in the row.
    """
    meter = Meter()
    started = time.perf_counter()
    try:
        stored = caption_video(
            video_id,
            cfg,
            meter,
            arm=arm,
            keyframes=keyframes,
            runs_found=runs_found,
            write=False,
        )
    except (CaptionError, IngestError) as exc:
        return {
            "arm": arm,
            "label": ARM_LABELS[arm],
            "model": None,
            "error": str(exc),
            "calls": 0,
        }
    wall = time.perf_counter() - started

    calls = len(stored.captions)
    with_text = sum(1 for c in stored.captions if c.has_text)
    chars = sum(len(c.text) for c in stored.captions)
    return {
        "arm": arm,
        "label": ARM_LABELS[arm],
        "model": stored.model,
        "error": None,
        "calls": calls,
        "tokens": stored.tokens,
        "tokens_per_call": round(stored.tokens / calls, 1) if calls else 0.0,
        "latency_s": round(stored.latency_s, 3),
        "s_per_call": round(stored.latency_s / calls, 3) if calls else 0.0,
        "wall_s": round(wall, 3),
        "cost_usd": round(meter.total_cost_usd(), 6),
        "with_text": with_text,
        "text_yield": round(with_text / calls, 4) if calls else 0.0,
        "chars": chars,
        "chars_per_caption": round(chars / with_text, 1) if with_text else 0.0,
        # Projected, not measured: what the whole video would cost at this arm's per-call rate
        # if every selected stretch were captioned. Labelled as a projection everywhere it is
        # printed, because `runs_found` stretches were selected and only `calls` were run.
        "projected_calls": runs_found,
        "projected_s": round(stored.latency_s / calls * runs_found, 1) if calls else 0.0,
        "projected_tokens": int(stored.tokens / calls * runs_found) if calls else 0,
        "captions": [
            {
                "frame": c.frame,
                "t_start": c.t_start,
                "t_end": c.t_end,
                "has_text": c.has_text,
                "chars": len(c.text),
            }
            for c in stored.captions
        ],
    }


def table(rows: list[dict]) -> str:
    head = (
        f"{'arm':<7} {'model (HF repo id)':<44} {'calls':>5} {'tokens':>7} "
        f"{'tok/call':>8} {'s/call':>7} {'yield':>6} {'chars':>6} {'$':>6}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        if r["error"]:
            lines.append(f"{r['label']:<7} {'— did not run —':<44} {r['error'][:60]}")
            continue
        lines.append(
            f"{r['label']:<7} {r['model']:<44} {r['calls']:5d} {r['tokens']:7d} "
            f"{r['tokens_per_call']:8.1f} {r['s_per_call']:7.2f} "
            f"{r['text_yield']:5.0%} {r['chars_per_caption']:6.0f} "
            f"{r['cost_usd']:6.2f}"
        )
    return "\n".join(lines)


def projection(rows: list[dict], duration_s: float) -> str:
    """What a whole video would cost per arm, at the measured per-call rate."""
    hours = duration_s / 3600 if duration_s else 0.0
    head = f"{'arm':<7} {'calls':>6} {'tokens':>8} {'seconds':>8} {'s/video-hour':>13} {'$/video-hour':>13}"
    lines = [head, "-" * len(head)]
    for r in rows:
        if r["error"]:
            continue
        per_hour = r["projected_s"] / hours if hours else 0.0
        lines.append(
            f"{r['label']:<7} {r['projected_calls']:6d} {r['projected_tokens']:8d} "
            f"{r['projected_s']:8.1f} {per_hour:13.1f} {0.0:13.4f}"
        )
    return "\n".join(lines)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Caption keyframes on both arms, and cost it.")
    ap.add_argument(
        "video", help="the media file in samples/, or a bare video_id already ingested"
    )
    ap.add_argument("--config", default=str(ROOT / "config.toml"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"keyframes per arm (default {DEFAULT_LIMIT}); 0 for every selected stretch",
    )
    ap.add_argument(
        "--arm",
        action="append",
        choices=list(ARM_ORDER),
        help="run only this arm (repeatable); default is both",
    )
    args = ap.parse_args(argv)

    # A caption is transcribed text off a slide, so a typographic quote is the expected case.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    cfg = load_config(Path(args.config))
    video = Path(args.video)
    if video.is_file():
        from src.chunk import resolve_video_id

        video_id, _ = resolve_video_id(video)
    else:
        video_id = args.video

    try:
        frames = frames_record(video_id)
    except CaptionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    records = frames["frames"]
    step_s = 1.0 / float(frames["fps"])
    threshold = float(cfg.get("caption.still_threshold"))
    min_run = int(cfg.get("caption.min_run_frames"))

    # Selected once, for both arms. See the module docstring.
    print(
        f"selecting keyframes: {len(records)} frames, still_threshold={threshold}, "
        f"min_run_frames={min_run} ...",
        flush=True,
    )
    try:
        # Selected UNLIMITED and truncated here, rather than asking select() for `--limit`
        # frames. Coverage is a property of the video, so it has to be summed over the whole
        # selection — and one ffmpeg pass is enough to get both numbers.
        all_keyframes, stage, found = select(
            Path(frames["dir"]),
            records,
            threshold=threshold,
            min_frames=min_run,
            step_s=step_s,
            fmt=str(frames["format"]),
            limit=None,
        )
    except (CaptionError, IngestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    keyframes = all_keyframes if args.limit == 0 else all_keyframes[: args.limit]

    duration_s = float(frames.get("covers_s") or 0.0)
    covered_s = sum(k["t_end"] - k["t_start"] for k in keyframes)
    still_s = sum(k["t_end"] - k["t_start"] for k in all_keyframes)
    reduction = len(records) / found if found else 0.0
    print(
        f"  {len(records)} frames -> {found} still stretches "
        f"({reduction:.1f}x fewer vision calls), scored in {stage.seconds:.1f}s"
    )
    if found and duration_s:
        print(
            f"  still stretches cover {still_s:.0f}s of {duration_s:.0f}s "
            f"({still_s / duration_s:.1%}) — this is what 'slide-heavy' measures"
        )
    if not keyframes:
        print(
            "\nNo still stretches were selected, so there is nothing to caption and no table "
            "to print. That is a result, not a failure: it is what a video with no slides in "
            "it looks like. Loosen caption.still_threshold or lower caption.min_run_frames to "
            "see what a looser rule would pick up — and read the grid in config.toml first, "
            "because a looser rule stops discriminating.",
            file=sys.stderr,
        )
        return 1

    arms = args.arm or list(ARM_ORDER)
    rows = []
    for arm in arms:
        print(f"\n{ARM_LABELS[arm]} arm ({arm}): {len(keyframes)} keyframes", flush=True)
        row = run_arm(
            video_id,
            arm,
            keyframes,
            cfg,
            runs_found=found,
        )
        rows.append(row)
        if row["error"]:
            print(f"  did not run — {row['error']}", file=sys.stderr)
        else:
            print(
                f"  {row['calls']} calls, {row['latency_s']:.1f}s, {row['tokens']} tokens, "
                f"{row['with_text']}/{row['calls']} with text"
            )

    payload = {
        "task": "VRAG-023",
        "video_id": video_id,
        "duration_s": duration_s,
        "config": cfg.fingerprint(),
        "prompt": str(cfg.get("caption.prompt")),
        "selection": {
            "frames_considered": len(records),
            "runs_found": found,
            "reduction": round(reduction, 2),
            "still_threshold": threshold,
            "min_run_frames": min_run,
            "captioned_per_arm": len(keyframes),
            "captioned_covers_s": round(covered_s, 1),
            "still_covers_s": round(still_s, 1),
            "still_covers_frac": round(still_s / duration_s, 4) if duration_s else 0.0,
            "scored_in_s": round(stage.seconds, 2),
        },
        "arms": rows,
        "note": (
            "cost_usd is 0.00 on both arms because both run on a free tier or locally; "
            "src/telemetry.py records rate 0.0 for each and deliberately carries no invented "
            "paid rate. projected_* columns are arithmetic from the measured per-call rate "
            "over runs_found stretches, not measured."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print(f"MEASURED — {len(keyframes)} keyframes per arm, video {video_id}")
    print(table(rows))
    if duration_s:
        print()
        print(f"PROJECTED to all {found} stretches of a {duration_s / 60:.0f}-minute video")
        print(projection(rows, duration_s))
    print()
    print(f"wrote {_rel(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
