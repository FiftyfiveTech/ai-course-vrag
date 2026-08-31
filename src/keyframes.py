"""Which frames are worth a vision call — VRAG-023.

    from src.keyframes import scene_scores, still_runs, keyframe, spans
    scores = scene_scores(Path("runs/bob-video/frames"), "jpg")
    runs = still_runs(scores, threshold=0.02, min_frames=3)

Why this module exists
----------------------
Ingest already sampled the frames (VRAG-005, `ingest.frames.fps = 0.2`), and there are more of
them than anyone would pay to caption: 1091 for the 91-minute client meeting, 680 for
bob-video, 3086 across the four dev videos. Captioning all of them is the cost, and it is the
cost whether or not the frames were worth reading.

A slide is worth reading and it holds still. That is the whole selection rule: find the
stretches where consecutive samples barely differ, and caption **one** frame per stretch
instead of every frame in it. Measured on the indexed corpus at `threshold=0.02,
min_frames=3` — the defaults in `config.toml` `[caption]`:

    video                           calls          still runs cover
    vector7-21aug-client-meeting    1091 ->  64    90.6% of 5455 s
    bob-video                        680 ->  62    39.6%
    701_-dfvdKf-KR0                  654 ->  12     6.1%
    611_H8fGd3fCJbg                  361 ->   7     6.9%

The coverage column is what makes "slide-heavy" a measured property rather than a label. The
same two levers select almost the whole screen-share meeting and almost none of the Bernini
documentary, which is the discrimination the feature needs — and neither number came from
looking at a video, so a new video is classified by the same rule.

Where the score comes from
--------------------------
ffmpeg's `select` filter computes a `scene` score per frame: how different this frame is from
the one before it. Run over the *already-extracted frames* as an image sequence, that is one
number per sample, and no video is decoded a second time:

    ffmpeg -framerate 1 -i runs/<stem>/frames/frame_%05d.jpg \\
           -vf "select='gte(scene,0)',metadata=print" -an -f null -

`gte(scene,0)` is always true, so nothing is filtered out and every frame prints its score —
the filter is being used as a measuring instrument, not as a filter. Parsing it out of stderr
is the same trick `ingest.frame_timestamps` already uses on `showinfo`, and for the same
reason: ffmpeg reports these numbers nowhere else.

The split in here
-----------------
`scene_scores` runs ffmpeg. Everything else is pure and takes the scores as an argument, so
the selection rule is unit-tested on lists of floats rather than on a machine with ffmpeg and
a corpus. That division is deliberate: the rule is what a bad number would come from, and the
subprocess is the part that cannot be tested cheaply.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.ingest import FFMPEG_TIMEOUT_S, IngestError, Stage, _run

# `metadata=print` writes one of these per frame. Anchored on the key so a stray float
# elsewhere in ffmpeg's chatter cannot be read as a score.
SCENE_RE = re.compile(r"\blavfi\.scene_score=([0-9]+(?:\.[0-9]+)?)")


def scene_args(frames_dir: Path, fmt: str) -> list[str]:
    """The measuring command. Pure, so a test can read it without ffmpeg installed.

    `-framerate 1` is not a claim about the video's frame rate and does not affect the
    scores — the image2 demuxer needs *some* rate, and one frame per second keeps the
    printed `pts_time` equal to the frame index, which is what makes a mismatch obvious
    when reading raw stderr by hand.

    `-loglevel info` rather than `ingest.QUIET`: `metadata=print` writes at info level, so
    quieting ffmpeg here would silence the only output this function wants. Same trade the
    frames pass in `ingest.frame_args` makes.
    """
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-framerate",
        "1",
        "-i",
        str(frames_dir / f"frame_%05d.{fmt}"),
        "-vf",
        "select='gte(scene,0)',metadata=print",
        "-an",
        "-f",
        "null",
        "-",
    ]


def scene_scores(frames_dir: Path, fmt: str = "jpg") -> tuple[list[float], Stage]:
    """How different each sampled frame is from the one before it, in frame order.

    The first element is always 0.0 and is **not** a measurement: frame 0 has no predecessor,
    and ffmpeg prints a zero for it. `still_runs` starts at index 1 for exactly that reason —
    counting it as "still" would silently make every video's first frame the start of a
    stretch it may have nothing to do with.
    """
    files = sorted(frames_dir.glob(f"frame_*.{fmt}"))
    if not files:
        raise IngestError(
            f"{frames_dir}: no frame_*.{fmt} files to score. Ingest the video first: "
            f"make chunks VIDEO=<the file in samples/>"
        )
    proc, stage = _run(
        "scene", scene_args(frames_dir, fmt), FFMPEG_TIMEOUT_S, subject=frames_dir
    )
    scores = [float(m) for m in SCENE_RE.findall(proc.stderr or "")]
    if len(scores) != len(files):
        raise IngestError(
            f"{frames_dir}: ffmpeg scored {len(scores)} frames but {len(files)} are on disk, "
            f"so a score cannot be attributed to a frame. Frames must be numbered "
            f"consecutively from frame_00001.{fmt} — the image2 demuxer stops at the first "
            f"gap."
        )
    return scores, stage


def still_runs(
    scores: list[float], threshold: float, min_frames: int
) -> list[tuple[int, int]]:
    """Stretches where the picture barely changes, as inclusive `(first, last)` frame indices.

    `scores[i]` is the difference between frame `i` and frame `i-1`, so a *run of k
    consecutive low scores covers k+1 frames* — the frame before the first low score is part
    of the same still stretch, and dropping it would shorten every span by one sample and
    lose the moment the slide actually appeared.

    `min_frames` is a floor on frames covered, not on scores seen. Two frames that happen to
    resemble each other is noise; three or more is something holding still. Measured: at
    `threshold=0.02`, dropping `min_frames` from 3 to 2 takes the Bernini documentary from 7
    selected stretches to 41 — i.e. it stops discriminating, because a documentary has plenty
    of two-frame coincidences and very few real slides.

    Index 0 is skipped: its score is ffmpeg's synthetic zero for a frame with no predecessor.
    """
    if min_frames < 2:
        raise ValueError(
            f"min_frames must be at least 2 — a stretch of one frame is a frame, not a "
            f"stretch, and there is no score to compare it with. Got {min_frames}."
        )
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(1, len(scores)):
        if scores[i] < threshold:
            # The still stretch begins at the *previous* frame, which this score compares to.
            if start is None:
                start = i - 1
        elif start is not None:
            if i - start >= min_frames:
                runs.append((start, i - 1))
            start = None
    # A stretch that runs to the last frame is never closed by a high score, so it has to be
    # closed here. Forgetting this drops the final slide of every deck that ends on one.
    if start is not None and len(scores) - start >= min_frames:
        runs.append((start, len(scores) - 1))
    return runs


def keyframe(run: tuple[int, int]) -> int:
    """Which frame of a still stretch to caption: the **last** one.

    Not the middle and not the first. A slide is often still being drawn when the stretch
    begins — a build animation, a cursor moving into place, a window still rendering — and it
    is complete by the end of the stretch it holds for. The last frame is the one that has all
    of the text on it, and the text is what a caption is for.
    """
    return run[1]


def spans(
    runs: list[tuple[int, int]], frames: list[dict[str, Any]], step_s: float
) -> list[tuple[float, float]]:
    """Each still stretch as `(t_start, t_end)` seconds on the video clock.

    Measured off the `t_s` values ingest recorded for the frames themselves, never off the
    sampling grid — the same rule `src.chunk` follows when it takes a chunk's span from the
    segments inside it. The grid says where a sample was *asked* for; `t_s` is ffmpeg's
    `pts_time` for the frame it actually kept, and on a variable-frame-rate source those
    differ.

    The stretch's end is the last frame's own timestamp plus one sampling step, because the
    slide is on screen until the next sample proves otherwise, not only at the instant it was
    photographed.
    """
    out = []
    for first, last in runs:
        t_start = float(frames[first]["t_s"])
        t_end = float(frames[last]["t_s"]) + step_s
        out.append((t_start, max(t_end, t_start)))
    return out


def select(
    frames_dir: Path,
    frames: list[dict[str, Any]],
    *,
    threshold: float,
    min_frames: int,
    step_s: float,
    fmt: str = "jpg",
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], Stage, int]:
    """Score, group, and return one keyframe per still stretch with its span.

    Returns the selected keyframes, the ffmpeg stage that scored them, and the number of
    stretches found *before* `limit` was applied — the ratio between that and the frames on
    disk is the cost reduction the arms are measured against, and it must not be silently
    changed by sampling fewer of them.
    """
    scores, stage = scene_scores(frames_dir, fmt)
    runs = still_runs(scores, threshold, min_frames)
    windows = spans(runs, frames, step_s)

    selected = []
    for run, (t_start, t_end) in zip(runs, windows):
        idx = keyframe(run)
        record = frames[idx]
        selected.append(
            {
                "frame": record["file"],
                "index": idx,
                "t_s": float(record["t_s"]),
                "t_start": t_start,
                "t_end": t_end,
                "run_frames": run[1] - run[0] + 1,
                "bytes": int(record.get("bytes", 0)),
            }
        )
    found = len(selected)
    if limit is not None and limit >= 0:
        selected = selected[:limit]
    return selected, stage, found
