"""Chunking — VRAG-014.

A timestamped transcript in; a table of time-windowed chunks out, each one carrying the
`video_id`, `t_start` and `t_end` that a citation is later built from.

    make chunks VIDEO=samples/one.mp4

Three things this module is careful about.

**The window is a grid, not a walk.** Windows start at 0, `hop`, 2·`hop`, … where
`hop = window_s - overlap_s`. Both levers live in `config.toml`, and `src.config` refuses to
default them. A fixed grid means the same transcript chunks identically on every run and on
either ASR arm, so a recall number measured last week is comparable to one measured today.

**A chunk's time range is measured, not computed.** Segments are never split, so the last
segment in a window usually runs past the window's end. The chunk records the range its
segments actually cover — `t_start` is the first segment's start, `t_end` the last one's end
— and keeps the grid window it came from alongside, so the two can be told apart. Copying
the grid bounds into `t_start`/`t_end` would be the same class of bug as VRAG-005's
`-vf fps=N`: a tidy number sitting next to content it does not describe.

**Nothing is dropped silently.** `verify()` re-derives the invariants from the chunks and
the segments they were built from, including the one the card names — every chunk has a real
time range that contains every segment in it — and every chunk is reachable from at least
one chunk per segment. The CLI prints the problem count and exits non-zero if it is not 0.

Two windows are dropped on purpose, and both are counted in the output rather than being
invisible: a window with no speech in it (indexing silence costs money and retrieves
nothing) and a window whose segments are exactly the previous window's (overlap can make
two neighbours identical when all the speech falls in their intersection; the duplicate is
index bloat, not recall).

Re-chunking is free. The ASR result is cached in `runs/<video>/transcript.json`, keyed on
the source file's sha256 and the model, so sweeping `window_s` costs $0.00 and makes no
model call. `--refresh` forces a new transcription.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from src.config import Config, ConfigError
from src.config import load as load_config
from src.ingest import OUT_ROOT, IngestError, ingest, sha256_file
from src.telemetry import Meter
from src.transcript import Segment, TranscriptError, bound_to_audio, transcribe

MANIFEST = Path("data/corpus/manifest.json")

# QA_SPEC §2 scores a citation correct when |citation.t_start - t_ref| <= 30. A chunk wider
# than that can hold the answer and still cite too early, so the dump says so out loud.
CITATION_TOLERANCE_S = 30.0


class ChunkError(Exception):
    """The transcript could not be chunked, or the chunks failed their own invariants."""


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of transcript, placed in one video at one time range."""

    chunk_id: str
    video_id: str
    t_start: float  # seconds from video start — the first segment's start
    t_end: float  # the last segment's end, which may run past window_t_end
    text: str
    n_segments: int
    segment_ids: tuple[int, ...]  # indices into the time-ordered segment list
    window_index: int  # which grid window produced it
    window_t_start: float
    window_t_end: float
    speakers: tuple[str, ...] = ()  # VRAG-026: distinct named speakers across this chunk's segments

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "video_id": self.video_id,
            "t_start": round(self.t_start, 3),
            "t_end": round(self.t_end, 3),
            "duration_s": round(self.duration_s, 3),
            "text": self.text,
            "chars": len(self.text),
            "n_segments": self.n_segments,
            "segment_ids": list(self.segment_ids),
            "speakers": list(self.speakers),
            "window_index": self.window_index,
            "window_t_start": round(self.window_t_start, 3),
            "window_t_end": round(self.window_t_end, 3),
        }


# --------------------------------------------------------------------------- config


def chunk_config(cfg: Config) -> dict[str, float]:
    """The two levers, validated. No defaults — see src/config.py."""
    window_s = cfg.get("chunk.window_s")
    overlap_s = cfg.get("chunk.overlap_s")
    for name, value in (("window_s", window_s), ("overlap_s", overlap_s)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfigError(f"{cfg.path}: chunk.{name} must be a number, got {value!r}")
    if window_s <= 0:
        raise ConfigError(f"{cfg.path}: chunk.window_s must be > 0, got {window_s}")
    if overlap_s < 0:
        raise ConfigError(f"{cfg.path}: chunk.overlap_s must be >= 0, got {overlap_s}")
    if overlap_s >= window_s:
        # hop would be <= 0: the grid would never advance and the walk would not terminate.
        raise ConfigError(
            f"{cfg.path}: chunk.overlap_s ({overlap_s}) must be less than chunk.window_s "
            f"({window_s}) — the windows advance by window_s - overlap_s, so an overlap "
            f"that large means the grid never moves."
        )
    return {
        "window_s": float(window_s),
        "overlap_s": float(overlap_s),
        "hop_s": float(window_s) - float(overlap_s),
    }


# --------------------------------------------------------------------------- chunking


def windows(span_s: float, window_s: float, hop_s: float) -> Iterator[tuple[int, float, float]]:
    """The grid: (index, t_start, t_end) for every window that can hold speech.

    Windows start at 0 and step by hop_s. The last one starts at or before span_s, so a
    segment ending at span_s is always inside some window.
    """
    if hop_s <= 0:
        raise ChunkError(f"hop_s must be > 0, got {hop_s}")
    count = int(math.floor(max(span_s, 0.0) / hop_s)) + 1
    for i in range(count):
        start = i * hop_s
        yield i, start, start + window_s


def _in_window(seg: Segment, w_start: float, w_end: float) -> bool:
    """True when seg has any part inside [w_start, w_end).

    A zero-length segment — some ASR responses carry them — has no part to overlap with, so
    it is placed by its point in time instead of being dropped.
    """
    if seg.t_end == seg.t_start:
        return w_start <= seg.t_start < w_end
    return seg.t_start < w_end and seg.t_end > w_start


def order_segments(segments: Iterable[Segment]) -> list[Segment]:
    """Segments in time order. The caller's list may be in whatever order the arm returned."""
    ordered = sorted(segments, key=lambda s: (s.t_start, s.t_end))
    for s in ordered:
        if s.t_start < 0 or s.t_end < s.t_start:
            raise ChunkError(
                f"segment has an impossible time range (t_start={s.t_start}, "
                f"t_end={s.t_end}): {s.text[:60]!r}"
            )
    return ordered


def chunk_segments(
    video_id: str, segments: Iterable[Segment], levers: dict[str, float]
) -> tuple[list[Chunk], dict[str, int]]:
    """Group segments into overlapping time windows. Returns the chunks and what was dropped.

    Segments are not split, so a chunk's range is the union of its segments' ranges. Empty
    windows and windows identical to their predecessor produce no chunk; both are counted.
    """
    ordered = order_segments(segments)
    stats = {"windows": 0, "empty": 0, "duplicate": 0}
    if not ordered:
        return [], stats

    span = max(s.t_end for s in ordered)
    chunks: list[Chunk] = []
    previous_ids: tuple[int, ...] | None = None

    for index, w_start, w_end in windows(span, levers["window_s"], levers["hop_s"]):
        stats["windows"] += 1
        ids = tuple(i for i, s in enumerate(ordered) if _in_window(s, w_start, w_end))
        if not ids:
            stats["empty"] += 1
            continue
        if ids == previous_ids:
            # Same segments as the chunk before it: a second copy buys no recall.
            stats["duplicate"] += 1
            continue
        previous_ids = ids
        members = [ordered[i] for i in ids]
        chunks.append(
            Chunk(
                chunk_id=f"{video_id}-{len(chunks):04d}",
                video_id=video_id,
                # float() so a Chunk's time range is the same type whatever the arm
                # returned; verify() and the JSON dump both assume a real number.
                t_start=float(min(s.t_start for s in members)),
                t_end=float(max(s.t_end for s in members)),
                text=" ".join(s.text for s in members),
                n_segments=len(members),
                segment_ids=ids,
                speakers=tuple(sorted({s.speaker for s in members if s.speaker})),
                window_index=index,
                window_t_start=w_start,
                window_t_end=w_end,
            )
        )
    return chunks, stats


# --------------------------------------------------------------------------- invariants


def _finite(value: Any) -> bool:
    """A real number — not None, not a bool wearing an int's clothes, not NaN or inf."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def verify(
    chunks: list[Chunk], segments: Iterable[Segment], duration_s: float | None = None
) -> list[str]:
    """Re-derive the invariants from the output. Returns one string per problem found.

    The card's criterion is the first four: no chunk loses its time range. The rest are the
    properties the retriever in VRAG-016 will assume without checking.
    """
    ordered = order_segments(segments)
    problems: list[str] = []

    if ordered and not chunks:
        problems.append(f"{len(ordered)} segments produced 0 chunks")

    seen_ids: set[str] = set()
    covered: set[int] = set()
    previous_start: float | None = None

    for c in chunks:
        # 1. a time range exists and is a real number
        if not all(_finite(v) for v in (c.t_start, c.t_end)):
            problems.append(
                f"{c.chunk_id}: t_start/t_end is not a finite number "
                f"({c.t_start!r}, {c.t_end!r})"
            )
            continue
        # 2. it runs forwards, and it is not a point
        if c.t_end <= c.t_start:
            problems.append(
                f"{c.chunk_id}: t_end {c.t_end} is not after t_start {c.t_start}"
            )
        if c.t_start < 0:
            problems.append(f"{c.chunk_id}: t_start {c.t_start} is before the video starts")
        # 3. it contains every segment it claims
        for i in c.segment_ids:
            if not 0 <= i < len(ordered):
                problems.append(f"{c.chunk_id}: segment id {i} is out of range")
                continue
            s = ordered[i]
            if s.t_start < c.t_start or s.t_end > c.t_end:
                problems.append(
                    f"{c.chunk_id}: range [{c.t_start}, {c.t_end}] does not contain "
                    f"segment {i} [{s.t_start}, {s.t_end}]"
                )
            covered.add(i)
        # 4. it is placed in a video
        if not c.video_id:
            problems.append(f"{c.chunk_id}: no video_id")
        # 5. identity and order
        if c.chunk_id in seen_ids:
            problems.append(f"{c.chunk_id}: duplicate chunk_id")
        seen_ids.add(c.chunk_id)
        if previous_start is not None and c.t_start < previous_start:
            problems.append(
                f"{c.chunk_id}: t_start {c.t_start} goes backwards from {previous_start}"
            )
        previous_start = c.t_start
        # 6. it has content, and no segment was silently emptied
        if not c.text.strip():
            problems.append(f"{c.chunk_id}: empty text")
        if c.n_segments != len(c.segment_ids):
            problems.append(
                f"{c.chunk_id}: n_segments {c.n_segments} != {len(c.segment_ids)} segment ids"
            )
        if duration_s is not None and c.t_end > duration_s + 1.0:
            problems.append(
                f"{c.chunk_id}: t_end {c.t_end} is past the end of the video ({duration_s})"
            )

    # 7. no speech was dropped on the floor
    missing = sorted(set(range(len(ordered))) - covered)
    if missing:
        shown = ", ".join(
            f"{i} [{ordered[i].t_start}, {ordered[i].t_end}]" for i in missing[:5]
        )
        problems.append(
            f"{len(missing)} segment(s) are in no chunk: {shown}"
            f"{' …' if len(missing) > 5 else ''}"
        )
    return problems


# --------------------------------------------------------------------------- video_id


def resolve_video_id(video: Path, manifest: Path = MANIFEST) -> tuple[str, str]:
    """The corpus video_id for a file, and where it came from.

    `make sample-real` writes `<video_id>_<youtube_id>.mp4`, so the id is recoverable from
    the filename — but only if the manifest agrees, because a chunk labelled with a video_id
    the corpus does not know is a citation that cannot be resolved. Anything else (the
    synthetic fixture, a one-off file) is identified by its stem.
    """
    stem = video.stem
    head = stem.split("_", 1)[0]
    try:
        videos = json.loads(manifest.read_text(encoding="utf-8"))["videos"]
    except (OSError, KeyError, json.JSONDecodeError):
        return stem, "filename stem (corpus manifest unreadable)"
    for v in videos:
        if str(v["video_id"]) == head:
            return str(v["video_id"]), f"{manifest.as_posix()} ({v['split']} split)"
    return stem, "filename stem (not a corpus video)"


# --------------------------------------------------------------------------- transcript


def _transcript_path(out_dir: Path) -> Path:
    return out_dir / "transcript.json"


def save_transcript(
    path: Path, segments: list[Segment], source_sha256: str, cfg: Config
) -> None:
    payload = {
        "task": "VRAG-008",
        "source_sha256": source_sha256,
        "arm": cfg.get("transcript.arm"),
        "model": cfg.get("transcript.model"),
        "language": cfg.get("transcript.language"),
        "segments": [
            {"t_start": round(s.t_start, 3), "t_end": round(s.t_end, 3), "text": s.text, "speaker": s.speaker}
            for s in segments
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_transcript(path: Path) -> tuple[list[Segment], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChunkError(f"cannot read the cached transcript {path}: {exc}") from exc
    raw = payload.get("segments")
    if not isinstance(raw, list):
        raise ChunkError(f"{path}: no 'segments' list")
    segments = []
    for i, s in enumerate(raw):
        try:
            segments.append(
                Segment(t_start=float(s["t_start"]), t_end=float(s["t_end"]), text=s["text"], speaker=s.get("speaker"))
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ChunkError(f"{path}: segment {i} is malformed ({exc})") from exc
    return segments, payload


def transcript_for(
    video: Path, cfg: Config, meter: Meter, out_dir: Path, refresh: bool
) -> tuple[list[Segment], dict[str, Any], str]:
    """Segments for one video, from cache when the cache is for these exact bytes.

    Returns the segments, the ingest result (or the cached media.json), and a one-word
    source so the dump can say whether a model was called.
    """
    media_path = out_dir / "media.json"
    digest = sha256_file(video)

    cached_media: dict[str, Any] | None = None
    if media_path.is_file():
        try:
            candidate = json.loads(media_path.read_text(encoding="utf-8"))
            if candidate.get("source", {}).get("sha256") == digest:
                cached_media = candidate
        except (OSError, json.JSONDecodeError):
            cached_media = None
    media = cached_media if cached_media is not None else ingest(video, cfg, out_dir.parent)

    cache = _transcript_path(out_dir)
    if not refresh and cache.is_file():
        segments, payload = load_transcript(cache)
        if (
            payload.get("source_sha256") == digest
            and payload.get("model") == cfg.get("transcript.model")
            and payload.get("arm") == cfg.get("transcript.arm")
        ):
            # A cached transcript is bounded on the way out as well as on the way in. The
            # arms clamp each piece to its own audio (transcript.bound_to_audio), but a
            # transcript.json written before they did still holds whisper's padded tail,
            # and re-transcribing 90 minutes to drop one bad segment is nine hosted calls
            # to fix something already on disk. Cheap, and it makes the cache and a fresh
            # run agree - which is the property that stops "works after a refresh" bugs.
            return (
                bound_to_audio(
                    segments, float(media["audio"]["duration_s"]), cache.name
                ),
                media,
                "cache",
            )

    wav = Path(media["audio"]["path"])
    if not wav.is_file():
        # media.json was reused but the wav it names is gone; re-run ingest for it.
        media = ingest(video, cfg, out_dir.parent)
        wav = Path(media["audio"]["path"])
    segments = transcribe(wav, cfg, meter)
    save_transcript(cache, segments, digest, cfg)
    return segments, media, cfg.get("transcript.arm")


# --------------------------------------------------------------------------- pipeline


def build(
    video: Path,
    cfg: Config,
    out_root: Path = OUT_ROOT,
    refresh: bool = False,
    video_id: str | None = None,
) -> dict[str, Any]:
    video = Path(video)
    if not video.is_file():
        raise ChunkError(f"{video}: not a file. `make sample` writes samples/one.mp4.")

    levers = chunk_config(cfg)
    meter = Meter()
    started = time.perf_counter()
    out_dir = out_root / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_id, id_source = resolve_video_id(video)
    if video_id:
        resolved_id, id_source = video_id, "--video-id"

    segments, media, transcript_source = transcript_for(video, cfg, meter, out_dir, refresh)
    duration_s = media["media"]["duration_s"]

    chunks, stats = chunk_segments(resolved_id, segments, levers)
    problems = verify(chunks, segments, duration_s)
    total = time.perf_counter() - started

    rows = [c.as_dict() for c in chunks]
    result = {
        "task": "VRAG-014",
        "video_id": resolved_id,
        "video_id_source": id_source,
        "source": {"path": video.as_posix(), "sha256": media["source"]["sha256"]},
        "duration_s": duration_s,
        "transcript": {
            "source": transcript_source,
            "model": cfg.get("transcript.model"),
            "segments": len(segments),
            "path": _transcript_path(out_dir).as_posix(),
        },
        "levers": levers,
        "counts": {
            "chunks": len(chunks),
            "windows": stats["windows"],
            "windows_empty": stats["empty"],
            "windows_duplicate": stats["duplicate"],
            "segment_placements": sum(c.n_segments for c in chunks),
            "chars": sum(len(c.text) for c in chunks),
            "transcript_chars": sum(len(s.text) for s in segments),
        },
        "coverage": {
            "t_start": rows[0]["t_start"] if rows else None,
            "t_end": max((r["t_end"] for r in rows), default=None),
            "max_chunk_duration_s": max((r["duration_s"] for r in rows), default=None),
            "mean_chunk_duration_s": round(
                sum(r["duration_s"] for r in rows) / len(rows), 3
            )
            if rows
            else None,
            "over_citation_tolerance": sum(
                1 for r in rows if r["duration_s"] > CITATION_TOLERANCE_S
            ),
            # A chunk can overhang its window at both ends by this much, so it is what sets
            # the real ceiling on chunk.window_s. Depends on the ASR arm, so it is measured.
            "longest_segment_s": round(
                max((s.t_end - s.t_start for s in segments), default=0.0), 3
            ),
        },
        "problems": problems,
        "config": cfg.fingerprint(),
        "timing": {"total_s": round(total, 3)},
        "chunks": rows,
        "telemetry": meter.summary_line(duration_s, wall_s=total),
    }

    manifest = out_dir / "chunks.json"
    manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result["manifest"] = manifest.as_posix()
    return result


# --------------------------------------------------------------------------- report


def _clip(text: str, width: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def report(result: dict[str, Any], out=None, text_width: int = 44) -> None:
    """Dump the chunk table. A chunk table not printed is not a chunk table.

    `out` is resolved here, not in the signature: a default of sys.stdout binds the stream
    that existed at import time, which is not the one a test — or a redirected gate log —
    is reading.
    """
    out = sys.stdout if out is None else out
    lv, cn, cv, tr = result["levers"], result["counts"], result["coverage"], result["transcript"]

    print(f"chunks — {result['source']['path']}", file=out)
    print(
        f"  video_id  {result['video_id']}  (from {result['video_id_source']})",
        file=out,
    )
    print(
        f"  levers    window={lv['window_s']}s overlap={lv['overlap_s']}s "
        f"hop={lv['hop_s']}s (from {result['config']['path']})",
        file=out,
    )
    print(
        f"  transcript {tr['segments']} segments from {tr['source']} · {tr['model']} · "
        f"{tr['path']}",
        file=out,
    )
    print(file=out)
    header = (
        f"  {'chunk_id':<14}{'t_start':>9}{'t_end':>9}{'dur':>7}{'segs':>6}{'chars':>7}  text"
    )
    print(header, file=out)
    print(f"  {'-' * (len(header) - 2)}", file=out)
    for r in result["chunks"]:
        print(
            f"  {r['chunk_id']:<14}{r['t_start']:>9.3f}{r['t_end']:>9.3f}"
            f"{r['duration_s']:>7.1f}{r['n_segments']:>6}{r['chars']:>7}  "
            f"{_clip(r['text'], text_width)}",
            file=out,
        )
    if not result["chunks"]:
        print("  (no chunks — the transcript has no segments)", file=out)
    print(file=out)

    span = (
        f"{cv['t_start']:.3f}–{cv['t_end']:.3f} s of {result['duration_s']:.2f} s"
        if cv["t_start"] is not None
        else f"nothing of {result['duration_s']:.2f} s"
    )
    print(
        f"  {cn['chunks']} chunks from {cn['windows']} windows "
        f"({cn['windows_empty']} empty, {cn['windows_duplicate']} duplicate) · covers {span}",
        file=out,
    )
    if cv["max_chunk_duration_s"] is not None:
        print(
            f"  duration  mean {cv['mean_chunk_duration_s']:.1f} s · max "
            f"{cv['max_chunk_duration_s']:.1f} s · longest segment "
            f"{cv['longest_segment_s']:.2f} s, which is what a chunk overhangs its window by",
            file=out,
        )
    print(
        f"  text      {cn['chars']} chars indexed from {cn['transcript_chars']} transcript "
        f"chars ({cn['chars'] / cn['transcript_chars']:.2f}× — the overlap)"
        if cn["transcript_chars"]
        else "  text      nothing to index",
        file=out,
    )
    print(
        f"  placement {cn['segment_placements']} segment slots for {tr['segments']} segments "
        f"· every segment in >=1 chunk",
        file=out,
    )
    if cv["over_citation_tolerance"]:
        # Not an invariant failure: a wide chunk still retrieves. It is a citation that
        # starts too early to be scored correct, which is a lever problem, not a bug.
        # Segments are not split, so a chunk overhangs its window at both ends and can run
        # to window_s + 2x the longest segment — which is why window_s alone is not the
        # bound and this has to be measured per ASR arm rather than derived.
        print(
            f"  WARN      {cv['over_citation_tolerance']} of {cn['chunks']} chunk(s) span "
            f"more than {CITATION_TOLERANCE_S:.0f} s, which is the +/-"
            f"{CITATION_TOLERANCE_S:.0f} s a citation is scored on (QA_SPEC 2), so they can "
            f"retrieve the right passage and still cite too early. A chunk runs to "
            f"window_s + 2x the longest segment, so lower chunk.window_s in "
            f"{result['config']['path']} until this reads 0.",
            file=out,
        )

    problems = result["problems"]
    if problems:
        print(f"\n  FAIL      {len(problems)} problem(s) — a chunk lost its time range:", file=out)
        for p in problems:
            print(f"              {p}", file=out)
    else:
        print(f"  invariants 0 problems in {cn['chunks']} chunks", file=out)

    print(f"\nwrote {result['manifest']}", file=out)
    print(result["telemetry"], file=out)


def main(argv: list[str] | None = None) -> int:
    # The table prints ids and text from real videos; a cp1252 console must not turn that
    # into an exit code that reads like a failed invariant (VRAG-013 hit this).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(
        description="Chunk one video's transcript into overlapping time windows (VRAG-014)."
    )
    ap.add_argument("video", help="path to the video file")
    ap.add_argument("--config", default="config.toml", help="the file holding the levers")
    ap.add_argument("--out", default=str(OUT_ROOT), help=f"output root (default {OUT_ROOT})")
    ap.add_argument(
        "--video-id",
        help="override the corpus video_id (default: from data/corpus/manifest.json, "
        "else the filename stem)",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="re-transcribe instead of reusing runs/<video>/transcript.json (costs an ASR call)",
    )
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
        result = build(
            Path(args.video), cfg, Path(args.out), refresh=args.refresh, video_id=args.video_id
        )
    except (ChunkError, ConfigError, IngestError, TranscriptError) as exc:
        print(f"FAIL — {exc}", file=sys.stderr)
        return 1

    report(result)
    sys.stdout.flush()
    # An invariant failure is the one thing this command exists to catch, so it is the exit
    # code, not a line in the middle of the output.
    return 1 if result["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
