"""Index one video — the wire between chunking (VRAG-014) and the vector store (VRAG-015).

    make index VIDEO=samples/181_8np5YKYx3sU.mp4

VRAG-015 shipped `embed_and_persist()` and VRAG-016 shipped the query side, but nothing
called the first from a command line, so the Phase 1 gate had an index to score and no way
to build one. This is that step and nothing more: chunk the video (which ingests and
transcribes it, or reuses the cache), embed the chunks, upsert them into Chroma, print what
went in.

Idempotent. Chunk ids are `<video_id>_<t_start>_<t_end>` and the store upserts, so
re-running the same video replaces its rows rather than duplicating them — but only while
the chunk levers hold. Changing `chunk.window_s` moves the boundaries, so the ids change
too and the old rows stay behind as orphans; `--reset` drops the collection first, which is
what you want when a lever moved. That is a real hazard for the gate: an index holding two
generations of chunks scores better than either, for no reason the levers explain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.chunk import build as build_chunks
from src.config import Config
from src.config import load as load_config
from src.embed import Chunk as EmbedChunk
from src.embed import EmbedError, embed_and_persist
from src.telemetry import Meter

MANIFEST = Path("data/corpus/manifest.json")
SAMPLES = Path("samples")


class IndexingError(Exception):
    """Indexing failed — message says which step and why."""


def to_embed_chunks(rows: list[dict]) -> list[EmbedChunk]:
    """chunks.json rows → the four fields the vector store keeps.

    Only these four cross over. The rest (segment ids, which grid window produced the
    chunk) is provenance for `make chunks` to print, not something a citation needs.
    """
    return [
        EmbedChunk(
            video_id=str(r["video_id"]),
            t_start=float(r["t_start"]),
            t_end=float(r["t_end"]),
            text=r["text"],
        )
        for r in rows
    ]


def reset_collection(cfg: Config) -> bool:
    """Delete the collection so a re-index cannot leave orphans behind. True if dropped."""
    import chromadb

    chroma_path = Path(cfg.get("embed.chroma_path"))
    if not chroma_path.exists():
        return False
    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection(cfg.get("embed.collection"))
    except Exception:
        return False
    return True


def index_video(
    video: Path,
    cfg: Config,
    *,
    refresh: bool = False,
    video_id: str | None = None,
) -> dict:
    """Chunk one video and put its chunks in the store. Returns a small report."""
    result = build_chunks(video, cfg, refresh=refresh, video_id=video_id)

    if result["problems"]:
        raise IndexingError(
            f"{video}: chunking reported {len(result['problems'])} problem(s) — indexing a "
            f"chunk whose time range does not hold its segments would index a citation that "
            f"points at the wrong moment. First: {result['problems'][0]}"
        )

    chunks = to_embed_chunks(result["chunks"])
    meter = Meter()
    try:
        indexed = embed_and_persist(chunks, cfg, meter)
    except EmbedError as exc:
        raise IndexingError(
            f"{video}: embedding failed — {exc}\n"
            f"The model in config.toml is {cfg.get('embed.model')!r}; it has to be pulled "
            f"in Ollama as hf.co/<that repo id> before this runs."
        ) from exc

    return {
        "video_id": result["video_id"],
        "duration_s": result["duration_s"],
        "transcript_source": result["transcript"]["source"],
        "segments": result["transcript"]["segments"],
        "chunks": len(chunks),
        "indexed": indexed,
        "collection": cfg.get("embed.collection"),
        "chroma_path": cfg.get("embed.chroma_path"),
        "model": cfg.get("embed.model"),
        "telemetry": meter.summary_line(result["duration_s"]),
        "chunk_telemetry": result["telemetry"],
    }


def dev_videos(manifest: Path = MANIFEST) -> list[dict]:
    """The dev side of the corpus split, lowest video_id first.

    Read rather than hard-coded: `make corpus` can re-select the corpus, and an index built
    over ids that are no longer dev would score the Phase 1 gate on the wrong videos.
    """
    videos = json.loads(manifest.read_text(encoding="utf-8"))["videos"]
    return sorted(
        (v for v in videos if v["split"] == "dev"), key=lambda v: str(v["video_id"])
    )


def local_file(video_id: str, samples: Path = SAMPLES) -> Path | None:
    """The already-fetched file for a corpus id, if there is one.

    `make sample-real` writes `<video_id>_<youtube_id>.<ext>` and yt-dlp picks the
    extension, so the id is a prefix match rather than a known filename.
    """
    if not samples.is_dir():
        return None
    hits = sorted(p for p in samples.glob(f"{video_id}_*") if p.is_file())
    return hits[0] if hits else None


def index_dev_split(cfg: Config, *, refresh: bool = False, reset: bool = True) -> dict:
    """Fetch (if needed) and index every dev video. Returns a per-video report.

    `reset` drops the collection once, before the first video: the chunk levers set the
    chunk ids, so an index built across a lever change holds two generations of rows and
    scores better than either of them.
    """
    from src.sample import fetch_real

    videos = dev_videos()
    if not videos:
        raise IndexingError(f"{MANIFEST}: no videos on the dev side of the split")

    if reset and reset_collection(cfg):
        print(f"dropped collection {cfg.get('embed.collection')!r} — re-indexing from empty")

    reports = []
    for v in videos:
        vid = str(v["video_id"])
        path = local_file(vid)
        if path is None:
            print(f"\nvideo {vid}: not in {SAMPLES.as_posix()}/, fetching from its manifest url")
            fetch_real(vid, SAMPLES, cfg, allow_heldout=False)
            path = local_file(vid)
            if path is None:
                raise IndexingError(
                    f"video {vid}: fetch reported success but nothing landed in "
                    f"{SAMPLES.as_posix()}/"
                )
        print(f"\nvideo {vid}: indexing {path.as_posix()}")
        r = index_video(path, cfg, refresh=refresh, video_id=vid)
        report(r)
        reports.append(r)

    return {
        "videos": [r["video_id"] for r in reports],
        "chunks": sum(r["chunks"] for r in reports),
        "indexed": sum(r["indexed"] for r in reports),
        "duration_s": sum(r["duration_s"] for r in reports),
        "reports": reports,
    }


def report(r: dict, out=None) -> None:
    out = out or sys.stdout
    print(f"indexed video {r['video_id']}", file=out)
    print(f"  duration    {r['duration_s']:.1f}s", file=out)
    print(f"  transcript  {r['segments']} segment(s) ({r['transcript_source']})", file=out)
    print(f"  chunks      {r['chunks']} -> {r['indexed']} row(s) upserted", file=out)
    print(f"  store       {r['collection']} at {r['chroma_path']}", file=out)
    print(f"  model       {r['model']}", file=out)
    print(f"  embed       {r['telemetry']}", file=out)
    print(f"  chunk+asr   {r['chunk_telemetry']}", file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "video", nargs="?", help="path to the video file to index (omit with --dev)"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="index every video on the dev side of the corpus split, fetching what is missing",
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--refresh", action="store_true", help="re-run ASR instead of using the cache"
    )
    parser.add_argument(
        "--video-id", default=None, help="override the corpus video_id for this file"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="drop the collection first — use after changing a chunk lever",
    )
    args = parser.parse_args(argv)

    # cp1252 consoles cannot print the em dashes and arrows below, and a UnicodeEncodeError
    # here would exit non-zero for a reason that has nothing to do with indexing.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    cfg = load_config(args.config)

    if args.reset and reset_collection(cfg):
        print(f"dropped collection {cfg.get('embed.collection')!r}")

    if args.dev:
        try:
            summary = index_dev_split(cfg, refresh=args.refresh, reset=True)
        except Exception as exc:
            print(f"FAIL - {exc}", file=sys.stderr)
            return 1
        print()
        print(
            f"dev split indexed: {len(summary['videos'])} video(s) "
            f"{summary['videos']}, {summary['indexed']} chunk(s) over "
            f"{summary['duration_s'] / 60:.1f} min of video"
        )
        return 0

    if not args.video:
        parser.error("give a video path, or --dev to index the whole dev split")

    try:
        r = index_video(
            Path(args.video), cfg, refresh=args.refresh, video_id=args.video_id
        )
    except Exception as exc:
        print(f"FAIL - {exc}", file=sys.stderr)
        return 1

    report(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
