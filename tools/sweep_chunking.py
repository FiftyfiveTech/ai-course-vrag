"""Chunking sweep — the measurements behind the VRAG-018 primer.

    make sweep            # the full grid, ~12 points, $0.00
    make sweep-dry        # chunk only, no embedding, seconds

The primer (docs/learning/primer-chunking-embeddings.md) makes three claims —
window trades recall against citability, overlap buys recall with duplicated text, and the
cost of both is paid in tokens and wall-clock rather than dollars while the embedder is
local. This is the script that produces the numbers those claims are made of, so a reader
can re-run it and disagree with the table rather than with the prose.

What it does not do
-------------------
It does not touch `config.toml` and it does not write to `./chroma`. Phase 1 passed at
window_s = 25.0 / overlap_s = 8.0 (VRAG-017) and a sweep that edits the levers in place
would silently re-tune a gate that has already been graded. Each grid point gets its own
Chroma directory under `runs/sweep/`, and the levers are overridden on an in-memory copy of
the config.

Why it re-chunks from the cached transcript
-------------------------------------------
`src.chunk.build()` would ingest and transcribe first; the transcripts are already on disk
from VRAG-017 and re-running ASR 12 times would cost money on the Groq arm for no new
information. So this calls `chunk_segments()` directly on `runs/<stem>/transcript.json`.
Consequence worth stating: the sweep is only valid for the ASR arm that produced those
transcripts, because the chunk-duration ceiling depends on how long that arm's segments
run. The output records the arm.

Only counts leave this script. `data/corpus/PROVENANCE.md` forbids redistributing
Video-MME text, so the JSON holds chunk counts, durations, character and word totals and
recall — never a chunk's text.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunk import chunk_segments, load_transcript
from src.config import Config
from src.config import load as load_config
from src.embed import Chunk as EmbedChunk
from src.embed import embed_and_persist
from src.retrieve import CITATION_TOLERANCE_S, retrieve
from src.telemetry import Meter

MANIFEST = ROOT / "data" / "corpus" / "manifest.json"
RUNS = ROOT / "runs"
SWEEP_ROOT = RUNS / "sweep"
DEV_DIR = ROOT / "evals" / "dev"
DEFAULT_OUT = ROOT / "docs" / "learning" / "data" / "chunking_sweep.json"

# k is pinned the way the Phase 1 gate pins it: this is recall@5, and a sweep that let k
# drift with the config would be comparing two different numbers across rows.
K = 5

# The grid: a cross, not a full crossing. Two one-dimensional cuts through the lever space
# share the shipped setting (25 / 8) at their intersection, so each cut answers one
# question with everything else held still — which is the only way a row teaches anything.
#
#   the window cut   overlap pinned at 8 s, window walked  — chunking vs recall
#   the overlap cut  window pinned at 25 s, overlap walked — the cost of overlap
SHIPPED = (25.0, 8.0)
WINDOW_CUT = [12.0, 15.0, 20.0, 25.0, 30.0, 45.0, 60.0]
OVERLAP_CUT = [0.0, 2.0, 5.0, 8.0, 12.0, 16.0]


def grid() -> list[tuple[float, float]]:
    """The (window_s, overlap_s) points, de-duplicated, in a readable order."""
    points = [(w, SHIPPED[1]) for w in WINDOW_CUT]
    points += [(SHIPPED[0], o) for o in OVERLAP_CUT if (SHIPPED[0], o) not in points]
    return points


def slug(window_s: float, overlap_s: float) -> str:
    return f"w{window_s:g}_o{overlap_s:g}"


# --------------------------------------------------------------------------- inputs


def dev_videos() -> list[dict]:
    """The dev-split videos, each with the run directory holding its cached transcript."""
    videos = json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"]
    out = []
    for v in videos:
        if v["split"] != "dev":
            continue
        video_id = str(v["video_id"])
        matches = sorted(d for d in RUNS.glob(f"{video_id}_*") if (d / "transcript.json").is_file())
        if not matches:
            raise SystemExit(
                f"no cached transcript for dev video {video_id} — expected "
                f"runs/{video_id}_*/transcript.json. Run `make index-dev` first."
            )
        out.append({"video_id": video_id, "run_dir": matches[0]})
    return sorted(out, key=lambda r: r["video_id"])


def dev_pairs() -> list[dict]:
    """Answerable dev pairs. Unanswerable ones are Phase 2's number, not recall's."""
    pairs = []
    for path in sorted(DEV_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                pairs.append(json.loads(line))
    return [p for p in pairs if not p.get("unanswerable", False)]


def variant(cfg: Config, window_s: float, overlap_s: float, chroma_path: Path) -> Config:
    """An in-memory config with the two levers moved and the store pointed somewhere else.

    `cfg.raw` is deliberately left alone, so `fingerprint()` still identifies the file on
    disk rather than pretending these overrides were ever in it. The overrides are recorded
    in the JSON instead.
    """
    data = copy.deepcopy(cfg.data)
    data["chunk"]["window_s"] = window_s
    data["chunk"]["overlap_s"] = overlap_s
    data["embed"]["chroma_path"] = str(chroma_path)
    data["embed"]["collection"] = f"sweep_{slug(window_s, overlap_s)}"
    return dataclasses.replace(cfg, data=data)


# --------------------------------------------------------------------------- one point


def chunk_dev(videos: list[dict], levers: dict[str, float]) -> dict:
    """Chunk every dev video at these levers. Returns the chunks and what they cost."""
    chunks: list[EmbedChunk] = []
    per_video = []
    transcript_chars = 0
    transcript_words = 0
    arms = set()

    for v in videos:
        segments, payload = load_transcript(v["run_dir"] / "transcript.json")
        arms.add(str(payload.get("arm", "?")))
        rows, stats = chunk_segments(v["video_id"], segments, levers)
        transcript_chars += sum(len(s.text) for s in segments)
        transcript_words += sum(len(s.text.split()) for s in segments)
        chunks += [
            EmbedChunk(video_id=c.video_id, t_start=c.t_start, t_end=c.t_end, text=c.text)
            for c in rows
        ]
        per_video.append(
            {
                "video_id": v["video_id"],
                "segments": len(segments),
                "chunks": len(rows),
                "windows": stats["windows"],
                "windows_empty": stats["empty"],
                "windows_duplicate": stats["duplicate"],
            }
        )

    durations = [c.t_end - c.t_start for c in chunks]
    chars = sum(len(c.text) for c in chunks)
    words = sum(len(c.text.split()) for c in chunks)
    return {
        "arm": sorted(arms)[0] if len(arms) == 1 else "mixed:" + ",".join(sorted(arms)),
        "chunks": chunks,
        "per_video": per_video,
        "counts": {
            "chunks": len(chunks),
            "chars_indexed": chars,
            "words_indexed": words,
            "transcript_chars": transcript_chars,
            "transcript_words": transcript_words,
            # The duplication factor: how many times the corpus is written into the index.
            # 1.0 means every word is stored once; 1.74 means 74% of the embedding bill is
            # text the index already holds.
            "duplication": round(chars / transcript_chars, 4) if transcript_chars else None,
        },
        "shape": {
            "mean_chunk_s": round(sum(durations) / len(durations), 3) if durations else None,
            "max_chunk_s": round(max(durations), 3) if durations else None,
            # A chunk longer than the citation tolerance can retrieve the right passage and
            # still be scored wrong, because the citation points at the chunk's start.
            "over_tolerance": sum(1 for d in durations if d > CITATION_TOLERANCE_S),
            "mean_words_per_chunk": round(words / len(chunks), 1) if chunks else None,
        },
    }


def score(cfg: Config, pairs: list[dict]) -> dict:
    """Retrieve once per pair at k=5 and record the rank each one hit at."""
    meter = Meter()
    rows = []
    started = time.perf_counter()
    for pair in pairs:
        results = retrieve(pair["question"], cfg, meter)[:K]
        t_ref, target = float(pair["t_ref"]), str(pair["video_id"])
        hit_rank = None
        nearest = None
        for rank, r in enumerate(results, start=1):
            if r.video_id != target:
                continue
            delta = abs(r.t_start - t_ref)
            if nearest is None or delta < nearest:
                nearest = round(delta, 2)
            if delta <= CITATION_TOLERANCE_S and hit_rank is None:
                hit_rank = rank
        rows.append(
            {
                "id": pair.get("id", "?"),
                "video_id": target,
                "hit_rank": hit_rank,
                "nearest_dt_s": nearest,
            }
        )
    wall = time.perf_counter() - started

    # recall@1/@3/@5 all come out of the same top-5 retrieval — a hit at rank r counts for
    # every k >= r. One query per pair, three numbers, no extra spend.
    def recall(k: int) -> float:
        return sum(1 for r in rows if r["hit_rank"] is not None and r["hit_rank"] <= k) / len(rows)

    return {
        "pairs": len(rows),
        "hits_at_5": sum(1 for r in rows if r["hit_rank"] is not None),
        "recall_at_1": round(recall(1), 4),
        "recall_at_3": round(recall(3), 4),
        "recall_at_5": round(recall(5), 4),
        "query_wall_s": round(wall, 2),
        "query_cost_usd": round(meter.total_cost_usd(), 6),
        "rows": rows,
    }


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def run_point(
    cfg: Config, videos: list[dict], pairs: list[dict], window_s: float, overlap_s: float,
    *, dry_run: bool,
) -> dict:
    levers = {"window_s": window_s, "overlap_s": overlap_s, "hop_s": window_s - overlap_s}
    chunked = chunk_dev(videos, levers)
    point = {
        "levers": levers,
        "shipped": (window_s, overlap_s) == SHIPPED,
        "arm": chunked["arm"],
        "counts": chunked["counts"],
        "shape": chunked["shape"],
        "per_video": chunked["per_video"],
    }
    if dry_run:
        return point

    store = SWEEP_ROOT / slug(window_s, overlap_s)
    if store.exists():
        shutil.rmtree(store)  # a stale collection would score two generations of chunks
    store.mkdir(parents=True)

    vcfg = variant(cfg, window_s, overlap_s, store)
    meter = Meter()
    started = time.perf_counter()
    indexed = embed_and_persist(chunked["chunks"], vcfg, meter)
    embed_wall = time.perf_counter() - started

    point["index"] = {
        "chunks_indexed": indexed,
        "embed_wall_s": round(embed_wall, 2),
        "embed_cost_usd": round(meter.total_cost_usd(), 6),
        "store_bytes": dir_bytes(store),
    }
    point["score"] = score(vcfg, pairs)
    return point


# --------------------------------------------------------------------------- report


def table(points: list[dict], *, dry_run: bool) -> str:
    head = f"{'window':>7} {'overlap':>8} {'hop':>5} {'chunks':>7} {'dup':>6} {'max s':>7} {'>tol':>5}"
    if not dry_run:
        head += f" {'r@1':>6} {'r@3':>6} {'r@5':>6} {'embed s':>8} {'MB':>6}"
    lines = [head, "-" * len(head)]
    for p in points:
        lv, c, sh = p["levers"], p["counts"], p["shape"]
        row = (
            f"{lv['window_s']:7.1f} {lv['overlap_s']:8.1f} {lv['hop_s']:5.1f} "
            f"{c['chunks']:7d} {c['duplication']:6.2f} {sh['max_chunk_s']:7.1f} "
            f"{sh['over_tolerance']:5d}"
        )
        if not dry_run:
            s, ix = p["score"], p["index"]
            row += (
                f" {s['recall_at_1']:6.2f} {s['recall_at_3']:6.2f} {s['recall_at_5']:6.2f}"
                f" {ix['embed_wall_s']:8.1f} {ix['store_bytes'] / 1e6:6.1f}"
            )
        lines.append(row + ("   <- shipped" if p["shipped"] else ""))
    return "\n".join(lines)


def _rel(path: Path) -> str:
    """Repo-relative when it can be, absolute when the caller pointed somewhere else."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sweep chunk.window_s / chunk.overlap_s.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write the JSON")
    ap.add_argument(
        "--dry-run", action="store_true", help="chunk only — no embedding, no recall"
    )
    args = ap.parse_args(argv)

    cfg = load_config(ROOT / "config.toml")
    videos = dev_videos()
    pairs = dev_pairs()
    if not pairs:
        raise SystemExit("no answerable dev pairs in evals/dev — nothing to score")

    points = grid()
    print(
        f"sweep: {len(points)} points · {len(videos)} dev videos · {len(pairs)} answerable pairs"
        f"{' · DRY RUN (no embedding)' if args.dry_run else ''}"
    )
    started = time.perf_counter()
    results = []
    for i, (w, o) in enumerate(points, start=1):
        print(f"  [{i}/{len(points)}] window={w:g}s overlap={o:g}s ... ", end="", flush=True)
        point = run_point(cfg, videos, pairs, w, o, dry_run=args.dry_run)
        got = point["counts"]["chunks"]
        tail = "" if args.dry_run else f", recall@5 = {point['score']['recall_at_5']:.4f}"
        print(f"{got} chunks{tail}")
        results.append(point)
    wall = time.perf_counter() - started

    payload = {
        "task": "VRAG-018",
        "k": K,
        "citation_tolerance_s": CITATION_TOLERANCE_S,
        "dry_run": args.dry_run,
        "base_config": cfg.fingerprint(),
        "shipped": {"window_s": SHIPPED[0], "overlap_s": SHIPPED[1]},
        "embed_model": cfg.get("embed.model"),
        "transcript_arm": results[0]["arm"] if results else None,
        "dev_videos": [v["video_id"] for v in videos],
        "answerable_pairs": len(pairs),
        "wall_s": round(wall, 1),
        "total_cost_usd": round(
            sum(
                p.get("index", {}).get("embed_cost_usd", 0.0)
                + p.get("score", {}).get("query_cost_usd", 0.0)
                for p in results
            ),
            6,
        ),
        "points": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print()
    print(table(results, dry_run=args.dry_run))
    print()
    print(
        f"{len(points)} points in {wall:.1f}s · ${payload['total_cost_usd']:.4f} "
        f"· wrote {_rel(args.out)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
