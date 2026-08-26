"""GATE Phase 1 - VRAG-017.

recall@5 on the 4 dev videos, computed and printed. Threshold **>= 0.80**.

    make gate                       # leakage first, then every gate
    uv run pytest tests/gates/gate_phase1.py -v -s

What recall@5 means here is fixed by QA_SPEC sec.2 and implemented once, in
`src.retrieve.recall_at_k`. A question counts as a hit when at least one of the top-5
results has the ground-truth `video_id` **and** a `t_start` within +/-30 s of `t_ref`:

    hit(pair) = exists r in top5 : r.video_id == pair.video_id and |r.t_start - pair.t_ref| <= 30
    recall@5  = hits / answerable dev pairs

Unanswerable pairs are not scored here - abstention is Phase 2's number (VRAG-021,
QA_SPEC sec.5). This gate measures one thing: does the retriever put the right moment in
front of the answer module at all.

Pass criteria
-------------
- `evals/dev`  intersect  `evals/heldout` = the empty set, and **dev is not empty**. A gate scored against no
  labels is not a gate; `recall = 0/0` is refused rather than reported.
- Every one of the 4 dev videos in `data/corpus/manifest.json` has at least one answerable
  dev pair (QA_SPEC sec.6 asks for the spread) - the criterion says "on the 4 dev videos", so
  a 0.80 earned on one video does not clear it.
- The Chroma index exists and holds chunks for all 4 dev video_ids. A missing video reads
  as a retrieval miss, which would blame the retriever for an un-run pipeline.
- `retrieve.top_k` is 5. This is recall@**5**; scoring it at another k is a different
  number wearing the same name.
- **recall@5 >= 0.80**, printed per-pair and in total.

Why the gate re-derives the number
----------------------------------
The scoring fixture walks the dev pairs itself - so it can print which question missed and
by how many seconds, which is the only useful output when the number is below threshold -
and then asserts its own arithmetic equals `recall_at_k()`. Two implementations of one
definition disagreeing is a bug worth failing on, and the Builder tunes against
`recall_at_k`, so that is the function that has to be right.

Tuning
------
The levers are `chunk.window_s`, `chunk.overlap_s` and `retrieve.top_k` (k is pinned at 5
here). Three attempts, each tuned on `evals/dev` only, then escalate - CLAUDE.md. Never on
`evals/heldout`: this file must not read it, and does not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Project root so imports resolve without an editable install.
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load as load_config
from src.leakage import collisions, load_split
from src.retrieve import CITATION_TOLERANCE_S, recall_at_k, retrieve
from src.telemetry import Meter

# THE THRESHOLD - the Phase 1 exit criterion.
RECALL_THRESHOLD = 0.80

# recall@K. Pinned, not read from config: a gate whose k moves with the config it is
# grading can be passed by editing the config.
K = 5

DEV_DIR = ROOT / "evals" / "dev"
HELDOUT_DIR = ROOT / "evals" / "heldout"
MANIFEST = ROOT / "data" / "corpus" / "manifest.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def dev_video_ids(manifest: Path = MANIFEST) -> list[str]:
    """The corpus video_ids on the dev side of the split. Four of them (VRAG-004)."""
    videos = json.loads(manifest.read_text(encoding="utf-8"))["videos"]
    return sorted(str(v["video_id"]) for v in videos if v["split"] == "dev")


def indexed_video_ids(cfg) -> tuple[set[str], int]:
    """The distinct video_ids in the Chroma collection, and the total chunk count.

    Returns (set(), 0) when the store or collection does not exist yet - the gate turns
    that into a FAIL naming the command to run, rather than a stack trace.
    """
    import chromadb

    chroma_path = Path(cfg.get("embed.chroma_path"))
    if not chroma_path.is_absolute():
        chroma_path = ROOT / chroma_path
    if not chroma_path.exists():
        return set(), 0

    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        collection = client.get_collection(cfg.get("embed.collection"))
    except Exception:
        return set(), 0

    count = collection.count()
    if count == 0:
        return set(), 0

    got = collection.get(include=["metadatas"])
    ids = {str((m or {}).get("video_id", "")) for m in (got.get("metadatas") or [])}
    return {i for i in ids if i}, count


def _best_for(pair: dict, results: list) -> tuple[int | None, float | None]:
    """Rank (1-based) and |dt| of the first result that satisfies the hit rule.

    When nothing hits, returns the rank and |dt| of the closest right-video result instead,
    so a miss says whether it was the wrong video or the wrong moment.
    """
    t_ref = float(pair["t_ref"])
    target = str(pair["video_id"])
    closest: tuple[int | None, float | None] = (None, None)
    for rank, r in enumerate(results, start=1):
        if r.video_id != target:
            continue
        delta = abs(r.t_start - t_ref)
        if delta <= CITATION_TOLERANCE_S:
            return rank, delta
        if closest[1] is None or delta < closest[1]:
            closest = (rank, delta)
    return closest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "config.toml")


@pytest.fixture(scope="module")
def dev_pairs() -> list[dict]:
    """Answerable dev pairs. Unanswerable ones are Phase 2's business, not recall's."""
    pairs = []
    for path in sorted(DEV_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return [p for p in pairs if not p.get("unanswerable", False)]


@pytest.fixture(scope="module")
def scored(cfg, dev_pairs):
    """Retrieve once per answerable dev pair and record why each one hit or missed."""
    if not dev_pairs:
        pytest.skip("no answerable dev pairs - test_dev_set_is_not_empty reports this")

    meter = Meter()
    rows = []
    for pair in dev_pairs:
        results = retrieve(pair["question"], cfg, meter)[:K]
        rank, delta = _best_for(pair, results)
        rows.append(
            {
                "id": pair.get("id", "?"),
                "video_id": str(pair["video_id"]),
                "t_ref": float(pair["t_ref"]),
                "question": pair["question"],
                "hit": rank is not None and delta is not None
                and delta <= CITATION_TOLERANCE_S,
                "rank": rank,
                "delta_s": delta,
                "videos_returned": [r.video_id for r in results],
            }
        )

    hits = sum(1 for r in rows if r["hit"])
    return {
        "rows": rows,
        "hits": hits,
        "total": len(rows),
        "recall": hits / len(rows),
        "meter": meter,
    }


# ---------------------------------------------------------------------------
# Preconditions - none of these are the number, all of them can void it
# ---------------------------------------------------------------------------


def test_no_leakage_before_scoring():
    """dev  intersect  heldout = the empty set. tests/gates/README: no gate result counts until this holds."""
    found = collisions(load_split(DEV_DIR), load_split(HELDOUT_DIR))
    assert not found, (
        f"{len(found)} dev/held-out collision(s) - a held-out label is in the dev set. "
        f"Stop, do not tune. First: {found[0] if found else ''}. See `make leakage-check`."
    )


def test_dev_set_is_not_empty(dev_pairs):
    """Refuse the vacuous score.

    With no labels `recall_at_k` returns 0.0 and 0/0 could as easily be written 1.0. Both
    are numbers about nothing, and a green gate here would read as "retrieval verified".
    """
    print(f"\ndev pairs: {len(dev_pairs)} answerable, from {DEV_DIR.as_posix()}")
    assert dev_pairs, (
        f"{DEV_DIR.as_posix()} holds no answerable pairs. recall@{K} over an empty set is "
        f"not 0.0 and not 1.0 - it is undefined, and this gate will not report it as "
        f"either. Write the dev cases (QA_SPEC sec.1 format, ids d001...) and re-run."
    )


def test_dev_pairs_cover_all_four_dev_videos(dev_pairs):
    """"on the 4 dev videos" - one video carrying the whole score does not clear this."""
    expected = dev_video_ids()
    covered = {p["video_id"] and str(p["video_id"]) for p in dev_pairs}
    per_video = {v: sum(1 for p in dev_pairs if str(p["video_id"]) == v) for v in expected}
    print(f"\ndev pairs per video: {per_video}")
    missing = [v for v in expected if v not in covered]
    assert not missing, (
        f"no answerable dev pair for video(s) {missing}. The criterion is recall@{K} on "
        f"the 4 dev videos {expected}; QA_SPEC sec.6 asks for at least one question each."
    )


def test_top_k_is_five(cfg):
    """recall@5 is scored at k=5, or it is a different number with the same name."""
    top_k = int(cfg.get("retrieve.top_k"))
    print(f"\nretrieve.top_k: {top_k}")
    assert top_k == K, (
        f"retrieve.top_k={top_k}, but this gate scores recall@{K}. Set it to {K} in "
        f"config.toml - k is the lever the gate pins, not one of the ones you tune."
    )


def test_index_covers_all_four_dev_videos(cfg):
    """An un-indexed video is a miss the retriever did not cause."""
    expected = dev_video_ids()
    indexed, count = indexed_video_ids(cfg)
    print(f"\nindex: {count} chunk(s), video_ids {sorted(indexed) or '{}'}")
    assert count > 0, (
        f"the Chroma collection {cfg.get('embed.collection')!r} at "
        f"{cfg.get('embed.chroma_path')} is empty or absent. Build it before scoring: "
        f"`make index VIDEO=samples/<id>_<youtube_id>.mp4` per dev video "
        f"(`make sample-real VIDEO_ID=<id>` fetches one)."
    )
    missing = [v for v in expected if v not in indexed]
    assert not missing, (
        f"dev video(s) {missing} are not in the index, so every question about them misses "
        f"for a reason that is not retrieval. Indexed: {sorted(indexed)}. Expected all of "
        f"{expected}. Run `make index VIDEO=...` for the missing ones."
    )


# ---------------------------------------------------------------------------
# THE NUMBER
# ---------------------------------------------------------------------------


def test_recall_at_5_matches_the_shipped_function(cfg, scored):
    """The gate's arithmetic and `src.retrieve.recall_at_k` must agree.

    The Builder tunes against `recall_at_k`, so that is the implementation that has to be
    right; this gate re-derives the same definition only to be able to print the per-pair
    detail. If the two disagree, one of them is wrong and neither number is usable.
    """
    shipped = recall_at_k(cfg, Meter(), k=K, dev_dir=DEV_DIR)
    print(f"\nrecall_at_k(): {shipped:.4f}   gate: {scored['recall']:.4f}")
    assert abs(shipped - scored["recall"]) < 1e-9, (
        f"src.retrieve.recall_at_k returned {shipped:.4f} but scoring the same pairs by "
        f"the QA_SPEC sec.2 rule here gives {scored['recall']:.4f}. One of the two "
        f"implementations of the hit rule is wrong."
    )


def test_recall_at_5_meets_threshold(scored):
    """recall@5 >= 0.80 on the 4 dev videos. This is the Phase 1 gate."""
    rows = scored["rows"]

    print(f"\nper-question (tolerance +/-{CITATION_TOLERANCE_S:.0f}s, k={K}):")
    for r in rows:
        mark = "HIT " if r["hit"] else "MISS"
        if r["rank"] is None:
            why = "right video not in top-5"
        else:
            why = f"rank {r['rank']}, dt={r['delta_s']:.1f}s"
        print(f"  {mark} {r['id']}  video {r['video_id']}  t_ref={r['t_ref']:.1f}s  {why}")
        if not r["hit"]:
            print(f"       q: {r['question']}")
            print(f"       returned videos: {r['videos_returned']}")

    # THE PHASE 1 NUMBER - must appear in output.
    print(
        f"\nrecall@{K} = {scored['recall']:.4f}  "
        f"({scored['hits']}/{scored['total']} answerable dev pairs)  "
        f"threshold {RECALL_THRESHOLD:.2f}"
    )
    sys.stdout.flush()

    assert scored["recall"] >= RECALL_THRESHOLD, (
        f"recall@{K} = {scored['recall']:.4f} < {RECALL_THRESHOLD:.2f}. "
        f"{scored['total'] - scored['hits']} of {scored['total']} pairs missed. Tune "
        f"chunk.window_s / chunk.overlap_s on evals/dev - three attempts, then escalate "
        f"(CLAUDE.md). Do not touch evals/heldout, and do not move the threshold."
    )


def test_cost_of_the_gate_is_printed(scored, cfg):
    """What the measurement cost. Local embeddings are $0.00 - printed, not assumed."""
    meter: Meter = scored["meter"]
    calls = len(meter._calls)
    latency = sum(c.latency_s for c in meter._calls)
    cost = sum(c.cost_usd for c in meter._calls)
    print(
        f"\ngate cost: {calls} embed call(s) over {scored['total']} question(s), "
        f"{latency:.2f}s total, ${cost:.4f}  (model {cfg.get('embed.model')})"
    )
    assert calls >= scored["total"], (
        f"{calls} embed call(s) for {scored['total']} question(s) - every scored question "
        f"must have been embedded and queried, not served from a stale result"
    )
