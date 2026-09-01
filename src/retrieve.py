"""Retrieve — VRAG-016.

Question → top-k chunks with video_id and time range.

Usage:

    from src.retrieve import retrieve, recall_at_k
    from src.config import load
    from src.telemetry import Meter

    cfg = load()
    meter = Meter()

    hits = retrieve("What does the performer place on the table?", cfg, meter)

    # Only video 611, whatever else the index holds. This is what an `@611` tag in a
    # question becomes — see src/mention.py.
    hits = retrieve("What is on the table?", cfg, meter, video_ids=["611"])

    for h in hits:
        print(f"{h.video_id}  {h.t_start:.1f}s–{h.t_end:.1f}s  score={h.score:.3f}")
        print(f"  {h.text[:80]}")

    # Scored against dev pairs (evals/dev/*.jsonl):
    score = recall_at_k(cfg, meter, k=5)
    print(f"recall@5 = {score:.4f}")

recall@5 definition (from QA_SPEC.md §2):
  A question is a hit when at least one of the top-k results has the correct
  video_id AND |result.t_start - t_ref| ≤ 30 s.
  recall@5 = hits / total answerable questions in dev.
  Unanswerable questions (unanswerable=true) are skipped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from src.config import Config
from src.telemetry import Meter

# Tolerance from QA_SPEC §2.
CITATION_TOLERANCE_S = 30.0

DEV_DIR = Path("evals/dev")


class RetrieveError(Exception):
    """Retrieval failed — message says which step and why."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One result from a Chroma query."""

    video_id: str
    t_start: float
    t_end: float
    text: str
    score: float  # cosine distance (lower = more similar)
    speakers: tuple[str, ...] = ()  # VRAG-026: from Chroma metadata, empty when unattributed


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def retrieve(
    question: str,
    cfg: Config,
    meter: Meter,
    video_ids: Sequence[str] | None = None,
) -> list[RetrievedChunk]:
    """Embed question and return top-k chunks from the Chroma collection.

    k is read from config: retrieve.top_k.
    Returns an empty list when the collection is empty.

    `video_ids` restricts the search to those videos and nothing else — it is what an
    `@source` tag in the question becomes (`src.mention`). It is a filter applied by the
    store *before* ranking, not a re-rank of the top k afterwards: taking k results and
    dropping the ones from other videos would return fewer than k, and often zero, because
    the excluded videos are exactly the ones that outranked the wanted one. None or empty
    means the whole index, which is the default and the untagged case.
    """
    k = int(cfg.get("retrieve.top_k"))
    model = cfg.get("embed.model")
    chroma_path = Path(cfg.get("embed.chroma_path"))
    collection_name = cfg.get("embed.collection")

    vector = _embed_question(question, model, meter)
    # Staged, not just called: opening the Chroma PersistentClient is ~1.35s on the first
    # query of a process and ~0.02s after, and until this span existed neither number was
    # visible anywhere — the meter recorded model calls only, so a 3.59s request reported
    # itself as 1.44s. See `make latency`.
    with meter.stage("retrieve.query"):
        return _query(vector, k, chroma_path, collection_name, where=where(video_ids))


def where(video_ids: Sequence[str] | None) -> dict | None:
    """Chroma's metadata filter for a set of videos, or None for no filter at all.

    `$in` even for a single id, so the one-tag and many-tag cases go down one path — a
    `{"video_id": "611"}` shortcut for the common case is a second filter shape to keep
    true, and Chroma treats a one-element `$in` identically.
    """
    ids = [str(v) for v in (video_ids or []) if str(v)]
    return {"video_id": {"$in": ids}} if ids else None


def recall_at_k(
    cfg: Config,
    meter: Meter,
    k: int | None = None,
    dev_dir: Path = DEV_DIR,
) -> float:
    """Compute recall@k on the dev eval set.

    Reads all *.jsonl files under dev_dir.  Skips unanswerable pairs.
    Returns 0.0 when there are no answerable dev pairs (vacuous).

    A pair is a hit when retrieve() returns at least one result with:
      - video_id matching the pair's video_id
      - |t_start - t_ref| ≤ CITATION_TOLERANCE_S
    """
    if k is None:
        k = int(cfg.get("retrieve.top_k"))

    pairs = _load_dev_pairs(dev_dir)
    answerable = [p for p in pairs if not p.get("unanswerable", False)]

    if not answerable:
        return 0.0

    hits = 0
    for pair in answerable:
        results = retrieve(pair["question"], cfg, meter)[:k]
        if _is_hit(pair, results):
            hits += 1

    return hits / len(answerable)


# ---------------------------------------------------------------------------
# Embedding the query
# ---------------------------------------------------------------------------


def _embed_question(question: str, model: str, meter: Meter) -> list[float]:
    """Embed a single question string and return its vector."""
    try:
        import ollama
    except ImportError as exc:
        raise RetrieveError("ollama package not installed — run `uv sync`") from exc

    from src.embed import _hf_to_ollama_tag, _parse_embed_response

    ollama_model = _hf_to_ollama_tag(model)
    try:
        with meter.span(model, tokens=len(question.split()), phase="retrieve.embed"):
            response = ollama.embed(model=ollama_model, input=[question])
    except Exception as exc:
        raise RetrieveError(
            f"ollama embed failed for question: {exc}\n"
            f"Make sure the model is pulled: ollama pull {ollama_model}"
        ) from exc

    vecs = _parse_embed_response(response)
    if not vecs:
        raise RetrieveError("ollama returned no embedding for the question")
    return vecs[0]


# ---------------------------------------------------------------------------
# Chroma query
# ---------------------------------------------------------------------------


def _query(
    vector: list[float],
    k: int,
    chroma_path: Path,
    collection_name: str,
    where: dict | None = None,
) -> list[RetrievedChunk]:
    try:
        import chromadb
    except ImportError as exc:
        raise RetrieveError("chromadb not installed — run `uv sync`") from exc

    if not chroma_path.exists():
        return []

    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        # Collection does not exist yet — index is empty.
        return []

    count = collection.count()
    if count == 0:
        return []

    # min() against the whole collection and not against the filtered subset: Chroma has no
    # count-with-a-where, and asking for more rows than the filter can supply returns fewer
    # rather than raising. The cap exists only because n_results above the collection size
    # is the error case; a filter narrowing it further is fine.
    actual_k = min(k, count)
    results = collection.query(
        query_embeddings=[vector],
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
        **({"where": where} if where else {}),
    )

    return _parse_query_results(results)


def indexed_video_ids(cfg: Config) -> list[str]:
    """The distinct video_ids the collection holds. Empty when there is no index yet.

    Read-only on purpose — `src.embed._get_collection` mkdirs and get_or_creates, so calling
    the writer's door from here would make listing what can be tagged *create* the empty
    index it is listing. Same rule `src.api.index_status` follows.
    """
    try:
        import chromadb
    except ImportError:
        return []

    chroma_path = Path(cfg.get("embed.chroma_path"))
    if not chroma_path.exists():
        return []
    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(str(cfg.get("embed.collection")))
        rows = collection.get(include=["metadatas"])
    except Exception:
        return []

    ids = {
        str(m.get("video_id"))
        for m in (rows.get("metadatas") or [])
        if m and m.get("video_id") is not None
    }
    return sorted(ids)


def _parse_query_results(results: dict) -> list[RetrievedChunk]:
    """Turn a Chroma query response into RetrievedChunk objects."""
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]

    chunks = []
    for doc, meta, dist in zip(docs, metas, dists):
        raw_speakers = meta.get("speakers", "[]")
        try:
            speakers = tuple(json.loads(raw_speakers))
        except (ValueError, TypeError):
            speakers = ()
        chunks.append(
            RetrievedChunk(
                video_id=str(meta.get("video_id", "")),
                t_start=float(meta.get("t_start", 0.0)),
                t_end=float(meta.get("t_end", 0.0)),
                text=doc,
                score=float(dist),
                speakers=speakers,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Dev pair loading and scoring
# ---------------------------------------------------------------------------


def _load_dev_pairs(dev_dir: Path) -> list[dict]:
    """Read all *.jsonl files from dev_dir and return parsed pairs."""
    pairs = []
    for path in sorted(dev_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def _is_hit(pair: dict, results: list[RetrievedChunk]) -> bool:
    """True if any result matches the pair's video_id and is within tolerance."""
    target_video = pair.get("video_id")
    t_ref = pair.get("t_ref")
    if target_video is None or t_ref is None:
        return False
    for r in results:
        if r.video_id == str(target_video):
            if abs(r.t_start - float(t_ref)) <= CITATION_TOLERANCE_S:
                return True
    return False
