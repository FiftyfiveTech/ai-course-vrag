"""Retrieve — VRAG-016.

Question → top-k chunks with video_id and time range.

Usage:

    from src.retrieve import retrieve, recall_at_k
    from src.config import load
    from src.telemetry import Meter

    cfg = load()
    meter = Meter()

    hits = retrieve("What does the performer place on the table?", cfg, meter)
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
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def retrieve(question: str, cfg: Config, meter: Meter) -> list[RetrievedChunk]:
    """Embed question and return top-k chunks from the Chroma collection.

    k is read from config: retrieve.top_k.
    Returns an empty list when the collection is empty.
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
        return _query(vector, k, chroma_path, collection_name)


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

    actual_k = min(k, count)
    results = collection.query(
        query_embeddings=[vector],
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
    )

    return _parse_query_results(results)


def _parse_query_results(results: dict) -> list[RetrievedChunk]:
    """Turn a Chroma query response into RetrievedChunk objects."""
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]

    chunks = []
    for doc, meta, dist in zip(docs, metas, dists):
        chunks.append(
            RetrievedChunk(
                video_id=str(meta.get("video_id", "")),
                t_start=float(meta.get("t_start", 0.0)),
                t_end=float(meta.get("t_end", 0.0)),
                text=doc,
                score=float(dist),
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
