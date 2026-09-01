"""Embed + persist — VRAG-015.

Embeds transcript chunks with nomic-ai/nomic-embed-text-v1.5 running locally
on Ollama, then persists them to a local Chroma collection.

Each chunk must carry video_id, t_start, t_end — these become Chroma metadata
and are what the retriever (VRAG-016) returns alongside the text so the answer
module can build a citation.

Usage (one call per ingested video, after chunking):

    from src.embed import Chunk, embed_and_persist
    from src.config import load
    from src.telemetry import Meter

    cfg = load()
    meter = Meter()
    chunks = [Chunk(video_id="181", t_start=0.0, t_end=30.0, text="..."), ...]
    n = embed_and_persist(chunks, cfg, meter)
    print(f"indexed {n} chunks")

The Chroma collection is created if it does not exist and appended to on
subsequent calls, so re-running ingest on a new video does not wipe the index.
To reset: delete the chroma_path directory or change the collection name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from src.config import Config
from src.telemetry import Meter

# nomic-embed-text-v1.5 produces 768-dimensional vectors.
EMBED_DIM = 768


class EmbedError(Exception):
    """Embedding or persistence failed — message says which step and why."""


@dataclass(frozen=True)
class Chunk:
    """One time-windowed unit of transcript text.

    Produced by the chunker (VRAG-014) and consumed here.  The three metadata
    fields are what the retriever returns so the answer module can cite a
    specific moment in a specific video.
    """

    video_id: str
    t_start: float  # seconds from video start
    t_end: float
    text: str
    speakers: list[str] = field(default_factory=list)  # VRAG-026: stored in metadata, never embedded

    def chunk_id(self) -> str:
        """Stable, unique id for this chunk within the Chroma collection."""
        return f"{self.video_id}_{self.t_start:.3f}_{self.t_end:.3f}"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def embed_and_persist(
    chunks: Sequence[Chunk],
    cfg: Config,
    meter: Meter,
) -> int:
    """Embed chunks and upsert them into the Chroma collection.

    Returns the number of chunks indexed.  Raises EmbedError on failure.
    Existing chunks with the same id are overwritten (idempotent re-runs).
    """
    if not chunks:
        return 0

    model = cfg.get("embed.model")
    chroma_path = Path(cfg.get("embed.chroma_path"))
    collection_name = cfg.get("embed.collection")
    batch_size = int(cfg.get("embed.batch_size"))

    collection = _get_collection(chroma_path, collection_name)
    total = 0

    for batch in _batches(list(chunks), batch_size):
        texts = [c.text for c in batch]
        embeddings = _embed_batch(texts, model, meter)
        _upsert(collection, batch, embeddings)
        total += len(batch)

    return total


# ---------------------------------------------------------------------------
# Ollama embedding
# ---------------------------------------------------------------------------


def _embed_batch(texts: list[str], model: str, meter: Meter) -> list[list[float]]:
    """Call Ollama to embed a list of texts. Returns one vector per text."""
    try:
        import ollama
    except ImportError as exc:
        raise EmbedError(
            "ollama package not installed — run `uv sync`"
        ) from exc

    ollama_model = _hf_to_ollama_tag(model)

    try:
        with meter.span(
            model, tokens=sum(len(t.split()) for t in texts), phase="index.embed"
        ):
            response = ollama.embed(model=ollama_model, input=texts)
    except Exception as exc:
        raise EmbedError(
            f"ollama embed failed for model {ollama_model!r}: {exc}\n"
            f"Make sure the model is pulled: ollama pull {ollama_model}"
        ) from exc

    embeddings = _parse_embed_response(response)
    if len(embeddings) != len(texts):
        raise EmbedError(
            f"ollama returned {len(embeddings)} embeddings for {len(texts)} texts"
        )
    return embeddings


def _hf_to_ollama_tag(hf_repo_id: str) -> str:
    """'nomic-ai/nomic-embed-text-v1.5' → 'hf.co/nomic-ai/nomic-embed-text-v1.5'"""
    if hf_repo_id.startswith("hf.co/"):
        return hf_repo_id
    return f"hf.co/{hf_repo_id}"


def _parse_embed_response(response) -> list[list[float]]:
    """Extract the embedding vectors from an Ollama embed response.

    Handles both object-style (SDK) and dict-style (test fakes).
    """
    if isinstance(response, dict):
        raw = response.get("embeddings", [])
    else:
        raw = getattr(response, "embeddings", []) or []
    return [list(map(float, vec)) for vec in raw]


# ---------------------------------------------------------------------------
# Chroma persistence
# ---------------------------------------------------------------------------


def _get_collection(chroma_path: Path, name: str):
    """Open or create the Chroma collection at chroma_path."""
    try:
        import chromadb
    except ImportError as exc:
        raise EmbedError(
            "chromadb not installed — run `uv sync`"
        ) from exc

    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def _upsert(collection, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Upsert a batch of chunks into the collection."""
    collection.upsert(
        ids=[c.chunk_id() for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {"video_id": c.video_id, "t_start": c.t_start, "t_end": c.t_end,
             "speakers": json.dumps(c.speakers)}
            for c in chunks
        ],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _batches(items: list, size: int):
    """Yield successive slices of `items` of length `size`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]
