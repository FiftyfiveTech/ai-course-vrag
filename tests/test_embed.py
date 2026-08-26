"""Tests for src/embed.py — VRAG-015.

No real Ollama or Chroma calls.  The module is tested by injecting fake
responses through the internal helpers, and the public function is tested by
patching _embed_batch and _get_collection.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from src.embed import (
    Chunk,
    EmbedError,
    _batches,
    _embed_batch,
    _hf_to_ollama_tag,
    _parse_embed_response,
    _upsert,
    embed_and_persist,
)
from src.telemetry import Meter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def meter():
    return Meter()


@pytest.fixture()
def cfg(tmp_path):
    from src.config import load
    p = tmp_path / "config.toml"
    p.write_text(
        "[embed]\n"
        'model = "nomic-ai/nomic-embed-text-v1.5"\n'
        'chroma_path = "./chroma_test"\n'
        'collection = "vrag_test"\n'
        "batch_size = 2\n"
    )
    return load(p)


@pytest.fixture()
def sample_chunks():
    return [
        Chunk(video_id="181", t_start=0.0, t_end=30.0, text="Hello world"),
        Chunk(video_id="181", t_start=30.0, t_end=60.0, text="Second chunk"),
        Chunk(video_id="521", t_start=0.0, t_end=30.0, text="Third chunk"),
    ]


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


def test_chunk_id_format():
    c = Chunk(video_id="181", t_start=0.0, t_end=30.0, text="x")
    assert c.chunk_id() == "181_0.000_30.000"


def test_chunk_id_unique_per_position():
    c1 = Chunk(video_id="181", t_start=0.0, t_end=30.0, text="a")
    c2 = Chunk(video_id="181", t_start=30.0, t_end=60.0, text="b")
    assert c1.chunk_id() != c2.chunk_id()


def test_chunk_id_unique_per_video():
    c1 = Chunk(video_id="181", t_start=0.0, t_end=30.0, text="a")
    c2 = Chunk(video_id="521", t_start=0.0, t_end=30.0, text="a")
    assert c1.chunk_id() != c2.chunk_id()


# ---------------------------------------------------------------------------
# _hf_to_ollama_tag
# ---------------------------------------------------------------------------


def test_hf_to_ollama_tag_adds_prefix():
    # A generic HF repo (no native Ollama entry) gets the hf.co/ prefix.
    assert _hf_to_ollama_tag("some-org/some-model") == "hf.co/some-org/some-model"


def test_hf_to_ollama_tag_already_prefixed():
    tag = "hf.co/some-org/some-model"
    assert _hf_to_ollama_tag(tag) == tag


def test_hf_to_ollama_tag_native_nomic():
    # nomic-embed-text-v1.5 HF repo is PyTorch, not GGUF — use Ollama's native name.
    assert _hf_to_ollama_tag("nomic-ai/nomic-embed-text-v1.5") == "nomic-embed-text"


def test_hf_to_ollama_tag_native_llama():
    # Llama 3.2 3B is a gated HF repo — use Ollama's native name instead.
    assert _hf_to_ollama_tag("meta-llama/Llama-3.2-3B-Instruct") == "llama3.2:3b"


# ---------------------------------------------------------------------------
# _parse_embed_response
# ---------------------------------------------------------------------------


def test_parse_embed_response_dict():
    response = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
    result = _parse_embed_response(response)
    assert len(result) == 2
    assert result[0] == [0.1, 0.2]


def test_parse_embed_response_object():
    response = SimpleNamespace(embeddings=[[1.0, 2.0]])
    result = _parse_embed_response(response)
    assert result == [[1.0, 2.0]]


def test_parse_embed_response_empty():
    assert _parse_embed_response({"embeddings": []}) == []


def test_parse_embed_response_no_key():
    assert _parse_embed_response({}) == []


def test_parse_embed_response_converts_to_float():
    response = {"embeddings": [[1, 2, 3]]}
    result = _parse_embed_response(response)
    assert all(isinstance(v, float) for v in result[0])


# ---------------------------------------------------------------------------
# _batches
# ---------------------------------------------------------------------------


def test_batches_even_split():
    result = list(_batches([1, 2, 3, 4], 2))
    assert result == [[1, 2], [3, 4]]


def test_batches_uneven_split():
    result = list(_batches([1, 2, 3], 2))
    assert result == [[1, 2], [3]]


def test_batches_larger_than_list():
    result = list(_batches([1, 2], 10))
    assert result == [[1, 2]]


def test_batches_empty():
    assert list(_batches([], 5)) == []


# ---------------------------------------------------------------------------
# embed_and_persist — public interface
# ---------------------------------------------------------------------------


def test_embed_and_persist_empty_chunks_returns_zero(cfg, meter):
    assert embed_and_persist([], cfg, meter) == 0


def test_embed_and_persist_returns_chunk_count(cfg, meter, sample_chunks, tmp_path):
    fake_embedding = [0.1] * 768
    fake_embeddings = [fake_embedding, fake_embedding]

    mock_collection = MagicMock()

    with patch("src.embed._embed_batch", return_value=fake_embeddings) as mock_embed, \
         patch("src.embed._get_collection", return_value=mock_collection):
        # batch_size=2, 3 chunks → 2 batches
        result = embed_and_persist(sample_chunks, cfg, meter)

    assert result == 3


def test_embed_and_persist_calls_upsert_per_batch(cfg, meter, sample_chunks):
    fake_embeddings = [[0.1] * 768, [0.2] * 768]

    mock_collection = MagicMock()

    with patch("src.embed._embed_batch", return_value=fake_embeddings), \
         patch("src.embed._get_collection", return_value=mock_collection):
        embed_and_persist(sample_chunks, cfg, meter)

    # 3 chunks, batch_size=2 → 2 upsert calls
    assert mock_collection.upsert.call_count == 2


def test_embed_and_persist_metadata_shape(cfg, meter):
    chunks = [Chunk(video_id="181", t_start=0.0, t_end=30.0, text="hello")]
    fake_embeddings = [[0.1] * 768]

    mock_collection = MagicMock()

    with patch("src.embed._embed_batch", return_value=fake_embeddings), \
         patch("src.embed._get_collection", return_value=mock_collection):
        embed_and_persist(chunks, cfg, meter)

    _, kwargs = mock_collection.upsert.call_args
    meta = kwargs["metadatas"][0]
    assert meta["video_id"] == "181"
    assert meta["t_start"] == 0.0
    assert meta["t_end"] == 30.0


def test_embed_and_persist_ids_match_chunk_ids(cfg, meter, sample_chunks):
    fake_embeddings = [[0.1] * 768, [0.2] * 768]
    mock_collection = MagicMock()

    with patch("src.embed._embed_batch", return_value=fake_embeddings), \
         patch("src.embed._get_collection", return_value=mock_collection):
        embed_and_persist(sample_chunks, cfg, meter)

    first_call_ids = mock_collection.upsert.call_args_list[0][1]["ids"]
    assert first_call_ids == [sample_chunks[0].chunk_id(), sample_chunks[1].chunk_id()]
