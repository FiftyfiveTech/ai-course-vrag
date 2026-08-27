"""Tests for src/retrieve.py — VRAG-016.

No real Ollama or Chroma calls.  Internal helpers are tested directly;
retrieve() and recall_at_k() are tested via patching.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.retrieve import (
    CITATION_TOLERANCE_S,
    RetrievedChunk,
    RetrieveError,
    _is_hit,
    _load_dev_pairs,
    _parse_query_results,
    _query,
    recall_at_k,
    retrieve,
    where,
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
        "[retrieve]\ntop_k = 5\n"
        "[embed]\n"
        'model = "nomic-ai/nomic-embed-text-v1.5"\n'
        'chroma_path = "./chroma_test"\n'
        'collection = "vrag_test"\n'
        "batch_size = 32\n"
    )
    return load(p)


@pytest.fixture()
def dev_dir(tmp_path):
    d = tmp_path / "dev"
    d.mkdir()
    return d


def _write_pairs(dev_dir: Path, pairs: list[dict]) -> None:
    p = dev_dir / "pairs.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in pairs), encoding="utf-8")


def _make_chunk(video_id="181", t_start=0.0, t_end=30.0, text="hello", score=0.1):
    return RetrievedChunk(video_id=video_id, t_start=t_start, t_end=t_end, text=text, score=score)


# ---------------------------------------------------------------------------
# _parse_query_results
# ---------------------------------------------------------------------------


def test_parse_query_results_basic():
    results = {
        "documents": [["chunk text"]],
        "metadatas": [[{"video_id": "181", "t_start": 0.0, "t_end": 30.0}]],
        "distances": [[0.12]],
    }
    chunks = _parse_query_results(results)
    assert len(chunks) == 1
    assert chunks[0].video_id == "181"
    assert chunks[0].t_start == 0.0
    assert chunks[0].score == 0.12
    assert chunks[0].text == "chunk text"


def test_parse_query_results_multiple():
    results = {
        "documents": [["a", "b"]],
        "metadatas": [
            [{"video_id": "181", "t_start": 0.0, "t_end": 30.0},
             {"video_id": "521", "t_start": 30.0, "t_end": 60.0}]
        ],
        "distances": [[0.1, 0.2]],
    }
    chunks = _parse_query_results(results)
    assert len(chunks) == 2
    assert chunks[1].video_id == "521"


def test_parse_query_results_empty():
    assert _parse_query_results({}) == []


def test_parse_query_results_missing_keys():
    results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    assert _parse_query_results(results) == []


# ---------------------------------------------------------------------------
# _load_dev_pairs
# ---------------------------------------------------------------------------


def test_load_dev_pairs_reads_jsonl(dev_dir):
    _write_pairs(dev_dir, [
        {"id": "d001", "question": "Q1", "video_id": "181", "t_ref": 10.0, "unanswerable": False},
    ])
    pairs = _load_dev_pairs(dev_dir)
    assert len(pairs) == 1
    assert pairs[0]["id"] == "d001"


def test_load_dev_pairs_multiple_files(dev_dir):
    (dev_dir / "a.jsonl").write_text('{"id":"d001"}\n{"id":"d002"}\n', encoding="utf-8")
    (dev_dir / "b.jsonl").write_text('{"id":"d003"}\n', encoding="utf-8")
    pairs = _load_dev_pairs(dev_dir)
    assert len(pairs) == 3


def test_load_dev_pairs_empty_dir(dev_dir):
    assert _load_dev_pairs(dev_dir) == []


def test_load_dev_pairs_skips_blank_lines(dev_dir):
    (dev_dir / "pairs.jsonl").write_text('{"id":"d001"}\n\n{"id":"d002"}\n', encoding="utf-8")
    assert len(_load_dev_pairs(dev_dir)) == 2


# ---------------------------------------------------------------------------
# _is_hit
# ---------------------------------------------------------------------------


def test_is_hit_correct_video_and_timestamp():
    pair = {"video_id": "181", "t_ref": 15.0}
    results = [_make_chunk(video_id="181", t_start=10.0)]
    assert _is_hit(pair, results) is True


def test_is_hit_wrong_video():
    pair = {"video_id": "181", "t_ref": 15.0}
    results = [_make_chunk(video_id="521", t_start=10.0)]
    assert _is_hit(pair, results) is False


def test_is_hit_timestamp_just_within_tolerance():
    pair = {"video_id": "181", "t_ref": 0.0}
    results = [_make_chunk(video_id="181", t_start=CITATION_TOLERANCE_S)]
    assert _is_hit(pair, results) is True


def test_is_hit_timestamp_just_outside_tolerance():
    pair = {"video_id": "181", "t_ref": 0.0}
    results = [_make_chunk(video_id="181", t_start=CITATION_TOLERANCE_S + 0.001)]
    assert _is_hit(pair, results) is False


def test_is_hit_hit_in_second_result():
    pair = {"video_id": "181", "t_ref": 15.0}
    results = [
        _make_chunk(video_id="521", t_start=15.0),  # wrong video
        _make_chunk(video_id="181", t_start=20.0),  # correct
    ]
    assert _is_hit(pair, results) is True


def test_is_hit_empty_results():
    pair = {"video_id": "181", "t_ref": 15.0}
    assert _is_hit(pair, []) is False


def test_is_hit_unanswerable_pair_no_crash():
    pair = {"video_id": None, "t_ref": None}
    assert _is_hit(pair, [_make_chunk()]) is False


# ---------------------------------------------------------------------------
# retrieve() dispatch
# ---------------------------------------------------------------------------


def test_retrieve_returns_list(cfg, meter):
    fake_vector = [0.1] * 768
    fake_chunks = [_make_chunk()]
    with patch("src.retrieve._embed_question", return_value=fake_vector), \
         patch("src.retrieve._query", return_value=fake_chunks):
        result = retrieve("what happened?", cfg, meter)
    assert result == fake_chunks


def test_retrieve_passes_k_from_config(cfg, meter):
    fake_vector = [0.1] * 768
    with patch("src.retrieve._embed_question", return_value=fake_vector), \
         patch("src.retrieve._query", return_value=[]) as mock_q:
        retrieve("question", cfg, meter)
    assert mock_q.call_args[0][1] == 5  # top_k from cfg


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


def test_recall_at_k_empty_dev_returns_zero(cfg, meter, dev_dir):
    score = recall_at_k(cfg, meter, dev_dir=dev_dir)
    assert score == 0.0


def test_recall_at_k_skips_unanswerable(cfg, meter, dev_dir):
    _write_pairs(dev_dir, [
        {"id": "d001", "question": "Q", "video_id": None, "t_ref": None, "unanswerable": True},
    ])
    score = recall_at_k(cfg, meter, dev_dir=dev_dir)
    assert score == 0.0


def test_recall_at_k_all_hits(cfg, meter, dev_dir):
    _write_pairs(dev_dir, [
        {"id": "d001", "question": "Q1", "video_id": "181", "t_ref": 10.0, "unanswerable": False},
        {"id": "d002", "question": "Q2", "video_id": "521", "t_ref": 20.0, "unanswerable": False},
    ])
    # patch retrieve to always return a hit
    def fake_retrieve(question, cfg, meter):
        if "Q1" in question:
            return [_make_chunk(video_id="181", t_start=10.0)]
        return [_make_chunk(video_id="521", t_start=20.0)]

    with patch("src.retrieve.retrieve", side_effect=fake_retrieve):
        score = recall_at_k(cfg, meter, dev_dir=dev_dir)
    assert score == 1.0


def test_recall_at_k_no_hits(cfg, meter, dev_dir):
    _write_pairs(dev_dir, [
        {"id": "d001", "question": "Q1", "video_id": "181", "t_ref": 10.0, "unanswerable": False},
    ])
    with patch("src.retrieve.retrieve", return_value=[_make_chunk(video_id="999", t_start=0.0)]):
        score = recall_at_k(cfg, meter, dev_dir=dev_dir)
    assert score == 0.0


def test_recall_at_k_partial(cfg, meter, dev_dir):
    _write_pairs(dev_dir, [
        {"id": "d001", "question": "Q1", "video_id": "181", "t_ref": 10.0, "unanswerable": False},
        {"id": "d002", "question": "Q2", "video_id": "521", "t_ref": 20.0, "unanswerable": False},
    ])
    call_count = [0]

    def fake_retrieve(question, cfg, meter):
        call_count[0] += 1
        if call_count[0] == 1:
            return [_make_chunk(video_id="181", t_start=10.0)]  # hit
        return [_make_chunk(video_id="999", t_start=0.0)]  # miss

    with patch("src.retrieve.retrieve", side_effect=fake_retrieve):
        score = recall_at_k(cfg, meter, dev_dir=dev_dir)
    assert score == 0.5


# ---------------------------------------------------------------------------
# Scoping to a set of videos — what an `@source` tag becomes (src/mention.py)
# ---------------------------------------------------------------------------


def test_no_scope_is_no_filter():
    # None and empty both mean the whole index. A `{"video_id": {"$in": []}}` sent to Chroma
    # would match nothing, which is the opposite of what "unscoped" means.
    assert where(None) is None
    assert where([]) is None
    assert where([""]) is None


def test_one_video_is_still_an_in_clause():
    # One filter shape for one tag and for five: a `{"video_id": "611"}` shortcut for the
    # common case is a second thing to keep true for no gain.
    assert where(["611"]) == {"video_id": {"$in": ["611"]}}


def test_several_videos_scope_to_their_union():
    assert where(["611", "181"]) == {"video_id": {"$in": ["611", "181"]}}


def test_the_scope_reaches_the_store_and_not_a_post_filter(cfg, meter):
    # The distinction the whole feature rests on. A filter applied after ranking would take
    # the top 5 and drop the ones from other videos, which returns fewer than 5 and often
    # zero — the excluded videos are exactly the ones that outranked the wanted one.
    with patch("src.retrieve._embed_question", return_value=[0.1] * 768), \
         patch("src.retrieve._query", return_value=[]) as mock_q:
        retrieve("question", cfg, meter, video_ids=["611"])
    assert mock_q.call_args.kwargs["where"] == {"video_id": {"$in": ["611"]}}


def test_an_unscoped_query_sends_no_filter(cfg, meter):
    with patch("src.retrieve._embed_question", return_value=[0.1] * 768), \
         patch("src.retrieve._query", return_value=[]) as mock_q:
        retrieve("question", cfg, meter)
    assert mock_q.call_args.kwargs["where"] is None


def test_the_where_clause_is_left_off_the_chroma_call_entirely_when_there_is_none(tmp_path):
    # Not `where=None`: chromadb has treated an explicit null filter differently from an
    # absent one across versions, and the unscoped path is every question anyone has asked
    # so far — it must go down the call it has always gone down.
    calls = {}

    class FakeCollection:
        def count(self):
            return 10

        def query(self, **kwargs):
            calls.update(kwargs)
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    class FakeClient:
        def __init__(self, path):
            pass

        def get_collection(self, name):
            return FakeCollection()

    chroma = tmp_path / "chroma"
    chroma.mkdir()
    with patch("chromadb.PersistentClient", FakeClient):
        _query([0.1] * 768, 5, chroma, "vrag", where=None)
    assert "where" not in calls

    with patch("chromadb.PersistentClient", FakeClient):
        _query([0.1] * 768, 5, chroma, "vrag", where={"video_id": {"$in": ["611"]}})
    assert calls["where"] == {"video_id": {"$in": ["611"]}}
