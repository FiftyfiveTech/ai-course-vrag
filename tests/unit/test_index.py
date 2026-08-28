"""Tests for src/index.py — VRAG-017.

No real Ollama, Chroma, ffmpeg, network or model calls. The two pieces worth pinning are
the row -> store translation (a dropped or restringed field there is a citation that cannot
be resolved) and the dev-split driver's refusal to invent work.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.index import (
    IndexingError,
    dev_videos,
    index_dev_split,
    index_video,
    local_file,
    report,
    to_embed_chunks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg(tmp_path):
    from src.config import load

    p = tmp_path / "config.toml"
    p.write_text(
        "[embed]\n"
        'model = "nomic-ai/nomic-embed-text-v1.5-GGUF"\n'
        'chroma_path = "./chroma_test"\n'
        'collection = "vrag_test"\n'
        "batch_size = 32\n",
        encoding="utf-8",
    )
    return load(p)


def _manifest(tmp_path: Path, videos: list[dict]) -> Path:
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"videos": videos}), encoding="utf-8")
    return p


def _chunk_result(problems=None, rows=None) -> dict:
    """The shape src.chunk.build returns, trimmed to what index_video reads."""
    return {
        "video_id": "181",
        "duration_s": 95.8,
        "transcript": {"source": "cache", "segments": 24},
        "levers": {"window_s": 25.0, "overlap_s": 8.0},
        "problems": problems or [],
        "telemetry": "$0.0000/video-hour  2330.4xrealtime",
        "chunks": rows
        if rows is not None
        else [
            {"video_id": "181", "t_start": 0.0, "t_end": 24.5, "text": "first"},
            {"video_id": "181", "t_start": 17.0, "t_end": 41.2, "text": "second"},
        ],
    }


# ---------------------------------------------------------------------------
# to_embed_chunks — the row -> store translation
# ---------------------------------------------------------------------------


def test_to_embed_chunks_carries_the_four_citation_fields():
    rows = [{"video_id": "611", "t_start": 79.5, "t_end": 104.0, "text": "hello"}]
    (chunk,) = to_embed_chunks(rows)
    assert chunk.video_id == "611"
    assert chunk.t_start == 79.5
    assert chunk.t_end == 104.0
    assert chunk.text == "hello"


def test_to_embed_chunks_stringifies_a_numeric_video_id():
    """The manifest holds "181" but a hand-written row can hold 181.

    QA_SPEC compares video_id for equality, and "181" != 181 would fail every question
    about that video for a reason no retrieval change can fix.
    """
    (chunk,) = to_embed_chunks(
        [{"video_id": 181, "t_start": 0, "t_end": 1, "text": "x"}]
    )
    assert chunk.video_id == "181"
    assert isinstance(chunk.t_start, float)


def test_to_embed_chunks_is_empty_for_no_rows():
    assert to_embed_chunks([]) == []


def test_chunk_id_is_stable_across_calls():
    """Re-indexing upserts rather than duplicating, which relies on a stable id."""
    rows = [{"video_id": "181", "t_start": 0.0, "t_end": 24.5, "text": "a"}]
    first = to_embed_chunks(rows)[0].chunk_id()
    second = to_embed_chunks(rows)[0].chunk_id()
    assert first == second == "181_0.000_24.500"


# ---------------------------------------------------------------------------
# index_video
# ---------------------------------------------------------------------------


def test_index_video_embeds_every_chunk(cfg, tmp_path):
    with patch("src.index.build_chunks", return_value=_chunk_result()) as build, patch(
        "src.index.embed_and_persist", return_value=2
    ) as embed:
        r = index_video(tmp_path / "v.mp4", cfg)
    assert build.called
    chunks_passed = embed.call_args.args[0]
    assert len(chunks_passed) == 2
    assert r["indexed"] == 2
    assert r["chunks"] == 2
    assert r["video_id"] == "181"
    assert r["model"] == "nomic-ai/nomic-embed-text-v1.5-GGUF"


def test_index_video_refuses_to_index_a_chunk_that_failed_verify(cfg, tmp_path):
    """A chunk whose range does not hold its segments is a citation pointing at the wrong
    moment. Indexing it would put that error in the store, where the gate cannot see it."""
    bad = _chunk_result(problems=["chunk 3 [10, 40] does not contain segment 7 [41, 44]"])
    with patch("src.index.build_chunks", return_value=bad), patch(
        "src.index.embed_and_persist"
    ) as embed:
        with pytest.raises(IndexingError) as exc:
            index_video(tmp_path / "v.mp4", cfg)
    assert "does not contain segment 7" in str(exc.value)
    assert not embed.called, "nothing may reach the store once verify has complained"


def test_index_video_names_the_model_when_embedding_fails(cfg, tmp_path):
    """The failure that cost this task an hour: the configured repo id was not pullable.

    The message has to name the id being used, or the next person reads "embedding failed"
    and goes looking at Chroma.
    """
    from src.embed import EmbedError

    with patch("src.index.build_chunks", return_value=_chunk_result()), patch(
        "src.index.embed_and_persist", side_effect=EmbedError("ollama embed failed: 400")
    ):
        with pytest.raises(IndexingError) as exc:
            index_video(tmp_path / "v.mp4", cfg)
    assert "nomic-ai/nomic-embed-text-v1.5-GGUF" in str(exc.value)


def test_index_video_reports_zero_chunks_without_calling_the_store(cfg, tmp_path):
    """A video with no speech chunks to nothing. That is a fact to print, not an error."""
    with patch("src.index.build_chunks", return_value=_chunk_result(rows=[])), patch(
        "src.index.embed_and_persist", return_value=0
    ):
        r = index_video(tmp_path / "v.mp4", cfg)
    assert r["chunks"] == 0 and r["indexed"] == 0


# ---------------------------------------------------------------------------
# dev split discovery
# ---------------------------------------------------------------------------


def test_dev_videos_reads_the_split_from_the_manifest(tmp_path):
    m = _manifest(
        tmp_path,
        [
            {"video_id": "611", "split": "dev"},
            {"video_id": "091", "split": "heldout"},
            {"video_id": "181", "split": "dev"},
        ],
    )
    assert [v["video_id"] for v in dev_videos(m)] == ["181", "611"]


def test_dev_videos_never_returns_a_heldout_id(tmp_path):
    """Indexing is not labelling, but a held-out id reaching a dev-only path is the kind of
    mistake that ends with the Builder tuning on the sealed split."""
    m = _manifest(tmp_path, [{"video_id": "091", "split": "heldout"}])
    assert dev_videos(m) == []


def test_local_file_matches_a_fetched_video_by_id_prefix(tmp_path):
    """yt-dlp picks the extension, so the id is a prefix match, not a known filename."""
    (tmp_path / "521_qJGqZ_g__So.mp4").write_bytes(b"x")
    found = local_file("521", tmp_path)
    assert found is not None and found.name == "521_qJGqZ_g__So.mp4"


def test_local_file_does_not_match_a_different_id_that_shares_a_prefix(tmp_path):
    """'52' must not claim '521_...'. The underscore in the glob is what stops it."""
    (tmp_path / "521_qJGqZ_g__So.mp4").write_bytes(b"x")
    assert local_file("52", tmp_path) is None


def test_local_file_matches_a_locally_ingested_video_named_after_its_id(tmp_path):
    """A video ingested here has no youtube id to append, so it lands as `<id>.<ext>`.

    Globbing only `<id>_*` made such a video answerable but unplayable: it indexed and got
    cited, then `media_url` found no file, `stream_url` came back null and the frontend drew
    "no playable copy" over a video sitting in samples/.
    """
    (tmp_path / "bob-video.mp4").write_bytes(b"x")
    found = local_file("bob-video", tmp_path)
    assert found is not None and found.name == "bob-video.mp4"


def test_local_file_does_not_match_a_shared_prefix_through_the_dotted_form(tmp_path):
    """The dot is as load-bearing as the underscore: '61' must not claim '611.mp4'."""
    (tmp_path / "611.mp4").write_bytes(b"x")
    assert local_file("61", tmp_path) is None


def test_local_file_is_none_when_samples_does_not_exist(tmp_path):
    assert local_file("181", tmp_path / "nope") is None


# ---------------------------------------------------------------------------
# index_dev_split
# ---------------------------------------------------------------------------


def test_index_dev_split_indexes_each_dev_video_with_its_corpus_id(cfg, tmp_path):
    videos = [{"video_id": "181", "split": "dev"}, {"video_id": "521", "split": "dev"}]
    with patch("src.index.dev_videos", return_value=videos), patch(
        "src.index.local_file", side_effect=lambda vid, *a: tmp_path / f"{vid}_x.mp4"
    ), patch("src.index.reset_collection", return_value=False), patch(
        "src.index.index_video",
        side_effect=lambda p, c, **kw: {**_chunk_result(), "video_id": kw["video_id"],
                                       "indexed": 3, "chunks": 3, "segments": 1,
                                       "transcript_source": "cache", "collection": "c",
                                       "chroma_path": "p", "model": "m",
                                       "telemetry": "t", "chunk_telemetry": "t"},
    ) as idx:
        summary = index_dev_split(cfg, reset=False)

    assert summary["videos"] == ["181", "521"]
    assert summary["indexed"] == 6
    # The corpus id is passed explicitly rather than left to the filename, because the
    # video_id in the store is what a citation resolves against.
    assert [c.kwargs["video_id"] for c in idx.call_args_list] == ["181", "521"]


def test_index_dev_split_fetches_only_what_is_missing(cfg, tmp_path):
    videos = [{"video_id": "181", "split": "dev"}, {"video_id": "521", "split": "dev"}]
    present = {"181": tmp_path / "181_x.mp4"}
    calls = []

    def fake_local(vid, *a):
        return present.get(vid)

    def fake_fetch(vid, out, cfg_, allow_heldout):
        calls.append(vid)
        present[vid] = tmp_path / f"{vid}_fetched.mp4"

    with patch("src.index.dev_videos", return_value=videos), patch(
        "src.index.local_file", side_effect=fake_local
    ), patch("src.index.reset_collection", return_value=False), patch(
        "src.sample.fetch_real", side_effect=fake_fetch
    ), patch(
        "src.index.index_video",
        return_value={**_chunk_result(), "indexed": 1, "chunks": 1, "segments": 1,
                      "transcript_source": "cache", "collection": "c", "chroma_path": "p",
                      "model": "m", "telemetry": "t", "chunk_telemetry": "t"},
    ):
        index_dev_split(cfg, reset=False)

    assert calls == ["521"], "the already-fetched video must not be downloaded again"


def test_index_dev_split_fetches_with_heldout_refused(cfg, tmp_path):
    """allow_heldout must be False. dev_videos() already filters, so this is the second
    lock on the same door — and it is the one src/sample.py enforces."""
    videos = [{"video_id": "181", "split": "dev"}]
    seen = {}

    def fake_fetch(vid, out, cfg_, allow_heldout):
        seen["allow_heldout"] = allow_heldout
        return None

    with patch("src.index.dev_videos", return_value=videos), patch(
        "src.index.local_file", side_effect=[None, tmp_path / "181_x.mp4"]
    ), patch("src.index.reset_collection", return_value=False), patch(
        "src.sample.fetch_real", side_effect=fake_fetch
    ), patch(
        "src.index.index_video",
        return_value={**_chunk_result(), "indexed": 1, "chunks": 1, "segments": 1,
                      "transcript_source": "cache", "collection": "c", "chroma_path": "p",
                      "model": "m", "telemetry": "t", "chunk_telemetry": "t"},
    ):
        index_dev_split(cfg, reset=False)

    assert seen["allow_heldout"] is False


def test_index_dev_split_fails_when_a_fetch_writes_nothing(cfg, tmp_path):
    """yt-dlp can exit 0 having written nothing. Indexing would then skip a dev video and
    the gate would score 3 videos while reporting 4."""
    with patch("src.index.dev_videos", return_value=[{"video_id": "181", "split": "dev"}]), patch(
        "src.index.local_file", return_value=None
    ), patch("src.index.reset_collection", return_value=False), patch(
        "src.sample.fetch_real", return_value=None
    ):
        with pytest.raises(IndexingError) as exc:
            index_dev_split(cfg, reset=False)
    assert "nothing landed" in str(exc.value)


def test_index_dev_split_refuses_an_empty_dev_split(cfg):
    with patch("src.index.dev_videos", return_value=[]):
        with pytest.raises(IndexingError) as exc:
            index_dev_split(cfg, reset=False)
    assert "dev side of the split" in str(exc.value)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_report_prints_the_counts_and_the_model(capsys):
    report(
        {
            "video_id": "181",
            "duration_s": 95.8,
            "transcript_source": "cache",
            "segments": 24,
            "chunks": 6,
            "indexed": 6,
            "collection": "vrag",
            "chroma_path": "./chroma",
            "model": "nomic-ai/nomic-embed-text-v1.5-GGUF",
            "telemetry": "$0.0000/video-hour  35.4xrealtime",
            "chunk_telemetry": "$0.0000/video-hour  2330.4xrealtime",
        }
    )
    out = capsys.readouterr().out
    assert "181" in out
    assert "6 -> 6 row(s) upserted" in out
    assert "nomic-ai/nomic-embed-text-v1.5-GGUF" in out
    assert "$0.0000/video-hour" in out
