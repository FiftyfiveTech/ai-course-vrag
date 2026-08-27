"""Tests for src/chunk.py — VRAG-014.

No ffmpeg, no network, no model calls: chunking is pure arithmetic over a list of
Segments, so every test here builds its own segments and asserts on the output.

The tests that matter most are the ones pinning the card's criterion — a chunk never
loses its time range — and the two design decisions that could be "fixed" later without
noticing what they were for: the grid is measured from segments rather than copied off the
window bounds, and an overlap really does place a boundary-straddling segment twice.
"""

from __future__ import annotations

import json

import pytest

from src.chunk import (
    CITATION_TOLERANCE_S,
    Chunk,
    ChunkError,
    chunk_config,
    chunk_segments,
    load_transcript,
    order_segments,
    report,
    resolve_video_id,
    save_transcript,
    verify,
    windows,
)
from src.config import Config, ConfigError
from src.transcript import Segment

from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def cfg_with(**chunk_table) -> Config:
    """A Config whose only content is a [chunk] table."""
    return Config(path=Path("test.toml"), raw=b"", data={"chunk": chunk_table})


LEVERS = {"window_s": 30.0, "overlap_s": 10.0, "hop_s": 20.0}


def speech(*spans: tuple[float, float]) -> list[Segment]:
    """Segments at the given (start, end) times, each with distinguishable text."""
    return [Segment(t_start=a, t_end=b, text=f"seg at {a}") for a, b in spans]


# ---------------------------------------------------------------------------
# Config levers
# ---------------------------------------------------------------------------


def test_chunk_config_reads_both_levers_and_derives_hop():
    levers = chunk_config(cfg_with(window_s=30.0, overlap_s=10.0))
    assert levers == {"window_s": 30.0, "overlap_s": 10.0, "hop_s": 20.0}


def test_chunk_config_accepts_zero_overlap():
    assert chunk_config(cfg_with(window_s=30.0, overlap_s=0))["hop_s"] == 30.0


def test_chunk_config_refuses_overlap_at_or_above_window():
    # hop would be <= 0 and the grid would never advance.
    with pytest.raises(ConfigError, match="less than"):
        chunk_config(cfg_with(window_s=30.0, overlap_s=30.0))
    with pytest.raises(ConfigError, match="less than"):
        chunk_config(cfg_with(window_s=30.0, overlap_s=45.0))


def test_chunk_config_refuses_non_positive_window():
    with pytest.raises(ConfigError, match="must be > 0"):
        chunk_config(cfg_with(window_s=0, overlap_s=0))


def test_chunk_config_refuses_negative_overlap():
    with pytest.raises(ConfigError, match="must be >= 0"):
        chunk_config(cfg_with(window_s=30.0, overlap_s=-5.0))


def test_chunk_config_refuses_non_numbers():
    with pytest.raises(ConfigError, match="must be a number"):
        chunk_config(cfg_with(window_s="30", overlap_s=10.0))


def test_chunk_config_has_no_defaults():
    # src/config.py's contract: a missing lever raises rather than falling back.
    with pytest.raises(ConfigError, match="chunk.overlap_s"):
        chunk_config(cfg_with(window_s=30.0))


def test_real_config_toml_levers_are_within_the_citation_tolerance():
    """The repo's own config has to be chunkable, and its window has to be citable.

    QA_SPEC 2 scores a citation on |t_start - t_ref| <= 30, and a citation points at the
    chunk it came from, so a window wider than that ships a scoring bug.
    """
    from src.config import load as load_config

    levers = chunk_config(load_config(Path(__file__).parent.parent.parent / "config.toml"))
    # Strictly under, not at: a chunk overhangs its window by the length of the segments at
    # each end, so a window sitting exactly on the tolerance has no room to overhang into.
    assert levers["window_s"] < CITATION_TOLERANCE_S
    assert levers["hop_s"] > 0


# ---------------------------------------------------------------------------
# The window grid
# ---------------------------------------------------------------------------


def test_windows_start_at_zero_and_step_by_hop():
    grid = list(windows(50.0, 30.0, 20.0))
    assert [(i, s) for i, s, _ in grid] == [(0, 0.0), (1, 20.0), (2, 40.0)]
    assert [e for _, _, e in grid] == [30.0, 50.0, 70.0]


def test_windows_last_one_starts_at_or_before_the_end_of_speech():
    for span in (0.0, 1.0, 19.9, 20.0, 20.1, 300.0):
        grid = list(windows(span, 30.0, 20.0))
        assert grid, f"no windows for span {span}"
        assert grid[-1][1] <= span or span == 0.0
        assert grid[-1][2] > span, f"span {span} falls outside the last window"


def test_windows_refuses_a_non_advancing_grid():
    with pytest.raises(ChunkError, match="hop_s"):
        list(windows(10.0, 30.0, 0.0))


# ---------------------------------------------------------------------------
# Chunk fields — the card's criterion
# ---------------------------------------------------------------------------


def test_every_chunk_carries_video_id_t_start_t_end():
    chunks, _ = chunk_segments("181", speech((0, 5), (25, 40), (70, 80)), LEVERS)
    assert chunks
    for c in chunks:
        assert c.video_id == "181"
        assert isinstance(c.t_start, float) and isinstance(c.t_end, float)
        assert c.t_end > c.t_start


def test_chunk_range_is_measured_from_its_segments_not_the_window():
    """A segment is never split, so the last one in a window runs past the window end."""
    # Window 0 is [0, 30); this segment starts inside it and ends well after.
    chunks, _ = chunk_segments("v", speech((28.0, 41.5)), LEVERS)
    first = chunks[0]
    assert first.window_t_start == 0.0 and first.window_t_end == 30.0
    assert first.t_start == 28.0
    assert first.t_end == 41.5, "t_end was clipped to the window and lost the segment's end"


def test_chunk_range_starts_at_the_first_segment_not_the_window_start():
    # Window 0 is [0, 30) but the speech starts at 12.
    chunks, _ = chunk_segments("v", speech((12.0, 18.0)), LEVERS)
    assert chunks[0].t_start == 12.0
    assert chunks[0].window_t_start == 0.0


def test_chunk_ids_are_unique_dense_and_prefixed_with_the_video():
    chunks, _ = chunk_segments("521", speech(*[(t, t + 4) for t in range(0, 200, 10)]), LEVERS)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == [f"521-{i:04d}" for i in range(len(chunks))]


def test_chunks_come_back_in_time_order():
    chunks, _ = chunk_segments("v", speech((100, 110), (0, 10), (50, 60)), LEVERS)
    starts = [c.t_start for c in chunks]
    assert starts == sorted(starts)


def test_chunk_text_joins_its_segments_in_order():
    segs = [
        Segment(0.0, 4.0, "first"),
        Segment(5.0, 9.0, "second"),
        Segment(10.0, 14.0, "third"),
    ]
    chunks, _ = chunk_segments("v", segs, LEVERS)
    assert chunks[0].text == "first second third"


def test_as_dict_round_trips_through_json_with_its_time_range():
    chunks, _ = chunk_segments("181", speech((0, 5), (25, 40)), LEVERS)
    for c, row in zip(chunks, [json.loads(json.dumps(c.as_dict())) for c in chunks]):
        assert row["video_id"] == c.video_id
        assert row["t_start"] == pytest.approx(c.t_start)
        assert row["t_end"] == pytest.approx(c.t_end)
        assert row["duration_s"] == pytest.approx(c.t_end - c.t_start)


# ---------------------------------------------------------------------------
# Coverage and overlap
# ---------------------------------------------------------------------------


def test_no_segment_is_dropped():
    segs = speech(*[(t, t + 3.5) for t in [0, 7, 19, 29, 31, 62, 63, 140]])
    chunks, _ = chunk_segments("v", segs, LEVERS)
    placed = {i for c in chunks for i in c.segment_ids}
    assert placed == set(range(len(segs)))


@pytest.mark.parametrize("window_s,overlap_s", [(30, 10), (30, 0), (45, 15), (10, 9), (60, 30)])
def test_no_segment_is_dropped_at_any_lever_setting(window_s, overlap_s):
    levers = chunk_config(cfg_with(window_s=float(window_s), overlap_s=float(overlap_s)))
    segs = speech(*[(t * 1.7, t * 1.7 + 2.2) for t in range(60)])
    chunks, _ = chunk_segments("v", segs, levers)
    placed = {i for c in chunks for i in c.segment_ids}
    assert placed == set(range(len(segs)))
    assert not verify(chunks, segs)


def test_a_segment_straddling_a_boundary_lands_in_two_chunks():
    """This is what the overlap is for: the sentence is retrievable whole from one of them."""
    # hop=20, so window 0 is [0,30) and window 1 is [20,50). 18–24 crosses 20.
    segs = speech((0, 6), (18, 24), (40, 46))
    chunks, _ = chunk_segments("v", segs, LEVERS)
    holders = [c.chunk_id for c in chunks if 1 in c.segment_ids]
    assert len(holders) == 2, f"the straddling segment is in {holders}, expected two chunks"


def test_zero_overlap_places_every_segment_exactly_once():
    segs = speech(*[(t, t + 2) for t in range(0, 120, 5)])
    levers = chunk_config(cfg_with(window_s=30.0, overlap_s=0.0))
    chunks, _ = chunk_segments("v", segs, levers)
    placements = [i for c in chunks for i in c.segment_ids]
    assert sorted(placements) == list(range(len(segs)))
    assert len(placements) == len(set(placements))


def test_overlap_costs_duplicated_text():
    segs = speech(*[(t, t + 2) for t in range(0, 200, 5)])
    with_overlap, _ = chunk_segments("v", segs, LEVERS)
    without, _ = chunk_segments("v", segs, chunk_config(cfg_with(window_s=30.0, overlap_s=0.0)))
    assert sum(len(c.text) for c in with_overlap) > sum(len(c.text) for c in without)


# ---------------------------------------------------------------------------
# Dropped windows — counted, not invisible
# ---------------------------------------------------------------------------


def test_a_silent_window_produces_no_chunk_but_is_counted():
    # Speech at 0–5 and at 200–205; everything between is silence.
    chunks, stats = chunk_segments("v", speech((0, 5), (200, 205)), LEVERS)
    assert len(chunks) == 2
    assert stats["empty"] > 0
    assert stats["windows"] == stats["empty"] + len(chunks) + stats["duplicate"]


def test_a_window_identical_to_its_predecessor_is_dropped_as_a_duplicate():
    # One segment at 21–24 sits inside windows [0,30), [20,50) — wait, not [0,30) only:
    # windows 0 and 1 both contain it and nothing else, so window 1 is a pure duplicate.
    chunks, stats = chunk_segments("v", speech((21.0, 24.0)), LEVERS)
    assert len(chunks) == 1, [c.chunk_id for c in chunks]
    assert stats["duplicate"] == 1


def test_dropping_duplicates_never_drops_a_segment():
    segs = speech((21.0, 24.0), (25.0, 26.0))
    chunks, stats = chunk_segments("v", segs, LEVERS)
    assert stats["duplicate"] >= 1
    assert not verify(chunks, segs)


def test_empty_transcript_gives_no_chunks_and_no_crash():
    chunks, stats = chunk_segments("v", [], LEVERS)
    assert chunks == []
    assert stats == {"windows": 0, "empty": 0, "duplicate": 0}


# ---------------------------------------------------------------------------
# Segment hygiene
# ---------------------------------------------------------------------------


def test_order_segments_sorts_by_time():
    segs = speech((10, 12), (0, 2), (5, 7))
    assert [s.t_start for s in order_segments(segs)] == [0, 5, 10]


def test_order_segments_refuses_a_backwards_segment():
    with pytest.raises(ChunkError, match="impossible time range"):
        order_segments([Segment(t_start=10.0, t_end=4.0, text="backwards")])


def test_order_segments_refuses_a_negative_start():
    with pytest.raises(ChunkError, match="impossible time range"):
        order_segments([Segment(t_start=-1.0, t_end=4.0, text="before the video")])


def test_a_zero_length_segment_is_placed_by_its_point_in_time():
    # Some ASR responses carry these. Dropping one would lose its text silently.
    segs = [Segment(t_start=22.0, t_end=22.0, text="blip"), Segment(35.0, 38.0, "after")]
    chunks, _ = chunk_segments("v", segs, LEVERS)
    assert {i for c in chunks for i in c.segment_ids} == {0, 1}


# ---------------------------------------------------------------------------
# verify() — the invariant checker itself
# ---------------------------------------------------------------------------


def test_verify_is_clean_on_real_output():
    segs = speech(*[(t * 3.3, t * 3.3 + 2.9) for t in range(40)])
    chunks, _ = chunk_segments("181", segs, LEVERS)
    assert verify(chunks, segs, duration_s=200.0) == []


def _chunk(**over) -> Chunk:
    base = dict(
        chunk_id="v-0000",
        video_id="v",
        t_start=0.0,
        t_end=10.0,
        text="hello",
        n_segments=1,
        segment_ids=(0,),
        window_index=0,
        window_t_start=0.0,
        window_t_end=30.0,
    )
    return Chunk(**{**base, **over})


def test_verify_catches_a_chunk_whose_range_is_a_point():
    segs = speech((0, 10))
    problems = verify([_chunk(t_start=4.0, t_end=4.0)], segs)
    assert any("not after" in p for p in problems)


def test_verify_catches_a_range_that_does_not_contain_its_segment():
    # The failure the card is really about: the chunk keeps text whose time it has lost.
    segs = speech((0, 25))
    problems = verify([_chunk(t_end=10.0)], segs)
    assert any("does not contain" in p for p in problems)


def test_verify_catches_a_dropped_segment():
    segs = speech((0, 10), (60, 70))
    problems = verify([_chunk()], segs)
    assert any("in no chunk" in p for p in problems)


def test_verify_catches_a_missing_video_id():
    problems = verify([_chunk(video_id="")], speech((0, 10)))
    assert any("no video_id" in p for p in problems)


def test_verify_catches_duplicate_chunk_ids():
    segs = speech((0, 10))
    problems = verify([_chunk(), _chunk()], segs)
    assert any("duplicate chunk_id" in p for p in problems)


def test_verify_catches_chunks_out_of_order():
    segs = speech((0, 10), (20, 30))
    late = _chunk(chunk_id="v-0000", t_start=20.0, t_end=30.0, segment_ids=(1,))
    early = _chunk(chunk_id="v-0001", t_start=0.0, t_end=10.0, segment_ids=(0,))
    assert any("goes backwards" in p for p in verify([late, early], segs))


def test_verify_catches_empty_text():
    assert any("empty text" in p for p in verify([_chunk(text="   ")], speech((0, 10))))


def test_verify_catches_a_chunk_past_the_end_of_the_video():
    segs = speech((0, 10))
    problems = verify([_chunk(t_end=10.0)], segs, duration_s=5.0)
    assert any("past the end" in p for p in problems)


def test_verify_catches_an_n_segments_that_disagrees_with_the_ids():
    problems = verify([_chunk(n_segments=7)], speech((0, 10)))
    assert any("n_segments" in p for p in problems)


def test_verify_catches_a_non_finite_time():
    problems = verify([_chunk(t_end=float("nan"))], speech((0, 10)))
    assert any("finite" in p for p in problems)


def test_verify_catches_a_none_time():
    assert any("finite" in p for p in verify([_chunk(t_end=None)], speech((0, 10))))


def test_verify_flags_segments_that_produced_nothing():
    assert any("0 chunks" in p for p in verify([], speech((0, 10))))


def test_verify_is_clean_on_an_empty_transcript():
    assert verify([], []) == []


# ---------------------------------------------------------------------------
# video_id resolution
# ---------------------------------------------------------------------------


def test_resolve_video_id_reads_the_corpus_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"videos": [{"video_id": "181", "split": "dev", "youtube_id": "abc"}]}),
        encoding="utf-8",
    )
    video_id, source = resolve_video_id(Path("samples/181_abc.mp4"), manifest)
    assert video_id == "181"
    assert "dev" in source


def test_resolve_video_id_falls_back_to_the_stem_for_a_non_corpus_file(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"videos": []}), encoding="utf-8")
    video_id, source = resolve_video_id(Path("samples/one.mp4"), manifest)
    assert video_id == "one"
    assert "not a corpus video" in source


def test_resolve_video_id_survives_a_missing_manifest(tmp_path):
    video_id, source = resolve_video_id(Path("samples/one.mp4"), tmp_path / "gone.json")
    assert video_id == "one"
    assert "unreadable" in source


# ---------------------------------------------------------------------------
# Transcript cache — the reason re-chunking is free
# ---------------------------------------------------------------------------


def test_transcript_cache_round_trips(tmp_path):
    cfg = Config(
        path=Path("test.toml"),
        raw=b"",
        data={"transcript": {"arm": "groq", "model": "openai/whisper-large-v3-turbo",
                             "language": "en"}},
    )
    segs = [Segment(0.0, 4.2, "hello there"), Segment(4.2, 9.9, "and again")]
    path = tmp_path / "transcript.json"
    save_transcript(path, segs, "deadbeef", cfg)

    loaded, payload = load_transcript(path)
    assert loaded == segs
    assert payload["source_sha256"] == "deadbeef"
    assert payload["model"] == "openai/whisper-large-v3-turbo"


def test_load_transcript_rejects_a_malformed_segment(tmp_path):
    path = tmp_path / "transcript.json"
    path.write_text(json.dumps({"segments": [{"t_start": 0.0, "text": "no end"}]}), encoding="utf-8")
    with pytest.raises(ChunkError, match="malformed"):
        load_transcript(path)


def test_load_transcript_rejects_a_file_with_no_segments(tmp_path):
    path = tmp_path / "transcript.json"
    path.write_text(json.dumps({"task": "VRAG-008"}), encoding="utf-8")
    with pytest.raises(ChunkError, match="no 'segments' list"):
        load_transcript(path)


# ---------------------------------------------------------------------------
# The dump itself
# ---------------------------------------------------------------------------


def _result_for(segs, video_id="181", duration_s=200.0):
    """A build()-shaped result assembled without touching ffmpeg or an ASR arm."""
    chunks, stats = chunk_segments(video_id, segs, LEVERS)
    rows = [c.as_dict() for c in chunks]
    return {
        "task": "VRAG-014",
        "video_id": video_id,
        "video_id_source": "test",
        "source": {"path": "samples/one.mp4", "sha256": "abc"},
        "duration_s": duration_s,
        "transcript": {
            "source": "cache",
            "model": "openai/whisper-large-v3-turbo",
            "segments": len(segs),
            "path": "runs/one/transcript.json",
        },
        "levers": LEVERS,
        "counts": {
            "chunks": len(chunks),
            "windows": stats["windows"],
            "windows_empty": stats["empty"],
            "windows_duplicate": stats["duplicate"],
            "segment_placements": sum(c.n_segments for c in chunks),
            "chars": sum(len(c.text) for c in chunks),
            "transcript_chars": sum(len(s.text) for s in segs),
        },
        "coverage": {
            "t_start": rows[0]["t_start"] if rows else None,
            "t_end": max((r["t_end"] for r in rows), default=None),
            "max_chunk_duration_s": max((r["duration_s"] for r in rows), default=None),
            "mean_chunk_duration_s": 10.0 if rows else None,
            "over_citation_tolerance": sum(
                1 for r in rows if r["duration_s"] > CITATION_TOLERANCE_S
            ),
            "longest_segment_s": round(
                max((s.t_end - s.t_start for s in segs), default=0.0), 3
            ),
        },
        "problems": verify(chunks, segs, duration_s),
        "config": {"path": "config.toml", "sha256": "0" * 64},
        "timing": {"total_s": 0.01},
        "chunks": rows,
        "telemetry": "$0.0000/video-hour  1.0×realtime",
        "manifest": "runs/one/chunks.json",
    }


def test_report_prints_one_row_per_chunk_with_its_time_range(capsys):
    segs = speech(*[(t * 6.0, t * 6.0 + 5.0) for t in range(20)])
    result = _result_for(segs)
    report(result)
    out = capsys.readouterr().out

    for row in result["chunks"]:
        assert row["chunk_id"] in out
        assert f"{row['t_start']:.3f}" in out
        assert f"{row['t_end']:.3f}" in out
    assert "video_id  181" in out
    assert "0 problems" in out


def test_report_says_so_when_there_is_nothing_to_chunk(capsys):
    report(_result_for([]))
    out = capsys.readouterr().out
    assert "no chunks" in out
    assert "nothing to index" in out


def test_report_warns_when_a_chunk_is_wider_than_the_citation_tolerance(capsys):
    # One long segment: the chunk has to span it, so it exceeds the +/-30 s rule.
    result = _result_for(speech((0.0, 95.0)))
    assert result["coverage"]["over_citation_tolerance"] == 1
    report(result)
    assert "WARN" in capsys.readouterr().out


def test_report_prints_the_problems_when_a_chunk_lost_its_range(capsys):
    result = _result_for(speech((0, 10)))
    result["problems"] = ["v-0000: range [0.0, 1.0] does not contain segment 0 [0.0, 10.0]"]
    report(result)
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "does not contain" in out
