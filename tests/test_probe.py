"""Tests for src/probe.py.

No Ollama, Chroma or network. `retrieve` is patched; the input parsing and the citation
line are tested directly, because those are the parts that decide whether a printed hit can
be checked by a human.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.probe import (
    ProbeError,
    cite,
    probe,
    read_questions,
    report,
    video_urls,
)
from src.retrieve import RetrievedChunk
from src.telemetry import Meter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg(tmp_path):
    from src.config import load

    p = tmp_path / "config.toml"
    p.write_text(
        "[retrieve]\ntop_k = 5\n[embed]\n"
        'model = "nomic-ai/nomic-embed-text-v1.5-GGUF:F16"\n'
        'chroma_path = "./chroma_test"\ncollection = "vrag_test"\nbatch_size = 32\n',
        encoding="utf-8",
    )
    return load(p)


def _hit(video_id="611", t_start=269.6, t_end=303.1, score=0.276, text="San Lorenzo"):
    return RetrievedChunk(
        video_id=video_id, t_start=t_start, t_end=t_end, text=text, score=score
    )


# ---------------------------------------------------------------------------
# read_questions - plain text
# ---------------------------------------------------------------------------


def test_read_questions_one_per_line(tmp_path):
    p = tmp_path / "q.txt"
    p.write_text("first question\nsecond question\n", encoding="utf-8")
    assert read_questions(p) == ["first question", "second question"]


def test_read_questions_skips_comments_and_blanks(tmp_path):
    """The starter file explains itself in comments; those must not become questions."""
    p = tmp_path / "q.txt"
    p.write_text("# a heading\n\nreal question\n\n#  another note\n", encoding="utf-8")
    assert read_questions(p) == ["real question"]


def test_read_questions_strips_surrounding_whitespace(tmp_path):
    p = tmp_path / "q.txt"
    p.write_text("   padded question   \n", encoding="utf-8")
    assert read_questions(p) == ["padded question"]


def test_read_questions_keeps_a_hash_inside_a_question(tmp_path):
    """Only a leading # is a comment - `#` mid-line is content."""
    p = tmp_path / "q.txt"
    p.write_text("what is the hashtag #papercutcraftpad for\n", encoding="utf-8")
    assert read_questions(p) == ["what is the hashtag #papercutcraftpad for"]


def test_read_questions_rejects_a_missing_file(tmp_path):
    with pytest.raises(ProbeError, match="not a file"):
        read_questions(tmp_path / "nope.txt")


def test_read_questions_rejects_a_file_with_only_comments(tmp_path):
    """Silently probing nothing would print a clean run that asked no questions."""
    p = tmp_path / "q.txt"
    p.write_text("# all comment\n\n", encoding="utf-8")
    with pytest.raises(ProbeError, match="no questions"):
        read_questions(p)


# ---------------------------------------------------------------------------
# read_questions - jsonl
# ---------------------------------------------------------------------------


def test_read_questions_from_jsonl_takes_the_question_field(tmp_path):
    """A labelled dev file can be probed without stripping it first."""
    p = tmp_path / "q.jsonl"
    p.write_text(
        json.dumps({"id": "d001", "question": "first", "video_id": "181", "t_ref": 30.0})
        + "\n"
        + json.dumps({"id": "d002", "question": "second", "unanswerable": True})
        + "\n",
        encoding="utf-8",
    )
    assert read_questions(p) == ["first", "second"]


def test_read_questions_from_jsonl_skips_rows_without_a_question(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(
        json.dumps({"id": "d001"}) + "\n" + json.dumps({"question": "real"}) + "\n",
        encoding="utf-8",
    )
    assert read_questions(p) == ["real"]


def test_read_questions_from_jsonl_names_the_bad_line(tmp_path):
    """A truncated file should say which line, not just "invalid JSON"."""
    p = tmp_path / "q.jsonl"
    p.write_text('{"question": "ok"}\nnot json at all\n', encoding="utf-8")
    with pytest.raises(ProbeError, match=r":2:"):
        read_questions(p)


def test_read_questions_reads_stdin(monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("piped question\n"))
    assert read_questions(Path("-")) == ["piped question"]


# ---------------------------------------------------------------------------
# video_urls
# ---------------------------------------------------------------------------


def test_video_urls_maps_id_to_url(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(
        json.dumps({"videos": [{"video_id": "611", "url": "https://example/611"}]}),
        encoding="utf-8",
    )
    assert video_urls(p) == {"611": "https://example/611"}


def test_video_urls_is_empty_rather_than_fatal_without_a_manifest(tmp_path):
    """A missing manifest costs you the links, not the run."""
    assert video_urls(tmp_path / "gone.json") == {}


def test_video_urls_survives_a_corrupt_manifest(tmp_path):
    p = tmp_path / "m.json"
    p.write_text("{ not json", encoding="utf-8")
    assert video_urls(p) == {}


# ---------------------------------------------------------------------------
# cite - the line a human checks the claim with
# ---------------------------------------------------------------------------


def test_cite_appends_a_youtube_link_at_the_cited_second():
    """The whole point: the timestamp has to be openable, or nobody verifies it."""
    urls = {"611": "https://www.youtube.com/watch?v=H8fGd3fCJbg"}
    line = cite(_hit(t_start=269.6), urls)
    assert "&t=269s" in line
    assert "video  611" in line
    assert "dist=0.276" in line


def test_cite_truncates_the_second_rather_than_rounding_up():
    """A link must never land after the moment it is citing."""
    urls = {"611": "https://www.youtube.com/watch?v=H8fGd3fCJbg"}
    assert "&t=269s" in cite(_hit(t_start=269.9), urls)


def test_cite_omits_the_link_for_an_unknown_video():
    line = cite(_hit(video_id="999"), {})
    assert "&t=" not in line
    assert "video  999" in line


def test_cite_omits_the_link_for_a_non_youtube_url():
    """`&t=` is a YouTube parameter. Appending it to another host invents a URL."""
    line = cite(_hit(), {"611": "https://vimeo.com/12345"})
    assert "&t=" not in line


# ---------------------------------------------------------------------------
# report and probe
# ---------------------------------------------------------------------------


def test_report_numbers_the_hits_from_one(capsys):
    report("a question", [_hit(), _hit(t_start=303.1)], {})
    out = capsys.readouterr().out
    assert "Q: a question" in out
    assert "1. video" in out and "2. video" in out


def test_report_collapses_whitespace_in_the_chunk_text(capsys):
    report("q", [_hit(text="line one\n  line   two")], {})
    assert "line one line two" in capsys.readouterr().out


def test_report_says_the_index_is_empty_rather_than_printing_nothing(capsys):
    """Zero hits and a working index look identical unless the output says so."""
    report("q", [], {})
    assert "make index-dev" in capsys.readouterr().out


def test_probe_asks_every_question_once(cfg, capsys):
    with patch("src.probe.retrieve", return_value=[_hit()]) as r:
        results = probe(["one", "two", "three"], cfg, Meter())
    assert r.call_count == 3
    assert [c.args[0] for c in r.call_args_list] == ["one", "two", "three"]
    assert len(results) == 3


def test_probe_prints_no_aggregate(cfg, capsys):
    """The module refuses to print a tally. A number here would get quoted as recall."""
    with patch("src.probe.retrieve", return_value=[_hit()]):
        probe(["one", "two"], cfg, Meter())
    out = capsys.readouterr().out.lower()
    for word in ("recall", "score", "hits", "passed", "total"):
        assert word not in out, f"probe output must not read as a score; found {word!r}"
