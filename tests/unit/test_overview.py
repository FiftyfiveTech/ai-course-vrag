"""src/overview.py — the fold. No network, no model, no index.

`windows` and `_merge` are the two pieces the fold added, and both are pure: one splits a
transcript, the other reconciles what came back. The model call between them is the gate's
job, not a unit test's.

Why there is a fold at all, since it is the thing these tests are about: no real transcript
fits one call on Groq's free tier. The tier meters tokens per minute, the limit is 8000, and
video 611 in one pass asked for 17152. The arithmetic is on `overview.max_context_chars` in
config.toml.
"""

from __future__ import annotations

import pytest

from schemas.overview import Overview, Person, Span, Topic
from src import overview as mod
from src.retrieve import RetrievedChunk


def chunk(t_start: float, t_end: float, text: str = "some words here") -> RetrievedChunk:
    return RetrievedChunk(
        video_id="611", t_start=t_start, t_end=t_end, text=text, score=0.1
    )


# --------------------------------------------------------------------------- windows


def test_no_chunks_is_no_windows():
    assert mod.windows([], 1000) == []


def test_a_transcript_that_fits_is_one_window():
    chunks = [chunk(0.0, 25.0), chunk(17.0, 42.0)]
    assert mod.windows(chunks, 10_000) == [chunks]


def test_every_window_is_under_the_ceiling():
    chunks = [chunk(i * 17.0, i * 17.0 + 25.0, "word " * 40) for i in range(40)]
    ceiling = 2_000
    groups = mod.windows(chunks, ceiling)
    assert len(groups) > 1, "this fixture is meant to need folding"
    for group in groups:
        assert len(mod.render_transcript(group)) <= ceiling


def test_the_fold_loses_no_chunk_and_keeps_them_in_order():
    """A dropped chunk is a stretch of the video the overview silently does not describe."""
    chunks = [chunk(i * 17.0, i * 17.0 + 25.0, "word " * 40) for i in range(40)]
    flat = [c for group in mod.windows(chunks, 2_000) for c in group]
    assert flat == chunks


def test_windows_are_contiguous_stretches():
    """A window is handed to the build prompt as though it were a whole transcript, and the
    topics it returns are 'the video in order'. That is only true of a contiguous stretch."""
    chunks = [chunk(i * 17.0, i * 17.0 + 25.0, "word " * 40) for i in range(40)]
    groups = mod.windows(chunks, 2_000)
    for group in groups:
        assert group == sorted(group, key=lambda c: c.t_start)
    for earlier, later in zip(groups, groups[1:]):
        assert earlier[-1].t_start <= later[0].t_start


def test_one_oversized_chunk_still_gets_a_window_rather_than_being_dropped():
    """It will 413, and the 413 names the numbers. Losing the stretch would say nothing."""
    chunks = [chunk(0.0, 25.0, "word " * 5000)]
    assert mod.windows(chunks, 100) == [chunks]


# --------------------------------------------------------------------------- merge


def person(name: str, t_start: float, described_as: str = "someone") -> Person:
    return Person(
        name=name,
        described_as=described_as,
        evidence=Span(t_start=t_start, t_end=t_start + 25.0),
    )


def part(abstract: str, people: list[Person], topics: list[Topic]) -> Overview:
    return Overview(abstract=abstract, people=people, topics=topics)


def topic(t_start: float, text: str) -> Topic:
    return Topic(t_start=t_start, t_end=t_start + 25.0, topic=text)


def test_merge_keeps_the_earliest_evidence_for_a_repeated_name(monkeypatch):
    """The first time a video names someone is the moment a viewer wants to jump to."""
    merged = _merge_with_fake_abstract(
        monkeypatch,
        [
            part("a", [person("Bernini", 900.0)], []),
            part("b", [person("Bernini", 21.8)], []),
        ],
    )
    assert len(merged.people) == 1
    assert merged.people[0].evidence.t_start == 21.8


def test_merge_matches_names_case_and_whitespace_insensitively(monkeypatch):
    merged = _merge_with_fake_abstract(
        monkeypatch,
        [
            part("a", [person("Pope Urban VIII", 100.0)], []),
            part("b", [person("  pope urban viii ", 200.0)], []),
        ],
    )
    assert len(merged.people) == 1


def test_merge_keeps_distinct_people(monkeypatch):
    merged = _merge_with_fake_abstract(
        monkeypatch,
        [
            part("a", [person("Bernini", 10.0)], []),
            part("b", [person("Caravaggio", 20.0)], []),
        ],
    )
    assert {p.name for p in merged.people} == {"Bernini", "Caravaggio"}


def test_merge_orders_topics_by_time(monkeypatch):
    merged = _merge_with_fake_abstract(
        monkeypatch,
        [
            part("a", [], [topic(500.0, "late"), topic(100.0, "early")]),
            part("b", [], [topic(900.0, "latest")]),
        ],
    )
    assert [t.topic for t in merged.topics] == ["early", "late", "latest"]


def test_merge_invents_no_span(monkeypatch):
    """The reason people and topics are merged in code and not by a model: every span in the
    stored document has to be a span off a real chunk, and concatenation cannot invent one."""
    parts = [
        part("a", [person("Bernini", 21.8)], [topic(100.0, "x")]),
        part("b", [person("Ovid", 1100.6)], [topic(900.0, "y")]),
    ]
    before = {(s.t_start, s.t_end) for p in parts for s in p.spans()}
    merged = _merge_with_fake_abstract(monkeypatch, parts)
    after = {(s.t_start, s.t_end) for s in merged.spans()}
    assert after <= before


def test_merge_refuses_an_empty_abstract(monkeypatch):
    with pytest.raises(mod.OverviewError, match="empty"):
        _merge_real(monkeypatch, [part("a", [], [])], raw='{"abstract": "   "}')


def test_merge_refuses_a_reply_that_is_not_the_abstract_object(monkeypatch):
    with pytest.raises(mod.OverviewError, match="abstract"):
        _merge_real(monkeypatch, [part("a", [], [])], raw='{"summary": "wrong key"}')


# --------------------------------------------------------------------------- helpers
#
# _merge makes exactly one model call, for the abstract. These drive it with a canned reply
# so the merge logic - which is the part with rules in it - is tested without a network.


def _merge_real(monkeypatch, parts, raw: str):
    import src.answer as answer_mod

    monkeypatch.setattr(answer_mod, "_ask", lambda *a, **k: (raw, 0, "fake"))
    monkeypatch.setattr(
        answer_mod, "build_messages", lambda *a, **k: ("system", "user")
    )

    class _Cfg:
        def get(self, key):
            return "prompts/overview_merge_v1.md"

    return mod._merge("611", parts, _Cfg(), None)


def _merge_with_fake_abstract(monkeypatch, parts):
    return _merge_real(monkeypatch, parts, raw='{"abstract": "A merged abstract."}')
