"""src/ask.py — VRAG-020. No network, no model, no index, no media.

`ask()` is one call to `src.answer.answer` and then pure rendering, so everything below
drives the rendering with a hand-built `AnswerRun`. What that leaves untested is whether the
model answers well, which belongs to `tests/gates/gate_phase2a.py`, and whether a browser
seeks — which is checked by opening the page.

The tests that matter here are the three ways the demo can be a lie:

* a citation that **looks** clickable and goes nowhere (a `<video src>` to a file that is not
  there, a `#t=` a browser ignores, a Windows backslash in an href);
* a page that needs the network, a server or a build step to render;
* a page that renders a question or an answer as markup instead of as text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from schemas.answer import Answer
from src.answer import AnswerRun
from src.answer import effective_model
from src.ask import (
    AskError,
    Cite,
    Source,
    ask,
    build_cites,
    clock,
    deep_link,
    load_manifest,
    page_path,
    passage_for,
    relative_src,
    render_page,
    resolve_source,
)
from src.config import Config
from src.config import load as load_config
from src.retrieve import RetrievedChunk
from src.telemetry import Meter

CONFIG = load_config("config.toml")
MANIFEST = Path("data/corpus/manifest.json")


# ---------------------------------------------------------------------------
# Fixtures — the pieces src.answer would have produced
# ---------------------------------------------------------------------------


def chunk(video_id="611", t_start=20.0, t_end=45.0, text="Bernini was eight."):
    return RetrievedChunk(
        video_id=video_id, t_start=t_start, t_end=t_end, text=text, score=0.3
    )


def run_with(answer_obj: Answer | None, hits: list[RetrievedChunk], error=None) -> AnswerRun:
    return AnswerRun(
        question="q", hits=hits, raw="{}", answer=answer_obj, error=error, tokens=10
    )


def answered(citations: list[dict], text="He was eight years old.") -> Answer:
    return Answer(answer=text, citations=citations, abstain=False)


def cfg_with_pad(pad: float, tmp_path: Path) -> Config:
    path = tmp_path / "config.toml"
    path.write_text(f"[ask]\npad_s = {pad}\n", encoding="utf-8")
    return load_config(path)


def page_for(
    run: AnswerRun, cites: list[Cite], out: Path, question="How old was Bernini?"
) -> str:
    return render_page(question, run, cites, CONFIG, out, Meter())


def a_cite(**kw) -> Cite:
    defaults = dict(
        n=1,
        video_id="611",
        t_start=20.0,
        t_end=45.0,
        seek_s=15.0,
        passage="Bernini was eight.",
        source=Source(video_id="611", local=None, url=None),
    )
    defaults.update(kw)
    return Cite(**defaults)


# ---------------------------------------------------------------------------
# The lever, and the pad it controls
# ---------------------------------------------------------------------------


def test_the_shipped_config_declares_the_pad():
    """`config.get` has no defaults, so a missing [ask] section is a crash at demo time."""
    assert float(CONFIG.get("ask.pad_s")) >= 0.0


def test_the_seek_is_padded_back_from_the_citation(tmp_path):
    run = run_with(answered([{"video_id": "611", "t_start": 20.0, "t_end": 45.0}]), [chunk()])
    cite = build_cites(run, cfg_with_pad(5.0, tmp_path), {})[0]
    assert cite.t_start == 20.0, "the citation itself is not moved"
    assert cite.seek_s == 15.0


def test_the_pad_never_seeks_before_the_start_of_the_video(tmp_path):
    """A citation at 2 s with a 5 s pad is -3 s, and a negative currentTime is ignored."""
    run = run_with(
        answered([{"video_id": "611", "t_start": 2.0, "t_end": 30.0}]),
        [chunk(t_start=2.0, t_end=30.0)],
    )
    cite = build_cites(run, cfg_with_pad(5.0, tmp_path), {})[0]
    assert cite.seek_s == 0.0


def test_an_abstention_has_no_citations_to_render(tmp_path):
    run = run_with(Answer.abstention(), [chunk()])
    assert build_cites(run, cfg_with_pad(5.0, tmp_path), {}) == []


def test_a_schema_invalid_reply_has_no_citations_to_render(tmp_path):
    run = run_with(None, [chunk()], error="answer: field required")
    assert build_cites(run, cfg_with_pad(5.0, tmp_path), {}) == []


# ---------------------------------------------------------------------------
# Which copy of the video the page points at
# ---------------------------------------------------------------------------


def test_a_fetched_video_is_found_by_its_id_prefix(tmp_path):
    """`make sample-real` writes `<video_id>_<youtube_id>.<ext>` and yt-dlp picks the ext."""
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "611_H8fGd3fCJbg.webm").write_bytes(b"")
    source = resolve_source("611", {}, samples)
    assert source.playable and source.local is not None
    assert source.local.name == "611_H8fGd3fCJbg.webm"


def test_an_unfetched_video_falls_back_to_the_manifest_url(tmp_path):
    """The clean-clone case: no media on disk, and the pointer is all there is."""
    records = {"611": {"url": "https://www.youtube.com/watch?v=H8fGd3fCJbg"}}
    source = resolve_source("611", records, tmp_path / "nothing-here")
    assert not source.playable
    assert source.linkable
    assert source.url == "https://www.youtube.com/watch?v=H8fGd3fCJbg"


def test_a_video_that_is_neither_on_disk_nor_in_the_manifest_is_not_linkable(tmp_path):
    source = resolve_source("999", {}, tmp_path / "nothing-here")
    assert not source.playable and not source.linkable


def test_every_dev_video_in_the_shipped_manifest_has_a_url():
    """The fallback player is only honest if the pointer it falls back to exists."""
    records = load_manifest(MANIFEST)
    assert records, f"{MANIFEST} holds no videos — run `make corpus`"
    missing = [vid for vid, rec in records.items() if not rec.get("url")]
    assert not missing, f"manifest videos with no url: {missing}"


# ---------------------------------------------------------------------------
# The deep link
# ---------------------------------------------------------------------------


def test_the_deep_link_carries_whole_seconds_with_the_s_suffix():
    """YouTube ignores `t=15.5` and opens at 0 — which reads as a broken citation."""
    link = deep_link("https://www.youtube.com/watch?v=abc", 15.7)
    assert "v=abc" in link
    assert "t=15s" in link
    assert "15.7" not in link


def test_the_deep_link_floors_rather_than_rounds():
    """Rounding up can land after the first word of the sentence being cited."""
    assert "t=15s" in deep_link("https://www.youtube.com/watch?v=abc", 15.99)


def test_the_deep_link_keeps_a_url_that_already_has_a_query():
    link = deep_link("https://example.com/v?id=7&list=x", 30.0)
    assert "id=7" in link and "list=x" in link and "t=30s" in link


def test_the_deep_link_never_emits_a_negative_time():
    assert "t=0s" in deep_link("https://www.youtube.com/watch?v=abc", -4.0)


# ---------------------------------------------------------------------------
# The <video src>
# ---------------------------------------------------------------------------


def test_the_video_src_is_relative_to_the_page_and_uses_forward_slashes():
    """A backslash in an href is an escape to a browser, not a separator — the player would
    silently show nothing, which is the failure this whole module exists to avoid."""
    src = relative_src(Path("samples/611_H8fGd3fCJbg.mp4"), Path("runs/ask/page.html"))
    assert src == "../../samples/611_H8fGd3fCJbg.mp4"
    assert "\\" not in src


def test_the_relative_src_resolves_back_to_the_media_file(tmp_path):
    """Computed, then walked back — the one property a relative path has to have."""
    media = tmp_path / "samples" / "611_x.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"")
    page = tmp_path / "runs" / "ask" / "p.html"
    page.parent.mkdir(parents=True)
    assert (page.parent / relative_src(media, page)).resolve() == media.resolve()


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def test_a_fetched_video_gets_a_player_and_a_seek_button(tmp_path):
    media = tmp_path / "samples" / "611_x.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"")
    page = tmp_path / "runs" / "ask" / "p.html"
    cite = a_cite(source=Source("611", local=media, url="https://youtu.be/x"))
    html = page_for(run_with(answered([]), [chunk()]), [cite], page)

    assert '<video id="v611"' in html
    assert 'src="../../samples/611_x.mp4#t=15.00"' in html
    assert 'data-video="v611"' in html and 'data-seek="15.00"' in html
    assert 'data-end="45.00"' in html


def test_an_unfetched_video_gets_a_link_instead_of_a_button(tmp_path):
    """No `<video>` at all rather than one pointing at a file that is not there."""
    page = tmp_path / "runs" / "ask" / "p.html"
    cite = a_cite(source=Source("611", local=None, url="https://www.youtube.com/watch?v=x"))
    html = page_for(run_with(answered([]), [chunk()]), [cite], page)

    assert "<video" not in html
    assert "t=15s" in html
    assert "make sample-real VIDEO_ID=611" in html


def test_a_citation_with_nowhere_to_go_is_not_rendered_as_a_link(tmp_path):
    """The one outcome worth avoiding is a control that looks clickable and is not."""
    page = tmp_path / "runs" / "ask" / "p.html"
    html = page_for(run_with(answered([]), [chunk()]), [a_cite()], page)

    assert '<span class="seek dead">' in html
    assert 'href="#"' not in html


def test_two_citations_on_one_video_share_a_single_player(tmp_path):
    media = tmp_path / "samples" / "611_x.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"")
    source = Source("611", local=media, url=None)
    cites = [
        a_cite(n=1, seek_s=15.0, source=source),
        a_cite(n=2, t_start=80.0, t_end=100.0, seek_s=75.0, source=source),
    ]
    html = page_for(run_with(answered([]), [chunk()]), cites, tmp_path / "p.html")

    assert html.count("<video") == 1
    assert html.count('id="v611"') == 1
    assert 'data-seek="15.00"' in html and 'data-seek="75.00"' in html
    assert 'id="c1"' in html and 'id="c2"' in html


def test_the_player_opens_at_the_earliest_citation_on_that_video(tmp_path):
    """The `#t=` media fragment is what positions the player before any JS runs."""
    media = tmp_path / "samples" / "611_x.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"")
    source = Source("611", local=media, url=None)
    cites = [
        a_cite(n=1, seek_s=75.0, source=source),
        a_cite(n=2, seek_s=15.0, source=source),
    ]
    html = page_for(run_with(answered([]), [chunk()]), cites, tmp_path / "p.html")
    assert "#t=15.00" in html


def test_the_page_needs_no_network_no_server_and_no_build_step(tmp_path):
    """Inline CSS and inline JS. A demo that needs a CDN is not a demo on a train."""
    media = tmp_path / "samples" / "611_x.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"")
    cite = a_cite(source=Source("611", local=media, url=None))
    html = page_for(run_with(answered([]), [chunk()]), [cite], tmp_path / "p.html")

    assert "<style>" in html and "<script>" in html
    assert "<script src=" not in html
    assert '<link rel="stylesheet"' not in html
    external = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
    assert external == [], f"the page reaches the network for {external}"


def test_the_question_and_the_answer_are_escaped(tmp_path):
    """A question is user input and an answer is model output. Neither is trusted markup."""
    run = run_with(answered([], text="<img src=x onerror=alert(1)>"), [chunk()])
    html = render_page(
        "<script>alert('q')</script>", run, [], CONFIG, tmp_path / "p.html", Meter()
    )
    assert "<script>alert('q')</script>" not in html
    assert "&lt;script&gt;" in html
    # The payload survives as *text* — escaped, so it is inert. What must not survive is the
    # tag: no `<img` that a parser would open.
    assert "<img" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


def test_the_passage_text_is_escaped_too(tmp_path):
    cite = a_cite(passage="he said <b>eight</b> & meant it")
    html = page_for(run_with(answered([]), [chunk()]), [cite], tmp_path / "p.html")
    assert "<b>eight</b>" not in html
    assert "&lt;b&gt;eight&lt;/b&gt; &amp; meant it" in html


def test_an_abstention_page_says_so_and_shows_no_player(tmp_path):
    run = run_with(Answer.abstention(), [chunk()])
    html = page_for(run, [], tmp_path / "p.html", question="What was Q3 revenue?")
    assert "abstain" in html
    assert "<video" not in html
    assert "QA_SPEC §4" in html


def test_a_schema_invalid_reply_is_shown_as_such_rather_than_as_an_answer(tmp_path):
    run = run_with(None, [chunk()], error="citations.0.t_end: does not run forward")
    html = page_for(run, [], tmp_path / "p.html")
    assert "did not validate" in html
    assert "citations.0.t_end" in html


def test_the_page_records_what_produced_it(tmp_path):
    """A demo page with no provenance is a screenshot: it cannot be re-run."""
    html = page_for(run_with(answered([]), [chunk()]), [], tmp_path / "p.html")
    for expected in (
        # The model the configured arm actually runs, not answer.model — those differ on
        # the local arm, and a footer naming the hosted id under a local run is provenance
        # that points at a number never measured on it.
        effective_model(CONFIG),
        str(CONFIG.get("embed.model")),
        str(CONFIG.get("answer.prompt")),
        "sha256:",
    ):
        assert expected in html, f"the footer never names {expected!r}"


def test_the_page_is_a_complete_html_document(tmp_path):
    html = page_for(run_with(answered([]), [chunk()]), [], tmp_path / "p.html")
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert '<meta charset="utf-8">' in html


# ---------------------------------------------------------------------------
# Odds and ends that are still user-visible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds, expected",
    [(0.0, "0:00"), (8.8, "0:08"), (65.0, "1:05"), (611.4, "10:11"), (3725.0, "1:02:05")],
)
def test_seconds_are_shown_as_a_timestamp(seconds, expected):
    """`2847.0` is not a timestamp to the person watching the demo."""
    assert clock(seconds) == expected


def test_the_passage_is_looked_up_from_the_chunk_the_citation_was_grounded_onto():
    hits = [chunk(t_start=20.0, text="the right one"), chunk(t_start=90.0, text="the wrong one")]
    assert passage_for("611", 20.0, hits) == "the right one"


def test_a_passage_lookup_that_misses_returns_empty_rather_than_the_wrong_text():
    """Grounding makes this unreachable; showing a neighbouring chunk's text would be worse
    than showing none, because it would look like the evidence for the answer."""
    assert passage_for("611", 500.0, [chunk(t_start=20.0)]) == ""


def test_the_page_filename_is_stable_for_the_same_question():
    """Re-asking overwrites its page instead of leaving a directory of near-identical files."""
    a = page_path("How old was Bernini?", Path("runs/ask"))
    b = page_path("  How   old was   Bernini?  ", Path("runs/ask"))
    assert a == b
    assert a.suffix == ".html"


def test_two_questions_that_slug_the_same_still_get_their_own_page():
    a = page_path("Is it a scalpel?")
    b = page_path("Is it a scalpel")
    assert a != b


def test_a_question_of_pure_punctuation_still_produces_a_usable_filename():
    name = page_path("?!?").name
    assert name.endswith("-question.html")
    assert "/" not in name and "\\" not in name


# ---------------------------------------------------------------------------
# ask() refuses rather than demoing nothing
# ---------------------------------------------------------------------------


def test_an_empty_index_is_refused_instead_of_answered(monkeypatch, tmp_path):
    """With nothing indexed every question abstains, so the demo would look like it works
    while proving nothing. The message names the command that fixes it."""
    import src.ask as mod

    monkeypatch.setattr(mod, "answer_question", lambda q, cfg, meter: run_with(None, []))
    with pytest.raises(AskError, match="make index-dev"):
        ask("anything", CONFIG, Meter(), out_dir=tmp_path)


def test_a_blank_question_is_refused_before_any_model_call(monkeypatch, tmp_path):
    import src.ask as mod

    def explode(*_args, **_kw):  # pragma: no cover - the point is that it is not called
        raise AssertionError("a blank question reached the model")

    monkeypatch.setattr(mod, "answer_question", explode)
    with pytest.raises(AskError, match="no question"):
        ask("   ", CONFIG, Meter(), out_dir=tmp_path)


def test_ask_writes_the_page_where_it_says_it_did(monkeypatch, tmp_path):
    import src.ask as mod

    run = run_with(answered([{"video_id": "611", "t_start": 20.0, "t_end": 45.0}]), [chunk()])
    monkeypatch.setattr(mod, "answer_question", lambda q, cfg, meter: run)

    _, cites, out = ask("How old was Bernini?", CONFIG, Meter(), out_dir=tmp_path)
    assert out is not None and out.is_file()
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert len(cites) == 1


def test_no_html_writes_nothing(monkeypatch, tmp_path):
    import src.ask as mod

    run = run_with(answered([{"video_id": "611", "t_start": 20.0, "t_end": 45.0}]), [chunk()])
    monkeypatch.setattr(mod, "answer_question", lambda q, cfg, meter: run)

    _, _, out = ask("q?", CONFIG, Meter(), out_dir=tmp_path, write=False)
    assert out is None
    assert list(tmp_path.iterdir()) == []
