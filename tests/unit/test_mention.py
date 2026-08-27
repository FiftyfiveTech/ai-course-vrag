"""src/mention.py — `@source`, the tag that scopes a question to one video.

No Ollama, no Chroma, no network: the catalogue's two inputs (the manifest, and the list of
video_ids the index holds) are both injected, which is the property that makes this file
possible at all and is why `catalogue()` takes them rather than reading them.

The failures worth catching here are the quiet ones:

* **a tag that silently does nothing.** `@nosuchthing` resolving to "search everything"
  would produce an answer that looks scoped, cites whatever it liked, and says nothing about
  the scope having been dropped. Every unresolvable tag raises instead.
* **a tag that reaches the wrong layer.** The token must leave the question text before it
  is embedded, and must arrive at the store as a filter. Half of that — leaving the text but
  never becoming a filter — is a working demo of nothing: the answer is just unscoped.
* **an email address read as a tag.** `ritika@fiftyfivetech.io` in a question is not a
  mention of a source called `fiftyfivetech`, and refusing the question because of it would
  be a refusal nobody could act on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.mention import (
    MentionError,
    Scope,
    Source,
    catalogue,
    resolve,
    scope,
    strip,
    tokens,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RECORDS = {
    "181": {
        "video_id": "181",
        "youtube_id": "8np5YKYx3sU",
        "split": "dev",
        "domain": "Artistic Performance",
        "sub_category": "Stage Play",
    },
    "611": {
        "video_id": "611",
        "youtube_id": "H8fGd3fCJbg",
        "split": "dev",
        "domain": "Knowledge",
        "sub_category": "Literature & Art",
    },
    "091": {
        "video_id": "091",
        "youtube_id": "_aVHf_jmWk8",
        "split": "heldout",
        "domain": "Film & Television",
        "sub_category": "Animation",
    },
}


@pytest.fixture()
def manifest(tmp_path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"videos": list(RECORDS.values())}), encoding="utf-8"
    )
    return path


@pytest.fixture()
def samples(tmp_path) -> Path:
    """A samples/ with one video fetched, the way `make sample-real` leaves it."""
    directory = tmp_path / "samples"
    directory.mkdir()
    (directory / "611_H8fGd3fCJbg.mp4").write_bytes(b"\0")
    return directory


@pytest.fixture()
def sources(manifest, samples) -> list[Source]:
    """The corpus above with 181 and 611 indexed, plus a non-corpus id someone indexed."""
    return catalogue(
        indexed=["181", "611", "bob-video"], manifest=manifest, samples=samples
    )


# ---------------------------------------------------------------------------
# Parsing the token out of the text
# ---------------------------------------------------------------------------


def test_a_tag_is_found():
    assert tokens("@611 what two tools do I need?") == ["611"]


def test_a_tag_is_found_mid_sentence():
    assert tokens("what does @611 say about it") == ["611"]


def test_several_tags_are_kept_in_the_order_they_were_typed():
    assert tokens("@701 and @181 both") == ["701", "181"]


def test_the_same_tag_twice_is_one_scope():
    assert tokens("@611 and again @611") == ["611"]


def test_the_same_tag_in_two_cases_is_one_scope():
    assert tokens("@Bob-Video vs @bob-video") == ["Bob-Video"]


@pytest.mark.parametrize(
    "question",
    [
        "mail me at ritika@fiftyfivetech.io",
        "the handle is user@example.com please",
        "prices@2x are wrong",
    ],
)
def test_an_email_address_is_not_a_tag(question):
    # A mention starts the string or follows whitespace. Without that, a question that
    # happens to contain an address is refused for naming a source nobody meant.
    assert tokens(question) == []


def test_a_bare_at_sign_is_not_a_tag():
    assert tokens("meet @ 5pm") == []
    assert tokens("what does @ mean") == []


def test_punctuation_after_a_tag_is_not_part_of_it():
    assert tokens("@611, what happens?") == ["611"]
    assert tokens("(@611) what happens?") == ["611"]
    assert tokens("@611-") == ["611"]


def test_a_filename_stem_is_a_single_tag():
    assert tokens("@611_H8fGd3fCJbg what happens?") == ["611_H8fGd3fCJbg"]


def test_stripping_takes_the_tags_out_and_closes_the_gap():
    assert strip("@611 what two tools?") == "what two tools?"
    assert strip("what does @611 say") == "what does say"
    assert strip("@701 @181 what happens") == "what happens"


def test_stripping_leaves_an_untagged_question_alone_but_normalises_it():
    assert strip("  what   two tools?  ") == "what two tools?"


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------


def test_the_catalogue_is_the_manifest_and_the_index_together(sources):
    # Neither alone is the set of things a person can ask about: the manifest names 091,
    # which was never fetched, and the index holds bob-video, which the manifest never named.
    assert [s.video_id for s in sources] == ["091", "181", "611", "bob-video"]


def test_a_non_decimal_id_sorts_after_the_corpus_ids(sources):
    assert sources[-1].video_id == "bob-video"


def test_only_what_the_index_holds_is_marked_indexed(sources):
    assert {s.video_id: s.indexed for s in sources} == {
        "091": False,
        "181": True,
        "611": True,
        "bob-video": True,
    }


def test_a_source_is_labelled_from_the_manifest_taxonomy(sources):
    assert _by_id(sources, "611").label == "Knowledge / Literature & Art"


def test_a_source_the_manifest_never_named_has_no_label_and_no_split(sources):
    stray = _by_id(sources, "bob-video")
    assert stray.label == ""
    assert stray.split is None


def test_a_source_carries_its_youtube_id_and_its_fetched_filename_as_aliases(sources):
    assert _by_id(sources, "611").aliases == ("H8fGd3fCJbg", "611_H8fGd3fCJbg")


def test_an_unfetched_source_carries_only_its_youtube_id(sources):
    assert _by_id(sources, "181").aliases == ("8np5YKYx3sU",)


def test_an_alias_that_cannot_be_typed_as_a_tag_is_not_advertised(tmp_path):
    # Dev video 701's youtube id is `-dfvdKf-KR0`, and `@-dfvdKf-KR0` does not parse: a
    # handle begins with a letter or a digit. Offering it in `make sources` or in the
    # frontend's picker would be offering a name that resolves to nothing when typed back.
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {"videos": [{"video_id": "701", "youtube_id": "-dfvdKf-KR0", "split": "dev"}]}
        ),
        encoding="utf-8",
    )
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "701_-dfvdKf-KR0.mp4").write_bytes(b"x")

    got = catalogue(indexed=["701"], manifest=manifest, samples=samples)[0]
    # The filename stem starts with a digit and survives; the bare youtube id does not.
    assert got.aliases == ("701_-dfvdKf-KR0",)


def test_every_advertised_alias_actually_resolves(sources):
    # The property the filter above exists to hold, asserted over the whole catalogue.
    for source in sources:
        for alias in source.aliases:
            assert tokens("@" + alias) == [alias]
            assert resolve(f"@{alias} what?", sources).video_ids == (source.video_id,)


def test_a_missing_manifest_leaves_the_indexed_ids_taggable(tmp_path):
    # A clone that has an index but never ran `make corpus`. Refusing every tag there would
    # be refusing to scope to videos that are demonstrably in the store.
    got = catalogue(
        indexed=["611"], manifest=tmp_path / "nope.json", samples=tmp_path / "nope"
    )
    assert [s.video_id for s in got] == ["611"]
    assert got[0].indexed is True


def test_the_catalogue_prefers_records_it_is_handed_over_the_file(manifest, samples):
    # src.api has the manifest loaded already and hands it over. If this re-read the file,
    # the endpoint would list one corpus and resolve `@` tags against another.
    got = catalogue(
        indexed=["999"],
        records={"999": {"video_id": "999", "split": "dev", "domain": "Sports"}},
        manifest=manifest,
        samples=samples,
    )
    assert [s.video_id for s in got] == ["999"]


# ---------------------------------------------------------------------------
# Resolving
# ---------------------------------------------------------------------------


def test_a_tag_resolves_to_a_video_id(sources):
    got = resolve("@611 what two tools?", sources)
    assert got.video_ids == ("611",)
    assert got.text == "what two tools?"
    assert got.raw == "@611 what two tools?"
    assert got.scoped is True


def test_a_youtube_id_resolves_to_the_same_video(sources):
    assert resolve("@H8fGd3fCJbg what?", sources).video_ids == ("611",)


def test_a_fetched_filename_resolves_to_the_same_video(sources):
    # The thing being tagged is, to the person typing, a file — so the filename has to work.
    assert resolve("@611_H8fGd3fCJbg what?", sources).video_ids == ("611",)


def test_a_tag_is_case_insensitive(sources):
    assert resolve("@BOB-VIDEO what?", sources).video_ids == ("bob-video",)


def test_two_tags_scope_to_both(sources):
    assert resolve("@611 @181 what?", sources).video_ids == ("611", "181")


def test_two_names_for_one_video_scope_to_it_once(sources):
    assert resolve("@611 @H8fGd3fCJbg what?", sources).video_ids == ("611",)


def test_an_untagged_question_is_unscoped(sources):
    got = resolve("what two tools?", sources)
    assert got.video_ids == ()
    assert got.scoped is False
    assert got.text == "what two tools?"


def test_a_tag_that_names_nothing_is_refused_and_not_ignored(sources):
    # The whole point. Widening back to the whole index would answer a different question
    # and say nothing about having done so.
    with pytest.raises(MentionError) as exc:
        resolve("@bernini what?", sources)
    assert "@bernini" in str(exc.value)


def test_the_refusal_lists_the_handles_that_do_work(sources):
    with pytest.raises(MentionError) as exc:
        resolve("@bernini what?", sources)
    message = str(exc.value)
    assert "@181" in message and "@611" in message and "@bob-video" in message


def test_the_refusal_does_not_offer_a_source_that_cannot_be_asked(sources):
    with pytest.raises(MentionError) as exc:
        resolve("@bernini what?", sources)
    assert "@091" not in str(exc.value)


def test_a_truncated_handle_gets_a_suggestion(sources):
    with pytest.raises(MentionError) as exc:
        resolve("@61 what?", sources)
    assert "Did you mean @611?" in str(exc.value)


def test_an_ambiguous_near_miss_gets_no_suggestion(manifest, samples):
    # Two candidates contain "1", and offering one of them is worse than offering none:
    # the user would take it, and the answer would be scoped to a video they did not name.
    sources = catalogue(indexed=["181", "611"], manifest=manifest, samples=samples)
    with pytest.raises(MentionError) as exc:
        resolve("@1 what?", sources)
    assert "Did you mean" not in str(exc.value)


def test_a_video_the_index_does_not_hold_is_refused_with_the_command_to_fix_it(
    manifest, samples
):
    sources = catalogue(indexed=["611"], manifest=manifest, samples=samples)
    with pytest.raises(MentionError) as exc:
        resolve("@181 what?", sources)
    message = str(exc.value)
    assert "make sample-real VIDEO_ID=181" in message
    assert "make index-dev" in message


def test_a_held_out_video_is_refused_by_name(manifest, samples):
    # It is not an unindexed dev video with a missing fetch — it is sealed, and telling
    # someone to run `make sample-real` on it would be telling them to break the seal.
    sources = catalogue(indexed=["611"], manifest=manifest, samples=samples)
    with pytest.raises(MentionError) as exc:
        resolve("@091 what?", sources)
    assert "held-out" in str(exc.value)
    assert "make sample-real" not in str(exc.value)


def test_a_tag_with_no_question_after_it_is_refused(sources):
    with pytest.raises(MentionError) as exc:
        resolve("@611", sources)
    assert "asks nothing" in str(exc.value)


def test_nothing_indexed_says_so_rather_than_listing_an_empty_menu(manifest, samples):
    sources = catalogue(indexed=[], manifest=manifest, samples=samples)
    with pytest.raises(MentionError) as exc:
        resolve("@611 what?", sources)
    assert "make index-dev" in str(exc.value)


# ---------------------------------------------------------------------------
# scope() — the entry point, and its short circuit
# ---------------------------------------------------------------------------


def test_an_untagged_question_never_touches_the_index(monkeypatch):
    # The common case must not pay for a manifest read, a samples/ glob and a full scan of
    # the collection's metadata to discover there was nothing to resolve.
    def boom(cfg, samples=None):
        raise AssertionError("built the catalogue for a question with no tag in it")

    monkeypatch.setattr("src.mention.from_config", boom)
    assert scope("what two tools?", cfg=None).video_ids == ()


def test_a_question_whose_at_sign_is_not_a_tag_never_touches_the_index(monkeypatch):
    monkeypatch.setattr(
        "src.mention.from_config",
        lambda cfg, samples=None: (_ for _ in ()).throw(AssertionError("no tag here")),
    )
    assert scope("mail me at ritika@fiftyfivetech.io", cfg=None).video_ids == ()


def test_scope_resolves_against_the_sources_it_is_given(sources):
    assert scope("@611 what?", cfg=None, sources=sources).video_ids == ("611",)


# ---------------------------------------------------------------------------
# Describing a scope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "video_ids, expected",
    [
        ((), "the whole index"),
        (("611",), "video 611"),
        (("611", "181"), "videos 611, 181"),
    ],
)
def test_a_scope_describes_itself(video_ids, expected):
    assert Scope(raw="q", text="q", video_ids=video_ids).describe() == expected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _by_id(sources: list[Source], video_id: str) -> Source:
    return next(s for s in sources if s.video_id == video_id)
