"""GATE — `evals/dev ∩ evals/heldout = ∅` by content hash. VRAG-013.

This is the gate every other gate depends on. A recall or accuracy number measured after a
held-out label reached `evals/dev/` is not a number, it is a memorisation score, so this
runs first and nothing downstream counts until it is green.

It is in `tests/gates/` rather than `tests/` deliberately: `make test` excludes this
directory, so a red seal cannot be mistaken for an ordinary unit-test failure and worked
around. `make gate` runs it.

Three groups of tests, and the split is the point:

1. **The repo as committed.** `evals/dev/` and `evals/heldout/` are disjoint right now.
   Today `evals/dev/` is empty, so this passes vacuously — group 3 is what makes the
   check itself trustworthy in the meantime.
2. **The negative control** — the VRAG-013 acceptance criterion. Plant a held-out pair in a
   throwaway dev directory and the check has to fail. A leak detector never seen to fire
   is indistinguishable from `return 0`.
3. **Normalisation.** The evasions a copy-paste produces on its own — different quote
   characters, a dropped question mark, changed case, doubled whitespace — must not slip
   past the hash, and unrelated pairs must not trip it.

Offline: temporary directories only, no network and no media.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.leakage import (
    DEV,
    HELDOUT,
    KINDS,
    LeakageError,
    check,
    collisions,
    fingerprint,
    load_split,
    normalize,
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def write_jsonl(path: Path, pairs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8"
    )
    return path


def dev_case(**overrides) -> dict:
    """A dev pair that shares nothing with the sealed set. QA_SPEC §1 field set."""
    pair = {
        "id": "d001",
        "question": "Which instrument opens the second segment of the workshop demo?",
        "unanswerable": False,
        "video_id": "181",
        "t_ref": 12.0,
        "answer_note": "A cello, first heard at 11.4 s as the camera pans left.",
    }
    pair.update(overrides)
    return pair


@pytest.fixture(scope="module")
def sealed() -> list[dict]:
    """The real held-out pairs, as the thing a leak would be a copy of."""
    pairs = load_split(HELDOUT)
    if not pairs:
        pytest.fail(f"{HELDOUT} holds no pairs — the sealed set is a VRAG-012 deliverable")
    return [pair for _, pair in pairs]


@pytest.fixture
def split_dirs(tmp_path: Path) -> tuple[Path, Path]:
    dev, heldout = tmp_path / "dev", tmp_path / "heldout"
    dev.mkdir()
    heldout.mkdir()
    return dev, heldout


# --------------------------------------------------------------------------------------
# 1. The repo as committed — the gate itself
# --------------------------------------------------------------------------------------


def test_committed_splits_are_disjoint(capsys):
    """The gate. Prints the intersection size, then asserts it is zero."""
    code = check(DEV, HELDOUT)
    captured = capsys.readouterr()
    print(captured.out, end="")  # so `make gate -s` and a failure report both show it
    assert code == 0, f"leakage gate FAILED:\n{captured.out}\n{captured.err}"


def test_dev_pairs_use_the_dev_id_namespace():
    """QA_SPEC §8: dev ids are `d…`, held-out ids are `q…`.

    Without separate namespaces both splits would independently number from `q001` and the
    id fingerprint would fire on every dev case ever written — a check that always fails is
    as useless as one that never does.
    """
    offenders = [
        str(ref) for ref, pair in load_split(DEV)
        if not str(pair.get("id", "")).startswith("d")
    ]
    assert not offenders, f"dev pair ids must start with 'd' (QA_SPEC §8): {offenders}"


# --------------------------------------------------------------------------------------
# 2. The negative control — VRAG-013's acceptance criterion
# --------------------------------------------------------------------------------------


def test_planting_a_heldout_pair_in_dev_fails_the_check(split_dirs, sealed, capsys):
    """The acceptance criterion: copy a sealed pair into dev, the check must go red."""
    dev, heldout = split_dirs
    write_jsonl(heldout / "heldout_v1.jsonl", sealed)
    write_jsonl(dev / "dev_v1.jsonl", [dev_case(), sealed[0]])

    code = check(dev, heldout)
    captured = capsys.readouterr()

    assert code == 1, f"a planted held-out pair was not detected:\n{captured.out}"
    # Every fingerprint should fire on a whole-row copy, and the report has to name the
    # sealed pair so the reviewer can see exactly what leaked.
    for kind in KINDS:
        assert f"{kind}:" in captured.err, f"{kind} collision not reported:\n{captured.err}"
    assert sealed[0]["id"] in captured.err


@pytest.mark.parametrize("kind", KINDS)
def test_planting_one_field_alone_fails_the_check(split_dirs, sealed, capsys, kind):
    """Each fingerprint stands on its own.

    A leak rarely arrives as a clean copy: an id reused, a question pasted under a new id,
    or the answer_note lifted while the question is reworded. Any one of the three is
    enough, so each is planted alone here — with the other two fields left as the
    unrelated dev case's.
    """
    dev, heldout = split_dirs
    write_jsonl(heldout / "heldout_v1.jsonl", sealed)
    write_jsonl(dev / "dev_v1.jsonl", [dev_case(**{kind: sealed[0][kind]})])

    code = check(dev, heldout)
    captured = capsys.readouterr()

    assert code == 1, f"planting {kind} alone was not detected:\n{captured.out}"
    assert f"{kind}:" in captured.err


def test_the_planted_pair_is_named_not_just_counted(split_dirs, sealed):
    """A collision has to point at both sides, or nobody can act on it."""
    dev, heldout = split_dirs
    write_jsonl(heldout / "heldout_v1.jsonl", sealed)
    planted = write_jsonl(dev / "dev_v1.jsonl", [dev_case(), dev_case(**sealed[2])])

    found = collisions(load_split(dev), load_split(heldout))

    assert found, "no collision found for a planted pair"
    assert all(c.dev.path == planted for c in found)
    assert {c.heldout.pair_id for c in found} == {sealed[2]["id"]}
    # Line 2 of the dev file, not line 1: the clean case must not be blamed.
    assert {c.dev.lineno for c in found} == {2}


def test_clean_dev_set_passes(split_dirs, sealed, capsys):
    """The other half of the control: a real dev set against the real sealed set is green.

    A check that fires on everything would pass the test above and still be worthless.
    """
    dev, heldout = split_dirs
    write_jsonl(heldout / "heldout_v1.jsonl", sealed)
    write_jsonl(dev / "dev_v1.jsonl", [dev_case(id=f"d{i:03d}", question=f"What happens at "
                                                 f"the {i} minute mark of the demo reel?")
                                       for i in range(1, 16)])

    code = check(dev, heldout)
    captured = capsys.readouterr()

    assert code == 0, f"a clean dev set was flagged:\n{captured.out}\n{captured.err}"
    assert "overlap  0" in captured.out


def test_shared_video_id_is_not_leakage(split_dirs, sealed, capsys):
    """The video split is public; only the labels are sealed.

    QA_SPEC §6 asks for an answerable question per corpus video, so held-out pairs point at
    dev videos by design. Flagging that would make the gate unpassable.
    """
    dev, heldout = split_dirs
    answerable = next(p for p in sealed if p["video_id"] is not None)
    write_jsonl(heldout / "heldout_v1.jsonl", sealed)
    write_jsonl(dev / "dev_v1.jsonl", [
        dev_case(video_id=answerable["video_id"], t_ref=answerable["t_ref"])
    ])

    assert check(dev, heldout) == 0, capsys.readouterr().err


def test_empty_heldout_does_not_pass(split_dirs, capsys):
    """0 ∩ 0 = ∅ is not a seal. An empty sealed set means the check is measuring nothing."""
    dev, heldout = split_dirs
    write_jsonl(dev / "dev_v1.jsonl", [dev_case()])

    assert check(dev, heldout) == 1
    assert "nothing to be blind to" in capsys.readouterr().err


def test_empty_dev_passes_but_says_it_is_vacuous(split_dirs, sealed, capsys):
    """Today's state of the repo. It passes — and reports why that is cheap."""
    dev, heldout = split_dirs
    write_jsonl(heldout / "heldout_v1.jsonl", sealed)

    assert check(dev, heldout) == 0
    assert "vacuous" in capsys.readouterr().out


def test_malformed_dev_file_is_not_skipped(split_dirs, sealed):
    """A file that cannot be parsed is a failure, not zero pairs.

    Swallowing it would report a clean intersection over cases nobody compared — the one
    way this gate could be green and wrong at the same time.
    """
    dev, heldout = split_dirs
    write_jsonl(heldout / "heldout_v1.jsonl", sealed)
    (dev / "dev_v1.jsonl").write_text('{"id": "d001", truncated\n', encoding="utf-8")

    with pytest.raises(LeakageError, match="not valid JSON"):
        check(dev, heldout)


# --------------------------------------------------------------------------------------
# 3. Normalisation — the evasions a copy-paste produces by itself
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mangle",
    [
        pytest.param(lambda q: q, id="verbatim"),
        pytest.param(str.upper, id="shouted"),
        pytest.param(str.lower, id="whispered"),
        pytest.param(lambda q: f"  {q}  ", id="padded"),
        pytest.param(lambda q: q.replace(" ", "  "), id="double-spaced"),
        pytest.param(lambda q: q.replace(" ", "\t"), id="tabbed"),
        pytest.param(lambda q: f"{q}\n", id="trailing-newline"),
    ],
)
def test_reformatted_question_still_collides(split_dirs, sealed, mangle):
    """Case, padding and stray whitespace are formatting, not new content."""
    dev, heldout = split_dirs
    question = mangle(sealed[0]["question"])
    write_jsonl(heldout / "heldout_v1.jsonl", sealed)
    write_jsonl(dev / "dev_v1.jsonl", [dev_case(question=question)])

    found = collisions(load_split(dev), load_split(heldout))
    assert [c.kind for c in found] == ["question"], f"{question!r} slipped past the hash"


def test_dropped_question_mark_still_collides(split_dirs, sealed):
    dev, heldout = split_dirs
    write_jsonl(heldout / "heldout_v1.jsonl", sealed)
    write_jsonl(dev / "dev_v1.jsonl", [dev_case(question=sealed[0]["question"].rstrip("?"))])

    assert [c.kind for c in collisions(load_split(dev), load_split(heldout))] == ["question"]


def test_smart_quotes_and_dashes_normalise():
    """The pasted-through-a-doc case: typographic characters are not a rewrite."""
    curly = "the performer’s cue — stage left"
    straight = "the performer's cue - stage left"
    assert normalize(curly) == normalize(straight)
    assert fingerprint(curly) == fingerprint(straight)


def test_paraphrase_is_not_caught_and_that_is_documented(split_dirs, sealed):
    """The honest limit of a content hash, pinned so nobody over-claims it.

    A rewritten question hashes differently. VRAG-013 asserts disjointness by hash; it does
    not and cannot assert semantic novelty. Review and the blind-labelling rule cover this.
    """
    dev, heldout = split_dirs
    write_jsonl(heldout / "heldout_v1.jsonl", sealed)
    write_jsonl(dev / "dev_v1.jsonl", [dev_case(question="Where was that performance filmed?")])

    assert not collisions(load_split(dev), load_split(heldout))


def test_empty_and_missing_fields_do_not_collide():
    """Two pairs both missing `answer_note` are not evidence of a leak."""
    assert fingerprint(None) is None
    assert fingerprint("") is None
    assert fingerprint("   ") is None
    assert fingerprint("?") is None


def test_unrelated_questions_do_not_collide(sealed):
    """Sanity: the sealed set's own questions all fingerprint differently."""
    digests = [fingerprint(p["question"]) for p in sealed]
    assert len(set(digests)) == len(sealed)
