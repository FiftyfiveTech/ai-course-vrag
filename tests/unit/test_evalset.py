"""Tests for the sealed Q&A set — VRAG-012.

Two halves, and the split matters.

The first half reads the committed `evals/heldout/heldout_v1.jsonl` and asserts it satisfies
`evals/QA_SPEC.md`. That is the seal itself, and it is the thing `make heldout-check` prints
a number for.

The second half feeds `validate()` deliberately broken pairs. A checker that has never been
seen to fail is not evidence of anything — if the counts rule silently stopped firing, the
first half would still be green on a set that no longer obeys the spec.

Offline: no network, no media. The part that cannot be tested here is whether each `t_ref` is
really the moment the answer appears — that came from watching the videos while labelling,
and QA_SPEC §6 puts that burden on the labeler.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.evalset import (
    ANSWERABLE,
    FIELDS,
    TOTAL,
    UNANSWERABLE,
    YES_NO_OPENERS,
    digest,
    load_pairs,
    load_videos,
    readme_digest,
    validate,
)

HELDOUT = Path("evals/heldout/heldout_v1.jsonl")
MANIFEST = Path("data/corpus/manifest.json")
README = Path("README.md")


@pytest.fixture(scope="module")
def pairs() -> list[dict]:
    if not HELDOUT.exists():
        pytest.fail(f"{HELDOUT} is missing — the held-out set is a VRAG-012 deliverable")
    return load_pairs(HELDOUT)


@pytest.fixture(scope="module")
def videos() -> dict[str, dict]:
    return load_videos(MANIFEST)


# --------------------------------------------------------------------------------------
# The committed set
# --------------------------------------------------------------------------------------


def test_committed_set_satisfies_the_spec(pairs, videos):
    assert validate(pairs, videos) == []


def test_counts_are_the_split_the_spec_fixes(pairs):
    assert len(pairs) == TOTAL
    assert sum(1 for p in pairs if not p["unanswerable"]) == ANSWERABLE
    assert sum(1 for p in pairs if p["unanswerable"]) == UNANSWERABLE


def test_every_corpus_video_has_an_answerable_question(pairs, videos):
    covered = {p["video_id"] for p in pairs if not p["unanswerable"]}
    assert covered == set(videos)


def test_readme_records_the_digest_of_the_committed_file():
    assert readme_digest(README) == digest(HELDOUT)


def test_answer_notes_say_where_the_answer_is(pairs):
    """QA_SPEC §1: the note is the labeler's record, so it has to carry a real reference."""
    for pair in pairs:
        if pair["unanswerable"]:
            continue
        assert " s" in pair["answer_note"], f"{pair['id']} note cites no timestamp"


def test_one_json_object_per_line():
    """The gate and the leakage check both read this line by line, not as one document."""
    lines = [ln for ln in HELDOUT.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == TOTAL
    for line in lines:
        assert set(json.loads(line)) == FIELDS


# --------------------------------------------------------------------------------------
# The checker itself
# --------------------------------------------------------------------------------------


def _mutate(pairs: list[dict], index: int, **changes) -> list[dict]:
    out = copy.deepcopy(pairs)
    out[index].update(changes)
    return out


def test_wrong_total_is_caught(pairs, videos):
    problems = validate(pairs[:-1], videos)
    assert any("expected 20" in p for p in problems)


def test_non_sequential_id_is_caught(pairs, videos):
    problems = validate(_mutate(pairs, 3, id="q999"), videos)
    assert any("sequential" in p for p in problems)


def test_unknown_video_id_is_caught(pairs, videos):
    problems = validate(_mutate(pairs, 0, video_id="404"), videos)
    assert any("not in" in p for p in problems)


def test_t_ref_past_the_bucket_ceiling_is_caught(pairs, videos):
    """Video 181 is a `short`, so a t_ref of 820 s is a mis-keyed 82.0 rather than a fact."""
    problems = validate(_mutate(pairs, 0, t_ref=820.0), videos)
    assert any("ceiling" in p for p in problems)


def test_unanswerable_pair_carrying_a_video_is_caught(pairs, videos):
    index = next(i for i, p in enumerate(pairs) if p["unanswerable"])
    problems = validate(_mutate(pairs, index, video_id="181"), videos)
    assert any("must be null" in p for p in problems)


def test_answerable_pair_missing_its_timestamp_is_caught(pairs, videos):
    problems = validate(_mutate(pairs, 0, t_ref=None), videos)
    assert any("number of seconds" in p for p in problems)


def test_yes_no_question_is_caught(pairs, videos):
    problems = validate(_mutate(pairs, 0, question="Is the stage lit in orange?"), videos)
    assert any("yes/no" in p for p in problems)


def test_committed_questions_open_with_none_of_the_yes_no_words(pairs):
    for pair in pairs:
        assert pair["question"].split()[0].lower() not in YES_NO_OPENERS


def test_extra_field_is_caught(pairs, videos):
    problems = validate(_mutate(pairs, 0, note_to_self="fix later"), videos)
    assert any("unexpected field" in p for p in problems)


def test_dropping_a_video_from_coverage_is_caught(pairs, videos):
    """Re-pointing 181's only question leaves that video with nothing measuring it."""
    problems = validate(_mutate(pairs, 0, video_id="611"), videos)
    assert any("coverage" in p for p in problems)


def test_empty_answer_note_is_caught(pairs, videos):
    problems = validate(_mutate(pairs, 0, answer_note="  "), videos)
    assert any("answer_note is empty" in p for p in problems)


def test_readme_digest_returns_none_when_nothing_is_recorded(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no digest here\n", encoding="utf-8")
    assert readme_digest(readme) is None


def test_readme_digest_reads_the_line_naming_the_file(tmp_path):
    sha = "a" * 64
    readme = tmp_path / "README.md"
    readme.write_text(f"| sha256 of `heldout_v1.jsonl` | `{sha}` |\n", encoding="utf-8")
    assert readme_digest(readme) == sha
