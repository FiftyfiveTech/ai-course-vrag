"""The sealed Q&A set — VRAG-012.

`evals/heldout/heldout_v1.jsonl` is the 20 pairs the MVP gate (VRAG-021) is scored on. The
rules it has to satisfy are not invented here: they are `evals/QA_SPEC.md`, written in
VRAG-011 so a Builder can predict every label without asking. This module is that spec
turned into a check that either passes or names the pair it failed on.

Two things make a "sealed" set actually sealed:

* **the bytes are pinned.** `--check` recomputes the file's sha256 and compares it to the
  one written into README.md. Editing a question after the tag is pushed changes the
  digest and the check fails, so a quiet re-label cannot pass unnoticed.
* **the number is computed, not claimed.** The check prints the counts, the per-video
  spread and the digest. That printed line is the evidence for the card.

What this module cannot check is the part that matters most: whether `t_ref` is really the
moment the answer appears. Only watching the video establishes that, which is why
QA_SPEC.md §6 puts the burden on the labeler and why the bucket ceiling below is described
as a typo catcher rather than a verification.

    make heldout-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

HELDOUT = Path("evals/heldout/heldout_v1.jsonl")
MANIFEST = Path("data/corpus/manifest.json")
README = Path("README.md")

# QA_SPEC.md §6: "Split: 17 answerable, 3 unanswerable." The gate formula in §5 divides by
# 20, so this is part of the contract rather than a preference — a 16/4 file would score on
# a denominator the spec does not define.
ANSWERABLE = 17
UNANSWERABLE = 3
TOTAL = ANSWERABLE + UNANSWERABLE

# QA_SPEC.md §1 fixes the field set. Exact, not "at least": an extra key is how a stray note
# or a half-finished edit lands in a sealed file without anyone noticing.
FIELDS = {"id", "question", "unanswerable", "video_id", "t_ref", "answer_note"}

# Video-MME's duration buckets (arXiv:2405.21075 §3 — short 0-2 min, medium 4-15 min, long
# 30-60 min). The manifest carries the bucket, not the runtime, so this is the only upper
# bound on t_ref available without the media. It catches a mis-keyed timestamp (t_ref=820
# on a 96-second clip). It is not verification; see the module docstring.
BUCKET_MAX_S = {"short": 120.0, "medium": 900.0, "long": 3600.0}

# QA_SPEC.md §6: "No yes/no questions." A question opening with one of these takes a yes/no
# answer, which is exactly the shape where a lucky guess is indistinguishable from
# retrieval that worked.
YES_NO_OPENERS = (
    "is", "are", "was", "were", "does", "do", "did", "can", "could",
    "will", "would", "has", "have", "had", "should", "am",
)


class EvalSetError(Exception):
    """The sealed set is unreadable, or the corpus manifest it refers to is."""


def load_pairs(path: Path = HELDOUT) -> list[dict[str, Any]]:
    """Parse the jsonl. Blank lines are skipped; every other line must be one object."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalSetError(f"cannot read {path}: {exc}") from exc
    pairs: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalSetError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise EvalSetError(f"{path}:{lineno} is {type(obj).__name__}, not an object")
        pairs.append(obj)
    return pairs


def load_videos(path: Path = MANIFEST) -> dict[str, dict[str, Any]]:
    try:
        videos = json.loads(path.read_text(encoding="utf-8"))["videos"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise EvalSetError(
            f"cannot read the corpus manifest {path} ({exc}) — run `make corpus`"
        ) from exc
    return {str(v["video_id"]): v for v in videos}


def validate(pairs: list[dict[str, Any]], videos: dict[str, dict[str, Any]]) -> list[str]:
    """Every way the set breaks the spec, not only the first. Empty list means valid."""
    problems: list[str] = []

    def fail(where: str, why: str) -> None:
        problems.append(f"{where}: {why}")

    if len(pairs) != TOTAL:
        fail("counts", f"{len(pairs)} pairs, expected {TOTAL} (QA_SPEC §6)")

    seen: set[str] = set()
    for i, pair in enumerate(pairs, start=1):
        where = f"pair {i} ({pair.get('id', 'no id')})"

        extra = set(pair) - FIELDS
        missing = FIELDS - set(pair)
        if extra:
            fail(where, f"unexpected field(s) {sorted(extra)} (QA_SPEC §1)")
        if missing:
            fail(where, f"missing field(s) {sorted(missing)} (QA_SPEC §1)")
            continue

        pair_id = pair["id"]
        expected_id = f"q{i:03d}"
        if pair_id != expected_id:
            fail(where, f"id is {pair_id!r}, expected {expected_id!r} — ids are sequential (QA_SPEC §1)")
        if pair_id in seen:
            fail(where, f"duplicate id {pair_id!r}")
        seen.add(pair_id)

        question = pair["question"]
        if not isinstance(question, str) or not question.strip():
            fail(where, "question is empty")
        elif not question.rstrip().endswith("?"):
            fail(where, "question does not end in '?'")
        elif question.split()[0].lower().strip("“\"'") in YES_NO_OPENERS:
            fail(where, f"opens with {question.split()[0]!r} — reads as yes/no (QA_SPEC §6)")

        note = pair["answer_note"]
        if not isinstance(note, str) or not note.strip():
            fail(where, "answer_note is empty — it is the labeler's record of what the answer is (QA_SPEC §1)")

        unanswerable = pair["unanswerable"]
        if not isinstance(unanswerable, bool):
            fail(where, f"unanswerable is {type(unanswerable).__name__}, must be a bool")
            continue

        if unanswerable:
            # QA_SPEC §3: an unanswerable pair carries no video and no timestamp. A
            # leftover video_id would make §4's abstention check ambiguous.
            if pair["video_id"] is not None:
                fail(where, f"unanswerable but video_id is {pair['video_id']!r}, must be null (QA_SPEC §3)")
            if pair["t_ref"] is not None:
                fail(where, f"unanswerable but t_ref is {pair['t_ref']!r}, must be null (QA_SPEC §3)")
            continue

        video_id = pair["video_id"]
        known = isinstance(video_id, str) and video_id in videos
        if not isinstance(video_id, str):
            fail(where, f"video_id is {type(video_id).__name__}, must be the manifest's string id")
        elif not known:
            fail(where, f"video_id {video_id!r} is not in {MANIFEST}")

        t_ref = pair["t_ref"]
        if not isinstance(t_ref, (int, float)) or isinstance(t_ref, bool):
            fail(where, f"t_ref is {t_ref!r}, must be a number of seconds (QA_SPEC §1)")
        elif t_ref < 0:
            fail(where, f"t_ref is {t_ref}, must be >= 0")
        elif known:
            bucket = videos[video_id]["duration"]
            cap = BUCKET_MAX_S[bucket]
            if t_ref > cap:
                fail(where, f"t_ref {t_ref}s is past {cap:.0f}s, the ceiling of video "
                            f"{video_id}'s {bucket} bucket")

    answerable = [p for p in pairs if p.get("unanswerable") is False]
    unanswerable = [p for p in pairs if p.get("unanswerable") is True]
    if len(answerable) != ANSWERABLE or len(unanswerable) != UNANSWERABLE:
        fail("split", f"{len(answerable)} answerable / {len(unanswerable)} unanswerable, "
                      f"expected {ANSWERABLE} / {UNANSWERABLE} (QA_SPEC §6)")

    # QA_SPEC §6: "at least one answerable question per dev video and at least one per
    # heldout video." A set that piles onto one video measures that video, not the pipeline.
    covered = {p["video_id"] for p in answerable if isinstance(p.get("video_id"), str)}
    uncovered = sorted(set(videos) - covered)
    if uncovered:
        fail("coverage", f"no answerable question for video(s) {uncovered} (QA_SPEC §6)")

    return problems


def digest(path: Path = HELDOUT) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise EvalSetError(f"cannot read {path}: {exc}") from exc


def readme_digest(path: Path = README, name: str = HELDOUT.name) -> str | None:
    """The sha256 README records for the sealed file, or None if it records none.

    The seal only means something if something outside the file asserts what the file
    should be. README is that something: it is reviewed in the PR, and it is what the
    acceptance criterion for VRAG-012 points at.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalSetError(f"cannot read {path}: {exc}") from exc
    for line in text.splitlines():
        if name in line:
            found = re.search(r"\b([0-9a-f]{64})\b", line)
            if found:
                return found.group(1)
    return None


def summary_line(pairs: list[dict[str, Any]], videos: dict[str, dict[str, Any]], sha: str) -> str:
    answerable = sum(1 for p in pairs if p.get("unanswerable") is False)
    unanswerable = sum(1 for p in pairs if p.get("unanswerable") is True)
    covered = len({p["video_id"] for p in pairs if isinstance(p.get("video_id"), str)})
    return (f"{len(pairs)} pairs — {answerable} answerable / {unanswerable} unanswerable · "
            f"{covered}/{len(videos)} videos covered · sha256 {sha}")


def check(path: Path = HELDOUT, manifest: Path = MANIFEST, readme: Path = README) -> int:
    pairs = load_pairs(path)
    videos = load_videos(manifest)
    problems = validate(pairs, videos)
    sha = digest(path)

    print(f"heldout set — {path.as_posix()}")
    print(f"  {summary_line(pairs, videos, sha)}")

    per_video = dict.fromkeys(videos, 0)
    for pair in pairs:
        video_id = pair.get("video_id")
        if isinstance(video_id, str) and video_id in per_video:
            per_video[video_id] += 1
    print("  per video  " + "  ".join(f"{v}:{n}" for v, n in per_video.items()))

    recorded = readme_digest(readme)
    if recorded is None:
        problems.append(f"seal: {readme} records no sha256 for {path.name} — the digest above "
                        f"has nothing to be checked against")
        print(f"  README     records no sha256 for {path.name}")
    elif recorded != sha:
        problems.append(f"seal: {readme} records sha256 {recorded}, the file hashes to {sha} — "
                        f"the sealed set was edited after it was recorded")
        print(f"  README     MISMATCH — records {recorded}")
    else:
        print("  README     matches")

    if problems:
        print(f"\nFAIL — {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("\nPASS — the sealed set satisfies evals/QA_SPEC.md and matches the sha256 in README.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate and seal the held-out Q&A set (VRAG-012).")
    ap.add_argument("--path", default=str(HELDOUT), help="the sealed jsonl")
    ap.add_argument("--manifest", default=str(MANIFEST), help="the corpus manifest")
    ap.add_argument("--readme", default=str(README), help="the file recording the sha256")
    ap.add_argument("--sha", action="store_true", help="print only the sha256 and exit")
    args = ap.parse_args(argv)

    try:
        if args.sha:
            print(digest(Path(args.path)))
            return 0
        return check(Path(args.path), Path(args.manifest), Path(args.readme))
    except EvalSetError as exc:
        print(f"FAIL — {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
