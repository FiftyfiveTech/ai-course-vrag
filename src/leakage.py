"""The blind-labelling seal — VRAG-013.

`evals/heldout/` is the Evaluator's. The Builder tunes on `evals/dev/` and never reads the
held-out labels, because a threshold tuned against the set it is later scored on measures
nothing. That rule is a promise until something checks it; this module is the check.

It computes `evals/dev ∩ evals/heldout` by content hash and names every pair on both sides
of any collision.

What "by content hash" means here — three fingerprints per pair, each a sha256 over one
normalised field:

* **id** — QA_SPEC §1 says ids are unique and never reused, so an id in both splits means a
  row was copied across. Held-out ids are `q001`…`q020`; dev ids are `d001`… (QA_SPEC §8),
  so the two namespaces cannot collide by accident, only by copying.
* **question** — the leak that actually costs us. A dev case asking a held-out question
  turns tuning into memorisation.
* **answer_note** — the note records what the answer is and where it appears. Copied into
  dev it hands over the answer even under a reworded question.

Normalisation (NFKC, casefolded, whitespace collapsed, smart quotes and dashes flattened,
trailing punctuation dropped) is there so a copy-paste that picked up different quote
characters or lost its question mark is still recognised as the same content.

An exact copied row trips all three fingerprints, which is why there is no fourth
whole-record hash: it would only restate what the three already said.

**What a hash cannot catch:** a held-out question rewritten in the Builder's own words. No
digest sees through a paraphrase. That case is caught by review, and by the fact that the
Builder has no reason to open `evals/heldout/` at all.

What is deliberately *not* leakage: `video_id`. The dev/held-out **video** split is public —
it is in `data/corpus/manifest.json` and the Builder has to know which videos to avoid — and
QA_SPEC §6 asks for at least one answerable question per corpus video, so held-out pairs
point at dev videos on purpose. Only the labels are sealed.

    make leakage-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, NamedTuple

DEV = Path("evals/dev")
HELDOUT = Path("evals/heldout")

# The fields compared, in report order. See the module docstring for why these three and
# why video_id is not among them.
KINDS = ("id", "question", "answer_note")

# Copy-paste noise, not content: the left column is what a paste can turn the right into.
PUNCT_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
})

# Dropped from the end of a normalised field: a question copied without its '?' is still
# the same question.
TRAILING = " \t?!.,;:'\"()"


class LeakageError(Exception):
    """A split directory or one of its files cannot be read."""


class PairRef(NamedTuple):
    """Where a pair came from, in a form a human can open."""

    path: Path
    lineno: int
    pair_id: str

    def __str__(self) -> str:
        return f"{self.path.as_posix()}:{self.lineno} ({self.pair_id})"


class Collision(NamedTuple):
    kind: str
    dev: PairRef
    heldout: PairRef


def normalize(value: Any) -> str:
    """Fold away everything that is formatting rather than content.

    Returns "" for a missing or non-string field. An absent field is not evidence of a
    leak, and hashing the string "None" would make every pair missing that field collide
    with every other one missing it.
    """
    if not isinstance(value, str):
        return ""
    folded = unicodedata.normalize("NFKC", value).translate(PUNCT_MAP)
    collapsed = " ".join(folded.split())
    return collapsed.rstrip(TRAILING).casefold()


def fingerprint(value: Any) -> str | None:
    """sha256 of the normalised field, or None when there is nothing to compare."""
    text = normalize(value)
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_split(directory: Path) -> list[tuple[PairRef, dict[str, Any]]]:
    """Every pair in every `*.jsonl` under `directory`, with its origin.

    A missing directory is not an error — `evals/dev/` is empty until the dev cases are
    written and this check has to be runnable before then. An unreadable or malformed file
    *is* an error: skipping it quietly would report a clean intersection over pairs that
    were never compared.
    """
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise LeakageError(f"{directory} is not a directory")

    out: list[tuple[PairRef, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.jsonl")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise LeakageError(f"cannot read {path}: {exc}") from exc
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LeakageError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise LeakageError(f"{path}:{lineno} is {type(obj).__name__}, not an object")
            pair_id = obj.get("id")
            out.append((PairRef(path, lineno, str(pair_id) if pair_id else "no id"), obj))
    return out


def index(pairs: list[tuple[PairRef, dict[str, Any]]]) -> dict[str, dict[str, list[PairRef]]]:
    """kind -> fingerprint -> the pairs carrying it.

    A list, not one ref: two pairs inside one split can share a fingerprint, and a
    collision should name all of them rather than whichever was read last.
    """
    by_kind: dict[str, dict[str, list[PairRef]]] = {kind: {} for kind in KINDS}
    for ref, pair in pairs:
        for kind in KINDS:
            digest = fingerprint(pair.get(kind))
            if digest is not None:
                by_kind[kind].setdefault(digest, []).append(ref)
    return by_kind


def collisions(
    dev: list[tuple[PairRef, dict[str, Any]]],
    heldout: list[tuple[PairRef, dict[str, Any]]],
) -> list[Collision]:
    """Every dev/held-out pairing that shares a fingerprint. Empty means disjoint."""
    dev_index, heldout_index = index(dev), index(heldout)
    found: list[Collision] = []
    for kind in KINDS:
        for digest in sorted(set(dev_index[kind]) & set(heldout_index[kind])):
            for dev_ref in dev_index[kind][digest]:
                for heldout_ref in heldout_index[kind][digest]:
                    found.append(Collision(kind, dev_ref, heldout_ref))
    return found


def check(dev_dir: Path = DEV, heldout_dir: Path = HELDOUT) -> int:
    """Print the intersection size and return a process exit code."""
    dev = load_split(dev_dir)
    heldout = load_split(heldout_dir)
    found = collisions(dev, heldout)

    dev_files = len({ref.path for ref, _ in dev})
    heldout_files = len({ref.path for ref, _ in heldout})

    print(f"leakage check — {dev_dir.as_posix()} vs {heldout_dir.as_posix()}")
    print(f"  dev      {len(dev):3d} pairs from {dev_files} file(s)")
    print(f"  heldout  {len(heldout):3d} pairs from {heldout_files} file(s)")
    print(f"  compared by sha256 over: {', '.join(KINDS)}")
    print(f"  overlap  {len(found)}")
    # stdout is block-buffered when piped, stderr is not, so without this the failure
    # detail below lands before the counts it refers to. This is a gate: the number has to
    # read first, in the same order on a terminal and in a redirected log.
    sys.stdout.flush()

    if not heldout:
        print(f"\nFAIL — {heldout_dir.as_posix()} holds no pairs, so there is nothing to be "
              f"blind to; this check must not pass on an empty held-out set", file=sys.stderr)
        return 1

    if found:
        print(f"\nFAIL — {len(found)} collision(s):", file=sys.stderr)
        for kind, dev_ref, heldout_ref in found:
            print(f"  {kind}: {dev_ref}  ==  {heldout_ref}", file=sys.stderr)
        print("\nA held-out label is in the dev set. Stop — do not tune, do not run a gate.\n"
              "Remove the dev case, then re-run. See CLAUDE.md, blind labelling.",
              file=sys.stderr)
        return 1

    if not dev:
        # Honest about what was actually established. 0 ∩ 20 = ∅ is true and worth nothing:
        # the seal starts meaning something the moment the first dev case is written.
        print(f"\nPASS (vacuous) — {dev_dir.as_posix()} holds no pairs yet, so the "
              f"intersection is empty by default rather than by discipline")
        return 0

    print(f"\nPASS — {len(dev)} dev and {len(heldout)} held-out pairs share no id, "
          f"question or answer_note")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Assert the dev and held-out eval splits are disjoint by content hash (VRAG-013)."
    )
    ap.add_argument("--dev", default=str(DEV), help="the dev split directory")
    ap.add_argument("--heldout", default=str(HELDOUT), help="the held-out split directory")
    args = ap.parse_args(argv)

    # A gate signals a leak with exit 1. On a cp1252 console an em dash or a question mark
    # lifted from a pair title raises UnicodeEncodeError, which also exits 1 — so an
    # encoding accident would be read as a leak. Never let the report be the thing that
    # fails. Only in main(): importers keep their own streams.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    try:
        return check(Path(args.dev), Path(args.heldout))
    except LeakageError as exc:
        print(f"FAIL — {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
