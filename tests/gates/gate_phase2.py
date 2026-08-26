"""GATE Phase 2 (MVP) — VRAG-021.

Scores the full retrieve→answer pipeline against the 20 sealed heldout pairs.

    pytest tests/gates/gate_phase2.py -v -s
    # or
    make gate

Pass criteria (QA_SPEC §5)
--------------------------
- score = (correct_answerable + correct_abstentions) / 20 ≥ 0.70
- 3/3 unanswerable pairs correctly abstained (hard requirement printed separately)

A pair is correct when (QA_SPEC §2):
  1. abstain is False
  2. At least one citation has the right video_id
  3. That citation has |t_start − t_ref| ≤ 30 s

An abstention is correct when the pair is unanswerable and abstain is True.

Pre-requisites
--------------
The corpus videos must be indexed in Chroma and the answer model available:

    make index-dev   # or index each heldout video_id manually
    # pull the answer model (see answer.arm in config.toml)

The supervisor re-runs this gate after reviewing the PR.  The printed numbers —
not anything written in a comment — are what count.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.answer import AnswerError, AnswerRun, answer
from src.config import load as load_config
from src.telemetry import Meter

HELDOUT = ROOT / "evals" / "heldout"
CITATION_TOLERANCE_S = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_pairs(directory: Path) -> list[dict]:
    pairs = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def _is_correct(pair: dict, run: AnswerRun) -> bool:
    """True when the run satisfies QA_SPEC §2 for this pair."""
    if not run.valid:
        return False  # schema validation failed
    ans = run.answer
    if pair.get("unanswerable", False):
        return ans.abstain
    if ans.abstain:
        return False
    target = str(pair["video_id"])
    t_ref = float(pair["t_ref"])
    for c in ans.citations:
        if str(c.video_id) == target:
            if abs(c.t_start - t_ref) <= CITATION_TOLERANCE_S:
                return True
    return False


# ---------------------------------------------------------------------------
# Module-scoped fixture: run the pipeline once for all test functions
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "config.toml")


@pytest.fixture(scope="module")
def gate_results(cfg):
    """Run answer() for every heldout pair. Returns list of (pair, AnswerRun|None, error)."""
    pairs = _load_pairs(HELDOUT)
    assert pairs, (
        f"No pairs found in {HELDOUT} — has evals/heldout/heldout_v1.jsonl been committed?"
    )

    meter = Meter()
    results = []
    for pair in pairs:
        try:
            run = answer(pair["question"], cfg, meter)
            results.append((pair, run, None))
        except AnswerError as exc:
            results.append((pair, None, str(exc)))

    return results


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------


def test_all_pairs_loaded(gate_results):
    total = len(gate_results)
    print(f"\nheldout pairs loaded: {total}")
    assert total == 20, f"expected 20 heldout pairs, got {total}"


def test_no_pipeline_errors(gate_results):
    errors = [(p["id"], e) for p, _, e in gate_results if e is not None]
    if errors:
        for id_, msg in errors:
            print(f"\n  ERROR {id_}: {msg}")
    assert not errors, (
        f"{len(errors)} pipeline error(s) — is the index built and the model available?\n"
        + "\n".join(f"  {id_}: {msg}" for id_, msg in errors)
    )


def test_abstentions(gate_results):
    """All 3 unanswerable pairs must be correctly abstained."""
    unanswerable = [(p, r) for p, r, e in gate_results if p.get("unanswerable") and e is None]
    abstained = sum(1 for _, r in unanswerable if r is not None and r.valid and r.answer.abstain)
    total_u = len(unanswerable)

    # THE PHASE 2 ABSTENTION NUMBER — must appear in output.
    print(f"\nabstentions: {abstained}/{total_u}")

    assert abstained == total_u, (
        f"only {abstained}/{total_u} unanswerable pairs correctly abstained"
    )


def test_score_at_least_70_percent(gate_results):
    """(correct_answerable + correct_abstentions) / 20 ≥ 0.70 (QA_SPEC §5)."""
    total = len(gate_results)
    correct = sum(
        1
        for pair, run, err in gate_results
        if err is None and run is not None and _is_correct(pair, run)
    )

    score = correct / total if total else 0.0

    answerable = [(p, r) for p, r, e in gate_results if not p.get("unanswerable") and e is None]
    correct_ans = sum(1 for p, r in answerable if r is not None and _is_correct(p, r))
    unanswerable = [(p, r) for p, r, e in gate_results if p.get("unanswerable") and e is None]
    correct_abst = sum(
        1 for p, r in unanswerable if r is not None and r.valid and r.answer.abstain
    )

    # THE PHASE 2 SCORE — must appear in output.
    print(f"\nscore: {correct}/{total}  ({score:.4f})")
    print(f"  answerable correct:  {correct_ans}/{len(answerable)}")
    print(f"  abstentions correct: {correct_abst}/{len(unanswerable)}")

    # Per-pair breakdown.
    print("\nper-pair results:")
    for pair, run, err in gate_results:
        if err:
            print(f"  {pair['id']} ERROR: {err[:60]}")
        elif run is None or not run.valid:
            print(f"  {pair['id']} INVALID (schema error)")
        elif _is_correct(pair, run):
            print(f"  {pair['id']} OK")
        else:
            if pair.get("unanswerable"):
                print(f"  {pair['id']} FAIL (expected abstain, got abstain={run.answer.abstain})")
            else:
                cits = [(c.video_id, c.t_start) for c in run.answer.citations]
                print(
                    f"  {pair['id']} FAIL want video={pair['video_id']} t={pair['t_ref']}"
                    f"  abstain={run.answer.abstain} cits={cits}"
                )

    assert score >= 0.70, (
        f"score {correct}/{total} = {score:.4f} is below the 0.70 MVP threshold"
    )
