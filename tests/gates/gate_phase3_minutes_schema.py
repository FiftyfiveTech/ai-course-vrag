"""GATE VRAG-030 — Minutes schema is valid on 100% of dev meetings.

    uv run pytest tests/gates/gate_phase3_minutes_schema.py -v -s

The card's criterion: `Minutes{summary, attendees[], decisions[], action_items[]}` and
`ActionItem{task, owner: str | None, due: str | None, evidence}`. Schema-valid on 100% of
the dev meetings, tally printed, as gate_phase2a does.

This gate has two parts:

**Schema correctness.** `Minutes.model_validate` and `json_schema()` are round-tripped
against fixture data — the same double-check `schemas.answer` tests use. The schema handed
to the model (json_schema) and the schema used to validate output (model_validate) must be
the same declaration.

**Tally on runs/*/minutes.json.** Every minutes file in the runs directory is validated and
the count is printed. A 0/0 result (no runs yet) prints and passes — the run directory fills
once VRAG-031 is wired up and a real meeting is captured. A validation failure on any
existing file is a hard FAIL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from schemas.minutes import ActionItem, Minutes, json_schema

RUNS = Path("runs")

# ---------------------------------------------------------------------------
# Fixtures — synthetic minutes that must be schema-valid
# ---------------------------------------------------------------------------

VALID_MINUTES = {
    "summary": "The team reviewed the migration timeline and assigned owners.",
    "attendees": ["Priya Nair", "Rohan Mehta"],
    "decisions": ["Use the Graph arm for all internal meetings from next sprint."],
    "action_items": [
        {
            "task": "Set up the tenant application access policy",
            "owner": "Rohan Mehta",
            "due": "2026-09-05",
            "evidence": "00:04:12 — Rohan said he would handle the policy before end of week.",
        },
        {
            "task": "Draft the eval spec for action item labelling",
            "owner": None,
            "due": None,
            "evidence": "00:11:30 — discussed but no owner volunteered.",
        },
    ],
}

MINIMAL_MINUTES = {
    "summary": "Short sync with no decisions.",
    "attendees": [],
    "decisions": [],
    "action_items": [],
}


# ---------------------------------------------------------------------------
# Schema correctness
# ---------------------------------------------------------------------------


def test_valid_minutes_parses():
    m = Minutes.model_validate(VALID_MINUTES)
    assert m.summary.startswith("The team")
    assert len(m.attendees) == 2
    assert len(m.decisions) == 1
    assert len(m.action_items) == 2


def test_owner_is_nullable():
    m = Minutes.model_validate(VALID_MINUTES)
    owners = [item.owner for item in m.action_items]
    assert "Rohan Mehta" in owners
    assert None in owners


def test_minimal_minutes_parses():
    m = Minutes.model_validate(MINIMAL_MINUTES)
    assert m.action_items == []


def test_missing_evidence_is_rejected():
    bad = {**VALID_MINUTES, "action_items": [{"task": "Do something", "owner": None, "due": None}]}
    with pytest.raises(ValidationError):
        Minutes.model_validate(bad)


def test_empty_task_is_rejected():
    bad = {
        **VALID_MINUTES,
        "action_items": [{"task": "", "owner": None, "due": None, "evidence": "00:01:00"}],
    }
    with pytest.raises(ValidationError):
        Minutes.model_validate(bad)


def test_extra_fields_are_rejected():
    bad = {**VALID_MINUTES, "unexpected_field": "oops"}
    with pytest.raises(ValidationError):
        Minutes.model_validate(bad)


def test_json_schema_round_trips():
    """json_schema() and model_validate must agree — same declaration, two jobs."""
    schema = json_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    assert set(props) == {"summary", "attendees", "decisions", "action_items"}
    # action_items items must have the ActionItem shape inlined (no $ref)
    items_schema = props["action_items"]["items"]
    assert "$ref" not in str(items_schema), "json_schema() must inline $ref for strict mode"
    assert set(items_schema["properties"]) == {"task", "owner", "due", "evidence"}


def test_action_item_owner_nullable_in_json_schema():
    schema = json_schema()
    owner_schema = schema["properties"]["action_items"]["items"]["properties"]["owner"]
    # owner must accept null — either anyOf with null type, or type includes null
    schema_str = json.dumps(owner_schema)
    assert "null" in schema_str, f"owner must be nullable in json_schema, got: {owner_schema}"


# ---------------------------------------------------------------------------
# Tally on runs/*/minutes.json
# ---------------------------------------------------------------------------


def _load_minutes_files() -> list[Path]:
    if not RUNS.is_dir():
        return []
    return sorted(RUNS.glob("*/minutes.json"))


def test_schema_valid_on_all_runs(capsys):
    """Validate every runs/*/minutes.json and print the tally.

    0/0 passes — the runs directory fills once VRAG-031 is wired up.
    Any file that fails validation is a hard FAIL.
    """
    paths = _load_minutes_files()
    valid = 0
    invalid = 0
    failures: list[str] = []

    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            Minutes.model_validate(raw)
            valid += 1
        except (json.JSONDecodeError, ValidationError) as exc:
            invalid += 1
            failures.append(f"  {path}: {exc}")

    total = valid + invalid
    print(f"\nminutes schema-valid: {valid}/{total} meetings")
    if failures:
        print("failures:")
        for f in failures:
            print(f)

    assert invalid == 0, (
        f"{invalid}/{total} minutes file(s) failed schema validation:\n"
        + "\n".join(failures)
    )
