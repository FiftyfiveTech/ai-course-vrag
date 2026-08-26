"""schemas/answer.py — the answer contract. VRAG-019.

What is tested here is what "schema-valid" means in the VRAG-019 gate, so each case below
is a decision about what the pipeline refuses to hand a user, not a type check.

No network, no model, no index.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from schemas.answer import ABSTAIN_TEXT, Answer, Citation, json_schema

GOOD = {
    "answer": "Eight.",
    "citations": [{"video_id": "611", "t_start": 21.8, "t_end": 47.8}],
    "abstain": False,
}


# --------------------------------------------------------------------- accepted


def test_a_well_formed_answer_validates():
    ans = Answer.model_validate(GOOD)
    assert ans.answer == "Eight."
    assert ans.abstain is False
    assert ans.citations[0].video_id == "611"
    assert ans.cited_videos() == ["611"]


def test_an_abstention_with_no_citations_validates():
    ans = Answer.model_validate({"answer": "Not covered.", "citations": [], "abstain": True})
    assert ans.abstain is True
    assert ans.citations == []


def test_abstention_helper_builds_a_valid_abstention():
    ans = Answer.abstention()
    assert ans.abstain is True and ans.citations == [] and ans.answer == ABSTAIN_TEXT


def test_citations_may_be_omitted_by_python_callers():
    """The default exists for `Answer.abstention()`; the wire schema still demands the key."""
    ans = Answer.model_validate({"answer": "x", "abstain": True})
    assert ans.citations == []


def test_a_zero_length_range_is_allowed():
    """t_end == t_start is a point in time, not a malformed range."""
    Citation.model_validate({"video_id": "1", "t_start": 5.0, "t_end": 5.0})


def test_a_json_string_validates_the_same_way():
    """`answer()` validates the model's raw text, not a dict it parsed itself."""
    assert Answer.model_validate_json(json.dumps(GOOD)).answer == "Eight."


# --------------------------------------------------------------------- refused


def test_an_extra_key_is_refused():
    """A key we did not ask for means the prompt has drifted; dropping it hides that."""
    with pytest.raises(ValidationError):
        Answer.model_validate({**GOOD, "confidence": 0.9})


def test_an_extra_key_on_a_citation_is_refused():
    bad = {**GOOD, "citations": [{**GOOD["citations"][0], "quote": "..."}]}
    with pytest.raises(ValidationError):
        Answer.model_validate(bad)


def test_a_missing_field_is_refused():
    for missing in ("answer", "abstain"):
        payload = {k: v for k, v in GOOD.items() if k != missing}
        with pytest.raises(ValidationError):
            Answer.model_validate(payload)


def test_a_backwards_time_range_is_refused():
    """Uncitable. transcript.drop_impossible refuses these on the way in as well."""
    with pytest.raises(ValidationError, match="does not run forward"):
        Citation.model_validate({"video_id": "611", "t_start": 90.0, "t_end": 12.0})


def test_a_negative_timestamp_is_refused():
    with pytest.raises(ValidationError):
        Citation.model_validate({"video_id": "611", "t_start": -1.0, "t_end": 5.0})


def test_an_empty_video_id_is_refused():
    with pytest.raises(ValidationError):
        Citation.model_validate({"video_id": "", "t_start": 1.0, "t_end": 2.0})


def test_abstaining_while_citing_is_refused():
    """QA_SPEC section 4: a citation on a declined question is incorrect regardless."""
    with pytest.raises(ValidationError, match="abstain is true"):
        Answer.model_validate({**GOOD, "abstain": True})


def test_answering_with_no_text_is_refused():
    with pytest.raises(ValidationError, match="answer text is empty"):
        Answer.model_validate({**GOOD, "answer": "   "})


# --------------------------------------------------------------------- the wire schema


def test_the_wire_schema_has_no_refs():
    """Strict structured-output mode does not follow $ref/$defs."""
    rendered = json.dumps(json_schema())
    assert "$ref" not in rendered
    assert "$defs" not in rendered


def test_every_object_in_the_wire_schema_is_closed_and_fully_required():
    """What strict mode means: no extra keys anywhere, no optional fields anywhere."""
    seen = 0

    def walk(node):
        nonlocal seen
        if isinstance(node, dict):
            if node.get("type") == "object" and isinstance(node.get("properties"), dict):
                seen += 1
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json_schema())
    assert seen == 2, f"expected the Answer object and the inlined Citation, saw {seen}"


def test_the_wire_schema_names_the_three_fields_the_card_asks_for():
    schema = json_schema()
    assert set(schema["properties"]) == {"answer", "citations", "abstain"}
    citation = schema["properties"]["citations"]["items"]
    assert set(citation["properties"]) == {"video_id", "t_start", "t_end"}


def test_the_wire_schema_and_the_validator_agree_on_a_real_reply():
    """The point of generating one from the other: they cannot disagree.

    A reply that satisfies the schema handed to the model must validate here. If this ever
    fails, a generation-time constraint is not the constraint being checked, and the gate's
    schema-valid number is measuring the wrong thing.
    """
    required = json_schema()["required"]
    assert set(GOOD) == set(required)
    Answer.model_validate(GOOD)
