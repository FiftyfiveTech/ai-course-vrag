"""Meeting minutes contract — VRAG-030.

`Minutes` is the shape a Teams meeting produces after Phase 4: a summary, who attended,
what was decided, and what was committed to. Expressed once here so nothing in the pipeline
assembles or parses it by hand.

    from schemas.minutes import Minutes, ActionItem

    minutes = Minutes.model_validate(raw_dict)
    print(minutes.summary)
    for item in minutes.action_items:
        print(item.owner, item.task)

Two rules that are not negotiable and are enforced here rather than in the caller:

**`owner` is nullable and stays nullable.** A wrong name is worse than no name in a document
that assigns work: an invented commitment attributed to a named colleague is the worst output
this pipeline can produce. `None` is the honest representation of "nobody said". Any code that
later validates owners against a roster (VRAG-032) operates on this field; if it were required,
every action item without a speaker would force a hallucinated name just to satisfy the schema.

**`evidence` is required on every action item.** An action item with no timestamp is a rumour
with formatting. The same rule `schemas.answer.Citation` enforces for Q&A citations applies
here: a minute that cannot be checked cannot be trusted. VRAG-033 grounds every item in a time
range; this field is where that range lands.

`json_schema()` follows the same pattern as `schemas.answer.json_schema()`: one declaration
that both constrains generation and validates output, so the two cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ActionItem(BaseModel):
    """One commitment made in the meeting."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, description="what was committed to")
    owner: str | None = Field(
        default=None,
        description="who committed — None when no speaker was identified; never invented",
    )
    due: str | None = Field(
        default=None,
        description="deadline if stated explicitly in the meeting, else None",
    )
    evidence: str = Field(
        min_length=1,
        description="timestamp or quote that grounds this item; required — an ungrounded "
        "action item is a rumour",
    )


class Minutes(BaseModel):
    """Structured minutes for one meeting."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, description="one-paragraph meeting summary")
    attendees: list[str] = Field(
        default_factory=list,
        description="display names of people who attended",
    )
    decisions: list[str] = Field(
        default_factory=list,
        description="decisions made — each a complete sentence",
    )
    action_items: list[ActionItem] = Field(
        default_factory=list,
        description="commitments made; owner is None when the speaker was not identified",
    )


def json_schema() -> dict[str, Any]:
    """Minutes as JSON Schema for structured-output generation.

    Same contract as `schemas.answer.json_schema()`: strict mode — every property required,
    no additionalProperties, no $ref indirection — so the schema handed to the model and the
    schema used to validate its output are the same declaration.
    """
    schema = Minutes.model_json_schema()
    defs = schema.pop("$defs", {})
    return _strictify(_inline_refs(schema, defs))


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target = defs.get(ref.split("/")[-1], {})
            return _inline_refs(dict(target), defs)
        return {k: _inline_refs(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def _strictify(node: Any) -> Any:
    if isinstance(node, dict):
        out = {k: _strictify(v) for k, v in node.items()}
        if out.get("type") == "object" and isinstance(out.get("properties"), dict):
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out
    if isinstance(node, list):
        return [_strictify(v) for v in node]
    return node
