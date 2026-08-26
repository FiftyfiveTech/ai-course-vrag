"""The answer contract - VRAG-019.

`Answer` is the shape QA_SPEC section 2 describes, expressed once so that nothing in the pipeline
parses model output by hand:

    {"answer": "...", "citations": [{"video_id": "611", "t_start": 79.5, "t_end": 85.0}],
     "abstain": false}

Two jobs, and they are separate on purpose.

**A wire contract.** `json_schema()` renders this model as the JSON Schema handed to the
model itself - Groq's strict `response_format`, Ollama's `format` - so the thing that
constrains generation and the thing that validates the result are the same declaration. A
schema written twice drifts, and the day it drifts the validator is the one that is wrong.

**A definition of valid.** `Answer.model_validate` is what "schema-valid" means in the
VRAG-019 gate, so what it rejects is a decision, not an accident:

* `extra="forbid"` - an extra key means the model answered a different question than the one
  the schema asked. Silently dropping it hides a prompt that has stopped working.
* time ranges run forward and start at or after zero. `t_end < t_start` is uncitable, and
  the chunker already refuses such ranges on the way in (`transcript.drop_impossible`);
  accepting one on the way out would put a range in front of a user that the index would
  have thrown away.
* `abstain` and `citations` are coupled. QA_SPEC section 4: a citation returned for a question the
  system declined is incorrect *regardless of what the citation says*. A response that
  declines and cites is not a near-miss to be repaired, it is incoherent, and repairing it
  quietly before validation would make the gate's "100% schema-valid" mean nothing.
* `answer` is non-empty unless `abstain` - the one field a user reads cannot be blank while
  the response claims to have answered.

What this model deliberately does **not** check is whether a citation points at a chunk that
was actually retrieved. A well-formed citation can still be invented. That is grounding, it
needs the retrieved set to check against, and it lives in `src.answer.ground` - after
validation, so the gate measures the model's own output rather than a repaired copy of it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# What a declined answer says when the pipeline itself forces the abstention (see
# src.answer.ground). Kept here because it has to satisfy this model's own rules.
ABSTAIN_TEXT = "I could not find this in the indexed videos."


class Citation(BaseModel):
    """One pointer into the corpus: which video, and the seconds to look at.

    `t_start` is the field the gate scores (QA_SPEC section 2 checks |t_start - t_ref| <= 30), so it
    is the one that has to be a real number off a real chunk rather than a plausible one.
    """

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1, description="corpus video_id, e.g. '611'")
    t_start: float = Field(ge=0.0, description="seconds from the start of the video")
    t_end: float = Field(ge=0.0, description="seconds from the start of the video")

    @model_validator(mode="after")
    def _range_runs_forward(self) -> "Citation":
        if self.t_end < self.t_start:
            raise ValueError(
                f"citation time range does not run forward: t_start={self.t_start}, "
                f"t_end={self.t_end}"
            )
        return self

    def __str__(self) -> str:
        return f"video {self.video_id} {self.t_start:.1f}s-{self.t_end:.1f}s"


class Answer(BaseModel):
    """A grounded answer, or a refusal. Never both, never neither."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="the answer text; empty only when abstaining")
    citations: list[Citation] = Field(
        default_factory=list, description="empty when abstaining"
    )
    abstain: bool = Field(description="true when the corpus does not answer the question")

    @model_validator(mode="after")
    def _abstention_is_coherent(self) -> "Answer":
        if self.abstain and self.citations:
            raise ValueError(
                f"abstain is true but {len(self.citations)} citation(s) were returned - "
                f"QA_SPEC section 4: a citation on a declined question is incorrect regardless of "
                f"what it says"
            )
        if not self.abstain and not self.answer.strip():
            raise ValueError("abstain is false but the answer text is empty")
        return self

    @classmethod
    def abstention(cls, text: str = ABSTAIN_TEXT) -> "Answer":
        """The refusal, built so it cannot be built wrong."""
        return cls(answer=text, citations=[], abstain=True)

    def cited_videos(self) -> list[str]:
        return sorted({c.video_id for c in self.citations})


def json_schema() -> dict[str, Any]:
    """This model as JSON Schema, in the dialect hosted structured-output modes want.

    Groq's `response_format={"type": "json_schema", ..., "strict": True}` and Ollama's
    `format=` both take a plain JSON Schema object, but strict mode adds two demands that
    Pydantic's default output does not meet:

    * every property must be listed in `required` - optional fields are not allowed at all,
      so `citations`, which has a default here, has to be declared required on the wire.
      A model that omits it is caught by the schema rather than by a validator.
    * `additionalProperties: false` on every object. `extra="forbid"` already puts it on
      each model, but `$defs`/`$ref` indirection is not accepted by strict mode either, so
      the reference to `Citation` is inlined.

    Round-tripped by the tests against `Answer.model_validate`, because the whole point of
    generating it from the model is that the two cannot disagree.
    """
    schema = Answer.model_json_schema()
    defs = schema.pop("$defs", {})
    return _strictify(_inline_refs(schema, defs))


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    """Replace every {"$ref": "#/$defs/X"} with a copy of X. Strict mode rejects $ref."""
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
    """Every object closed, every property required - what strict mode means."""
    if isinstance(node, dict):
        out = {k: _strictify(v) for k, v in node.items()}
        if out.get("type") == "object" and isinstance(out.get("properties"), dict):
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out
    if isinstance(node, list):
        return [_strictify(v) for v in node]
    return node
