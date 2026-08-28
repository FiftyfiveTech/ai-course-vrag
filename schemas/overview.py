"""The overview contract — what a whole video is, in one document.

`schemas/answer.py` declares what the model may return for a *question*. This declares what
it may return for a *video*, and the two are separate for the same reason `AskResponse` is
separate from `Answer`: they are scored differently and they fail differently.

An `Overview` is built once, at index time, from the whole transcript — not from five
retrieved passages. That is the entire point. "What is this video about?" and "who is taking
part?" have no answer in any single 25-second chunk, so the extractive path
(`prompts/answer_v1.md`, rules 2 and 3) is *correct* to decline them. This document is where
the synthesis lives instead, and it is written to disk so the synthesis is paid for once per
video rather than once per question.

Every claim carries seconds
---------------------------
`Person.evidence` and `Topic` both carry a `t_start`/`t_end` off a real chunk. That is not
decoration — `src.overview.as_chunks` projects those spans into `RetrievedChunk`s and hands
them to `src.answer.ground`, so an answer built from an overview is grounded by exactly the
same code, and cited by exactly the same code, as an answer built from retrieval. A person
the model cannot point at is a person it has to leave out.

No speaker labels, and the schema says so
-----------------------------------------
`speaker_labels` is `False` and there is no way to set it true, because nothing in this
pipeline does diarization: `runs/<stem>/transcript.json` segments carry `t_start`, `t_end`
and `text` and nothing else (`src/transcript.py`). So `people` is *who is named in the
transcript*, not *who is speaking* — the two are different claims and only the first one is
recoverable here. The field exists to keep that distinction in the contract rather than in a
comment, so that the day diarization is added, the readers that have to change are the ones
this field is checked in.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.answer import _inline_refs, _strictify


class Span(BaseModel):
    """Seconds into the video, off a real chunk. The same three numbers a citation is."""

    model_config = ConfigDict(extra="forbid")

    t_start: float = Field(ge=0.0, description="seconds from the start of the video")
    t_end: float = Field(ge=0.0, description="seconds from the start of the video")

    @model_validator(mode="after")
    def _range_runs_forward(self) -> "Span":
        if self.t_end < self.t_start:
            raise ValueError(
                f"span does not run forward: t_start={self.t_start}, t_end={self.t_end}"
            )
        return self

    def __str__(self) -> str:
        return f"{self.t_start:.1f}s-{self.t_end:.1f}s"


class Person(BaseModel):
    """Someone the transcript names.

    `described_as` is what the video says about them and nothing more — a role it states, a
    thing it says they did. It is not an inference from how much they talk, because how much
    they talk is exactly what an undiarized transcript does not record.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="the name as the transcript says it")
    described_as: str = Field(
        description="what the video says about them; empty when it says only the name"
    )
    evidence: Span = Field(description="where the name is said")


class Topic(BaseModel):
    """One stretch of the video, and what is discussed in it."""

    model_config = ConfigDict(extra="forbid")

    t_start: float = Field(ge=0.0, description="seconds from the start of the video")
    t_end: float = Field(ge=0.0, description="seconds from the start of the video")
    topic: str = Field(min_length=1, description="one clause, not a sentence")

    @model_validator(mode="after")
    def _range_runs_forward(self) -> "Topic":
        if self.t_end < self.t_start:
            raise ValueError(
                f"topic {self.topic!r} does not run forward: t_start={self.t_start}, "
                f"t_end={self.t_end}"
            )
        return self


class Overview(BaseModel):
    """What one video is, built from its whole transcript."""

    model_config = ConfigDict(extra="forbid")

    abstract: str = Field(
        min_length=1,
        description="three to five sentences saying what the video is and what happens in it",
    )
    people: list[Person] = Field(
        default_factory=list,
        description="everyone the transcript names; empty when it names nobody",
    )
    topics: list[Topic] = Field(
        default_factory=list, description="the video in order, as stretches"
    )

    def spans(self) -> list[Span]:
        """Every span this document can cite, people first, then the timeline."""
        out = [p.evidence for p in self.people]
        out.extend(Span(t_start=t.t_start, t_end=t.t_end) for t in self.topics)
        return out


class StoredOverview(BaseModel):
    """An `Overview` on disk, with what is needed to know whether it is still true.

    `source_sha256` is copied from the transcript this was built from, so
    `src.overview.build` can skip a video whose media has not changed and rebuild one whose
    has. Without it, re-indexing either pays for every overview again or serves a description
    of a video that no longer exists — and the second failure is silent.

    The model and prompt digests are here for the same reason they are in `Provenance`: an
    overview that cannot be attributed to the prompt that produced it cannot be disagreed
    with.
    """

    model_config = ConfigDict(extra="forbid")

    task: Literal["VRAG-OVERVIEW"] = "VRAG-OVERVIEW"
    video_id: str = Field(min_length=1)
    source_sha256: str = Field(default="", description="the transcript's source digest")
    model: str = Field(default="", description="HF repo id that produced it")
    prompt: str = Field(default="", description="path of the prompt file")
    prompt_sha256: str = Field(default="")
    chunks: int = Field(default=0, ge=0, description="chunks the overview was built from")
    speaker_labels: Literal[False] = Field(
        default=False,
        description=(
            "always false: nothing in this pipeline diarizes, so `people` is who the "
            "transcript names, not who is speaking"
        ),
    )
    overview: Overview


def json_schema() -> dict[str, Any]:
    """`Overview` as the JSON Schema handed to the model at generation time.

    Same treatment `schemas.answer.json_schema` gives `Answer`, reusing its two helpers
    rather than restating them: strict mode wants every object closed, every property
    required, and no `$ref` indirection. Only `Overview` is rendered — the disk wrapper is
    this module's business, not the model's.
    """
    schema = Overview.model_json_schema()
    defs = schema.pop("$defs", {})
    return _strictify(_inline_refs(schema, defs))
