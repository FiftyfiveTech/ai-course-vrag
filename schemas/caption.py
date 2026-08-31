"""The caption contract — what one keyframe says, and what a run of them cost.

`schemas/overview.py` declares what a whole video is; this declares what is written on its
screen. Both carry seconds off something real, and for the same reason: a claim a player
cannot seek to is a claim nobody can check.

What is different here, and why there is no JSON schema
-------------------------------------------------------
`Overview` is generated under strict mode — `schemas.overview.json_schema()` constrains the
model at generation time. A caption is not, and that is a deliberate asymmetry rather than an
omission.

VRAG-023's deliverable is a **two-arm cost table**: the same keyframes through a hosted vision
model and a local one, with the arm as the only variable. The hosted arm (NVIDIA NIM) and the
local arm (Ollama) do not offer the same structured-output guarantees, so constraining
generation would make the arms differ in *how they were asked* as well as in what ran — and
the table could no longer attribute a latency or a token count to the arm. So both arms are
asked for plain text under the same prompt, and the one bit of structure that is needed is
carried by a sentinel the prompt defines: `NO_TEXT`.

`has_text` is that sentinel, normalised (`src.caption.parse_reply`). It is not decoration —
it is the yield number the cost table reports. A cost per call is only interesting next to
how often the call found anything, because a selection rule that picks blank frames is cheap
and useless in the same breath.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Caption(BaseModel):
    """What one keyframe has written on it, and the stretch of video it stands for."""

    model_config = ConfigDict(extra="forbid")

    frame: str = Field(
        min_length=1, description="the frame file this was read from, e.g. frame_00042.jpg"
    )
    t_start: float = Field(ge=0.0, description="seconds; the start of the still stretch")
    t_end: float = Field(ge=0.0, description="seconds; the end of the still stretch")
    text: str = Field(
        description="what is written on the frame, verbatim; empty when there is none"
    )
    has_text: bool = Field(
        description="false when the model replied NO_TEXT; the yield the cost table reports"
    )
    run_frames: int = Field(
        default=1,
        ge=1,
        description="how many sampled frames this one keyframe stands in for",
    )

    @model_validator(mode="after")
    def _range_runs_forward(self) -> "Caption":
        if self.t_end < self.t_start:
            raise ValueError(
                f"{self.frame}: span does not run forward: t_start={self.t_start}, "
                f"t_end={self.t_end}"
            )
        return self

    @model_validator(mode="after")
    def _text_and_flag_agree(self) -> "Caption":
        """`has_text` is derived from the reply, so it cannot disagree with `text`.

        Both directions are real failures rather than tidiness. A caption with text and
        `has_text=False` is undercounted yield; one with no text and `has_text=True` is a
        `NO_TEXT` reply that was not recognised — the sentinel changed, or the model wrapped
        it in a sentence — and the yield number silently becomes 100%.
        """
        if bool(self.text.strip()) != self.has_text:
            raise ValueError(
                f"{self.frame}: has_text={self.has_text} but text is "
                f"{'empty' if not self.text.strip() else 'not empty'}. The flag is derived "
                f"from the reply in src.caption.parse_reply; they cannot be set separately."
            )
        return self


class StoredCaptions(BaseModel):
    """One video's captions on disk, with what is needed to know what produced them.

    `arm` and `model` are both here, and both are needed: the arm is the lever that was set
    and the model is the HF repo id that answered. Everywhere else in this pipeline those two
    can come apart (`answer.fallback` drops a 429 onto the local model), so recording only the
    lever would attribute one model's numbers to another. This is the same reason
    `StoredOverview` carries `model` and `AnswerRun` carries `used`.

    `frames_considered` and `runs_found` are the cost reduction, stored rather than recomputed:
    they are the numerator and denominator of the ratio the two-arm table is read against, and
    a `--limit`ed run must still record the full selection it sampled from.
    """

    model_config = ConfigDict(extra="forbid")

    task: Literal["VRAG-023"] = "VRAG-023"
    video_id: str = Field(min_length=1)
    arm: str = Field(default="", description="the lever: 'nim' or 'ollama'")
    model: str = Field(default="", description="HF repo id that actually answered")
    prompt: str = Field(default="", description="path of the prompt file")
    prompt_sha256: str = Field(default="")
    threshold: float = Field(default=0.0, description="caption.still_threshold as run")
    min_run_frames: int = Field(default=0, description="caption.min_run_frames as run")
    frames_considered: int = Field(
        default=0, ge=0, description="frames on disk the scorer read"
    )
    runs_found: int = Field(
        default=0, ge=0, description="still stretches found, before any --limit"
    )
    tokens: int = Field(default=0, ge=0, description="tokens the arm reported, all calls")
    latency_s: float = Field(default=0.0, ge=0.0, description="sum of vision-call latency")
    captions: list[Caption] = Field(default_factory=list)

    def text_yield(self) -> float:
        """Fraction of captioned keyframes that had readable text on them.

        Zero captions is 0.0 rather than an error: a video with no still stretches at all is a
        real and informative outcome — it is what "not slide-heavy" looks like.
        """
        if not self.captions:
            return 0.0
        return sum(1 for c in self.captions if c.has_text) / len(self.captions)

    def reduction(self) -> float:
        """How many times fewer vision calls the selection bought. 1.0 when it bought nothing."""
        if not self.runs_found:
            return 0.0
        return self.frames_considered / self.runs_found
