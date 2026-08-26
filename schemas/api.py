"""The HTTP contract — what a frontend can rely on.

`schemas/answer.py` declares what the *model* is allowed to return. This declares what the
*API* returns, and the two are deliberately not the same object. `Answer` is three fields
because a citation is three numbers by design; a browser needs more than that to render one —
a clock label, the passage it came from, and a url it can actually play. Widening `Answer` to
carry those would put presentation into the contract the gate measures schema-validity
against, so the presentation lives here instead and `AskResponse` is built *from* a validated
`Answer` (see `src.api.to_response`).

Declared rather than hand-assembled for the same reason `schemas/answer.py` is: FastAPI
renders these models into the OpenAPI document at `/docs`, so the response shape a frontend
codes against and the response shape the server produces come off one declaration. A response
dict built inline would be documented by nothing and could drift by a rename.

Three things a client has to be able to tell apart, and the reason each field exists:

* **answered / declined / broken.** `abstain` is the system declining because the corpus does
  not cover the question — a correct outcome under QA_SPEC §4, not an error. `schema_valid`
  false is the model having produced something that is not an `Answer` at all; `error` says
  what. Both come back 200, because the request was fine and a frontend that renders a 500
  for an abstention would be reporting a bug that is not there.
* **playable / linkable / neither.** `stream_url` is set when the media is on the machine
  running the API and can be range-served (`GET /media/{video_id}`); `source_url` is the
  manifest url with the timestamp on it, which is all a clean clone has, because the corpus
  is pointers and not copies (`data/corpus/PROVENANCE.md`). Both can be null for a cited
  video that is neither — the citation is still returned, and a client that finds no url
  should say so rather than render a dead control.
* **what produced this.** `provenance` and `spend` are the same lines the CLI prints and the
  demo page footers. An answer with no provenance cannot be re-run or disagreed with, and
  that does not stop being true because the transport changed.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    """`POST /ask` — one question, and nothing else.

    `extra="forbid"` so a client that sends `top_k` or `temperature` gets a 422 telling it
    so, rather than having the field silently ignored and wondering why the lever did
    nothing. That is the deliberate part: the retrieval and generation levers live in
    `config.toml` and only there (`src/config.py`), so a run's numbers are attributable to a
    config fingerprint. A per-request override would make two answers from the same server
    incomparable and there would be no record of why.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=1000,
        description="the question to answer against the indexed videos",
    )


class CitationOut(BaseModel):
    """One citation, with everything a player needs to jump to it."""

    n: int = Field(description="1-based index, matching the order the model cited them")
    video_id: str
    t_start: float = Field(description="seconds; the chunk boundary the model cited")
    t_end: float
    seek_s: float = Field(
        description=(
            "seconds to seek the player to — t_start less ask.pad_s, floored at 0. A chunk "
            "boundary is a grid line on the video clock and not where the sentence starts."
        )
    )
    label: str = Field(description="human-readable, e.g. 'video 611 · 1:19–1:25'")
    passage: str = Field(description="the retrieved text this citation was grounded onto")
    stream_url: str | None = Field(
        default=None,
        description="range-served media on this host, or null when it was never fetched",
    )
    source_url: str | None = Field(
        default=None,
        description="the manifest url with the timestamp on it, or null if there is none",
    )


class Spend(BaseModel):
    """What this one request cost. Every model call goes through the shared meter."""

    calls: int
    latency_s: float
    cost_usd: float


class Provenance(BaseModel):
    """Which prompt, which config bytes, which models. Re-runnable or it did not happen."""

    arm: str
    answer_model: str = Field(description="HF repo id, not a provider's wire id")
    embed_model: str
    top_k: int
    retrieved: int = Field(description="passages actually returned by retrieval")
    prompt: str
    prompt_sha256: str
    config: str
    config_sha256: str


class AskResponse(BaseModel):
    """`POST /ask` — the answer, its citations, and what produced it."""

    question: str
    answer: str = Field(description="the answer text, or the refusal when abstain is true")
    abstain: bool = Field(description="true when the corpus does not answer the question")
    schema_valid: bool = Field(
        description="false when the model's reply was not a valid Answer; see error"
    )
    error: str | None = Field(
        default=None, description="why the reply did not validate, when schema_valid is false"
    )
    citations: list[CitationOut] = Field(
        default_factory=list, description="empty on an abstention — QA_SPEC §4"
    )
    repairs: list[str] = Field(
        default_factory=list,
        description="what grounding had to fix: dropped or snapped citations",
    )
    spend: Spend
    provenance: Provenance


class IndexStatus(BaseModel):
    """Whether there is anything to answer from."""

    ready: bool = Field(description="false when the collection is absent or empty")
    collection: str
    path: str
    chunks: int
    videos: list[str] = Field(
        default_factory=list, description="video_ids present in the collection"
    )


class Health(BaseModel):
    """`GET /health` — can this server answer a question right now, and if not, why not.

    A frontend needs this before it shows a question box. An empty index does not fail: every
    question abstains and the UI looks like it is working while answering nothing, which is
    the failure mode `src.ask` refuses outright. `ready` false with `detail` naming the
    command to run is the honest version of that.
    """

    ready: bool
    detail: str = Field(description="what to do about it when ready is false")
    index: IndexStatus
    arm: str
    answer_model: str
    embed_model: str
    media_served: bool = Field(
        description="whether GET /media/{video_id} will serve local corpus media"
    )
    config: str
    config_sha256: str


class Video(BaseModel):
    """`GET /videos` — one cited-or-citable video and where it can be watched."""

    video_id: str
    split: str | None = Field(default=None, description="'dev' or 'heldout' per the manifest")
    indexed: bool = Field(description="whether the collection holds chunks for this video")
    stream_url: str | None = None
    source_url: str | None = None


class Problem(BaseModel):
    """The error body. One shape for every non-2xx this app produces.

    FastAPI's default is `{"detail": ...}` where detail is sometimes a string and sometimes a
    list of validation errors, so a client ends up type-switching on it. `error` is always a
    string and `hint` is the next command to run when there is one.
    """

    error: str
    hint: str | None = None
