# Answer whole-video questions ("what is this about?", "who is taking part?")

## Context

Asking `@vector7-21aug-client-meeting what is this video about?` gets an abstention today.
That is not a bug — it is an unimplemented question class, and it fails at both ends of the
pipeline:

1. **Retrieval is point-seeking.** [answer.py:143](src/answer.py#L143) embeds the question and
   takes `top_k = 5` chunks. That meeting is 321 chunks of 25 s; five of them is 1.5 % of the
   video, chosen by similarity to a query with no semantic target. Whatever comes back is
   arbitrary.
2. **The prompt is deliberately extractive.** [answer_v1.md](prompts/answer_v1.md) rule 2 says
   decline unless a passage *states* the answer, rule 3 says "if you have to infer, estimate,
   or fill a gap, decline instead", rule 5 caps the reply at one or two sentences. No passage
   ever states what a video is about. **Abstaining is the current contract behaving correctly.**

Every one of the 15 dev pairs is extractive point-fact QA (`d001`–`d015`), and
`gate_phase2a` rewards exactly that separation. So the fix must add a second mode beside the
extractive one, not loosen the extractive one.

A third fact shapes the answer to "who is taking part": **there is no diarization anywhere in
this repo.** `runs/<stem>/transcript.json` segments carry `t_start`, `t_end`, `text` and
nothing else. Participants can only be named from what people say out loud.

Decisions taken: **explicit control** (no wording heuristics), **precomputed at ingest**, and
**name who is named, state the limit**.

## Design

One overview document per video, built once at index time, answered against on request.

```
make index  ──►  runs/<stem>/overview.json   { abstract, people[], topics[] }
                          │
POST /ask {question, mode:"overview"}  ──►  overview doc as context
                          │                 (~1k tokens, one small call)
                          └──►  Answer + citations ── ground() ── player seeks
```

Citations still work because every `people` and `topics` entry carries a `t_start`/`t_end`
taken from a real chunk. Those spans are handed to the existing
[`ground()`](src/answer.py#L234) as `RetrievedChunk`s, so grounding, `to_citation`,
`stream_url` and the click-to-seek player all run unchanged — no second citation path.

## Changes

**1. `schemas/overview.py` (new)** — the document the builder model must return, declared the
way [`schemas/answer.py`](schemas/answer.py) is so the schema handed to the model and the
schema it is validated against come off one declaration.

```
Overview: video_id, abstract (3-5 sentences), people[], topics[], speaker_labels=False
Person:   name, described_as, evidence {t_start, t_end}
Topic:    t_start, t_end, topic
```

**2. `prompts/overview_v1.md` (new)** — whole-transcript synthesis. Same `## System` /
`## User` split [`load_prompt`](src/answer.py#L203) already parses; `{{context}}` and
`{{question}}` are the only tokens it may use. It must state, in the file's own commentary and
in the system rules, that the transcript has **no speaker labels**: list only people who are
named or self-identify, attach the timestamp where the name is said, and never infer a
participant from tone or turn-taking. Synthesis across the whole video is permitted here —
that is the one rule that differs from `answer_v1.md`.

**3. `src/overview.py` (new)** — three functions:

- `build(video_id, cfg, meter) -> Overview` — reads every chunk for the video in `t_start`
  order and makes one model call. Read the chunks from Chroma with
  `collection.get(where={"video_id": ...})`, the same read-only pattern
  [`indexed_video_ids`](src/retrieve.py#L220) uses, so the overview describes the corpus that
  is actually answerable rather than a `runs/` directory that may be ahead of the index.
- `path_for(video_id)` — `runs/<stem>/overview.json`, where `<stem>` comes from
  [`local_file(video_id, SAMPLES)`](src/index.py#L128) (the function fixed earlier this
  session, so it resolves `bob-video` as well as `611_H8fGd3fCJbg`). Fall back to scanning
  `runs/*/chunks.json` for the `video_id` when the media is no longer on disk.
- `load(video_id)` / `as_chunks(overview)` — read the JSON back, and project its evidence
  spans into `RetrievedChunk`s for `ground()`.

The largest transcript here is ~12 k tokens, comfortably inside `openai/gpt-oss-120b`'s
context on Groq, so one pass is enough. Guard it: above a configured character ceiling, **fail
with a message naming the video and the ceiling** rather than truncating silently. Map-reduce
folding is future work, not this change.

**4. `src/index.py`** — call `overview.build` at the end of
[`index_video`](src/index.py#L73), after chunks are embedded, and add the path to the returned
report so `make index` prints it. Skip when `overview.json` exists and the transcript
`source_sha256` matches, so re-indexing is free; `--refresh` rebuilds.

**5. `Makefile`** — `make overview VIDEO=…` to rebuild one, mirroring `chunks:` / `index:`.

**6. `src/answer.py`** — an overview branch in [`answer()`](src/answer.py#L125). When
`mode == "overview"`: require exactly one video in `scope.video_ids`, load its overview,
render it as context, call the model with `prompts/overview_v1.md`, then run the *existing*
`ground()` against `as_chunks(overview)`. A missing overview is an `AnswerError` naming
`make overview VIDEO=…`, not a silent fall-through to the extractive path. Everything else in
`answer()` is untouched — same `Answer` schema, same `AnswerRun`, same abstention semantics.

**7. `schemas/api.py` + `src/api.py`** — add `mode: Literal["extractive","overview"] =
"extractive"` to `AskRequest`, and `mode` to `Provenance`. Extend the `extra="forbid"`
docstring rather than contradicting it: the rule it states is that **levers** (`top_k`,
`temperature`) stay in `config.toml` so a run is attributable to a config fingerprint. `mode`
is not a lever — it is part of what was asked, like the question itself — and it is recorded
in `provenance` so the run stays re-runnable.

**8. `src/mention.py`** — [`_label`](src/mention.py#L206) returns `""` for any video without a
manifest record, which is why `bob-video` and `vector7-21aug-client-meeting` show a blank
label in `/videos` and in the `@` picker. Fall back to the overview's first sentence when the
manifest has nothing. Small, and it fixes a visible gap on the same data.

**9. `web/app.js` + `web/styles.css`** — an **About this video** button on each row of the `@`
source menu ([`app.js:297-310`](web/app.js#L297-L310)), the one place every indexed source is
already listed. Clicking it posts `{question: "What is this video about, and who takes part?",
mode: "overview"}` scoped to that handle. The button renders only for rows whose source has an
overview, so it is never a control that looks live and is not — `GET /videos` gains a
`has_overview` boolean for that.

**10. Evals — `evals/overview/overview_v1.jsonl` (new dir), and `tests/gates/gate_overview.py`.**

> ⚠️ **Not `evals/dev/`.** Both [`recall_at_k`](src/retrieve.py#L192) and
> [`leakage.load_split`](src/leakage.py#L125) glob `evals/dev/*.jsonl`. A file dropped there
> silently joins the recall@5 denominator and the `gate_phase2` / `gate_phase2a` abstention
> rates, moving three recorded numbers that have nothing to do with this feature.

Two pairs per ingested video ("what is this about", "who takes part"), and the gate computes
and prints, in the style of `gate_phase2a`:

- every stored `overview.json` validates against `schemas.overview` — threshold 1.00
- every `people[].evidence` and `topics[]` span lands inside a real chunk of that video
- overview answers do **not** abstain, and every citation survives `ground()` unrepaired
- the cost of the gate, printed

## Verification

```
make overview VIDEO=samples/bob-video.mp4
python -c "import json;print(json.load(open('runs/bob-video/overview.json'))['abstract'])"

uv run pytest tests/unit -q                       # nothing extractive moved
uv run pytest tests/gates/gate_phase2a.py -q      # abstention selectivity unchanged
uv run pytest tests/gates/gate_overview.py -q -s  # the new numbers, printed

make api PORT=8077
curl -s :8077/ask -H 'content-type: application/json' \
  -d '{"question":"@vector7-21aug-client-meeting who is taking part?","mode":"overview"}'
```

Then in the browser: type `@`, pick the meeting, click **About this video**, and confirm the
answer names the people who are named, says it cannot attribute turns to voices, and that its
citations seek the player to the moments where those names are spoken.

`gate_phase2.py::test_score_at_least_70_percent` is **already red on a clean tree** (verified
this session by stashing) — it is a live-model gate and unrelated to this work. It should not
be read as a regression from these changes.

## The trade-off you are choosing

With an explicit control and no wording heuristic, a user who *types* "what is this video
about?" instead of clicking the button still gets today's abstention. To keep that from being
a dead end without adding auto-routing: when a scoped question abstains and that video has an
overview, the answer bubble offers a **Try: About this video** link. The user still clicks —
nothing is inferred from their wording — but the path is visible at the moment they need it.

---

## Implementation status — 2026-08-28

Paused part-way through, at the user's request. What exists on the branch:

**Done**

| File | State |
|---|---|
| `schemas/overview.py` | new — `Overview` / `Person` / `Topic` / `Span` / `StoredOverview`, `json_schema()` reusing `schemas.answer._inline_refs` and `_strictify` |
| `prompts/overview_v1.md` | new — the build prompt (whole transcript → document) |
| `prompts/overview_answer_v1.md` | new — the query-time prompt (document → `Answer`) |
| `src/overview.py` | new — `path_for`, `load`, `has_overview`, `chunks_for`, `render_transcript`, `render_overview`, `as_chunks`, `build`, `_validate_spans`, CLI |
| `src/answer.py` | `build_messages` gained `prompt=` / `context=`; `_ask` / `_groq_arm` / `_ollama_arm` gained `schema` / `schema_name` / `max_tokens`; `EXTRACTIVE` / `OVERVIEW` constants; `AnswerRun.mode`; `answer(..., mode=)` and `_overview_run` |
| `config.toml` | new `[overview]` section — `prompt`, `answer_prompt`, `max_tokens`, `max_context_chars` |

**Not started** — items 4, 5, 7, 8, 9, 10 of the plan above: the `index_video` hook, the
`make overview` target, the `mode` field on `AskRequest` / `Provenance`, the `_label`
fallback, the frontend control, and `evals/overview/` + `tests/gates/gate_overview.py`.
No tests have been written yet for what is done.

## What the first real build run showed

`uv run python -m src.overview 611 --config config.toml` produced a document, so the schema,
the prompt loading, the Chroma read and the disk write all work end to end. The **content was
poor**, and the reason is in the first line of its own output:

```
video 611  (107 chunks, bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M)
```

It ran on the **local 3B fallback, not `openai/gpt-oss-120b`** — `src.answer._ask` catches a
Groq failure and falls back to Ollama by design. Two symptoms followed, and both are what a
3B model does with a 50 000-character context rather than anything the prompt got wrong:

- `people` contained "Baroque period", "Victorian England", "20th century" and "Rome" —
  rule 2 of `overview_v1.md` says a person is someone the transcript *names*, and it was not
  followed.
- every span clustered in the last 300 seconds of a 1800-second video (1509 s – 1804 s),
  i.e. it copied timestamps off the passages nearest the end of its context.

**The next step is to find out why the Groq arm fell back**, then rebuild with `--refresh`
and judge the prompt on `gpt-oss-120b` output. The warning naming the cause goes to stderr
from `src/answer.py` and was cut off by the `tail` on that run; re-run and read the head of
stderr. Until that is known, nothing about the quality of `overview_v1.md` has actually been
measured — the run above measures the fallback model, not the prompt.

`runs/611_H8fGd3fCJbg/overview.json` currently holds that poor document. It is stale by
intent, not by accident: rebuild it with `--refresh` before reading anything into it.
