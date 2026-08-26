# ai-course-template

Starting scaffold for the FiftyFive AI engineering course. One repo per track, grown phase by
phase by hand.

## Use it

```bash
gh repo create FiftyfiveTech/ai-course-<track> --template FiftyfiveTech/ai-course-template --public --clone
cd ai-course-<track>
make setup
```

Then, in order: replace `<TRACK>` in `CLAUDE.md` and `pyproject.toml`, protect `main`, add the
other person as a collaborator with push access.

## Layout

| Path | Holds |
|---|---|
| `src/` | The system. Small modules, one job each. |
| `config.toml` | Every cost or quality lever — frame sampling rate, audio target, sample spec. Read only by `src/config.py`, which refuses to default a missing lever. |
| `samples/` | Sample videos, **generated or fetched, never committed**. `make sample` writes a synthetic clip; `make sample-real VIDEO_ID=…` pulls one dev video from its manifest url. |
| `runs/` | Ingest output, one directory per video: `audio.wav`, `frames/`, `media.json`. Gitignored. |
| `data/corpus/` | The 10-video pilot corpus: **pointers only**, never media. `manifest.json` + `PROVENANCE.md` (licence, provenance, how the split was chosen). |
| `prompts/` | Versioned prompt files. `answer_v1.md` is the Phase 2 answering prompt; its `## System` / `## User` sections are the messages, the rest is commentary. Never inline a prompt in code. |
| `schemas/` | Pydantic models. `answer.py` is the `{answer, citations[], abstain}` contract — one declaration, used both to constrain generation and to validate the reply. |
| `evals/dev/` | **Builder** tunes here. 15 cases (`dev_v1.jsonl`): 12 answerable, 3 not. Written from the transcripts, not from watching — see [evals/dev/README.md](evals/dev/README.md) before quoting a number from them. |
| `evals/heldout/` | **Evaluator** only. Sealed Wednesday, tagged `heldout-v1`. The Builder never reads it. |
| `evals/QA_SPEC.md` | What a correct citation is (±30 s), what counts as unanswerable, how the gate scores. |
| `tests/gates/` | One script per phase gate. It prints the number; the number decides. |
| `STANDUP.md` | Daily log. Two minutes, append-only. |

## The sealed evaluation set

`evals/heldout/heldout_v1.jsonl` is the 20 pairs the MVP gate (VRAG-021) is scored on — 17
answerable, 3 unanswerable, at least one answerable question per corpus video. Every `t_ref`
was checked against the video it points at before the file was written; `answer_note` on each
pair records what was seen or heard and when.

| | |
|---|---|
| sha256 of `evals/heldout/heldout_v1.jsonl` | `74398cbae0956271962bde9a3b51b89db766da0ae1d65802c1c56a81ab0d1084` |
| Tag | `heldout-v1` |
| Contract | [evals/QA_SPEC.md](evals/QA_SPEC.md) |
| Check | `make heldout-check` |

The digest is the seal. `make heldout-check` re-hashes the file and compares it to the line
above, so a question edited after the tag was pushed fails the check instead of quietly
changing what the gate measures. It also re-derives the counts and the per-video spread from
the file rather than trusting this table.

**The Builder does not open this file.** The dev/held-out *video* split is public — it is in
`data/corpus/manifest.json`, and the Builder has to know which six videos to avoid — but the
held-out Q&A labels are the Evaluator's. `tests/gates/test_no_leakage.py` (VRAG-013) enforces
`evals/dev ∩ evals/heldout = ∅` by content hash before any gate result counts.

## The blind-labelling seal

`tests/gates/test_no_leakage.py` (VRAG-013) asserts `evals/dev ∩ evals/heldout = ∅` by content
hash. It runs first in `make gate`, because a recall or accuracy number measured after a held-out
label reached the dev set is a memorisation score, not a result.

| | |
|---|---|
| Compared | `id`, `question`, `answer_note` — sha256 of each, normalised for case, whitespace and typographic punctuation |
| Not compared | `video_id`. The **video** split is public (`data/corpus/manifest.json`) and QA_SPEC §6 asks for held-out questions on dev videos |
| Not caught | A held-out question rewritten in different words. No digest sees through a paraphrase — that one is on review |
| Check | `make leakage-check` (prints the intersection size), or `make gate` |

Id namespaces keep the two splits apart: held-out is `q001`…`q020`, dev is `d001`… (QA_SPEC §8).
Without that both would number from `q001` and the id check would fire on work that leaked nothing.

`evals/dev/` is still empty, so today the check passes **vacuously** and says so in its output. It
starts meaning something with the first dev case.

## Chunking

`src/chunk.py` (VRAG-014) turns one video's transcript into the units retrieval works on.
Windows are a fixed grid on the video clock — 0, `hop`, 2·`hop`, … where
`hop = window_s - overlap_s` — so the same transcript chunks identically on every run and on
either ASR arm. Every chunk carries the `video_id`, `t_start` and `t_end` a citation is built
from, and because segments are never split, that range is **measured from the segments in the
chunk** rather than copied off the window bounds.

```bash
make chunks VIDEO=samples/181_8np5YKYx3sU.mp4    # dumps the table; exits non-zero on a problem
```

| | |
|---|---|
| Levers | `chunk.window_s`, `chunk.overlap_s` in `config.toml`. No defaults — a missing one raises |
| Invariants | Every chunk has a forward, finite range that contains every segment in it; every segment is in ≥1 chunk; ids unique, order monotonic, nothing past the end of the video |
| Dropped on purpose | A window with no speech, and a window whose segments are exactly its predecessor's. Both are counted in the output, never silent |
| Free to re-run | The ASR result is cached in `runs/<video>/transcript.json` per source sha256, so sweeping `window_s` costs $0.00 and makes no model call |

`window_s = 25.0` is a measured value, not a preference. QA_SPEC §2 scores a citation on
`|citation.t_start − t_ref| ≤ 30`, so a chunk wider than 30 s can retrieve the right passage
and still be marked wrong. 30 s is *not* the ceiling that implies: a chunk overhangs its window
at both ends, so it runs to `window_s + 2 × (longest segment)`. On dev video 181 `window_s = 30`
produced a 35.7 s chunk — 2 of 5 past the tolerance. 25.0 is the widest setting where none are.

## Indexing

`src/index.py` (VRAG-017) is the step between chunking and retrieval: chunk a video, embed the
chunks, upsert them into the local Chroma collection. `embed_and_persist()` landed in VRAG-015
with no caller, so the Phase 1 gate had an index to score and no way to build one.

```bash
make index VIDEO=samples/181_8np5YKYx3sU.mp4   # one video
make index-dev                                  # every dev video, fetching what is missing
```

| | |
|---|---|
| Model | `nomic-ai/nomic-embed-text-v1.5-GGUF:F16` on Ollama, 768-dim, $0.00 |
| Idempotent | Chunk ids are `<video_id>_<t_start>_<t_end>` and the store upserts, so re-running a video replaces its rows |
| `--reset` | Drops the collection first. Use it after moving a chunk lever: the levers set the ids, so an index built across a `window_s` change holds two generations of rows and scores better than either |
| Refuses | A video whose chunks failed `verify()`. Indexing a chunk whose range does not hold its segments puts a wrong citation in the store, where the gate cannot see it |

The repo id carries `-GGUF` for a measured reason. `ollama pull hf.co/nomic-ai/nomic-embed-text-v1.5`
returns `400: Repository is not GGUF or is not compatible with llama.cpp` — that repo ships
Sentence-Transformers weights, and `src/embed.py` builds `hf.co/<repo id>`, so the configured
model could never have been pulled. The `-GGUF` repo is the same model from the same publisher,
converted. Same shape as the Groq whisper wire id: the HF repo id is the name, but only the
runnable variant is a name the runtime accepts.

### Tag the quantisation. It is the biggest quality lever here.

`ollama pull hf.co/<repo>` with no tag takes the repo's **smallest** file. For this repo that
is `Q2_K` — 2-bit weights on a 137 M-parameter encoder — and nothing anywhere says so. Swept
on `evals/dev` (346 chunks, 12 answerable pairs, recall@5 by the QA_SPEC §2 rule):

| Weights | Query prefix | recall@5 |
|---|---|---|
| `Q2_K` (what the untagged pull gives you) | none | 5/12 = 0.4167 |
| **`F16`** | **none** | **11/12 = 0.9167** |
| `F16` | `search_query: ` / `search_document: ` | 10/12 = 0.8333 |
| `Q2_K` | `search_query: ` / `search_document: ` | 2/12 = 0.1667 |

Same code, same chunks, same questions: **0.4167 → 0.9167 on 225 MB more weights.** A Phase 1
gate run before this sweep would have blamed the chunker, the window size, or the questions.

The second row pair is the one worth remembering. nomic documents task prefixes —
`search_document:` on what you index, `search_query:` on what you ask — so using them looks
like a free win. On F16 they **cost** 0.08, so the pipeline deliberately does not use them.
Neither result was predictable from the model card.

## The Phase 1 gate

`tests/gates/gate_phase1.py` (VRAG-017) computes recall@5 over `evals/dev` and asserts
**≥ 0.80**. A question is a hit when one of the top 5 results has the ground-truth `video_id`
and a `t_start` within ±30 s of `t_ref` — QA_SPEC §2, implemented once in
`src.retrieve.recall_at_k`.

```bash
make gate-phase1        # leakage first, then this gate alone
```

Five things run before the number, because each of them can void it:

| Precondition | Why it is not optional |
|---|---|
| `evals/dev ∩ evals/heldout = ∅` **and dev non-empty** | `recall = 0/0` could as easily be written 1.0. A green gate over no labels reads as "retrieval verified" |
| ≥1 answerable pair per dev video | The criterion is "on the 4 dev videos"; a 0.80 earned on one video does not clear it |
| The index holds chunks for all 4 dev `video_id`s | An un-indexed video is a miss the retriever did not cause |
| `retrieve.top_k == 5` | Pinned in the gate, not read from config — a gate whose k moves with the config it grades can be passed by editing the config |
| The gate's scoring agrees with `recall_at_k()` to 1e-9 | Two implementations of one definition disagreeing is a bug, and the Builder tunes against `recall_at_k` |

The gate prints per-question `HIT`/`MISS` with the rank and Δt, so a number under threshold says
which question missed and whether it was the wrong video or the wrong moment.

```
recall@5 = 0.9167  (11/12 answerable dev pairs)  threshold 0.80
```

Reached in **one** dev-tuned attempt, and the attempt was the quantisation tag above
(0.4167 -> 0.9167) — no chunk lever moved. The one remaining miss is `d010`, where the right
video is rank 5 but 98.8 s off the reference: one line of dialogue in a 54-minute drama,
against 191 chunks from that video alone.

Getting there also needed two things that were not retrieval at all. `evals/dev/` had no
labels and no board task that wrote them, and dev videos `611` and `701` could not be
transcribed — Groq returns `413 Request Entity Too Large` above its upload cap, and 16 kHz
mono s16le is 32 kB/s, so the arm topped out near 13 min of video. `transcript.max_upload_mb`
and the splitting behind it (VRAG-017) is what made 84 of the 98 dev minutes reachable.

Read the number with [evals/dev/README.md](evals/dev/README.md) next to it. **These dev labels
were written from the transcripts, not from watching the videos**, so each question shares
vocabulary with the chunk it is meant to find and 0.9167 is optimistic by construction. It
says the retrieval path is wired correctly and that F16 weights fixed a real defect. It is not
evidence that retrieval works on questions a user would actually ask. `d001` also hits at
exactly `dt = 30.0 s`, dead on the tolerance boundary — a rounding change either way flips it.

## Answering with citations

`src/answer.py` (VRAG-019) is the half of the pipeline retrieval cannot do: read the five
passages and either answer from them with a timestamp, or say the corpus does not cover it.

```bash
make answer Q="how old was Bernini when he met the Pope?"
make answer-dev          # every evals/dev pair, with the schema-valid tally
```

```
Q: How old was Bernini when he was first presented to the Pope?
   A: He was eight years old.
      cite video 611 21.8s-47.8s
```

Four steps, and each is somewhere a wrong answer can come from:

```
retrieve  ->  render_context  ->  the model  ->  validate  ->  ground
```

| | |
|---|---|
| Contract | `schemas/answer.py` — `{answer, citations[{video_id, t_start, t_end}], abstain}`, the shape QA_SPEC §2 describes |
| Prompt | `prompts/answer_v1.md`, versioned on disk, never inlined. Its `## System` / `## User` sections are the two messages; everything else in the file is commentary and is not sent |
| Model | `openai/gpt-oss-120b` on Groq's free tier, `temperature = 0.0` — which removes the sampling variance and, measured, **not** the run-to-run variance |
| Levers | `answer.arm`, `answer.model`, `answer.prompt`, `answer.temperature`, `answer.max_tokens` |

### One declaration, two jobs

`schemas.answer.json_schema()` renders the Pydantic model as the JSON Schema handed to the
model in Groq's strict `response_format` (and Ollama's `format`), and
`Answer.model_validate` checks the reply. Both come off one declaration, so the constraint on
generation and the definition of valid cannot drift — the day they drift, the validator is the
one that is wrong and nothing notices. Strict mode needs two things Pydantic's default output
does not give: every property in `required` and no `$ref`, so the `Citation` definition is
inlined and every object is closed.

What the model refuses is a decision, not a type check. `extra="forbid"` — an unexpected key
means the prompt has drifted, and dropping it hides that. Time ranges must run forward, the
same rule `transcript.drop_impossible` applies on the way in. And `abstain` is coupled to
`citations`: QA_SPEC §4 says a citation on a declined question is incorrect *regardless of
what it says*, so a reply that declines and cites is not a near-miss to be repaired, it is
incoherent.

### Grounding: a valid citation can still point nowhere

Validation proves a citation is well formed. It cannot prove it is real — `{"video_id":
"611", "t_start": 412.0}` is a perfectly valid object for a question whose passages were all
from video 701. So `ground()` runs **after** validation and matches every citation back to
the retrieved set:

| What it finds | What it does |
|---|---|
| No retrieved passage from that `video_id` | Drops it. The citation was invented |
| A timestamp a few seconds off a retrieved passage | Snaps it onto that passage's exact range. Bad copying is not a new claim |
| Nothing survives, and the reply was not already an abstention | The reply becomes an abstention |

That last row only moves in the safe direction, and it is worth being explicit about why.
QA_SPEC §2 needs at least one citation on the ground-truth video for an answerable question
to count, so a reply whose every citation was invented is *already* scored incorrect and
abstaining cannot lose a point that was available. On an unanswerable question it converts a
hallucination into a correct abstention. And a citation that goes nowhere is worse for a user
than no answer — which is the reason that is not about the scoring rule.

Grounding runs after validation, never before, so the gate's schema-valid number is measured
on what the model produced rather than on a repaired copy of it.

## The VRAG-019 gate

`tests/gates/gate_phase2a.py`. **Not** the Phase 2 exit gate — that is VRAG-021, it scores
QA_SPEC §5 accuracy on `evals/heldout`, and it belongs to the Evaluator. This one asserts the
two things that have to hold before that number means anything.

```bash
make gate-phase2a       # leakage first, then this gate alone
```

```
schema-valid = 1.0000  (15/15 dev pairs)  threshold 1.00
abstentions = 3/3 planted unanswerable pairs  threshold 3/3
abstention rate: 1.0000 on 3 unanswerable pairs, 0.0833 on 12 answerable (1/12)  ceiling 0.25
selectivity = 0.9167  (an abstain-everything module scores 0.0000)
```

**The refusal is the hard half.** `d013`–`d015` ask for a craft pad's sales figures, what
Bernini was paid for *Apollo and Daphne*, and the name of Mike Ross's grandmother. All three
are about videos that *are* indexed, so retrieval returns five confident, on-topic passages
for each — the abstention has to come from reading them and noticing the fact is not there,
not from an empty result. And a 120-billion-parameter model has read about Bernini and has
seen *Suits*, so the corpus is not the only place an answer could come from. 3/3 is the number
that says it did not come from anywhere else.

**The third number is a rate, and that is a correction.** It was a per-pair assertion first
— zero abstentions among the answerable pairs whose ground-truth moment was retrieved — and
it passed, then failed the next run with nothing changed. Two findings came out of that, and
both are why the check is now a rate with a ceiling.

`openai/gpt-oss-120b` on Groq is **not reproducible at `temperature = 0.0`**. Six identical
calls for `d001` — same code, same config, same prompt — returned 4 answers, 2 abstentions,
and three different answer texts. The six lines are recorded in `config.toml` next to the
lever. Temperature 0.0 pins the sampler, not the arithmetic, so a gate the supervisor re-runs
and compares needs margin, and a threshold of exactly 0 on a 12-pair denominator has none.

And `d001` is a pair no answerer can win. The line asked about is "I'm gonna graduate" at
`t_ref = 30.0 s`, and **no retrieved passage contains it** — the chunk that does is outside
the top 5. What satisfies the QA_SPEC §2 hit rule instead is a chunk from video 181 starting
at `0.0 s` **whose entire text is `.`**, plus a lyric chunk at `51.0 s`; both are inside the
±30 s tolerance, so the rule is met by passages that do not hold the answer. `make gate-phase1`
duly scores `d001` a HIT at `dt = 30.0 s`, exactly on the boundary flagged in the VRAG-017
standup. So abstaining is honest and answering would be a guess that §2 scores correct anyway
— a per-pair rule demanding an answer would be demanding the guess.

Two things follow, and neither is fixed here. The index holds a **text-empty chunk**, which
`src/chunk.py` is supposed to drop and does not because `.` is a non-empty segment: that is a
VRAG-014/017 finding, and fixing it would move a recorded Phase 1 number that belongs to
another gate. And a timestamp-only hit rule cannot tell "the answer was retrieved" from "a
passage near the right second was retrieved" — that is the Evaluator's contract to change, not
the Builder's. Both are on the card.

What the rate does measure is **selectivity**: the refusal fires on the questions the corpus
cannot answer and not on the ones it can. Accuracy is a different number, it is QA_SPEC §5,
and VRAG-021 scores it on `evals/heldout`.

Three passes of the same code, config and prompt over all 15 dev pairs, to size the ceiling
against that variance rather than guess it:

| pass | schema-valid | abstain (unanswerable) | abstain (answerable) |
|---|---|---|---|
| 1 | 15/15 | 3/3 | 1/12 |
| 2 | 15/15 | 3/3 | 1/12 |
| 3 | 15/15 | 3/3 | 2/12 |

`d001` is the only pair that moves. `d010` abstains in all three — it is the retrieval miss.
Every other pair answers in all three. The two asserted numbers are stable; the third has a
one-pair wobble, and the 0.25 ceiling (3 of 12) is one pair of margin over the worst pass.


### The local arm is a different model, and its number is different

`answer.arm = "ollama"` runs the same four steps with no key and no network. It is not a
formality — it works, and it is what a run with no credential falls back to:

```
make answer Q="how old was Bernini when he met the Pope?"   # arm = "ollama"
   A: Bernini was eight years old when he was first presented to the Pope.
      cite video 611 21.8s-47.8s
```

But the README promise about re-measuring rather than assuming applies to it, so it was
measured. `bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M` over all 15 dev pairs:

| | `openai/gpt-oss-120b` (Groq) | `Llama-3.2-3B-Instruct` (local) |
|---|---|---|
| schema-valid | 15/15 | **12/15** |
| abstained on the 3 unanswerable | 3/3 | **1/3** |
| abstained on an answerable pair | 1/12 | 0/12 |
| wall time, 15 questions | ~85 s | ~50 min |

All three schema failures are **the same failure**, and it is the one rule in
`schemas/answer.py` that is a judgement call rather than a type check:

```
d015  SCHEMA-INVALID  abstain is true but 1 citation(s) were returned
      raw: {"answer": "The passage does not mention the name of Mike Ross's grandmother.",
            "citations": [{"video_id": "701", "t_start": 0.0, "t_end": 0.0}], "abstain": true}
```

The 3B model declines correctly and then attaches a zero-timestamp citation anyway. Under
QA_SPEC §4 that response is incorrect regardless of what the citation says, so the schema
refuses it — which means two semantically correct refusals are counted as invalid rather than
as abstentions. That is a **deliberate choice and a reversible one**: moving the
`abstain ⇒ no citations` rule out of the schema and into `ground()`, which already normalises
answers into abstentions, would score the local arm 15/15 with 3/3 abstentions and 2 repairs.
It is not done here for one reason — it would make the VRAG-019 criterion (*schema-valid on
100% of dev*) easier to pass, and the gate is measured on the configured arm, where the strict
rule already scores 15/15. Flagged for review rather than decided quietly.

The useful part is that the strict rule found a real behaviour difference between two models
that both claim to honour a JSON schema. Constrained decoding gets the *shape* right in both;
only the larger model gets the *coherence* right.


| Precondition | Why it is not optional |
|---|---|
| `evals/dev ∩ evals/heldout = ∅` | `tests/gates/README` — no gate result counts until this holds |
| dev has at least one planted unanswerable pair | Otherwise half the criterion cannot be measured at all |
| The index covers all 4 dev videos | With nothing indexed, every abstention is free and 3/3 says nothing |
| The prompt resolves to a file, and its sha256 is printed | Which prompt produced the number, recorded next to it |
| The wire schema still carries the three fields the card names | If it does not, "schema-valid" is measuring something else |
| Every returned citation names a retrieved passage | The property `ground()` exists for, asserted rather than assumed |

15 hosted calls and 15 local embeddings, ~1.5 min, **$0.00** on the free tier — two orders of
magnitude slower than the other gates in that directory.

## Rules that live in this repo

`CLAUDE.md` carries the full contract. The short version:

- Models and datasets are named by **Hugging Face repo id**. The provider is only where it runs.
- **Zero spend.** A paid call is a STOP-and-ask, never a judgement call.
- A phase is done when its gate **prints the number**, not when the code looks right.
- Every PR is reviewed by the other person. `main` is protected; self-merges are the one thing
  the Friday retro always checks.
- Tasks come from the Odoo board via the `odoo-board` MCP server, not from this README.
