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
| `prompts/` | Versioned prompt files (`extract_v1.md`, `extract_v2.md`, …). Never inline a prompt in code. |
| `schemas/` | Pydantic models. Structured output is validated, not parsed by hand. |
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

## Rules that live in this repo

`CLAUDE.md` carries the full contract. The short version:

- Models and datasets are named by **Hugging Face repo id**. The provider is only where it runs.
- **Zero spend.** A paid call is a STOP-and-ask, never a judgement call.
- A phase is done when its gate **prints the number**, not when the code looks right.
- Every PR is reviewed by the other person. `main` is protected; self-merges are the one thing
  the Friday retro always checks.
- Tasks come from the Odoo board via the `odoo-board` MCP server, not from this README.
