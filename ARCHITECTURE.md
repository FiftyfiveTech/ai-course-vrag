# VRAG Architecture

Two questions this document answers, which is all it tries to answer:

1. **What gets indexed?** — the shape of what goes into the vector store and how it gets there.
2. **How is a citation validated?** — the exact rule that decides whether a retrieved moment is correct.

For the full pipeline narrative, measures, and lever commentary, see the README.

---

## Pipeline at a glance

```
video file
    │
    ├─ ffmpeg ──► audio.wav (16 kHz mono s16le)
    │                 │
    │             ASR arm ──► transcript.json  (segments: text + start + end)
    │                               │
    │                          src/chunk.py ──► chunks  (time-windowed, overlapping)
    │                               │
    ├─ ffmpeg ──► frames/          │
    │             (0.2 fps, 768 px) │
    │                          src/embed.py ──► Chroma collection
    │                               │
    └─────────────────────────────  │
                               src/retrieve.py ──► top-k chunks
                                    │
                               src/answer.py ──► {answer, citations[], abstain}
```

Everything that feeds retrieval — the transcript, the chunks, the embeddings — is derived
from the video file. The frames directory exists for Phase 0 inspection and future vision
work; it is **not** indexed in the current pipeline.

---

## What gets indexed

### The unit: a chunk

`src/chunk.py` turns one video's transcript into a sequence of overlapping time windows.
Each window is one **chunk** — the unit the embedding and the retriever operate on.

```python
@dataclass(frozen=True)
class Chunk:
    video_id: str      # corpus id, e.g. "611"
    t_start:  float    # seconds — measured from the segments inside the chunk
    t_end:    float    # seconds — measured from the segments inside the chunk
    text:     str      # concatenated segment text for this window
```

`t_start` and `t_end` are **measured from the segments**, not copied off the window grid.
A window boundary is a grid line; a segment is never split across windows. So a chunk's
recorded range is always the range that its segments actually span.

The chunk id that enters Chroma is `{video_id}_{t_start:.3f}_{t_end:.3f}` — stable across
runs and unique within the collection as long as the chunk levers are not moved mid-build.

### The window parameters

| Lever | Default | Why |
|---|---|---|
| `chunk.window_s` | 25.0 s | widest setting where no chunk overhangs the ±30 s citation tolerance |
| `chunk.overlap_s` | 8.0 s | a sentence straddling a boundary lands whole in at least one chunk |

`window_s = 25.0` is a **measured value**. QA_SPEC §2 scores on `|t_start − t_ref| ≤ 30`.
A chunk overhangs its window at both ends by up to one segment length, so the effective
maximum is `window_s + 2 × (longest segment)`. At `window_s = 30` on dev video 181, two
of five chunks ran to 35.7 s and fell outside the tolerance. 25.0 is the widest setting
that keeps every chunk on dev citable.

### What the Chroma collection holds

One collection (`config.toml: embed.collection`, default `"vrag"`), cosine similarity,
768-dimensional vectors from `nomic-ai/nomic-embed-text-v1.5-GGUF:F16` on Ollama.

Each row:

| Field | Type | Contents |
|---|---|---|
| `id` | string | `{video_id}_{t_start:.3f}_{t_end:.3f}` |
| embedding | float[768] | F16 vector of the chunk text |
| document | string | the raw chunk text |
| `metadata.video_id` | string | corpus video id |
| `metadata.t_start` | float | window start in seconds |
| `metadata.t_end` | float | window end in seconds |

The store upserts, so re-running `make index` on an already-indexed video replaces its
rows rather than duplicating them. Use `--reset` after moving a chunk lever: lever changes
alter the chunk ids, so an index built across a lever change holds two generations of rows
and scores better than either alone.

### The embedding model matters more than the chunker

Swept on dev (346 chunks, 12 answerable pairs, recall@5):

| Weights | recall@5 |
|---|---|
| Q2_K (what `ollama pull hf.co/<repo>` gives with no tag) | 5/12 = 0.42 |
| **F16** | **11/12 = 0.92** |

Same code, same chunks, same questions. The `:F16` tag in `embed.model` is not
decoration — omitting it lets Ollama pick the smallest file in the repo, which here is
2-bit weights on a 137 M-parameter encoder.

---

## How a citation is validated

### The QA_SPEC §2 rule

A pipeline response is **correct** when all three conditions hold:

1. `abstain` is `false`
2. At least one citation has `video_id` equal to the ground-truth `video_id`
3. At least one citation on the correct video has `|citation.t_start − t_ref| ≤ 30 s`

A response is **incorrect** when any condition fails — including when the right video is
cited but the timestamp is more than 30 s off.

For **unanswerable** questions (QA_SPEC §3–4): the response is correct only if
`abstain` is `true`. A citation on a declined question is incorrect regardless of what it
says — there is no partial credit.

### Where the citation comes from

```
retrieved passage ──► model generation ──► schema validation ──► grounding ──► final citation
```

`src/answer.py` runs four steps after retrieval:

1. **Render context.** Top-5 chunks formatted into the prompt (`prompts/answer_v1.md`).

2. **Generate.** `openai/gpt-oss-120b` (Groq) or `bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M`
   (Ollama), constrained by `schemas/answer.py`'s JSON Schema via `response_format`.

3. **Validate.** `Answer.model_validate()` checks the reply against the same Pydantic model
   that generated the schema — the constraint on generation and the definition of valid cannot
   drift. Strict mode (`extra="forbid"`): an unexpected key means the prompt has drifted.
   A reply that sets `abstain=true` and includes citations is **schema-invalid** — QA_SPEC §4
   says that response is incorrect regardless, so the schema refuses it at source.

4. **Ground.** `ground()` matches every citation back to the retrieved set:

   | Citation state | Action |
   |---|---|
   | `video_id` not in retrieved set | Dropped — the citation was invented |
   | Timestamp a few seconds off a retrieved passage | Snapped to that passage's exact range |
   | Nothing survives and reply was not an abstention | Reply becomes an abstention |

   Grounding runs **after** validation, so the gate's schema-valid count is measured on
   what the model produced, not on a repaired copy of it. The last row only moves in the
   safe direction: a citation that goes nowhere is worse for a user than no answer.

### The schema: one declaration, two jobs

`schemas/answer.py` defines `Answer` and `Citation` as Pydantic models.
`schemas.answer.json_schema()` renders them into the JSON Schema sent to the model.
`Answer.model_validate()` checks the model's reply.

Both come off one declaration. The constraint on generation and the definition of valid are
the same object — the day they drift, the validator is wrong and nothing notices.

### Scoring (VRAG-021, the Phase 2 gate)

```
score = (correct_answerable + correct_abstentions) / 20
```

- `total_pairs` = 20 (17 answerable, 3 unanswerable)
- Gate threshold: **≥ 0.70** (≥ 14/20 correct)
- Run: `make gate-phase2` — leakage check runs first

---

## Files and modules

| Path | Role |
|---|---|
| `src/ingest.py` | ffmpeg: extract audio + frames |
| `src/transcript.py` | ASR: audio → segments with timestamps |
| `src/chunk.py` | segments → overlapping time-window chunks |
| `src/embed.py` | chunks → Chroma (embed + upsert) |
| `src/index.py` | CLI entry point: chunk + embed one video |
| `src/retrieve.py` | question → top-k chunks from Chroma |
| `src/answer.py` | chunks + question → Answer (generate + validate + ground) |
| `src/ask.py` | CLI + HTML demo renderer |
| `schemas/answer.py` | Pydantic models: Answer, Citation, json_schema() |
| `prompts/answer_v1.md` | Versioned prompt; ## System / ## User sections are the messages |
| `config.toml` | Every cost/quality lever — no defaults, a missing key raises |
| `evals/QA_SPEC.md` | Full citation-correctness contract |
| `tests/gates/gate_phase1.py` | recall@5 on dev, threshold ≥ 0.80 |
| `tests/gates/gate_phase2a.py` | schema-valid + abstention rate on dev |
| `tests/gates/gate_phase2.py` | accuracy on 20 heldout pairs, threshold ≥ 0.70 |
