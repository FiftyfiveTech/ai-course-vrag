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
| `evals/dev/` | **Builder** tunes here. 15 cases. |
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

## Rules that live in this repo

`CLAUDE.md` carries the full contract. The short version:

- Models and datasets are named by **Hugging Face repo id**. The provider is only where it runs.
- **Zero spend.** A paid call is a STOP-and-ask, never a judgement call.
- A phase is done when its gate **prints the number**, not when the code looks right.
- Every PR is reviewed by the other person. `main` is protected; self-merges are the one thing
  the Friday retro always checks.
- Tasks come from the Odoo board via the `odoo-board` MCP server, not from this README.
