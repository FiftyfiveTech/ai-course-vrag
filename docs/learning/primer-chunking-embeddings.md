# Chunking, embeddings, and the cost of overlap

**VRAG-018.** A concept primer for the video-RAG build, written against measurements taken on
this repo's dev set rather than against the received wisdom about chunk sizes. The interactive
version — same numbers, with the levers movable — is [`coach.html`](coach.html).

The headline, before the reasoning, because it is not what the received wisdom predicts:

> On this dev set the chunking levers barely move recall. Six of the seven window widths
> measured — 12, 15, 20, 25, 30 and 45 seconds — return the **identical** recall@5 of 0.9167.
> What the window actually buys is index size: 1300 chunks and 9.8 MB at 12 s, 114 chunks and
> 1.9 MB at 60 s. The one lever that moved recall in this project moved it by 0.50, and it was
> not a chunking lever at all — it was the quantisation tag on the embedding model.

---

## Where these numbers come from

```bash
make sweep        # 12 points, ~22 min of local embedding, $0.0000
make coach        # rebuild coach.html from what the sweep wrote
```

| | |
|---|---|
| Grid | 12 points: the window walked at the shipped overlap (8 s), the overlap walked at the shipped window (25 s) |
| Scored on | `evals/dev` — 12 answerable pairs across the 4 dev videos. **Never `evals/heldout`** |
| Hit rule | QA_SPEC §2: a result with the right `video_id` and \|`t_start` − `t_ref`\| ≤ 30 s |
| Embedder | `nomic-ai/nomic-embed-text-v1.5-GGUF:F16` on Ollama, 768-dim |
| Transcripts | `openai/whisper-large-v3-turbo` via Groq, read from the `runs/` cache — no ASR call was made |
| Cost | `12 points in 1334.9s · $0.0000` |
| Raw | [`data/chunking_sweep.json`](data/chunking_sweep.json) — counts only, never chunk text |

The sweep never writes `config.toml` and never touches `./chroma`. Phase 1 was graded at
`window_s = 25.0` / `overlap_s = 8.0`; a sweep that edited the levers in place would silently
re-tune a gate that has already been passed. Each point gets its own store under `runs/sweep/`.

One validity check worth stating up front: at the shipped setting the sweep produces 346 chunks
and recall@5 = 0.9167 — the number `make gate-phase1` recorded when Phase 1 was graded, reached
down a completely separate code path. Two independent paths to one number is the reason to trust
the other eleven rows.

That check also caught something while it was being made, and §6 is where it is written up: the
gate on this working copy now prints **0.8333**, not because any lever moved but because the
index has since acquired two videos that are not in the corpus.

---

## 1. What video RAG does, and the one thing video changes

The pipeline is four steps: a transcript is cut into **chunks**, each chunk becomes one
**vector**, a question becomes a vector, and the nearest chunks come back.

The thing that makes this different from document RAG is in the last step. The unit you
retrieve is also the unit you **cite** — a citation here is `(video_id, t_start, t_end)`, and
`t_start` is the start of the chunk that was retrieved. So the geometry of a chunk is two
decisions at once:

- **a retrieval decision** — what is inside the chunk decides what queries can find it;
- **a correctness decision** — where the chunk *starts* decides whether the citation lands
  inside the ±30 s tolerance.

A document chunker only has to worry about the first. That second constraint is the whole
reason `chunk.window_s` has a ceiling that is not a matter of taste, and §4 is about how this
sweep found the derivation of that ceiling to be wrong.

---

## 2. Embeddings, in exactly the amount you need to reason about chunk size

An embedding model maps a passage to a fixed-length list of numbers — 768 of them here. Two
passages that mean similar things land near each other; retrieval is "give me the chunks whose
vectors are closest to the question's vector."

Three consequences do all the work in this primer.

**One chunk is one point.** Everything inside the chunk is compressed into a single position.
A chunk covering one topic sits squarely on that topic. A chunk covering three topics sits at
their average, which may be near none of them. This — not a token limit — is the real argument
against very wide chunks, and it is why recall@5 finally breaks at a 60 s window.

**The encoder decides what "similar" means, and it is replaceable.** Nothing about the chunk
changes when you swap weights; the entire geometry does. In this repo, pulling the same model
at `Q2_K` instead of `:F16` moved recall@5 from 0.9167 to 0.4167 — 225 MB of weights, one tag
on one `ollama pull`, and a larger swing than every chunking lever on this page combined. That
sweep is in the [README](../../README.md#tag-the-quantisation-it-is-the-biggest-quality-lever-here).

**Nothing in the index knows why.** A vector store returns a distance, not a reason. When
recall is low, the index cannot tell you whether the chunk was badly cut, badly encoded, or
simply not the answer — which is why the ordered checklist in §7 exists and why guessing at
`window_s` first is the expensive way to debug.

The practical reading: the encoder is the **first** thing to verify and the **last** thing to
tune. The chunk levers are cheap to sweep and, on the evidence below, mostly inert.

---

## 3. Chunking vs recall: what the window actually buys

Overlap pinned at 8 s, window walked. `>tol` counts chunks longer than the 30 s citation
tolerance — see §4 for why that column is not the same as "wrong".

| window | hop | chunks | duplication | longest chunk | >tol | recall@1 | recall@5 | embed | store |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 s | 4 s | 1300 | 3.82× | 53.8 s | 20 | 0.67 | **0.9167** | 199.8 s | 9.8 MB |
| 15 s | 7 s | 812 | 2.85× | 53.8 s | 23 | 0.83 | **0.9167** | 139.9 s | 5.7 MB |
| 20 s | 12 s | 486 | 2.10× | 53.8 s | 49 | 0.83 | **0.9167** | 101.0 s | 3.8 MB |
| **25 s** | **17 s** | **346** | **1.79×** | **53.8 s** | **120** | **0.75** | **0.9167** | **82.8 s** | **3.0 MB** |
| 30 s | 22 s | 268 | 1.61× | 60.0 s | 214 | 0.83 | **0.9167** | 73.3 s | 2.8 MB |
| 45 s | 37 s | 159 | 1.36× | 80.9 s | 156 | 0.75 | **0.9167** | 69.1 s | 2.1 MB |
| 60 s | 52 s | 114 | 1.27× | 92.8 s | 112 | 0.58 | 0.8333 | 69.9 s | 1.9 MB |

Bold row is the shipped setting.

**recall@5 is flat across a 4× range of window widths.** 12 s and 45 s score the same. That is
not a suspiciously tidy result — it is what the per-pair detail says too: at every one of those
six settings, eleven of the twelve pairs hit and the twelfth (`d010`) misses. The same eleven,
the same one.

**Only the 60 s window breaks it**, and it breaks in the way §2 predicts: the vector stops being
about any one thing. Two pairs miss instead of one, and `d006`'s nearest right-video result is
54.5 s off the reference — the chunk containing the answer now starts nearly a minute before it.

**recall@1 wobbles between 0.58 and 0.83 with no trend.** Do not read that column as a ranking.
Twelve pairs means one pair is 0.0833; a column that moves by one or two pairs with no monotone
shape is measurement noise, and treating it as signal is how a lever gets tuned to fit a
coincidence.

**What does move monotonically is cost.** 12 s → 60 s is 1300 → 114 chunks, 199.8 s → 69.9 s of
embedding, 9.8 MB → 1.9 MB on disk. Over the flat range, the window is not buying recall. It is
buying index size, and the correct way to choose within a tie is the constraint in §4.

---

## 4. The ceiling on `window_s`, and the derivation this sweep falsified

`config.toml` and the README both justify `window_s = 25.0` like this: a citation points at the
chunk's start, the tolerance is ±30 s, so a chunk must not run much past 30 s. Because segments
are never split, a chunk overhangs its grid window at both ends, so the bound is

```
max chunk duration  ≤  window_s + 2 × (longest segment)
```

and — measured on dev video 181, whose longest whisper segment is 4.16 s — `25.0` was recorded
as *"the widest setting that keeps every chunk on dev citable."*

**Across all four dev videos that claim is false.** At the shipped setting, 120 of 346 chunks
are longer than 30 s and the longest is 53.8 s.

The formula was right. The input was one video. Longest ASR segment, per dev video:

| video | segments | longest segment |
|---|---:|---:|
| 181 | 24 | 4.16 s |
| 521 | 122 | 15.71 s |
| 611 | 256 | **29.98 s** |
| 701 | 1143 | 27.34 s |

With 611's 29.98 s segment the bound is `25 + 2 × 29.98 = 85.0 s`, not 33.3 s. The 53.8 s chunk
is a real one and it is the same chunk at every window width, which is the tell — it is not the
grid that made it, it is the segments:

```
chunk 611 [111.36 → 165.17]  = 53.81 s, three segments
    111.36 – 141.34   (29.98 s)
    141.36 – 148.36   ( 7.00 s)
    148.36 – 165.17   (16.81 s)
```

A segment is placed in a window if **any part** of it overlaps (`src/chunk.py::_in_window`), so
a long segment clipping a window's left edge drags `t_start` far backwards while another
clipping the right edge drags `t_end` far forwards. At a 12 s window that chunk is still 53.8 s
long — narrowing the window cannot fix a chunk whose length is set by its segments.

### Why this did not cost recall, and why it still matters

A chunk longer than the tolerance is **not** automatically a wrong citation. The rule is
\|`t_start` − `t_ref`\| ≤ 30, and `t_ref` lies inside the chunk, so the citation only fails when
the referenced moment sits more than 30 s *after the chunk's own start*. A 53.8 s chunk is
therefore a chunk that **can** be uncitable, for the references in its last 23.8 s. That is why
the column is headed `>tol` and not "wrong".

On these 12 dev pairs it never bit. So the shipped `window_s = 25.0` is fine — but the sentence
written next to it is not the reason it is fine, and that distinction is the entire point of
measuring. A number that is right for a reason you have disproved is a number that will move
the wrong way the next time something changes. The obvious next change is the ASR arm: swap
away from `openai/whisper-large-v3-turbo` and the segment lengths change, so the `>tol` column
has to be re-measured before any of this can be quoted.

---

## 5. The cost of overlap

Window pinned at 25 s, overlap walked.

| overlap | hop | chunks | duplication | words indexed | recall@1 | recall@5 | embed | store |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 s | 25 s | 234 | 1.22× | 17,249 | 0.67 | 0.8333 | 68.2 s | 2.2 MB |
| 2 s | 23 s | 255 | 1.32× | 18,649 | 0.58 | **0.9167** | 75.8 s | 2.7 MB |
| 5 s | 20 s | 293 | 1.53× | 21,729 | 0.50 | **0.9167** | 92.8 s | 2.7 MB |
| **8 s** | **17 s** | **346** | **1.79×** | **25,369** | **0.75** | **0.9167** | **82.8 s** | **3.0 MB** |
| 12 s | 13 s | 449 | 2.32× | 32,914 | 0.67 | **0.9167** | 142.6 s | 3.8 MB |
| 16 s | 9 s | 647 | 3.34× | 47,397 | 0.83 | **0.9167** | 195.3 s | 5.2 MB |

*Duplication = indexed characters ÷ transcript characters. The four dev transcripts hold 14,458
words between them; every row above indexes more than that, which is what the lever does.*

### What overlap is for

Exactly one failure: a sentence cut in half by a window boundary, present in two chunks and
whole in neither. Neither half embeds to the meaning the whole sentence had, so a question about
it matches nothing well.

**This sweep caught that failure in the act.** At `overlap_s = 0`, one extra pair misses —
`d003` — and the nearest result from the right video starts **35.0 s** from the reference,
against a tolerance of 30 s. Over the line by five seconds: a boundary landing in the wrong
place. Every overlap from 2 s upwards recovers it.

### What overlap costs

That one pair is the *entire* visible benefit on this set, and it is bought at 0 s → 2 s. The
shipped 8 s recovers nothing further that 2 s had not already recovered, and 16 s recovers
nothing beyond that. Meanwhile the bill rises the whole way: 1.22× → 3.34× duplication, 2.2 MB
→ 5.2 MB, 68 s → 195 s of embedding.

Duplication is predictable to first order — a passage falls in roughly `window ÷ hop` windows:

| | window ÷ hop | measured | excess |
|---|---:|---:|---:|
| overlap 0 s | 1.00× | 1.22× | 1.22 |
| overlap 2 s | 1.09× | 1.32× | 1.21 |
| overlap 5 s | 1.25× | 1.53× | 1.23 |
| overlap 8 s | 1.47× | 1.79× | 1.22 |
| overlap 12 s | 1.92× | 2.32× | 1.21 |
| overlap 16 s | 2.78× | 3.34× | 1.20 |

The excess column is flat at ~1.21, and it has a cause worth knowing:

> **`overlap_s = 0` does not mean no duplication.** It measures 1.22× — 22% of the corpus is
> still indexed twice. Segments are never split and membership is any-overlap, so every segment
> straddling a window boundary is placed in *both* windows even when the windows themselves do
> not overlap. The floor on duplication is set by segment length, not by this lever.

For wide windows the excess shrinks (1.10 at 60 s) because the overhang is a fixed number of
seconds against a longer window. `window ÷ hop` is the number to reason with; the measured value
is that times a corpus-specific overhang factor.

### The part where the dollars are missing

The sweep's total cost is **$0.0000** — 5,363 chunks embedded across 12 points, zero dollars.
That is an artefact of where the model runs, not evidence that overlap is free.
`nomic-embed-text-v1.5-GGUF:F16` is rated at `0.0` in `src/telemetry.py` because it runs on this
machine, so the cost meter correctly reports nothing.

Overlap's real price is in the three columns that are not dollars: **tokens** (25,369 indexed
words against 14,458 real ones), **seconds** (82.8 s of embedding against 68.2 s), and
**megabytes** (3.0 against 2.2). Move the embedder to a metered endpoint and the duplication
factor becomes the multiplier on the invoice, directly — 1.79× at the shipped setting, 3.34× at
16 s. Nothing else about the pipeline changes; the number just stops being zero.

That is the general shape worth carrying out of this project: a zero in a cost column often
means *this workload has not met a price yet*, and the quantity that will be priced is already
sitting in the table next to it.

---

## 6. How to read a recall number from 12 pairs without fooling yourself

Everything above is measured on 12 answerable pairs. Five things follow, and none of them are
optional caveats.

**The ruler has 12 marks.** One pair is 0.0833 of recall. Two settings that differ by one pair
are not ranked by this measurement — they are tied, and the tie-break has to be something that
*is* measured precisely: chunk citability, index size, embedding time.

**The labels are optimistic by construction.** [`evals/dev/README.md`](../../evals/dev/README.md)
is explicit: these questions were written *from the transcripts*, not from watching the videos.
Question and target chunk share vocabulary a real user's question would not, nothing tests the
visual channel, and video 181 is a 96-second music video whose three questions are lyric lookups
against six chunks. 0.9167 says the retrieval path is wired up correctly. It is not evidence
that retrieval works.

**One hard pair dominates the residual.** `d010` misses at all twelve settings, every time on
video 701 — a 54-minute TV drama, 191 chunks at the shipped setting, one line of dialogue. No
chunking lever recovers it, which is useful information: the remaining 0.0833 is not a chunking
problem and sweeping harder will not find it.

**The number belongs to the index, not to the chunker.** This was found by re-running the gate
after the sweep and getting a different number than the sweep had:

```
tests/gates/gate_phase1.py      recall@5 = 0.8333  (10/12 answerable dev pairs)  threshold 0.80
tools/sweep_chunking.py 25/8    recall@5 = 0.9167  (11/12)
```

Same levers, same questions, same embedder. The difference is what else is in the store. The
sweep builds a clean index of the 4 corpus videos — 346 chunks. The working `./chroma` had since
been given two videos that are not in the corpus at all, and holds 867:

| video | chunks | in the corpus? |
|---|---:|---|
| 181, 521, 611, 701 | 346 | yes — the dev split |
| `bob-video` | 200 | no |
| `vector7-21aug-client-meeting` | 321 | no |

(That table is a snapshot of one working copy. `./chroma` is a build artefact and gitignored, so
it is *not* reproducible from a clean clone — which is the point being made, not a caveat on it.)

Scored against the same live index but filtered to the four corpus videos, it is 0.9167 again.
The 521 extra chunks are not wrong, and nothing is misconfigured — they simply compete for the
five slots, and one pair (`d003`, the fragile one, three chunks of song lyrics) loses its place.

The lesson generalises past this repo: **recall@k is a property of the whole index**, so it falls
as the index grows even when retrieval has not got worse. A number quoted without saying what was
in the store when it was taken is not reproducible, which is why the sweep records the corpus it
built and why `make gate-phase1` prints the index contents before the number. It also means a
recall number measured on a 4-video pilot is an upper bound on the same pipeline over 50 videos.

**Tune on dev, three attempts, then escalate.** `evals/heldout/` is sealed and tagged
`heldout-v1`; the Builder never reads it, and `make leakage-check` asserts the two sets share no
labels before any gate result counts. A recall number tuned against the set it is scored on is
the failure this course exists to avoid.

---

## 7. When recall is low, check these in this order

The long form of this list — with what each check has actually caught in this repo — is on
[`coach.html`](coach.html). The order is the point: it runs cheapest-and-most-likely first.

1. **The embedder's quantisation.** `ollama list` — an untagged pull takes the repo's *smallest*
   file. Cost of getting this wrong here: 0.50 of recall.
2. **What else is in the index.** `make sources` lists every `video_id` the store holds. Content
   outside the corpus competes for the k slots; here it cost 0.0833 with no lever touched (§6).
3. **One generation of chunks in the index, or two.** Chunk ids encode the timestamps, so moving
   a lever leaves the old rows behind as orphans. `make index-dev INDEX_FLAGS=--reset` after any
   lever moves.
4. **`dev ∩ heldout = ∅`.** `make leakage-check`. No gate result counts until this passes.
5. **Wrong video, or right video at the wrong second?** `make gate-phase1` prints per-pair rank
   and Δt. These are different bugs; only the second one is about chunk geometry.
6. **Only then, move a window.** `make sweep-dry` shows what a lever does to the index in
   seconds; `make sweep` shows what it does to recall in twenty minutes.

---

## Glossary

| Term | Meaning here |
|---|---|
| **window_s** | How many seconds of video one chunk covers on the grid |
| **overlap_s** | How much of a window the next window repeats |
| **hop** | `window_s − overlap_s` — how far the grid advances |
| **duplication** | Indexed characters ÷ transcript characters. 1.00× = every word stored once |
| **recall@k** | Fraction of answerable pairs where one of the top k results has the right `video_id` and a `t_start` within ±30 s of `t_ref` (QA_SPEC §2) |
| **citation tolerance** | The ±30 s in that rule. Measured from the chunk's `t_start`, not its middle |
| **`>tol`** | Chunks longer than 30 s — chunks that *can* hold a moment more than 30 s after their own start, and so *can* produce an uncitable hit |
| **segment** | One ASR output unit. Never split by the chunker, which is why chunks overhang their windows |

---

## Reproduce all of it

```bash
make index-dev                # once: fetch, transcribe and cache the 4 dev videos
make sweep                    # the 12 points above — ~22 min, $0.0000
make coach                    # rebuild coach.html from the sweep
make gate-phase1              # the shipped setting, scored by the gate rather than the sweep
uv run pytest tests/unit/test_primer_numbers.py -v   # every number in this file, checked
```

The last one exists because a primer that quotes numbers is a primer that can drift from them.
`tests/unit/test_primer_numbers.py` parses the tables above out of this markdown file and
asserts each cell against `data/chunking_sweep.json`. Re-run `make sweep` and it will tell you
exactly which sentences in this document have gone stale.
