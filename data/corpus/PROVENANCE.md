# Corpus provenance and licence — VRAG-004

The pilot corpus is **10 videos, 4 dev / 6 held-out**, listed in
[manifest.json](manifest.json). This file records where they came from and under what terms.

Regenerate: `make corpus` · Verify: `make corpus-check`

## What is in this repo, and what is not

**In:** ten pointers. For each video — Video-MME's index (`video_id`), the YouTube id and
url it is hosted at, its duration bucket, domain, and sub-category, plus our dev/held-out
label.

**Not in:** any video file, any audio, and **any of Video-MME's own questions or answers**.
Our Q&A pairs are written from scratch under `evals/` (VRAG-011, VRAG-012).

This is not a disk-space decision. See the licence below.

## Provenance chain

| | |
|---|---|
| Requested repo id | `lmms-lab/Video-MME` |
| Resolves to | `lmms-eval/Video-MME` — the org was renamed; the old id still redirects |
| Config / split | `videomme` / `test` |
| Revision pinned | `ead1408f75b618502df9a1d8e0950166bf0a2a0b` |
| File actually read | `videomme/test-00000-of-00001.parquet` (405 KB on the wire) |
| Read how | `datasets.load_dataset(..., streaming=True)` |
| Files **not** read | 20 × `videos_chunked_*.zip`, **101.0 GB** |
| Upstream project | [MME-Benchmarks/Video-MME](https://github.com/MME-Benchmarks/Video-MME) · paper [arXiv:2405.21075](https://arxiv.org/abs/2405.21075) |

The revision sha is the anchor: the manifest carries no generation timestamp, so
`make corpus-check` re-streams that revision and asserts the file is reproduced
byte-for-byte. If upstream re-publishes the dataset, the check fails loudly instead of
drifting.

## Licence

The HF dataset card declares **no `license:` field**, and neither the HF repo nor the
upstream GitHub repo ships a `LICENSE` file. The terms are stated in prose in the upstream
README ([§ Dataset](https://github.com/MME-Benchmarks/Video-MME#-dataset)), verbatim:

> Video-MME is only used for academic research. Commercial use in any form is prohibited.
> The copyright of all videos belongs to the video owners.
> If there is any infringement in Video-MME, please email videomme2024@gmail.com and we
> will remove it immediately.
> Without prior approval, you cannot distribute, publish, copy, disseminate, or modify
> Video-MME in whole or in part.
> You must strictly comply with the above restrictions.

**How that binds us.**

- *"only used for academic research … commercial use prohibited"* — this is
  non-commercial coursework, which the terms permit.
- *"cannot distribute, publish, copy … in whole or in part"* — so the repo holds pointers,
  not content. This is why the acceptance criterion says *nothing bulk-downloaded*: pulling
  the 101 GB of `videos_chunked_*.zip` into a shared repo would be redistribution, and
  copying the benchmark's Q&A rows in would be copying it "in part".
- *"copyright of all videos belongs to the video owners"* — each video stays the property of
  its original YouTube uploader. Anyone reproducing this work fetches each video themselves
  from the recorded url, under YouTube's terms, and keeps it out of version control.

`.gitignore` blocks video and audio extensions so a local working copy cannot be committed
by accident. `tests/test_corpus.py` asserts the manifest carries no media and no benchmark
Q&A text.

## How the 10 were chosen

Video-MME's test split is 2700 QA rows over 900 videos — 300 each `short` / `medium` /
`long`, across 6 domains and 30 sub-categories. Selection is deterministic (no seed, sorted
tie-breaks) so the split is re-derivable rather than merely asserted:

1. Collapse the streamed rows to one record per `videoID`.
2. Fill a per-duration quota — **3 short / 3 medium / 4 long**. Long videos get the extra
   slot: they are where chunking and retrieval actually get hard, so the corpus should not
   be mostly short clips.
3. Inside each bucket, walk domains **least-used-first counting across all buckets**, and
   prefer a video whose sub-category is still unused; lowest `video_id` breaks ties.
4. Assign each bucket's dev slots (**1 short / 1 medium / 2 long**) to the domains least
   represented in dev so far.

Steps 3 and 4 are what make both halves varied rather than just the whole. The measured
result:

| | videos | domains | durations |
|---|---|---|---|
| dev | 4 | 4 | short, medium, long |
| held-out | 6 | 6 | short, medium, long |
| **total** | **10** | **6 of 6** | 3 short, 3 medium, 4 long — 10 distinct sub-categories |

Both halves span all three durations on purpose: a held-out set with no long videos would
not test the thing most likely to break.

## Relationship to the blind-labelling rule

The dev/held-out **video** split is public — it is in this manifest, and the Builder has to
know which six videos are off-limits in order to avoid tuning on them. What gets sealed is
the **held-out Q&A labels** under `evals/heldout/` (VRAG-012, tagged `heldout-v1`), which
the Builder never reads. `tests/gates/test_no_leakage.py` (VRAG-013) enforces that
separately, by content hash.
