# evals/dev — provenance, and what the number from it is worth

15 pairs in `dev_v1.jsonl`: 12 answerable (3 per dev video) and 3 unanswerable. Format is
QA_SPEC §1; ids are `d001`…`d015` per QA_SPEC §8.

## Read this before quoting a recall number from this set

**These questions were written from the transcripts, not from watching the videos.** Every
pair carries `"derived_from": "transcript"` so it cannot be mistaken for a
video-verified label. `t_ref` is the start of the transcript segment the answer is spoken
in, and it was checked against that segment — not against the picture.

That makes the recall@5 measured on this set **optimistic, and not the Phase 1 result**.
The reason is structural, not a matter of care:

- Retrieval searches embedded transcript chunks. These questions were written by reading
  the same transcript. So the question and the target chunk share vocabulary that a real
  user's question would not, and the retriever gets a hint no real question carries.
- Nothing here tests the visual channel. A question whose answer is only on screen — a
  colour, a gesture, something written on a slide — cannot be written from a transcript, so
  the set contains none, and the pipeline is never asked for one.
- Video 181 is a 96-second music video whose only speech is song lyrics, over 6 indexed
  chunks. Its three questions are lyric lookups against a near-exhaustive index. They will
  hit almost regardless of how good retrieval is.

This is the failure mode the course exists to avoid: **1.0000 on self-written labels,
0.5195 on real ones.** A number from this set is a smoke test that the retrieval path is
wired up. It is not evidence that retrieval works.

## What would make this set count

Human labelling against the video, per QA_SPEC §6 — scrub to the timestamp, confirm the
answer is there, and write the question the way a user would ask it without having read the
transcript first. Then drop the `derived_from` field, because the label no longer needs the
caveat.

## Provenance of the transcripts these came from

| video | duration | ASR | segments | note |
|---|---|---|---|---|
| 181 | 95.8 s | `openai/whisper-large-v3-turbo` via Groq | 24 | music video; lyrics only, 30–88 s |
| 521 | 713.6 s | same | 122 | paper-cutting tutorial, single presenter |
| 611 | 1805.0 s | same, 3 split uploads | 256 | documentary narration |
| 701 | 3268.5 s | same, 6 split uploads | 1143 | TV drama, heavy dialogue; 2 segments dropped for a non-forward time range |

611 and 701 needed `transcript.max_upload_mb` splitting (VRAG-017) — Groq returns 413 above
its cap, and before that neither video could be transcribed at all.
