# VRAG Q&A Evaluation Spec

This document is the contract between the Evaluator (who writes labels) and the Builder
(who tunes the pipeline).  A builder must be able to predict every label from this spec
alone, without asking for clarification.

---

## 1. Q&A pair format

Each pair is a JSON object with these fields:

```jsonc
{
  "id":           "q001",          // string — unique, sequential, never reused
  "question":     "...",           // the question a user would ask
  "unanswerable": false,           // true → no answer exists in the corpus
  "video_id":     "611",           // corpus video_id; null when unanswerable
  "t_ref":        82.0,            // ground-truth timestamp in seconds; null when unanswerable
  "answer_note":  "..."            // one sentence: what the correct answer is and where
                                   // it appears in the video — for the labeler only,
                                   // never shown to the model
}
```

`t_ref` is the single point in the video where the answer is clearest.  For events that
span several seconds, pick the moment the key fact first appears on screen or in speech.

---

## 2. What a correct response looks like

The pipeline returns a JSON object matching `schemas/answer.py`:

```jsonc
{
  "answer":   "...",              // the answer text
  "citations": [
    {
      "video_id": "611",
      "t_start":  79.5,           // seconds
      "t_end":    85.0            // seconds
    }
  ],
  "abstain":  false               // true → the system declines to answer
}
```

A response is **correct** when **all three** of the following hold:

1. `abstain` is `false`
2. At least one citation has `video_id` equal to the ground-truth `video_id`
3. At least one citation with the correct `video_id` has `t_start` within **±30 s** of `t_ref`

Condition 3 is checked as: `|citation.t_start − t_ref| ≤ 30`.

A response is **incorrect** when any of those three conditions fails, including when the
right video is cited but the timestamp is more than 30 s off.

---

## 3. What makes a question unanswerable

A question is **unanswerable** when the answer cannot be derived from the 10 corpus
videos.  Label it `unanswerable: true, video_id: null, t_ref: null` when:

- The fact is not mentioned or shown in any of the 10 videos (the most common case).
- The question asks about something outside the video, e.g. "who directed this film?"
  when the director is never stated in the video.
- The question is ambiguous enough that any timestamp would be a guess.

Do **not** mark a question unanswerable just because it is hard — if the answer is
findable by a careful human watching the video, it is answerable.

---

## 4. Correct abstention

When `unanswerable` is `true`, the response is **correct** only if `abstain` is `true`.
If the system returns any citation for an unanswerable question, the response is incorrect
regardless of what the citation says.

---

## 5. Scoring (VRAG-021 gate)

```
score = (correct_answerable + correct_abstentions) / total_pairs
```

Where:
- `correct_answerable` — answerable pairs where the response satisfies all three
  conditions in §2
- `correct_abstentions` — unanswerable pairs where `abstain` is `true`
- `total_pairs` — 20 (the full sealed set)

The MVP gate threshold is **≥ 0.70** (≥ 14 / 20 correct).

Abstentions are worth the same as answerable pairs.  Missing all 3 abstentions while
getting every answerable pair right scores 17/20 = 0.85, which passes — but getting all
3 abstentions wrong while getting 14 answerable pairs right scores 14/20 = 0.70, which
just passes.  Both outcomes are acceptable at MVP.

---

## 6. Labeling rules for VRAG-012

These rules apply when writing the 20 held-out pairs.

**Split**: 17 answerable, 3 unanswerable.

**One answer per question.**  Do not write questions that have two valid answers in
different videos.  If you notice a question works for two videos, rewrite it to be
specific to one.

**t_ref must be verifiable.**  Before committing a pair, watch or scrub the video to the
timestamp and confirm the answer is visible or audible within ±30 s.  If you cannot
verify it in under two minutes, replace the question.

**Questions must be natural.**  Write the question as a user would type it, not as a
retrieval query.  "What colour is the performer's costume when they first appear?" not
"costume colour first appearance performer".

**No yes/no questions.**  The answer must name something specific (a time, a colour, an
action, a number) so that partial answers are obviously distinguishable from correct ones.

**Spread across videos.**  Aim for at least one answerable question per dev video and at
least one per heldout video.  Do not put all questions on one video.

**Unanswerable questions must be plausible.**  Write them so they look like they should
be answerable — a question about a topic the video touches on but does not actually
resolve.  "What is the performer's real name?" for a stage-play video is good.
"What is 2 + 2?" is not.

---

## 7. Worked examples

### Answerable — correct response

```
Question:  "What object does the performer place on the table at the start of the act?"
video_id:  "791"
t_ref:     47.0
```

Response:
```json
{
  "answer": "A red scarf",
  "citations": [{"video_id": "791", "t_start": 45.2, "t_end": 49.0}],
  "abstain": false
}
```

Verdict: **correct** — video_id matches, |45.2 − 47.0| = 1.8 ≤ 30.

---

### Answerable — wrong timestamp

Same question, response:
```json
{
  "answer": "A red scarf",
  "citations": [{"video_id": "791", "t_start": 120.0, "t_end": 125.0}],
  "abstain": false
}
```

Verdict: **incorrect** — |120.0 − 47.0| = 73 > 30.

---

### Answerable — wrong video

Same question, response:
```json
{
  "answer": "A red scarf",
  "citations": [{"video_id": "611", "t_start": 47.0, "t_end": 50.0}],
  "abstain": false
}
```

Verdict: **incorrect** — `video_id` does not match.

---

### Unanswerable — correct abstention

```
Question:  "What is the magician's real name?"
unanswerable: true
```

Response:
```json
{"answer": "", "citations": [], "abstain": true}
```

Verdict: **correct**.

---

### Unanswerable — incorrect (hallucinated citation)

Same question, response:
```json
{
  "answer": "David Chen",
  "citations": [{"video_id": "791", "t_start": 10.0, "t_end": 12.0}],
  "abstain": false
}
```

Verdict: **incorrect** — system should have abstained.

---

## 8. Files

| Path | Contents |
|------|----------|
| `evals/dev/` | Dev pairs — Builder tunes here; Evaluator writes them |
| `evals/heldout/` | Held-out pairs — Evaluator only; sealed Wednesday; tagged `heldout-v1` |

Both directories hold `.jsonl` files: one JSON object per line, each matching the format
in §1.

The gate (`tests/gates/test_no_leakage.py`) asserts `dev ∩ heldout = ∅` by content hash
before any gate result counts.
