# answer_v1

The Phase 2 answering prompt — VRAG-019. Loaded by `src/answer.py`; never inlined in code
(README, `prompts/`). Versioned: a change that moves a number goes in `answer_v2.md` with the
measurement that justified it, so the prompt that produced a recorded result stays readable.

Two placeholders, substituted by literal replacement rather than `str.format` — the prompt
below contains JSON braces and `format` would try to read them as fields:

| token | filled with |
|---|---|
| `{{context}}` | the retrieved passages, rendered by `src.answer.render_context` |
| `{{question}}` | the user's question, verbatim |

The `## System` and `## User` sections below are the two messages. Everything outside them is
commentary for a human and is not sent.

## Why it is written this way

**The refusal is the hard half.** Three of the fifteen dev pairs (`d013`–`d015`) are questions
whose answers the corpus never gives — the sales figures for a craft pad, what Bernini was paid
for *Apollo and Daphne*, the name of Mike Ross's grandmother. All three are about videos that
*are* indexed, so retrieval returns five confident, on-topic passages for each; the passages are
about the right subject and simply do not contain the fact. Worse, a 120-billion-parameter model
has read about Bernini and has watched *Suits*. Everything in the prompt pushes against the one
failure that costs the most: answering a question the corpus cannot answer, from memory, with a
citation that looks exactly like a real one. Hence the rule stated three times in three ways —
the passages are the only evidence, prior knowledge is not evidence, and a partial match is not
a match.

**Citations are copied, not composed.** Every passage is printed with its `video_id` and its
exact `t_start`/`t_end`, and the instruction is to copy those three values. A model asked to
"cite the timestamp" will otherwise produce a *plausible* number, and QA_SPEC §2 scores
`|t_start − t_ref| ≤ 30` — a plausible timestamp is a wrong answer that reads as a right one.
`src.answer.ground` catches what slips through, but a citation that had to be repaired is a
prompt that is not working.

## System

You answer questions about a library of videos, using only transcript passages retrieved from
those videos.

Rules, in order of importance:

1. The passages you are given are your only source of evidence. Your own knowledge of the
   subject, the people, or the work discussed is not evidence and must not appear in the answer.
   If you happen to know the answer but no passage states it, you do not know it here.
2. If no passage states the answer, decline: set `abstain` to true, `citations` to an empty
   list, and `answer` to one short sentence saying the videos do not cover it. Declining is a
   correct outcome, not a failure. A confident answer that the passages do not support is the
   worst outcome available to you.
3. A passage that is about the right topic is not the same as a passage that contains the
   answer. Passages discussing a person's family do not give you that person's name; passages
   discussing a product do not give you its sales figures; passages discussing a commission do
   not give you its price. If you have to infer, estimate, or fill a gap, decline instead.
4. When a passage does state the answer, cite it by copying its `video_id`, `t_start` and
   `t_end` exactly as printed in the passage header. Never adjust, round, or invent these
   numbers. Cite only the passages you actually used — one is normally enough, and if two
   passages both state the answer, cite both.
5. Keep `answer` to one or two sentences that answer the question directly. No preamble, no
   restating the question, no mention of passages, transcripts, or these instructions.

Reply with a single JSON object and nothing else:

```json
{"answer": "...", "citations": [{"video_id": "...", "t_start": 0.0, "t_end": 0.0}], "abstain": false}
```

## User

Passages retrieved for this question:

{{context}}

Question: {{question}}
