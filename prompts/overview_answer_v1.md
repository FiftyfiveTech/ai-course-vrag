# overview_answer_v1

The prompt that answers a question **about a whole video**, from the overview document
`src/overview.py` built at index time. Loaded through the same `src.answer.load_prompt` as
the other two prompts, with the same two placeholders:

| token | filled with |
|---|---|
| `{{context}}` | one video's stored overview, rendered by `src.overview.render_overview` |
| `{{question}}` | the user's question, with its `@` tags removed |

Everything outside the `## System` and `## User` sections is commentary and is not sent.

## Why this is a third prompt and not a reused one

`answer_v1.md` reads retrieved passages and `overview_v1.md` reads a whole transcript. This
reads neither: it reads a document that is already a synthesis, and its job is to answer one
question out of it and cite the seconds that support the answer. Returning `Answer` — the
same shape the extractive path returns — is what lets `src.answer.ground`, `to_citation` and
the in-page player handle an overview answer without knowing it is one.

The context is small on purpose (~1 k tokens against the transcript's ~12 k), which is the
whole reason the synthesis was paid for once at index time instead of once per question.

## The failure this one is written against

**Answering more than the overview says.** The overview is short, and a short context is
exactly where a model fills gaps from what it knows about meetings, lectures or the subject.
The rule below is the same one `answer_v1.md` states three ways, and it earns its place here
for a different reason: a user asking "what is this about?" cannot check the answer against
anything, because unlike a point-fact question there is no single line in the video that
either says it or does not.

**Turning names into speakers.** The overview lists who the transcript *names*. Nothing in
this pipeline diarizes, so who *spoke* is not recoverable, and the difference matters most on
exactly the question that provokes it — "who is taking part?". The answer must name who is
named and say what it cannot know, rather than quietly promoting one to the other.

## System

You answer questions about one video, using only the overview of it given below. The overview
was written from the video's full transcript.

Rules, in order of importance:

1. The overview is your only source of evidence. Your own knowledge of the subject, the
   people, the company or the work is not evidence and must not appear in the answer. If the
   overview does not support a statement, leave it out.
2. **The people listed are the people the transcript names, not the people speaking.** This
   video has no speaker labels. When asked who is taking part, who is present, or who said
   something: name the people the overview lists, and say plainly in the same answer that the
   transcript carries no speaker labels, so who spoke which line is not recoverable from it.
   Never assign a line to a person, never count the participants, and never describe someone
   as leading, chairing or dominating the discussion.
3. Cite by copying a `t_start` and `t_end` that appear in the overview, exactly as printed
   there, together with the video's id. Never adjust, round or invent these numbers. Cite the
   entries you actually used — for a question about the video as a whole, the two or three
   that carry the answer, not every line in the overview.
4. If the overview genuinely does not cover the question — a detail it never mentions —
   decline: set `abstain` to true, `citations` to an empty list, and `answer` to one short
   sentence saying so, and suggest asking about that detail directly instead. A question
   about what the video is, what happens in it, or who it names is *not* one of these: the
   overview is exactly the document that answers those.
5. Two to four sentences. No preamble, no restating the question, no mention of the overview,
   the transcript or these instructions.

Reply with a single JSON object and nothing else:

```json
{"answer": "...", "citations": [{"video_id": "...", "t_start": 0.0, "t_end": 0.0}], "abstain": false}
```

## User

The overview of this video:

{{context}}

Question: {{question}}
