# overview_v1

The whole-video prompt. Loaded by `src/overview.py` through the same
`src.answer.load_prompt` that reads `answer_v1.md`, so the `## System` / `## User` split and
the two placeholders work identically:

| token | filled with |
|---|---|
| `{{context}}` | every chunk of one video, in time order, rendered by `src.overview.render_transcript` |
| `{{question}}` | the instruction for this pass — `src.overview.BUILD_TASK` |

Everything outside those two H2 sections is commentary and is not sent.

## Why this prompt exists at all

`answer_v1.md` is extractive by design and declines anything it has to synthesise: "a passage
that is about the right topic is not the same as a passage that contains the answer... if you
have to infer, estimate, or fill a gap, decline instead." That rule is load-bearing — it is
what makes the abstention selective on `d013`–`d015` — and it is also why "what is this video
about?" comes back declined. No 25-second passage ever *states* what a video is about.

So this is a second prompt rather than a loosened first one. It runs **once per video, at
index time, over the whole transcript**, and synthesis is the job rather than the failure.
The extractive path is untouched and the gate that measures it does not move.

## The two failures this prompt is written against

**Inventing participants.** Nothing in this pipeline diarizes. Whisper returns
`t_start`/`t_end`/`text` and no speaker id, so the transcript records *what was said* and not
*who said it*. A model reading a meeting transcript will happily infer three participants
from turn-taking, and every one of those is a guess presented as a finding. Hence the rule
that a person is listed only when the transcript **says a name**, and the requirement that
each one carries the second the name is said — a name with no timestamp is a name that came
from somewhere other than the transcript, and the timestamp is what makes that checkable.

**Timestamps that are plausible rather than real.** Same failure `answer_v1.md` guards, same
reason: `src.overview.build` validates every span against the chunk list it just sent, and
`src.answer.ground` snaps a citation onto a real chunk before a user ever sees it. A model
that composes round numbers makes both of those do work they should not have to do.

## System

You are given the complete transcript of one video, in time order, as numbered passages. Each
passage header carries the seconds it covers.

Your job is to describe the video as a whole. Unlike a question-answering task, synthesising
across the whole transcript is exactly what is wanted here.

Rules, in order of importance:

1. The transcript is your only source. Do not use anything you know about the subject, the
   people, the company or the work from outside it. If the transcript does not support a
   statement, leave the statement out.
2. **The transcript has no speaker labels.** It records what was said, not who said it. So
   `people` is the list of everyone the transcript *names* — someone introduced, addressed by
   name, or naming themselves. Never infer a participant from turn-taking, tone, or the shape
   of the conversation, and never guess how many people are present. If nobody is named,
   `people` is an empty list, and that is a correct result.
3. Every person you list carries `evidence`: the `t_start` and `t_end` of the passage where
   the name is said, copied exactly as printed in that passage's header. Never adjust, round
   or invent these numbers. The same applies to every `t_start`/`t_end` in `topics`.
4. `abstract` is three to five sentences saying what the video is, what happens in it, and
   what it is for. Write it for someone deciding whether to watch. No preamble, no "this
   video", no mention of transcripts, passages or these instructions.
5. `topics` walks the video from start to finish in order, one entry per stretch that is
   actually about something different. Each `topic` is one clause, not a sentence. Use the
   real boundaries — a passage's `t_start` for the entry's start, a later passage's `t_end`
   for its end. Ten to twenty entries for an hour of video; fewer for a short one.
6. `described_as` says what the transcript says about that person — a role it states, a thing
   it says they did. If it says only the name, leave `described_as` empty. Do not describe
   them by how much they talk: that is the one thing an undiarized transcript cannot tell you.

Reply with a single JSON object and nothing else:

```json
{"abstract": "...",
 "people": [{"name": "...", "described_as": "...", "evidence": {"t_start": 0.0, "t_end": 0.0}}],
 "topics": [{"t_start": 0.0, "t_end": 0.0, "topic": "..."}]}
```

## User

The complete transcript of this video, in time order:

{{context}}

{{question}}
