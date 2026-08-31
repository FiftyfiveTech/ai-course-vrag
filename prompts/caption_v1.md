# caption_v1

The keyframe caption prompt — loaded by `src/caption.py`, never inlined in code (README,
`prompts/`). Versioned like every other prompt here: a change that moves a number goes in
`caption_v2.md` next to the measurement.

Two placeholders, substituted by literal replacement rather than `str.format`:

| token | filled with |
|---|---|
| `{{context}}` | which video and which second this frame was sampled at |
| `{{question}}` | `src.caption.CAPTION_TASK` |

The image itself is **not** in this file. It travels beside the text as an inline
`data:image/jpeg;base64,` part on the hosted arm and as an `images:` entry on the local arm —
that difference is in `src/caption.py`, and it is the only thing about the two arms that
differs, because the two-arm cost table needs the arm to be the only variable.

## Why the reply is plain text and not JSON

Everywhere else in this pipeline a model that returns a document is constrained at generation
time (`schemas/answer.py`, `schemas/overview.py`). Not here. The hosted arm (NVIDIA NIM) and
the local arm (Ollama, `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF`) do not offer the same strict
structured-output guarantees, and VRAG-023 exists to compare the two arms on cost. Asking them
differently would put "how it was asked" into a table that is supposed to isolate "what ran".

So the reply is text, and the one bit of structure needed is a sentinel: **`NO_TEXT`**. It is
normalised in `src.caption.parse_reply` into `Caption.has_text`, which is the yield column of
the cost table. `schemas/caption.py` refuses a caption whose flag and text disagree, so a
sentinel the model stops emitting fails loudly instead of quietly reporting 100% yield.

## The failure to design against

A vision model asked "what is in this image?" describes the image. That is the wrong output
twice over.

It is wrong for retrieval: "a slide with a bar chart on a blue background" contains none of
the words a question would be asked in, so the caption is unretrievable however good the
embedder is. The words on the slide are the searchable content; the fact that it is a slide is
not.

And it is wrong for grounding. This pipeline's rule is that a claim is either read off the
source or declined — `answer_v1.md` rules 2 and 3, and the reason `d013`-`d015` abstain
correctly. A frame shows text; it does not show what the text means. A caption that explains
the slide has inferred, and an inference stored next to a timestamp is indistinguishable from
something the video actually said.

The second failure is the frame that has nothing on it. A talking head holds still just as
well as a slide does, so the selection rule in `src/keyframes.py` will hand this prompt frames
with no text on them — by design, because the alternative is a second heuristic guessing at
text density. `NO_TEXT` is how those come back countable rather than as a paragraph describing
someone's face.

## System

You read text off a single still frame taken from a video. You are an OCR step, not a
describer.

Rules:

1. Transcribe the text that is visible in the frame, verbatim. Keep the wording, the numbers
   and the spelling exactly as they appear on screen.
2. Preserve the reading order and the line breaks of the original. A slide's structure — a
   title, then bullets — is part of what it says.
3. Transcribe only what is legible. Do not complete a word that is cut off at the edge of the
   frame, and do not guess at text that is too small or too blurred to read.
4. Do not describe the image. No layout, no colours, no fonts, no "a slide showing", no
   mention of people, rooms or backgrounds. If it is not written text, it is not your output.
5. Do not explain, summarise, translate or comment on the text. Transcribe it.
6. If the frame contains no readable text at all, reply with exactly `NO_TEXT` and nothing
   else. A frame of someone talking, a blank screen, or a photograph with no captions is
   `NO_TEXT`.
7. Reply with the transcription alone. No preamble, no "here is the text", no quotation marks
   around the whole thing, no trailing remark.

## User

{{context}}

{{question}}
