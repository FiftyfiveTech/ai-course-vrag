# overview_merge_v1

The reduce step of the overview fold — loaded by `src/overview.py:_merge`, never inlined in
code (README, `prompts/`). Versioned like every other prompt here: a change that moves a
number goes in `overview_merge_v2.md` next to the measurement.

Two placeholders, substituted by literal replacement rather than `str.format`, because the
prompt below contains JSON braces:

| token | filled with |
|---|---|
| `{{context}}` | the per-window abstracts, in order, each labelled `[part n of N]` |
| `{{question}}` | `src.overview.MERGE_TASK` |

## Why this prompt exists at all

No real transcript fits one call on Groq's free tier. The arithmetic is on
`overview.max_context_chars` in `config.toml`: the tier meters **tokens per minute**, the
limit is 8 000, and building video 611 in one pass asked for 17 152. So the transcript is
folded — each window summarised on its own, then the partials merged.

`people` and `topics` are merged **in code**, not here. Every span in the stored document has
to be a span off a real chunk, and a model asked to reconcile two documents will adjust a
timestamp to make them line up. This prompt is therefore given **no timestamps at all**: it
cannot corrupt a citation because it never sees one.

The abstract is the one field that needs a model. Eight window abstracts stapled together is
not an abstract of a video — it is eight abstracts.

## The failure to design against

Each part was written by a model that could see only its own window, so each one introduces
the video as though it were the whole thing ("This video is about...", "The speaker then
explains..."). Concatenating that reads as a video that restarts eight times. The merged
abstract has to describe one continuous thing, which means dropping the seams rather than
narrating them — no "in the first part", no "the video then moves on to".

The other failure is padding. A part that says little should contribute little; a merged
abstract that gives equal weight to every part because there were equal numbers of them
describes the *fold*, not the video.

## System

You are given abstracts of consecutive parts of a single video, in order. Write one abstract
of the whole video.

Rules, in order of importance:

1. Use only what the parts say. They are your only source. Do not add background, context, or
   anything you happen to know about the subject, the people, or the work discussed.
2. Write about the video as one continuous thing. Never mention parts, sections, windows,
   segments, or the order they arrived in. Phrases like "in the first part", "the video then
   moves on to", or "the final section covers" describe how this text was assembled and not
   what the video is.
3. Three to five sentences. Say what the video is, what happens in it, and what it is for.
4. Weight by substance, not by position. If six of eight parts describe one long discussion,
   the abstract is mostly about that discussion. A part that says little earns little.
5. Do not include timestamps, durations, or any numbers presented as seconds. You have not
   been given any, and inventing one would put a moment in front of a reader that the video
   may not have.

Reply with a single JSON object and nothing else:

```json
{"abstract": "..."}
```

## User

Abstracts of the parts of this video, in order:

{{context}}

Task: {{question}}
