# Teams notetaker — the backlog

Thirteen tickets, VRAG-026 … VRAG-038, for turning this repo into a Teams meeting notetaker
that produces minutes.

**Why this is a file and not the board.** The board is the source of truth for what to do
next (CLAUDE.md) and these belong there. They are here because they cannot be written from
here: the `odoo-board` MCP server is read-and-progress only — `my_tasks`, `task`, `note`,
`start`, `request_review`, `set_done` — and has no create tool. So this file is a staging
area, not a competing plan. Each ticket below is written to be pasted into Odoo unchanged:
the **Acceptance** block is the card description, because on this board the description *is*
the acceptance criterion and a supervisor re-runs the command in it.

Delete this file once the cards exist.

## The state that produced this

VRAG-000 … VRAG-023 are Done; the Phase 2 MVP gate passed. VRAG-024 (Friday retro) and
VRAG-025 (supervisor gate re-run) are Saurabh and Yash closing the track out. So this is new
work after a finished track, not a continuation of one — see the open questions.

`src/graph.py` landed on `feat/vrag-graph-client` (PR #32): app-only Graph auth, the VTT
parser that lifts `<v Name>` voice tags, and a check command. It is not wired into the
pipeline. Nothing imports it, there is no `[capture] arm`, and no gate is touched.

Access to a real transcript is currently blocked by two tenant-side grants, neither of them
a code problem, both measured 2026-08-31 and reported by `make graph-check`:

| Blocker | Fix | Blocks |
|---|---|---|
| Teams application access policy missing | `New-CsApplicationAccessPolicy` + `Grant-…` | meeting metadata, recordings |
| Tenant transcript switch off (default, enforced 29 Jul 2026) | `Set-CsTeamsMeetingConfiguration -EnableGraphTranscriptAccess true -EnableAttributedTranscripts true` | transcripts |

Ask for **both** halves of the second one. `EnableGraphTranscriptAccess` alone yields
readable, valid, *unattributed* VTT — worth exactly what whisper already gives us for free.

## Order

```
VRAG-026 ─┬─ VRAG-027 ── VRAG-028 ── VRAG-029 ─┐
          │                                    ├─ VRAG-032 ── VRAG-033
VRAG-030 ─┴─ VRAG-031 ──────────────────────---┘

VRAG-034 (start now, in parallel) ── VRAG-035, VRAG-036

VRAG-037, VRAG-038 last
```

**Startable today, with no tenant access:** VRAG-026, VRAG-030, and the labelling half of
VRAG-034. VRAG-026 is the keystone — every Phase 4 ticket is worthless without it, and
`parse_vtt` already produces `VttCue.speaker`, so it is testable end to end offline.

---

# Phase 3 — Capture

## VRAG-026 · `Segment` carries a speaker, and nothing downstream drops it

The pipeline's seam is `list[Segment]` and `Segment` is `t_start`, `t_end`, `text`. Teams
knows who was speaking and this repo has nowhere to put it: `src/graph.to_segments()` throws
the name away at the boundary today, and `src/overview.py` tells the model outright that the
transcript has no speaker labels.

Most action items in a real meeting are self-assignments — "I'll take that", "leave it with
me". Without a speaker those sentences have **no recoverable owner**; the information is not
in the text at all. This ticket is what makes minutes possible.

**Two constraints, both easy to get wrong once and expensive to redo:**

- A chunk spans several segments, so `Chunk` takes `speakers: list[str]`, not a single name.
- **The speaker rides as metadata and never enters the embedded text.** Embedding names moves
  recall@5, and VRAG-017 is a recorded number scored once — the same reasoning that keeps
  `[caption] index = false`. Embedding names is a separate ticket with its own gate.

**Acceptance.** `Segment.speaker: str | None` and `Chunk.speakers: list[str]`. A test walks
one attributed VTT the whole way — `parse_vtt` → `Segment` → `Chunk` → Chroma metadata →
`RetrievedChunk` → citation — and asserts the name survives every hop. `make gate` is
unchanged and still passes: print the recall@5 line before and after and show they match.

*Touches:* `src/transcript.py`, `src/chunk.py`, `src/embed.py`, `src/index.py`,
`src/retrieve.py`, `src/graph.py`. *Depends on:* nothing. **Start here.**

## VRAG-027 · `[capture] arm` — a second producer of `Segment[]`

`transcribe()` has one producer: whisper over a wav. Graph is a second one, for meetings in
our own tenant, and it is free and better — Teams already did the ASR and attached names.
Follow the `[transcript] arm` idiom exactly: one interface, two arms, chosen in config.toml
with no Python edit.

Cache per meeting id the way transcripts cache per source sha256, and record which arm
produced a stored transcript — flipping the arm must invalidate, not silently reuse.

**Acceptance.** `make capture MEETING=<id>` writes `runs/<id>/transcript.json` and prints
*N segments, M attributed, K distinct speakers*. `[capture] arm = "file"` reproduces today's
behaviour byte for byte: show the sha256 of a transcript built before and after this ticket.

*Depends on:* VRAG-026. *Needs tenant access.*

## VRAG-028 · A run directory for a meeting, which has no video

`ingest()` writes `audio.wav`, `frames/` and `media.json`, and every downstream step assumes
that shape. A Graph meeting has none of it — no media file at all, just text and times. Decide
what `media.json` means when there is no media rather than writing a stub that lies about a
duration nothing measured.

**Acceptance.** `make index MEETING=<id>` indexes a meeting with no `audio.wav` and no
`frames/`, and `make ask` returns a citation into it. `make demo` on a video is unaffected —
run it and print the same `media.json` timing block as before.

*Depends on:* VRAG-027. *Needs tenant access.*

## VRAG-029 · Capture the participant roster

Who was in the room is ground truth that the transcript does not contain, and it is what
VRAG-032 validates owners against. Graph gives the organiser and the attendee list; the
`<v Name>` tags give who actually spoke. They are not the same set and the difference is
informative — a speaker absent from the roster means the roster is stale or the name is wrong.

**Acceptance.** `runs/<id>/roster.json` holds organiser and attendees. The command prints the
roster and lists every VTT speaker **not** in it. On one real meeting, that list is empty or
each entry is explained.

*Depends on:* VRAG-027. *Needs tenant access.*

---

# Phase 4 — Minutes

## VRAG-030 · `schemas/minutes.py`

Declared, not hand-assembled, for the reason `schemas/answer.py` is: structured output is
validated, never parsed by hand.

`Minutes{summary, attendees[], decisions[], action_items[]}` and
`ActionItem{task, owner: str | None, due: str | None, evidence}`.

`owner` is **nullable and stays nullable**. A wrong name is worse than no name in a document
that assigns work: an invented commitment attributed to a named colleague is the worst output
this system can produce, and a null is the honest representation of "nobody said".

**Acceptance.** Schema-valid on 100% of the dev meetings. Print the tally, as
`gate_phase2a` does.

*Depends on:* nothing. **Startable today.**

## VRAG-031 · Build minutes by folding the transcript

No real meeting fits one call on the free tier — the same wall `src/overview.py` already hit
and solved. Reuse its fold: `windows()`, one summary per window, then merge. Merge people and
decisions **in code, never by a model**, so no span can be invented.

**Acceptance.** `make minutes MEETING=<id>` writes `runs/<id>/minutes.json` and prints one
line per window plus the fold count. Runs on the longest available meeting without a 413.

*Depends on:* VRAG-030, VRAG-027.

## VRAG-032 · Validate every owner against the roster

The cheap ticket with the highest payoff. An `owner` that is not in the roster is a
hallucination, and it gets rejected exactly the way `src/answer.py`'s `ground()` rejects an
ungrounded span. Draw owners from the roster, not from whoever the transcript happens to name.

**Acceptance.** The command prints owners accepted and owners rejected, with the rejected
name and the sentence it came from. A planted owner who was never in the meeting is rejected —
show it in the output.

*Depends on:* VRAG-029, VRAG-031.

## VRAG-033 · Ground every decision and action item in a time range

A minute nobody can check is a rumour. Same contract as a citation: each decision and action
item carries the `t_start`/`t_end` it came from, and the run refuses to emit one that has none.

**Acceptance.** Ungrounded item count is 0 on every dev meeting, printed. One item's
timestamp opens the meeting at the moment it was said.

*Depends on:* VRAG-031.

---

# Phase 5 — Measurement

## VRAG-034 · `evals/minutes/` — labelled minutes, dev and sealed held-out

The long pole, and the only thing that turns "the minutes look good" into a number. Same
discipline as VRAG-011/012/013: a written spec for what a correct action item *is* before
anything is labelled, a dev split and a sealed held-out split, and a leakage test.

**Blind labelling applies.** `evals/minutes/heldout/` belongs to the Evaluator and the Builder
never reads it. Roles swap weekly — check who holds which this week before starting.

**Acceptance.** A spec saying what counts as a correct action item, a correct owner and a
correct decision. Dev and held-out labelled and sealed, held-out hashed into the README.
`make leakage-check` passes over the new set and prints an overlap of 0.

*Depends on:* the PRD question below. **Labelling can start now.**

## VRAG-035 · GATE — attribution quality

**Misattribution rate is reported separately from unattributed rate.** These are not the same
failure and averaging them into one accuracy number hides the one that matters: a wrong name
is a document assigning work to someone who never agreed to it, an absent name is a gap a
human fills in. One is a defect, the other is a to-do.

**Acceptance.** Both numbers printed with the command that produced them, on the sealed set.
Thresholds set on dev only, before the held-out run.

*Depends on:* VRAG-034, VRAG-026.

## VRAG-036 · GATE — action items found, and found correctly

**Acceptance.** Precision and recall for action items on the sealed set, plus owner accuracy
among the items that were found. Printed, with the command. Tuned on dev only; max three
attempts, then escalate.

*Depends on:* VRAG-034, VRAG-032.

---

# Phase 6 — Operations

## VRAG-037 · Know that a meeting happened

Graph cannot enumerate meetings — `GET /users/{id}/onlineMeetings` unfiltered is a 400, so a
meeting is reachable only by id or join url. Two routes, and the choice is an infrastructure
decision, not a preference:

- **Change notifications** on transcripts. The right shape, and it needs a publicly reachable
  HTTPS `notificationUrl`, a subscription created *before* transcription starts, and a
  `lifecycleNotificationUrl` to survive past an hour.
- **Polling** `getAllTranscripts` per organiser. Already implemented as `--list`, needs no new
  infrastructure, and is the fallback if the endpoint cannot be exposed.

**Acceptance.** A meeting that ends produces minutes with no human running a command. Print
the delay between the meeting ending and the minutes existing.

*Depends on:* VRAG-031.

## VRAG-038 · The deliverable

Minutes over HTTP next to `/ask`, or a page — decide when Phase 5 has numbers, since what the
gate measures is what the artifact should show. Whatever it is, it carries the same provenance
line the CLI prints: an artifact with no provenance cannot be disagreed with.

**Acceptance.** One command or one URL produces the minutes for a real meeting, with per-item
timestamps and the provenance footer.

*Depends on:* VRAG-033.

---

# Open questions — decide before Phase 4

**1. Does the PRD allow the eval set we need?** MOM labels need real meetings. The PRD says
*"No client-recorded sessions in v1"*, and the only real meeting on disk is
`vector7-21aug-client-meeting`, a client recording. Either the PRD moves or VRAG-034 labels
internal meetings instead. This shapes the schema, so it is cheaper to settle before VRAG-030
than after.

**2. Does this belong on this board?** VRAG-024's description says "one change to carry into
FIXR" — the VRAG track is closing and the course moves on. So VRAG-026+ is either a new epic
here or the start of the next track. Saurabh's call, and cheaper to ask than to renumber
thirteen cards.

**3. Which arm is this for?** Everything above assumes the Graph arm, which reaches meetings
in **our** tenant only. A client's meeting in a client's tenant is out of reach whatever is
granted, and needs a guest bot instead. Phases 4, 5 and 6 are arm-independent — they consume
`Segment[]` and do not care where it came from. Phase 3 is not. If client meetings are in
scope for v1, that is a parallel capture ticket, not a change to any of these.
