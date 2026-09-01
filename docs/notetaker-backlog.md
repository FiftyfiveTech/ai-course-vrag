# Teams notetaker — the backlog

Thirteen tickets, VRAG-026 … VRAG-038, for turning this repo into a Teams meeting notetaker
that produces minutes.

**Why this is a file and not the board.** These belong on the board — it is the source of
truth for what to do next (CLAUDE.md) — and they cannot be written from here: the
`odoo-board` MCP server is read-and-progress only (`my_tasks`, `task`, `note`, `start`,
`request_review`, `set_done`) with no create tool. So this is a staging area, not a competing
plan. Every ticket below is uniform and pastes into Odoo unchanged: the **Acceptance** block
is the card description, because on this board the description *is* the acceptance criterion
and a supervisor re-runs the command in it.

**Delete this file once the cards exist.**

Every acceptance criterion is a command and the number it must print. A phase is done when
its number is computed and printed, not when it looks right (CLAUDE.md).

---

## Where this starts from

VRAG-000 … VRAG-023 are Done and the Phase 2 MVP gate passed. VRAG-024 (Friday retro) and
VRAG-025 (supervisor gate re-run) are Saurabh and Yash closing the track out. This is new work
after a finished track — see Open question 2.

`src/graph.py` is merged (PR #32, #33): app-only Graph auth, the VTT parser that lifts
`<v Name>` voice tags, a check command, and transcript discovery via `--list`. It is **not**
wired into the pipeline — nothing imports it, there is no `[capture] arm`, and no gate is
touched.

### Access is blocked by two tenant grants, neither of them a code problem

Both measured 2026-08-31 and reported by `make graph-check`:

| Blocker | Fix | Blocks |
|---|---|---|
| Teams application access policy missing | `New-CsApplicationAccessPolicy` + `Grant-CsApplicationAccessPolicy` | meeting metadata, recordings |
| Tenant transcript switch off — default, enforced 29 Jul 2026 | `Set-CsTeamsMeetingConfiguration -EnableGraphTranscriptAccess true -EnableAttributedTranscripts true -Identity Global` | transcripts |

Ask for **both halves** of the second. `EnableGraphTranscriptAccess` alone yields readable,
valid, *unattributed* VTT — worth exactly what whisper already gives us for free.

### Order

```
VRAG-026 ─┬─ VRAG-027 ── VRAG-028 ── VRAG-029 ─┐
          │                                    ├─ VRAG-032 ── VRAG-033 ── VRAG-038
VRAG-030 ─┴─ VRAG-031 ──────────────────────---┘                     └─ VRAG-037

VRAG-034 (start now, in parallel) ── VRAG-035, VRAG-036
```

### Startable today, with no tenant access

**VRAG-026**, **VRAG-030**, and the labelling half of **VRAG-034**.

| Ticket | Phase | Depends on | Needs access |
|---|---|---|---|
| VRAG-026 · `Segment` carries a speaker | 3 | — | no |
| VRAG-027 · `[capture] arm` | 3 | 026 | yes |
| VRAG-028 · Run directory with no video | 3 | 027 | yes |
| VRAG-029 · Participant roster | 3 | 027 | yes |
| VRAG-030 · `schemas/minutes.py` | 4 | — | no |
| VRAG-031 · Fold the transcript into minutes | 4 | 030, 027 | yes |
| VRAG-032 · Validate owners against the roster | 4 | 029, 031 | yes |
| VRAG-033 · Ground every decision and action item | 4 | 031 | yes |
| VRAG-034 · `evals/minutes/` labelled + sealed | 5 | Open Q1 | no |
| VRAG-035 · GATE — attribution quality | 5 | 034, 026 | yes |
| VRAG-036 · GATE — action items | 5 | 034, 032 | yes |
| VRAG-037 · Know that a meeting happened | 6 | 031 | yes |
| VRAG-038 · The deliverable | 6 | 033 | yes |

---

# Phase 3 — Capture

## VRAG-026 · `Segment` carries a speaker, and nothing downstream drops it

**Phase** 3 — Capture · **Depends on** — · **Needs tenant access** no · **Start here**

### Why

The pipeline's seam is `list[Segment]`, and `Segment` is `t_start`, `t_end`, `text`. Teams
knows who was speaking and this repo has nowhere to put it: `src/graph.to_segments()` throws
the name away at the boundary today, and `src/overview.py` tells the model outright that the
transcript has no speaker labels.

Most action items in a real meeting are self-assignments — "I'll take that", "leave it with
me". Without a speaker those sentences have **no recoverable owner**: the information is not
in the text at all, so any owner assigned to them is invented. This ticket is what makes
minutes possible, and every Phase 4 ticket is worthless without it.

### Acceptance

`Segment.speaker: str | None` and `Chunk.speakers: list[str]`.

A test walks one attributed VTT the whole way — `parse_vtt` → `Segment` → `Chunk` → Chroma
metadata → `RetrievedChunk` → citation — and asserts the name survives every hop.

`make gate` is unchanged and still passes. Print the recall@5 line before and after and show
they match.

### Notes

Two constraints, cheap now and expensive to redo later:

- A chunk spans several segments, so it is `speakers: list[str]`, not a single name.
- **The speaker rides as metadata and never enters the embedded text.** Embedding names moves
  recall@5, and VRAG-017 is a recorded number scored once — the same reasoning that keeps
  `[caption] index = false`. Embedding names is a separate ticket with its own gate.

**Touches** `src/transcript.py`, `src/chunk.py`, `src/embed.py`, `src/index.py`,
`src/retrieve.py`, `src/graph.py`

---

## VRAG-027 · `[capture] arm` — a second producer of `Segment[]`

**Phase** 3 — Capture · **Depends on** VRAG-026 · **Needs tenant access** yes

### Why

`transcribe()` has one producer: whisper over a wav. Graph is a second one for meetings in our
own tenant, and it is free and better — Teams already did the ASR and attached names. Follow
the `[transcript] arm` idiom exactly: one interface, two arms, chosen in `config.toml` with no
Python edit.

### Acceptance

`make capture MEETING=<id>` writes `runs/<id>/transcript.json` and prints
*N segments, M attributed, K distinct speakers*.

`[capture] arm = "file"` reproduces today's behaviour byte for byte: print the sha256 of a
transcript built before and after this ticket and show they match.

### Notes

Cache per meeting id the way transcripts cache per source sha256, and record which arm
produced a stored transcript — flipping the arm must invalidate the cache, not silently reuse
it. That exact bug is documented in `config.toml [transcript]`.

---

## VRAG-028 · A run directory for a meeting, which has no video

**Phase** 3 — Capture · **Depends on** VRAG-027 · **Needs tenant access** yes

### Why

`ingest()` writes `audio.wav`, `frames/` and `media.json`, and every downstream step assumes
that shape. A Graph meeting has none of it — no media file at all, just text and times.

### Acceptance

`make index MEETING=<id>` indexes a meeting with no `audio.wav` and no `frames/`, and
`make ask` returns a citation into it.

`make demo` on a video is unaffected — run it and print the same `media.json` timing block as
before.

### Notes

Decide what `media.json` means when there is no media, rather than writing a stub that lies
about a duration nothing measured.

---

## VRAG-029 · Capture the participant roster

**Phase** 3 — Capture · **Depends on** VRAG-027 · **Needs tenant access** yes

### Why

Who was in the room is ground truth the transcript does not contain, and it is what VRAG-032
validates owners against. Graph gives the organiser and the attendee list; the `<v Name>` tags
give who actually spoke. They are not the same set, and the difference is informative — a
speaker absent from the roster means the roster is stale or the name is wrong.

### Acceptance

`runs/<id>/roster.json` holds organiser and attendees. The command prints the roster and lists
every VTT speaker **not** in it. On one real meeting that list is empty, or each entry is
explained.

---

# Phase 4 — Minutes

## VRAG-030 · `schemas/minutes.py`

**Phase** 4 — Minutes · **Depends on** — · **Needs tenant access** no · **Startable today**

### Why

Declared, not hand-assembled, for the reason `schemas/answer.py` is: structured output is
validated, never parsed by hand, and FastAPI renders it into the contract a frontend codes
against.

### Acceptance

`Minutes{summary, attendees[], decisions[], action_items[]}` and
`ActionItem{task, owner: str | None, due: str | None, evidence}`.

Schema-valid on 100% of the dev meetings, tally printed, as `gate_phase2a` does.

### Notes

`owner` is **nullable and stays nullable**. A wrong name is worse than no name in a document
that assigns work: an invented commitment attributed to a named colleague is the worst output
this system can produce, and null is the honest representation of "nobody said".

---

## VRAG-031 · Build minutes by folding the transcript

**Phase** 4 — Minutes · **Depends on** VRAG-030, VRAG-027 · **Needs tenant access** yes

### Why

No real meeting fits one call on the free tier — the same wall `src/overview.py` already hit
and solved. Reuse its fold: `windows()`, one summary per window, then merge.

### Acceptance

`make minutes MEETING=<id>` writes `runs/<id>/minutes.json` and prints one line per window
plus the fold count. Runs on the longest available meeting without a 413.

### Notes

Merge people and decisions **in code, never by a model**, so no span can be invented — the
rule `src/overview.py` already follows.

---

## VRAG-032 · Validate every owner against the roster

**Phase** 4 — Minutes · **Depends on** VRAG-029, VRAG-031 · **Needs tenant access** yes

### Why

The cheap ticket with the highest payoff. An `owner` not in the roster is a hallucination, and
it gets rejected exactly the way `src/answer.py`'s `ground()` rejects an ungrounded span. Draw
owners from the roster — who was actually there — not from whoever the transcript happens to
name.

### Acceptance

The command prints owners accepted and owners rejected, with each rejected name and the
sentence it came from. A planted owner who was never in the meeting is rejected — show it in
the output.

---

## VRAG-033 · Ground every decision and action item in a time range

**Phase** 4 — Minutes · **Depends on** VRAG-031 · **Needs tenant access** yes

### Why

A minute nobody can check is a rumour. Same contract as a citation.

### Acceptance

Ungrounded item count is 0 on every dev meeting, printed. One item's timestamp opens the
meeting at the moment it was said.

### Notes

The run refuses to emit an item that has no time range, rather than emitting it unmarked.

---

# Phase 5 — Measurement

## VRAG-034 · `evals/minutes/` — labelled minutes, dev and sealed held-out

**Phase** 5 — Measurement · **Depends on** Open question 1 · **Needs tenant access** no

### Why

The long pole, and the only thing that turns "the minutes look good" into a number. Same
discipline as VRAG-011/012/013: a written spec for what a correct action item *is* before
anything is labelled, a dev split and a sealed held-out split, and a leakage test.

### Acceptance

A spec saying what counts as a correct action item, a correct owner and a correct decision.
Dev and held-out labelled and sealed, held-out hashed into the README. `make leakage-check`
passes over the new set and prints an overlap of **0**.

### Notes

**Blind labelling applies.** `evals/minutes/heldout/` belongs to the Evaluator and the Builder
never reads it. Roles swap weekly — check who holds which this week before starting.

---

## VRAG-035 · GATE — attribution quality

**Phase** 5 — Measurement · **Depends on** VRAG-034, VRAG-026 · **Needs tenant access** yes

### Why

**Misattribution rate is reported separately from unattributed rate.** These are not the same
failure and averaging them into one accuracy number hides the one that matters: a wrong name
is a document assigning work to someone who never agreed to it, an absent name is a gap a
human fills in. One is a defect, the other is a to-do.

### Acceptance

Both numbers printed with the command that produced them, on the sealed set. Thresholds set on
dev only, before the held-out run.

---

## VRAG-036 · GATE — action items found, and found correctly

**Phase** 5 — Measurement · **Depends on** VRAG-034, VRAG-032 · **Needs tenant access** yes

### Acceptance

Precision and recall for action items on the sealed set, plus owner accuracy among the items
that were found. Printed, with the command.

### Notes

Tuned on `evals/dev` only; maximum three attempts, then escalate (CLAUDE.md).

---

# Phase 6 — Operations

## VRAG-037 · Know that a meeting happened

**Phase** 6 — Operations · **Depends on** VRAG-031 · **Needs tenant access** yes

### Why

Graph cannot enumerate meetings — `GET /users/{id}/onlineMeetings` unfiltered is a 400, so a
meeting is reachable only by id or join url. Two routes, and the choice is an infrastructure
decision rather than a preference:

- **Change notifications** on transcripts. The right shape, and it needs a publicly reachable
  HTTPS `notificationUrl`, a subscription created *before* transcription starts, and a
  `lifecycleNotificationUrl` to survive past an hour.
- **Polling** `getAllTranscripts` per organiser. Already implemented as `--list`, needs no new
  infrastructure, and is the fallback if the endpoint cannot be exposed.

### Acceptance

A meeting that ends produces minutes with no human running a command. Print the delay between
the meeting ending and the minutes existing.

---

## VRAG-038 · The deliverable

**Phase** 6 — Operations · **Depends on** VRAG-033 · **Needs tenant access** yes

### Why

Minutes over HTTP next to `/ask`, or a page. Decide which when Phase 5 has numbers, since what
the gate measures is what the artifact should show.

### Acceptance

One command or one URL produces the minutes for a real meeting, with per-item timestamps and
the provenance footer.

### Notes

Carries the same provenance line the CLI prints. An artifact with no provenance cannot be
re-run or disagreed with, and that does not stop being true because the transport changed.

---

# Open questions — decide before Phase 4

## 1. Does the PRD allow the eval set we need?

MOM labels need real meetings. The PRD says **"No client-recorded sessions in v1"**, and the
only real meeting on disk is `vector7-21aug-client-meeting`, a client recording. Either the
PRD moves or VRAG-034 labels internal meetings instead.

This shapes the schema, so it is cheaper to settle before VRAG-030 than after. **Blocks
VRAG-034.**

## 2. Does this belong on this board?

VRAG-024's description says "one change to carry into FIXR" — the VRAG track is closing and
the course moves on. So VRAG-026+ is either a new epic here or the start of the next track.
Saurabh's call, and cheaper to ask than to renumber thirteen cards.

## 3. Which capture arm is this for?

Everything above assumes the **Graph arm**, which reaches meetings in *our* tenant only. A
client's meeting in a client's tenant is out of reach whatever is granted, and needs a guest
bot instead.

Phases 4, 5 and 6 are arm-independent — they consume `Segment[]` and do not care where it came
from. Phase 3 is not. If client-tenant meetings are in scope for v1, that is a parallel
capture ticket, not a change to any of these.
