# VRAG Minutes Evaluation Spec

This document is the contract between the Evaluator (who writes labels) and the Builder
(who tunes the pipeline). A Builder must be able to predict every label from this spec
alone, without asking for clarification.

---

## 1. Eval pair format

Each pair is a JSON object on one line of a `.jsonl` file:

```jsonc
{
  "id":           "m-dev-001",     // string — unique, sequential, never reused
  "question":     "...",           // the meeting title/description — used for leakage detection
  "answer_note":  "...",           // one sentence: key facts the labeller used to score this meeting
  "reference": {
    "summary":      "...",         // one-paragraph meeting summary
    "attendees":    ["..."],       // display names of people who attended
    "decisions":    ["..."],       // decisions made — each a complete sentence
    "action_items": [
      {
        "task":     "...",         // what was committed to
        "owner":    "..." | null,  // who committed — null when no speaker was identified
        "due":      "..." | null,  // deadline if stated explicitly, else null
        "evidence": "..."          // timestamp or quote grounding this item
      }
    ]
  }
}
```

Dev ids use `m-dev-NNN`. Held-out ids use `m-hld-NNN`. Both namespaces are disjoint by
design so the leakage check catches any copy-paste by id alone.

---

## 2. What a correct action item is

An action item is a **specific commitment** made by a named participant or the group.

**Correct:**
- The task text is a close paraphrase of what was actually said — not a generalisation
- The speaker explicitly accepted or volunteered the task ("I'll take that", "I can do that")
- The evidence timestamp points to the exact moment the commitment was made

**Incorrect:**
- Vague statements ("someone should look into this") are not action items — they are
  background discussion
- Inferred commitments ("they were talking about X so they probably meant Y") are not
  action items — if it was not said, it is not an item
- Decisions dressed as action items ("we decided to migrate") — these go in decisions

---

## 3. What a correct owner is

Owner is the **named speaker** who explicitly accepted the task.

**Rules:**
- `"I'll take that"` — owner is the speaker whose voice tag precedes this line
- `"Can you handle X?"` followed by `"Yes"` — owner is the person who said yes
- `"Someone should do X"` / `"Let's do X"` / group agreement — owner is `null`
- If no voice tag is present (unattributed cue) — owner is `null`
- Owner must be a name that appears in the attendees list (VRAG-032 enforces this)

**Never invent an owner.** A `null` owner is the honest answer when the speaker was not
identified. An invented name attributed to a real colleague is the worst possible output.

---

## 4. What a correct decision is

A decision is a **collective choice** explicitly made during the meeting.

**Correct:**
- "We decided to move to the new schema first" — explicit group agreement
- "Agreed: we will use the Graph arm for all internal meetings" — ratified choice

**Incorrect:**
- An action item restated as a decision ("Rohan will draft the script" is an action item)
- Background context ("We have been using whisper so far" — this is not a decision)
- Tentative statements ("We might want to consider...") — not a decision until agreed

---

## 5. What a correct summary is

The summary is **one paragraph** covering:
- The purpose of the meeting
- The main topics discussed
- The outcome (what was decided or committed to)

It must not contain made-up facts. Every sentence must be grounded in the meeting content.

---

## 6. Scoring (for the gate — VRAG-035, VRAG-036)

**Action item recall:** fraction of reference action items found by the pipeline.
An item is found when the pipeline's task text matches the reference task text closely
enough that a human would call them the same commitment (paraphrase is allowed; invention
is not).

**Owner accuracy:** among found action items, fraction where the pipeline's owner matches
the reference owner exactly (string match on display name, or both are null).

**Misattribution rate:** fraction of found items where the pipeline assigned a non-null
owner when the reference owner is null, or assigned the wrong name.

**Unattributed rate:** fraction of found items where the pipeline returned null owner when
the reference owner is non-null.

Misattribution and unattribution are reported separately — they are different failures.

---

## 7. Blind labelling

`evals/minutes/heldout/` belongs to the **Evaluator**. The Builder tunes on
`evals/minutes/dev/` only and never reads the held-out labels.

`make leakage-check-minutes` asserts `dev ∩ heldout = ∅` by content hash on `id`,
`question`, and `answer_note`. If it fails, stop — do not tune, do not run a gate.

Roles swap weekly. Check who holds which before starting to label.
