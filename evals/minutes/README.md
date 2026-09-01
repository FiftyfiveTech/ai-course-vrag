# evals/minutes — Meeting minutes evaluation set

Labelled minutes pairs for VRAG-035 (attribution quality) and VRAG-036 (action item gate).
See [MINUTES_SPEC.md](MINUTES_SPEC.md) for what counts as a correct action item, owner, and decision.

## Splits

| Split | File | Pairs | Sealed |
|-------|------|-------|--------|
| dev | `dev/dev_minutes_v1.jsonl` | 3 | no — Builder tunes on this |
| heldout | `heldout/heldout_minutes_v1.jsonl` | 4 | **yes — Evaluator only** |

## Held-out seal

```
sha256  4410b1f5a3008dc29179bf06a644ef083562c143197d394bed378c8ce2d1e7a1
file    evals/minutes/heldout/heldout_minutes_v1.jsonl
sealed  2026-08-31
```

The Builder must not open `evals/minutes/heldout/`. The Evaluator re-runs the gate on this
file and compares the hash before scoring.

## Leakage check

```
make leakage-check
```

The second invocation covers this directory pair. Must print `overlap  0`.
