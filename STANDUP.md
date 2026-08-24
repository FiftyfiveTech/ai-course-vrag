# Standup log

Append at the end of each working session. Two minutes. Newest at the bottom.

Format — one block per session:

```
## YYYY-MM-DD — <name> (Builder|Evaluator)
Did:      what actually landed, with the task id
Number:   any measured number + the command that produced it (or "none today")
Blocked:  what is in the way, or "nothing"
Next:     the single next task id
```

Rule: a number goes in this file only if it appeared in your terminal. No number is better than a
remembered one.

## 2026-08-24 — Ritika (Evaluator)
Did:      VRAG-004 — 10-video corpus selected from lmms-lab/Video-MME by streaming the
          annotation parquet only. manifest.json + PROVENANCE.md committed under
          data/corpus/. Licence is not on the HF card: it lives in the upstream README and
          forbids redistributing the benchmark "in whole or in part", so the manifest holds
          ids + urls and none of the media or the benchmark's own Q&A.
Number:   10 videos = 4 dev + 6 held-out; 6/6 domains; 10 distinct sub-categories;
          3 short / 3 medium / 4 long. Cold-cache footprint 22 KB, 0 of 20 video archives
          (101.0 GB) fetched. 10/10 pointers resolve.
          `make corpus` · `make corpus-check` · `make corpus-pointers` · `make test` (39 passed)
Blocked:  the odoo-board MCP is authenticated as vimal, so VRAG-004 (assigned to ritika)
          is read-only from here — start/note/request_review could not be posted.
Next:     VRAG-011 (evals/QA_SPEC.md) — the corpus is in place, so the Q&A contract is next.
