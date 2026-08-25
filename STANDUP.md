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

## 2026-08-25 — Vimal (Builder)
Did:      VRAG-005 — ingest pipeline. `src/ingest.py` (ffprobe metadata, 16 kHz mono wav,
          timestamped frame sampling, media.json per run), `src/config.py` + `config.toml`
          holding the levers, `src/sample.py` with two fixtures, `make sample` /
          `make sample-real` / `make demo`. Unblocked the card first: ffmpeg and ffprobe
          were simply not installed (winget Gyan.FFmpeg 9.0), which is what `make doctor`
          had been failing on since VRAG-001.
          Caught one real bug by looking at a frame instead of trusting the code: `-vf fps=N`
          keeps the frame from the middle of each interval but relabels it with the
          interval's start, so media.json said t=25.0 next to a picture whose burnt-in
          timecode read 27.48. Switched to `select` + `-fps_mode passthrough` and now read
          each timestamp back out of `showinfo` instead of computing it.
Number:   `make doctor` → 11 PASS 1 WARN 0 FAIL (was 9/1/2)
          `make demo VIDEO=samples/one.mp4` → 30.00 s in, 6 frames at fps=0.2 covering
          30.0 s, wav 16000 Hz 1 ch 0.96 MB, total 0.577 s, 52.02× realtime
          `make test` → 97 passed (84 passed / 13 skipped with ffmpeg off PATH)
          Same code + a config with fps=1.0 → 30 frames instead of 6, which is the
          "not hardcoded" half of the criterion.
Blocked:  nothing. Worth flagging for the retro: this session ran on Ritika's machine but
          the odoo-board MCP is authenticated as vimal, so start/note/request_review for
          VRAG-005 were posted as Vimal. The board's owner field and who typed are not the
          same person this week.
Next:     VRAG-006 (cost meter) — ingest already records per-stage latency and ×realtime,
          so $/video-hour is the missing half.
