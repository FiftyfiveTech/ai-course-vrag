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

## 2026-08-25 — Ritika (Evaluator)
Did:      VRAG-006 — shared cost/latency meter. `src/telemetry.py`: Meter class with
          span() context manager, log(), total_cost_usd(), summary_line(). Cost rates
          keyed by HF repo id: openai/whisper-large-v3-turbo @ $0.04/audio-hour,
          nomic-ai/nomic-embed-text-v1.5 @ $0.00 (local). Wired into src/ingest.py —
          meter created at pipeline start, summary_line stored in result["telemetry"]
          and printed as the last line of every run. 16 tests in tests/test_telemetry.py.
          PR #6 opened (feat/vrag-006 → dev).
Number:   `.venv/bin/python -m src.ingest samples/one.mp4 --config config.toml`
          → last line: $0.0000/video-hour  52.6×realtime
          `.venv/bin/pytest tests -q --ignore=tests/gates` → 113 passed (was 97, +16)
Blocked:  nothing. uv run exits 120 with no output in this environment — using
          .venv/bin/python and .venv/bin/pytest directly as workaround.
Next:     VRAG-011 (evals/QA_SPEC.md) — corpus is in place, cost meter is wired;
          Q&A spec is the next Evaluator task.

## 2026-08-25 — Ritika (Evaluator)
Did:      VRAG-011 — wrote evals/QA_SPEC.md (232 lines). Defines Q&A pair JSON
          format (id, question, unanswerable, video_id, t_ref, answer_note),
          correct-citation rule (right video_id AND |t_start − t_ref| ≤ 30 s AND
          abstain: false), unanswerable definition, abstention scoring, MVP gate
          formula (≥ 0.70 on 20 pairs), labeling rules for VRAG-012 (17 answerable
          + 3 unanswerable, spread across videos, no yes/no, t_ref verified by
          watching), and 5 worked examples covering all verdict cases.
          PR #7 opened (feat/vrag-011 → dev).
Number:   none today (spec document, no computed number)
Blocked:  nothing
Next:     VRAG-012 (label + seal 20 held-out Q&A pairs) — spec is in place.

## 2026-08-25 — Ritika (Evaluator)
Did:      VRAG-008 — timestamped transcript module. src/transcript.py:
          transcribe(wav, cfg, meter) dispatches to the arm in config.toml.
          Groq arm: whisper-large-v3-turbo via Groq SDK, verbose_json segments,
          cost logged through Meter at $0.04/audio-hour.
          Ollama arm: same model pulled locally via hf.co/openai/whisper-large-v3-turbo,
          cost $0.00. Both return list[Segment(t_start, t_end, text)].
          config.toml: [transcript] arm/model/language levers added.
          pyproject.toml: groq>=0.9, ollama>=0.3 added.
          21 tests in tests/test_transcript.py (no real API calls).
          PR #8 opened (feat/vrag-008 → dev).
Number:   .venv/bin/pytest tests -q --ignore=tests/gates → 134 passed (was 113, +21)
Blocked:  nothing. End-to-end test (real Groq call) deferred — needs GROQ_API_KEY
          wired in environment; unit tests cover all logic paths.
Next:     VRAG-012 (label + seal 20 held-out Q&A pairs) — QA_SPEC.md is in place.

## 2026-08-25 — Ritika (Evaluator)
Did:      VRAG-008 end-to-end test with real Groq call on samples/one.mp4.
Number:   .venv/bin/python -c "from src.transcript import transcribe; ..."
          → 2 segments, $0.0400/video-hour  29.6×realtime
          (synthetic video produces "Thank you." — expected; real corpus videos
          will produce actual transcript text)
Blocked:  nothing
Next:     VRAG-012 (label + seal 20 held-out Q&A pairs)

## 2026-08-25 — Ritika (Evaluator)
Did:      VRAG-010 — Phase 0 gate. tests/gates/gate_phase0.py: 9 checks that
          run the full pipeline end-to-end (ingest + transcript) from a
          synthetic video. Asserts: media.json, frames, audio.wav, transcript
          ran without crash, x_realtime > 1.0, $/video-hour computed and
          printed, ingest telemetry line present. PR #10 opened (frat/vrag-010 → dev).
Number:   .venv/bin/pytest tests/gates/gate_phase0.py -v -s
          → frames: 6 × jpg · audio: 0.96 MB · transcript: 2 segment(s)
            x_realtime: 26.0
            $0.0400/video-hour  17.7×realtime
            9 passed in 5.36s
Blocked:  nothing
Next:     VRAG-012 (label + seal 20 held-out Q&A pairs)
