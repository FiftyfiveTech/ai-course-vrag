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

## 2026-08-25 — Ritika (Evaluator)
Did:      VRAG-012 — sealed the held-out Q&A set. evals/heldout/heldout_v1.jsonl: 20 pairs,
          17 answerable + 3 unanswerable, at least one answerable question per corpus video,
          written against evals/QA_SPEC.md. src/evalset.py turns the spec into a check;
          `make heldout-check` prints the counts, the per-video spread and the sha256, and
          compares that digest to the one recorded in README.md. Tag heldout-v1 pushed at
          92b2763. PR #11 (feat/vrag-012 → dev).
          QA_SPEC §6 requires each t_ref to be verified by watching, and the repo holds
          pointers not media, so all ten videos were fetched from their manifest urls into a
          scratchpad outside the repo. None of the ten carries a YouTube caption track, so
          verification was frames-at-1s-intervals for on-screen facts and a whisper
          transcript of the ±30 s window for spoken ones. answer_note on every pair records
          what was seen or heard and the seconds it spans.
          Caught one real bug the same way VRAG-005 caught its frame-timestamp bug — by
          checking rather than assuming: core.autocrlf=true on our Windows checkouts rewrites
          LF to CRLF, which changes the file's bytes and so its sha256. The seal would have
          failed on a fresh clone for a reason with nothing to do with the labels.
          .gitattributes now pins *.jsonl and data/corpus/manifest.json to LF.
Number:   `make heldout-check` → 20 pairs — 17 answerable / 3 unanswerable · 10/10 videos
          covered · sha256 74398cbae0956271962bde9a3b51b89db766da0ae1d65802c1c56a81ab0d1084
          · README matches · PASS
          Same command after cloning the branch fresh into a scratchpad → same digest, PASS.
          Same command with one t_ref edited from 2.0 to 12.0 → FAIL, exit 1, digest
          e618b124… against the recorded 74398cba…
          `.venv/Scripts/pytest tests -q --ignore=tests/gates` → 119 passed, 13 skipped
          (was 113 passed; +19 in tests/test_evalset.py, 13 skips are the ffmpeg-dependent
          ingest tests with ffmpeg off PATH in this shell)
          After merging dev — VRAG-008 landed while this branch was open — the same
          command gives 140 passed, 13 skipped (119 + VRAG-008's 21), and
          `make heldout-check` prints the same sha256, so the heldout-v1 tag still
          describes the file it was cut against.
          Verification cost: 11 whisper windows, 689 audio-seconds, openai/whisper-large-v3-turbo
          on Groq's free tier — $0.00 spent, $0.0077 at the paid rate src/telemetry.py models.
Blocked:  nothing.
          Flagging for the retro, not blocking: 5 of the 17 answerable questions turn on
          something on screen rather than something said (q001, q002, q010, q012, q014).
          q002 had to — video 091 has no speech at all. If the index ends up transcript-only,
          those five are unretrievable however good retrieval is. VRAG-023 (keyframe captions)
          is the stretch task that closes it, and the call is worth making before VRAG-021,
          since the gate is scored once.
          Also noticed: README's "Deliberately missing" section still lists src/telemetry.py
          as absent. It landed in VRAG-006. Left alone here rather than widen this diff.
Next:     VRAG-013 (tests/gates/test_no_leakage.py) — the seal exists now, so the dev ∩
          heldout = ∅ check is what makes the blind-labelling rule enforced rather than
          agreed.

## 2026-08-26 — Ritika (Evaluator)
Did:      VRAG-015 — embed + persist. src/embed.py: embed_and_persist(chunks,
          cfg, meter) batches Chunk objects, embeds via Ollama
          (nomic-ai/nomic-embed-text-v1.5), upserts into local Chroma with
          video_id/t_start/t_end metadata. Chunk dataclass defined here —
          VRAG-014 chunker will produce this shape. Idempotent via
          Chunk.chunk_id() = {video_id}_{t_start:.3f}_{t_end:.3f}.
          config.toml: [embed] section added. chromadb>=0.5 added to
          pyproject.toml. 19 tests in tests/test_embed.py (no real Ollama/Chroma).
          PR #13 opened (feat/vrag-015 → dev).
Number:   .venv/bin/pytest tests -q --ignore=tests/gates → 172 passed (was 140, +19 embed + 13 skipped resolved)
          Cost: $0.00 — nomic-embed-text-v1.5 is local Ollama, rate $0.00 in telemetry.
Blocked:  end-to-end test deferred — needs `ollama pull hf.co/nomic-ai/nomic-embed-text-v1.5`.
          VRAG-014 (chunking, Vimal) not merged yet; Chunk shape agreed here.
Next:     VRAG-016 (retrieve: question → top-k chunks; recall@5 on dev)

## 2026-08-26 — Ritika (Evaluator)
Did:      VRAG-016 — retriever. src/retrieve.py: retrieve(question, cfg, meter)
          embeds question via Ollama (same model as embed), queries Chroma,
          returns list[RetrievedChunk(video_id, t_start, t_end, text, score)].
          recall_at_k(cfg, meter) loads evals/dev/*.jsonl, runs retrieve() per
          answerable pair, scores hit as: correct video_id AND
          |t_start - t_ref| <= 30 s (QA_SPEC tolerance). Returns 0.0 vacuously
          when dev is empty. config.toml: [retrieve] top_k = 5 added.
          22 tests in tests/test_retrieve.py (no real Ollama/Chroma calls).
          PR #15 opened (feat/vrag-016 → dev).
Number:   .venv/bin/pytest tests -q --ignore=tests/gates → 194 passed (was 172, +22)
          recall_at_k returns 0.0 vacuously today — evals/dev/ has no pairs yet.
          Real number available once dev pairs are written (VRAG-016 scoring task).
Blocked:  nothing. evals/dev/ is empty so recall@5 cannot be measured yet.
          Need dev Q&A pairs before the Phase 1 gate (VRAG-017) can be run.
Next:     VRAG-021 (Gate Phase 2) or wait for dev pairs to score recall@5.

## 2026-08-25 — Ritika (Evaluator)
Did:      VRAG-013 — the blind-labelling seal. src/leakage.py computes
          evals/dev ∩ evals/heldout by content hash; tests/gates/test_no_leakage.py is the
          gate (24 tests). Three fingerprints per pair, sha256 over one normalised field
          each: id, question, answer_note. Normalisation folds NFKC, case, whitespace,
          smart quotes/dashes and trailing punctuation, so a copy-paste that changed only
          formatting is still caught.
          video_id is deliberately NOT compared — the video split is public and QA_SPEC §6
          asks for held-out questions on dev videos, so flagging it would make the gate
          unpassable. Wrote a test that pins that, so nobody "fixes" it later.
          Two things the design had to settle:
          (1) Id namespaces. Both splits would independently number from q001, so the id
          fingerprint would fire on every dev case ever written — a check that always fails
          is as useless as one that never does. heldout stays q001…q020 (it is sealed);
          dev takes d001…. Recorded in QA_SPEC §8 with the reason. Sealed file untouched —
          `make heldout-check` still prints the same sha256.
          (2) evals/dev/ is empty, so the check passes vacuously today. It says so:
          "PASS (vacuous) — evals/dev holds no pairs yet, so the intersection is empty by
          default rather than by discipline". A silent green here would be the worst
          outcome — it reads as verified when nothing was compared. Empty *heldout* is a
          FAIL for the same reason: 0 ∩ 0 = ∅ is not a seal.
          Two bugs found by running it rather than reading it, both in the report path:
          - '∩' in the output raised UnicodeEncodeError on this cp1252 console and exited 1.
            A gate signals a leak with exit 1, so an encoding accident was indistinguishable
            from a leak. main() now reconfigures stdout/stderr to utf-8/replace, and the
            printed strings are cp1252-safe. Pre-existing in src/evalset.py too (its em
            dashes render as '?'), left alone there — flagging for the retro.
          - stdout is block-buffered when piped, stderr is not, so the FAIL detail printed
            before the counts it referred to. Explicit flush; the number reads first on a
            terminal and in a redirected log.
          Wired `make gate` to depend on leakage-check, so a leak stops the run instead of
          appearing as one red line among the phase gates — tests/gates/README.md already
          said no gate result counts until this passes.
          Also removed README's "Deliberately missing" section: it listed src/telemetry.py
          (landed in VRAG-006) and test_no_leakage.py (this task). Flagged it last session,
          both files exist now, so the section went rather than growing more stale.
Number:   `make leakage-check` clean → dev 0 pairs / heldout 20 pairs / compared by sha256
          over id, question, answer_note / overlap 0 → PASS (vacuous), exit 0
          Negative control — `head -1 evals/heldout/heldout_v1.jsonl > evals/dev/LEAK_PROBE.jsonl`
          then `make leakage-check` → overlap 3, FAIL naming all three fingerprints and both
          sides (evals/dev/LEAK_PROBE.jsonl:1 (q001) == evals/heldout/heldout_v1.jsonl:1
          (q001)), exit 1. `make gate` with the same probe → exit 2, stops before the phase
          gates. Probe removed; `git status --short` clean on evals/.
          `uv run pytest tests/gates -q` → 24 passed
          `uv run pytest tests -q --ignore=tests/gates` → 140 passed, 13 skipped (unchanged;
          the new tests are all in tests/gates/, which make test excludes on purpose so a
          red seal cannot be mistaken for an ordinary unit-test failure)
          `make heldout-check` → same sha256 74398cbae0956271962bde9a3b51b89db766da0ae1d65802c1c56a81ab0d1084, PASS
          $0.00 — no model calls in this task.
Blocked:  nothing.
          Flagging, not blocking: a content hash cannot catch a held-out question rewritten
          from memory in different words. Documented in the module docstring, in QA_SPEC §8
          and in a test that pins the limitation rather than papering over it. That case is
          on review and on the Builder not opening the file — worth saying out loud at the
          retro so the gate is not over-trusted.
          Also: there is no board task that writes evals/dev/. README says 15 cases and
          VRAG-016/017 score recall@5 on dev, so the cases have to exist by then. Until they
          do this gate is vacuous, which is exactly what its output says.
Next:     VRAG-009 (failure tests: no audio track, zero-length, unreadable codec) is the
          next unstarted one of mine.

## 2026-08-26 — Ritika (Evaluator)
Did:      VRAG-021 — Phase 2 gate + answer module (VRAG-019 prerequisite).
          prompts/answer_v1.md: system prompt for the answer LLM. src/answer.py:
          answer_question(question, chunks, cfg, meter) → Answer(answer, citations[], abstain).
          Dispatches on answer.arm in config; ollama arm calls hf.co/<model> via ollama.chat()
          with temperature=0. _parse_response() handles markdown fences and prose around
          the JSON. config.toml: [answer] section added (arm=ollama,
          model=meta-llama/Llama-3.2-3B-Instruct). Telemetry RATES updated.
          tests/gates/gate_phase2.py: scores the full retrieve→answer pipeline against
          evals/heldout/heldout_v1.jsonl (20 sealed pairs). Prints score (correct/20)
          and abstention count (abstained/3). Asserts ≥0.70 and all abstentions.
          20 tests in tests/test_answer.py (no real Ollama calls).
          PR opened (feat/vrag-021 → dev).
Number:   .venv/bin/pytest tests -q --ignore=tests/gates → 273 passed (was 194, +20 answer tests + 59 chunk tests now on PATH)
          Gate cannot be run yet — corpus not indexed in Chroma.
          Pre-requisites before gate_phase2.py can produce real numbers:
            ollama pull hf.co/meta-llama/Llama-3.2-3B-Instruct
            ollama pull hf.co/nomic-ai/nomic-embed-text-v1.5
            make sample-real VIDEO_ID=<each of 10 corpus ids>
            uv run python -m src.chunk <video>  # for each video
            uv run python -m src.embed <video>  # for each video (VRAG-015 interface)
Blocked:  VRAG-019 (answer module, assigned to Vimal, Backlog) — built here as part of gate
          prerequisites. Corpus not yet indexed: need 10 corpus videos fetched and embedded
          before gate_phase2.py produces a real number.
Next:     run the gate once corpus is indexed; VRAG-020 (demo CLI) when VRAG-019 is merged.

## 2026-08-25 — Ritika (VRAG-014, Builder-side task on the board)
Did:      VRAG-014 — time-window chunking. src/chunk.py: a fixed grid on the video clock
          (0, hop, 2·hop, … where hop = window_s − overlap_s), both levers in config.toml
          with no defaults. Every Chunk carries video_id, t_start, t_end, text, the segment
          ids it holds and the grid window it came from. `make chunks VIDEO=…` dumps the
          table and exits non-zero if verify() finds a problem. 59 tests in
          tests/test_chunk.py, none of which need ffmpeg, network or a model call.
          Segments are never split, so a chunk's t_start/t_end are *measured* from the
          segments in it, not copied off the window bounds — the same class of bug as
          VRAG-005's `-vf fps=N`. verify() re-derives that from the output rather than
          trusting the constructor: every chunk's range must contain every segment it
          claims, and every segment must be in ≥1 chunk.
          Two windows are dropped on purpose and both are counted in the dump: one with no
          speech (indexing silence costs money and retrieves nothing) and one whose segments
          are exactly its predecessor's (overlap can make two neighbours identical when all
          the speech falls in their intersection — that is index bloat, not recall).
          The ASR result is cached in runs/<video>/transcript.json per source sha256 + model,
          so re-chunking after moving a lever costs $0.00 and makes no model call. That is
          what made the sweep below affordable.
          One design decision was wrong and a measurement caught it. window_s = 30.0 was
          derived from QA_SPEC §2 (a citation is scored on |t_start − t_ref| ≤ 30, and a
          citation points at the chunk it came from). The derivation missed that a chunk
          overhangs its window at *both* ends, so it runs to window_s + 2 × longest segment.
          On dev video 181 that produced a 35.7 s chunk — 2 of 5 past the tolerance, i.e.
          chunks that can retrieve the right passage and still be scored wrong. Swept on the
          cached transcript and retuned to 25.0 / 8.0. The reasoning and the table are in
          config.toml next to the values, and the dump now prints the longest segment beside
          the chunk durations so the bound is visible rather than remembered.
Number:   `make chunks VIDEO=samples/181_8np5YKYx3sU.mp4` (dev split, fetched from the
          manifest url; gitignored) → video_id 181 from data/corpus/manifest.json (dev
          split), window=25.0s overlap=8.0s hop=17.0s, 24 segments → 6 chunks from 6 windows
          (0 empty, 0 duplicate), covers 0.000–88.520 s of 95.77 s, mean 15.6 s / max 29.0 s,
          longest segment 4.16 s, 1022 chars indexed from 622 transcript chars (1.64× — the
          overlap), 37 segment slots for 24 segments, **invariants 0 problems in 6 chunks**,
          exit 0. $0.0000/video-hour on the re-run (cached transcript, no model call); the
          first run that transcribed it printed $0.0400/video-hour 16.3×realtime.
          Lever sweep on the cached transcript, $0.00, no model call:
            window  overlap  chunks  max chunk  over 30 s  problems
            30.0    10.0     5       35.7 s     2          0
            25.0     8.0     6       29.0 s     0          0
            22.0     8.0     7       24.4 s     0          0
          Negative control — patched `t_end=float(w_end)` (copy the grid bound instead of
          measuring the segments, the exact bug the module exists to prevent) and re-ran the
          same command → `FAIL 3 problem(s) — a chunk lost its time range`, naming
          `181-0002: range [32.0, 59.0] does not contain segment 11 [58.0, 61.0]`,
          `181-0003: range [51.0, 76.0] does not contain segment 18 [75.34, 76.84]` and
          `181-0005: t_end 110.0 is past the end of the video (95.774)`, exit 1 (make
          reports 2). Patch reverted; `grep -c PROBE src/chunk.py` → 0 and the same command
          is back to 0 problems, exit 0.
          `uv run pytest tests -q --ignore=tests/gates` → 212 passed (was 140 passed /
          13 skipped; +59 in tests/test_chunk.py, and the 13 ingest skips run now because
          ffmpeg is on PATH in this shell)
          `make gate` → leakage PASS (vacuous), 24 passed
          `uv run pytest tests/gates/gate_phase0.py -q` → 9 passed
          `make chunks VIDEO=samples/one.mp4` → 1 chunk, 0 problems (the synthetic fixture
          transcribes to one segment, so it proves the wiring, not the windowing)
Blocked:  nothing.
          Flagging, not blocking, three things for the retro:
          (1) window_s cannot *guarantee* a citable chunk on its own — the bound depends on
          how long the ASR arm's segments run, so it has to be re-swept if the arm changes.
          The dump prints the longest segment next to the durations for exactly that reason.
          The clean fix belongs to VRAG-019: cite the supporting segment's t_start, not the
          chunk's. Worth deciding there rather than papering over it here.
          (2) `make gate` runs `pytest tests/gates -q`, which does not collect
          gate_phase0.py — the filename does not match pytest's `test_*.py` pattern, so the
          Phase 0 gate only runs when named explicitly. Ran it by hand (9 passed). Not fixed
          here because renaming another person's gate mid-week is their call.
          (3) The board has VRAG-014 assigned to vimal and the MCP authenticates as vimal,
          so start/note/request_review were posted as Vimal again. Same mismatch as VRAG-005.
Next:     VRAG-009 (failure tests: no audio track, zero-length, unreadable codec) is still
          the next unstarted one on the board.
