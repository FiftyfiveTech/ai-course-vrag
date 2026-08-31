.PHONY: overview captions caption-arms sources setup doctor corpus corpus-check corpus-pointers heldout-check leakage-check sample sample-real sample-broken test gate gate-phase1 gate-phase2a demo chunks index index-dev probe answer answer-dev ask api openapi latency graph-check sweep sweep-dry coach clean docker-build docker-up docker-down docker-doctor docker-test docker-gate docker-shell
.DEFAULT_GOAL := help

# The video the demo ingests, and the file the levers are read from. Both overridable:
#   make demo VIDEO=samples/181_8np5YKYx3sU.mp4
VIDEO ?= samples/one.mp4
CONFIG ?= config.toml
QUESTIONS ?= evals/probe_questions.txt

# The question `make answer` and `make ask` put to the corpus. Overridable:
#   make ask Q="how old was Bernini when he met the Pope?"
#
# Prefix it with @<video_id> to answer from one video and nothing else - `make sources` lists
# what can be tagged, and an unknown tag is refused rather than silently ignored:
#   make ask Q="@611 how old was Bernini when he met the Pope?"
Q ?= What two tools does the presenter say you need to make your first paper cut?

# Extra flags for `make ask`. --open also opens the page in a browser; left out of the
# default so that a supervisor re-running the command gets a path and not a browser tab.
ASK_FLAGS ?=

# Where `make api` binds. Both default to the [api] section of config.toml when left empty,
# so the levers stay in one place and these are the per-run override:
#   make api PORT=9000
#   make api HOST=0.0.0.0        # read the licence note in config.toml [api] first
HOST ?=
PORT ?=
API_FLAGS ?=

# Which session `make latency` reports on. Empty means the most recent one that recorded a
# span, which is what you want right after the run you just did:
#   make latency
#   make latency SESSION=20260827-111949-38708
#   make latency LATENCY_FLAGS=--list
SESSION ?=
LATENCY_FLAGS ?=

# Extra flags for `make index` / `make index-dev` — most often --reset.
INDEX_FLAGS ?=

# Extra flags for `make overview` — most often --refresh, which rebuilds a stored one.
OVERVIEW_FLAGS ?=

# Extra flags for `make captions` — most often --arm ollama, or --limit N to cap vision calls.
CAPTION_FLAGS ?=

# Extra flags for `make graph-check`. The two that matter:
#   --no-probe                                      report the config, send nothing to Microsoft
#   --user <organiser upn> --meeting '<join url>'    fetch a real transcript and print its speakers
#   --vtt <file>                                    parse a WebVTT file on disk; no credentials
GRAPH_FLAGS ?=

# Extra flags for `make caption-arms` — most often --limit N (0 for every selected stretch).
ARMS_FLAGS ?=

help:
	@echo "make setup   create the venv and install deps (uv)"
	@echo "make doctor  check every dependency and credential; non-zero on FAIL"
	@echo "make corpus  re-select the 10-video corpus (streams annotations only)"
	@echo "make corpus-check  assert the committed manifest reproduces byte-for-byte"
	@echo "make corpus-pointers  assert all 10 videos still resolve at their source url"
	@echo "make heldout-check  validate the sealed Q&A set and re-hash it against README"
	@echo "make leakage-check  assert evals/dev and evals/heldout share no labels"
	@echo "make sample  generate samples/one.mp4 locally (offline; no video is committed)"
	@echo "make sample-real VIDEO_ID=<id>  fetch one dev corpus video from its recorded url"
	@echo "make sample-broken  write the five deliberately broken ingest fixtures (VRAG-009)"
	@echo "make test    unit tests (tests/unit)"
	@echo "make gate    run every phase gate in tests/gates/"
	@echo "make demo    ingest VIDEO -> wav + sampled frames + media.json"
	@echo "make chunks  dump the chunk table for VIDEO; non-zero if a chunk lost its time range"
	@echo "make index   chunk VIDEO and put its chunks in the Chroma collection"
	@echo "make index-dev  fetch + index all 4 dev videos, then print the index contents"
	@echo "make gate-phase1  just the Phase 1 gate: recall@5 on dev, threshold 0.80"
	@echo "make probe   ask the index a file of plain questions and read the hits; no score"
	@echo "make answer  Q=\"...\"  answer one question with citations, or abstain"
	@echo "make answer-dev  answer every evals/dev pair; prints the schema-valid tally"
	@echo "make gate-phase2a  the VRAG-019 gate: schema-valid on all of dev + abstention"
	@echo "make ask     Q=\"...\"  THE DEMO: answer + a static player that jumps to the citation"
	@echo "make api     the same answer over HTTP, for a frontend to call; /docs for the schema"
	@echo "make openapi print the OpenAPI document and exit; no server, no network"
	@echo "make latency which phase ate the wall clock last session; LATENCY_FLAGS=--list for older"
	@echo "make graph-check  can we read a Teams meeting's own transcript; GRAPH_FLAGS=--no-probe"
	@echo "make overview VIDEO=<file|id>  build the whole-video document the overview mode answers from"
	@echo "make sources which videos an @tag can name; scope a question with make ask Q=\"@611 ...\""
	@echo "make captions VIDEO=<file|id>  read the text off slide-heavy keyframes (VRAG-023)"
	@echo "make caption-arms VIDEO=<file|id>  THE VRAG-023 TABLE: hosted vs local, cost measured"
	@echo "make sweep    re-measure the chunking sweep behind the VRAG-018 primer (~22 min, zero spend)"
	@echo "make sweep-dry  the same grid, chunk counts only - no embedding, seconds"
	@echo "make coach   open the coach page: the sweep as a chart you can move the levers on"
	@echo ""
	@echo "containers (compose.yaml: app + ollama; see the Dockerfile for what is in the image)"
	@echo "make docker-build   build the toolbox image"
	@echo "make docker-up      start ollama, pull the F16 embedding model, start the api"
	@echo "make docker-doctor  the same env check, inside the container; non-zero on FAIL"
	@echo "make docker-test    unit tests inside the container"
	@echo "make docker-gate    every phase gate inside the container — the reproducible run"
	@echo "make docker-down    stop everything"

setup:
	@command -v uv >/dev/null || { echo "uv not installed: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	uv sync
	@test -f .env || { cp .env.example .env; echo "wrote .env from .env.example — fill it in"; }
	@echo "ok. next: make doctor"

doctor:
	uv run python -m src.doctor

corpus:
	uv run python -m src.corpus

# Network check: re-streams the pinned revision and diffs against the committed manifest.
corpus-check:
	uv run python -m src.corpus --check

# We hold pointers, not copies, so a video can vanish out from under us. Run before ingest.
corpus-pointers:
	uv run python -m src.corpus --verify-pointers

# The seal on evals/heldout/. Offline: re-derives the counts and spread from the file and
# compares its sha256 to the one recorded in README.md.
heldout-check:
	uv run python -m src.evalset

# The blind-labelling seal (VRAG-013). Offline. `make gate` asserts the same thing under
# pytest; this prints the intersection size on its own, which is what goes on the card.
leakage-check:
	uv run python -m src.leakage

# tests/unit is the whole unit suite and tests/gates is the phase gates; the split is the
# directory layout now rather than an --ignore flag, because VRAG-009's acceptance criterion
# is the command `pytest tests/unit` and a gate command has to be one that resolves.
test:
	uv run pytest tests/unit -q

# Leakage first, and it prints its number before anything else runs: tests/gates/README.md
# says no gate result counts until dev and held-out are known to be disjoint, so a leak has
# to stop the run rather than show up as one red line among the phase gates.
#
# Since VRAG-019 this makes 15 hosted calls (gate_phase2a) and runs for about two minutes.
# $0.00 on Groq's free tier, but it needs GROQ_API_KEY; a gate that cannot run is a FAIL here
# and not a skip, the same way gate_phase1 fails on a missing index.
gate: leakage-check
	@test -n "$$(ls tests/gates/*.py 2>/dev/null)" || { echo "no gates written yet — see tests/gates/README.md"; exit 1; }
	uv run pytest tests/gates -q

# No video is in git — the licence forbids it — so the fixture is generated, not shipped.
samples/one.mp4:
	uv run python -m src.sample --out $@ --config $(CONFIG)

sample: samples/one.mp4

# Pointers, not copies: this fetches one video from the url recorded in the manifest, which
# is what data/corpus/PROVENANCE.md tells a reproducer to do. Refuses held-out ids.
sample-real:
	@test -n "$(VIDEO_ID)" || { echo "usage: make sample-real VIDEO_ID=<video_id from data/corpus/manifest.json, dev split>"; exit 1; }
	uv run python -m src.sample --real $(VIDEO_ID) --config $(CONFIG)

# The five failure fixtures VRAG-009 tests against: no audio track, no duration, 0 bytes,
# cut off before the moov atom, and noise named .mp4. Each line says what is wrong with the
# file it wrote. Offline; three of the five need no ffmpeg at all.
#
# The tests build their own copies in tmp_path, so this target is for looking at a failure by
# hand: `uv run python -m src.ingest samples/broken/truncated.mp4` should exit 1 and say why.
sample-broken:
	uv run python -m src.sample --broken all --config $(CONFIG)

# VIDEO is a prerequisite so the gate command works from a clean clone: samples/one.mp4 has
# a rule and gets built; any other path has to already exist.
demo: $(VIDEO)
	uv run python -m src.ingest $(VIDEO) --config $(CONFIG)

# The VRAG-014 gate: transcript -> time windows -> chunk table, one row per chunk with its
# video_id and time range, and a non-zero exit if any chunk's range does not hold the
# segments in it. The ASR result is cached per source sha256, so re-running after changing
# chunk.window_s re-chunks for $0.00 and makes no model call.
chunks: $(VIDEO)
	uv run python -m src.chunk $(VIDEO) --config $(CONFIG)

# The step VRAG-015/016 left out: embed_and_persist() had no caller, so the Phase 1 gate
# had an index to score and no way to build one. Chunking is cached per source sha256, so
# re-indexing after a lever change costs $0.00 in ASR.
index: $(VIDEO)
	uv run python -m src.index $(VIDEO) --config $(CONFIG)

# Everything the Phase 1 gate needs, for all four dev videos. Reads the dev split out of
# data/corpus/manifest.json rather than hard-coding ids, so re-selecting the corpus
# (`make corpus`) cannot leave this pointing at videos that are no longer dev. Fetches any
# video that is not already in samples/ (pointers, not copies — PROVENANCE) and refuses
# held-out ids outright.
# INDEX_FLAGS is how --reset gets through. It matters after a chunk lever moves: chunk ids
# carry the timestamps, so the old rows survive a re-index as orphans and the store ends up
# holding two generations that score better than either (VRAG-018 §7, and see `sources`).
index-dev:
	uv run python -m src.index --dev --config $(CONFIG) $(INDEX_FLAGS)

# Unlabelled questions in, hits out, no number. The gate says how often the right moment is
# in the top 5; it cannot say that the right passage keeps landing at rank 4, or that a
# question the corpus never covers still comes back looking confident. QUESTIONS takes a .txt
# (one per line) or a .jsonl with a 'question' field; - reads stdin.
probe:
	uv run python -m src.probe $(QUESTIONS) --config $(CONFIG)

# What `@` accepts in a question. A question that tags a source is answered from that
# source alone - the tag becomes a metadata filter on the Chroma query, so nothing from
# any other video can be retrieved, cited or grounded onto (src/mention.py). This lists
# the handles, and lists what is NOT taggable too, because "why does @091 not work" is
# the question it exists to answer.
#
# No model call and no network; it reads the manifest and the local store.
sources:
	uv run python -m src.mention --config $(CONFIG)

# One question, answered with citations - VRAG-019. Retrieval, then generation constrained by
# schemas/answer.py, then grounding. Needs the index (`make index-dev`) and, on the groq arm,
# GROQ_API_KEY; `answer.arm = "ollama"` in config.toml runs it with neither.
answer:
	uv run python -m src.answer "$(Q)" --config $(CONFIG)

# Every pair in evals/dev, with the schema-valid / abstention tally at the end. Prints no
# accuracy: that is QA_SPEC section 5, scored on evals/heldout by VRAG-021.
answer-dev:
	uv run python -m src.answer --dev --config $(CONFIG)

# The VRAG-019 gate: schema-valid on 100% of evals/dev, and an abstention on every planted
# unanswerable pair. NOT the Phase 2 exit gate - that is VRAG-021 on evals/heldout. Leakage
# first, as every gate here does. 15 hosted calls, ~2.5 min, $0.00 on the free tier.
gate-phase2a: leakage-check
	uv run pytest tests/gates/gate_phase2a.py -v -s

# The Phase 1 exit gate on its own. Leakage first for the same reason `make gate` does it:
# tests/gates/README says no gate result counts until dev and held-out are known disjoint.
gate-phase1: leakage-check
	uv run pytest tests/gates/gate_phase1.py -v -s

# What a whole video IS, as opposed to what it says at one moment. Built once, at index
# time, into runs/<stem>/overview.json, and answered against by `mode: "overview"`.
#
# Reads the chunks back out of Chroma, so the video has to be indexed first (`make index`).
#
# It FOLDS, and that is not an optimisation. No real transcript fits one call on Groq's free
# tier: the tier meters tokens per minute, the limit is 8000, and video 611 in one pass asked
# for 17152. So the transcript is cut into windows of overview.max_context_chars, each is
# summarised on its own, and the partials are merged — people and topics in code, so no span
# can be invented, and one small call for the abstract. Expect a few minutes and one line of
# progress per window on stderr; 611 is 6 windows in ~3 min.
#
#   make overview VIDEO=samples/bob-video.mp4
#   make overview VIDEO=611 OVERVIEW_FLAGS=--refresh
overview:
	uv run python -m src.overview "$(VIDEO)" --config $(CONFIG) $(OVERVIEW_FLAGS)

# VRAG-023, THE STRETCH TASK. Reads the text off the frames where something is holding still
# on screen — a slide, a shared window, a title card — and writes runs/<stem>/captions.json.
#
# Needs the frames ingest already sampled (`make chunks` or `make index`), and nothing else:
# it does not read the transcript and it does not touch the index.
#
# The cost lever is the SELECTION, not the model. `ingest.frames.fps = 0.2` leaves 1091 frames
# for the 91-minute client meeting; captioning all of them is 1091 vision calls. A slide holds
# still, so ffmpeg's per-frame scene score picks out the stretches where the picture barely
# changes and one keyframe stands in for each — 1091 -> 64. The grid behind the two levers that
# do that is in config.toml [caption]; it is what makes "slide-heavy" measured rather than said.
#
# Arms: hosted is NVIDIA NIM (NVIDIA_API_KEY) — Groq serves no vision model, which was checked.
# Local is Ollama, and needs the pull, WITH the tag and from a repo that has an mmproj:
#   ollama pull hf.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M
#
#   make captions VIDEO=samples/vector7-21aug-client-meeting.mp4
#   make captions VIDEO=611 CAPTION_FLAGS="--arm ollama --limit 3"
captions:
	uv run python -m src.caption "$(VIDEO)" --config $(CONFIG) $(CAPTION_FLAGS)

# THE VRAG-023 DELIVERABLE: the same keyframes through both arms, and what each one cost.
#
# Selects once and hands the identical list to both arms, so the arm is the only variable.
# Writes docs/learning/data/caption_arms.json and prints two tables — what was measured, and
# what a whole video is projected to cost at that per-call rate.
#
# Zero spend: NIM's free tier and a local model, $0.00 on both rows. What differs between the
# arms is tokens and seconds, which is what the table is actually for.
#
# Defaults to 10 keyframes per arm — the point is a per-call number, and 20 calls teach what
# 128 would. ARMS_FLAGS="--limit 0" captions every selected stretch.
#
#   make caption-arms VIDEO=samples/vector7-21aug-client-meeting.mp4
#   make caption-arms VIDEO=611 ARMS_FLAGS="--limit 5 --arm ollama"
caption-arms:
	uv run python tools/caption_arms.py "$(VIDEO)" --config $(CONFIG) $(ARMS_FLAGS)

# THE DEMO - VRAG-020. Question in, answer out, and a static HTML page under runs/ask/
# whose citations seek a player to the second they came from. No server, no build step:
# the CSS and JS are inline and the page is opened by double-clicking it.
#
# Needs what `make answer` needs - the index (`make index-dev`) and, on the groq arm,
# GROQ_API_KEY. It refuses outright on an empty index rather than answering from nothing:
# every question would abstain and the demo would look like it works.
#
# The player embeds the local video when samples/ has it and falls back to the manifest
# url with &t= when it does not, because the corpus is pointers and not copies
# (data/corpus/PROVENANCE.md). Both are a clickable timestamp; only one needs the media.
ask:
	uv run python -m src.ask "$(Q)" --config $(CONFIG) $(ASK_FLAGS)

# THE DEMO, over HTTP. Same pipeline as `make ask` — retrieval, generation against the answer
# schema, grounding, the padded seek — behind four endpoints, plus the frontend that calls
# them:
#
#   GET  /                the UI (web/) — a question box, the answer, its citations, a player
#   GET  /health          is there an index, which arm, which config bytes
#   POST /ask             {"question": "..."} -> answer + citations + provenance
#   GET  /videos          which videos are indexed, and where each can be watched
#   GET  /media/{id}      the local media file, range-served so a player can actually seek
#
# Open http://127.0.0.1:8000 and ask. The UI is static files served by this same app, so it
# needs no build step and no api.cors_origins entry — same origin — and editing web/app.js and
# reloading the tab is the whole edit loop. /?q=... asks on load, so a question is a link.
#
# Interactive schema at /docs. Needs what `make ask` needs — the index (`make index-dev`) and,
# on the groq arm, GROQ_API_KEY — but it starts without them and says so on /health, because a
# server that refuses to boot cannot tell a frontend why.
#
# Binds to 127.0.0.1 by default and serves corpus media only off this disk. Both are the
# licence, not caution: the corpus is pointers, not copies (data/corpus/PROVENANCE.md). Read
# the [api] comments in config.toml before putting this on a public address.
#
# Add --reload while working on a handler: `make api API_FLAGS=--reload`.
api:
	uv run python -m src.api --config $(CONFIG) 		$(if $(HOST),--host $(HOST),) $(if $(PORT),--port $(PORT),) $(API_FLAGS)

# The API contract as a file, for generating a frontend client or diffing what changed. No
# server is started and nothing is called, so this works with no index and no key.
# Which part of the pipeline consumed the time, for the last run that recorded any.
#
# Every model call and every instrumented stage appends a span to
# runs/telemetry/<session>.jsonl as it happens, so this works after the process is gone —
# which is the only time anyone asks. `make api` is ended with Ctrl+C and a killed process
# runs no atexit hook, so the log is written as it goes rather than flushed at the end.
#
# Offline: no index, no key, no network. It is a reader over JSONL and will report on a log
# copied off another machine. The row to read first is `unattributed` — wall time inside no
# span. That row is how 1.35s of Chroma client construction was found, back when a 3.59s
# request reported itself as 1.44s because only model calls were instrumented.
latency:
	uv run python -m src.latency $(SESSION) $(LATENCY_FLAGS)

# @, not echoed: this is meant to be piped (`make openapi > openapi.json`), and make's own
# echo of the command line would be the first thing in the file and not valid JSON.
openapi:
	@uv run python -m src.api --config $(CONFIG) --print-openapi

# Microsoft Graph, app-only — can this machine read a Teams meeting's own transcript?
#
# The reason to want that: Teams already transcribed its own meetings AND already knows who
# was speaking. It writes `<v Priya Nair>` voice tags into the VTT. Whisper over an audio
# file cannot recover a name from audio, and diarisation only ever yields anonymous clusters
# ("Speaker 1"), so this is the only free source of ATTRIBUTED text in the project — which is
# what minutes need, because minutes assign commitments to named people.
#
# Prints a doctor-style table, cheapest claim first, and exits non-zero on the first thing
# that is actually wrong: credentials -> token -> roles -> reachability. The roles section
# reads the token's own `roles` claim, which is how "did IT actually grant the permission"
# gets a yes/no without making a call and guessing at a 403 — a 403 has two causes and only
# one of them is a missing role (see src/graph.py GRAPH_HINTS).
#
# Needs GRAPH_TENANT_ID / GRAPH_CLIENT_ID / GRAPH_CLIENT_SECRET; `.env.example` lists the
# names and `make doctor` reports them as optional WARNs, because no gate reads them. Zero
# spend: Graph is covered by the tenant's own licences and nothing here is metered.
#
# The green table is NOT the answer. Whether Teams hands *this* tenant attributed text is a
# per-tenant switch, and the only way to know is to read one real transcript:
#
#   make graph-check
#   make graph-check GRAPH_FLAGS="--user amy@contoso.com --meeting 'https://teams.microsoft.com/l/meetup-join/...'"
#   make graph-check GRAPH_FLAGS="--vtt samples/meeting.vtt"     # offline, no credentials
graph-check:
	uv run python -m src.graph --config $(CONFIG) $(GRAPH_FLAGS)

# The measurements behind the VRAG-018 primer: recall@5, chunk shape and index size across a
# 12-point grid of chunk.window_s x chunk.overlap_s. Writes docs/learning/data/chunking_sweep.json,
# which docs/learning/coach.html reads and docs/learning/primer-chunking-embeddings.md quotes.
#
# Reads config.toml and never writes it, and never touches ./chroma - each grid point gets its
# own store under runs/sweep/. That is deliberate: Phase 1 was graded at window_s = 25.0 /
# overlap_s = 8.0, and a sweep that edited the levers in place would re-tune a passed gate.
#
# Needs the cached transcripts (`make index-dev` once) and Ollama. No ASR call and no hosted
# call: ~20 min of local embedding, $0.00.
sweep:
	uv run python tools/sweep_chunking.py $(SWEEP_FLAGS)

# The same grid with the embedding and the scoring skipped - chunk counts, durations and the
# duplication factor only. Seconds, no Ollama, no store written. Use it to see what a lever
# does to the index before paying 20 minutes to find out what it does to recall.
sweep-dry:
	uv run python tools/sweep_chunking.py --dry-run --out runs/sweep/dry.json

# The coach page - VRAG-018. The sweep as two charts and a lever you can move, plus the four
# checks to run before blaming the chunker. Standalone HTML with the numbers inlined at build
# time by tools/build_coach.py, so it opens off the filesystem with no server and no fetch.
coach: docs/learning/coach.html
	@echo "open docs/learning/coach.html"

docs/learning/coach.html: tools/build_coach.py docs/learning/data/chunking_sweep.json
	uv run python tools/build_coach.py

# ----------------------------------------------------------------------------- containers
#
# The image is a toolbox, not a server: it runs the pipeline, the gates and the API, because
# this repo's discipline is that a supervisor re-runs a gate command and compares output
# (CLAUDE.md). An image that could only serve /ask could not be handed to one.
#
# Two services (compose.yaml): `app`, and `ollama` with its model volume. Ollama is not
# optional — src/embed.py and src/retrieve.py call `ollama.embed` and there is no hosted
# embedding arm, so nothing retrieves without it.
#
# Secrets are read from your shell, never from a committed file. Export them first, or put
# them in ~/.config/ai-course-vrag.env and source it:
#   export GROQ_API_KEY=... HF_TOKEN=... NVIDIA_API_KEY=...

COMPOSE ?= docker compose

docker-build:
	$(COMPOSE) build

# Brings up ollama, waits for it to be healthy, pulls the F16 embedding model into its
# volume, then starts the app. The first run is slow — ~274 MB of weights — and every run
# after it is not, because the volume persists.
docker-up:
	$(COMPOSE) up -d
	@echo "api on http://127.0.0.1:8000  —  next: make docker-doctor"

docker-down:
	$(COMPOSE) down

# Same check the Dockerfile uses as its HEALTHCHECK: ffmpeg, ffprobe, the daemon over
# OLLAMA_HOST, the embedding model's tag, every credential. Non-zero if a required one fails.
# Run this before believing anything else in this section.
docker-doctor:
	$(COMPOSE) exec app make doctor

docker-test:
	$(COMPOSE) exec app make test

# The point of containerising at all: a gate run that does not depend on what is installed
# on the machine that runs it. Needs GROQ_API_KEY in the environment `make docker-up` saw —
# gate_phase2a makes 15 hosted calls and a gate that cannot run is a FAIL, not a skip.
docker-gate:
	$(COMPOSE) exec app make gate

# A shell in the app container, for when a target above is not the question.
docker-shell:
	$(COMPOSE) exec app bash

clean:
	rm -rf .venv .pytest_cache **/__pycache__ .devids
