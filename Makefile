.PHONY: setup doctor corpus corpus-check corpus-pointers heldout-check leakage-check sample sample-real sample-broken test gate gate-phase1 gate-phase2a demo chunks index index-dev probe answer answer-dev ask api openapi latency clean
.DEFAULT_GOAL := help

# The video the demo ingests, and the file the levers are read from. Both overridable:
#   make demo VIDEO=samples/181_8np5YKYx3sU.mp4
VIDEO ?= samples/one.mp4
CONFIG ?= config.toml
QUESTIONS ?= evals/probe_questions.txt

# The question `make answer` and `make ask` put to the corpus. Overridable:
#   make ask Q="how old was Bernini when he met the Pope?"
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
index-dev:
	uv run python -m src.index --dev --config $(CONFIG)

# Unlabelled questions in, hits out, no number. The gate says how often the right moment is
# in the top 5; it cannot say that the right passage keeps landing at rank 4, or that a
# question the corpus never covers still comes back looking confident. QUESTIONS takes a .txt
# (one per line) or a .jsonl with a 'question' field; - reads stdin.
probe:
	uv run python -m src.probe $(QUESTIONS) --config $(CONFIG)

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

clean:
	rm -rf .venv .pytest_cache **/__pycache__ .devids
