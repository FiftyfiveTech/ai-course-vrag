.PHONY: setup doctor corpus corpus-check corpus-pointers heldout-check leakage-check sample sample-real test gate gate-phase1 demo chunks index index-dev clean
.DEFAULT_GOAL := help

# The video the demo ingests, and the file the levers are read from. Both overridable:
#   make demo VIDEO=samples/181_8np5YKYx3sU.mp4
VIDEO ?= samples/one.mp4
CONFIG ?= config.toml

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
	@echo "make test    unit tests"
	@echo "make gate    run every phase gate in tests/gates/"
	@echo "make demo    ingest VIDEO -> wav + sampled frames + media.json"
	@echo "make chunks  dump the chunk table for VIDEO; non-zero if a chunk lost its time range"
	@echo "make index   chunk VIDEO and put its chunks in the Chroma collection"
	@echo "make index-dev  fetch + index all 4 dev videos, then print the index contents"
	@echo "make gate-phase1  just the Phase 1 gate: recall@5 on dev, threshold 0.80"

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

test:
	uv run pytest tests -q --ignore=tests/gates

# Leakage first, and it prints its number before anything else runs: tests/gates/README.md
# says no gate result counts until dev and held-out are known to be disjoint, so a leak has
# to stop the run rather than show up as one red line among the phase gates.
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

# The Phase 1 exit gate on its own. Leakage first for the same reason `make gate` does it:
# tests/gates/README says no gate result counts until dev and held-out are known disjoint.
gate-phase1: leakage-check
	uv run pytest tests/gates/gate_phase1.py -v -s

clean:
	rm -rf .venv .pytest_cache **/__pycache__ .devids
