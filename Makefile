.PHONY: setup doctor corpus corpus-check corpus-pointers heldout-check sample sample-real test gate demo clean
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
	@echo "make sample  generate samples/one.mp4 locally (offline; no video is committed)"
	@echo "make sample-real VIDEO_ID=<id>  fetch one dev corpus video from its recorded url"
	@echo "make test    unit tests"
	@echo "make gate    run every phase gate in tests/gates/"
	@echo "make demo    ingest VIDEO -> wav + sampled frames + media.json"

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

test:
	uv run pytest tests -q --ignore=tests/gates

gate:
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

clean:
	rm -rf .venv .pytest_cache **/__pycache__
