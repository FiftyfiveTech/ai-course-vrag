.PHONY: setup doctor corpus corpus-check corpus-pointers heldout-check leakage-check sample sample-real test gate gate-phase1 gate-phase2a demo chunks index index-dev probe answer answer-dev clean
.DEFAULT_GOAL := help

# The video the demo ingests, and the file the levers are read from. Both overridable:
#   make demo VIDEO=samples/181_8np5YKYx3sU.mp4
VIDEO ?= samples/one.mp4
CONFIG ?= config.toml
QUESTIONS ?= evals/probe_questions.txt

# The question `make answer` asks. Overridable:  make answer Q="how old was Bernini?"
Q ?= What two tools does the presenter say you need to make your first paper cut?

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
	@echo "make probe   ask the index a file of plain questions and read the hits; no score"
	@echo "make answer  Q=\"...\"  answer one question with citations, or abstain"
	@echo "make answer-dev  answer every evals/dev pair; prints the schema-valid tally"
	@echo "make gate-phase2a  the VRAG-019 gate: schema-valid on all of dev + abstention"

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

clean:
	rm -rf .venv .pytest_cache **/__pycache__ .devids
