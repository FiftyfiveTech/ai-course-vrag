.PHONY: setup doctor corpus corpus-check corpus-pointers test gate demo clean
.DEFAULT_GOAL := help

help:
	@echo "make setup   create the venv and install deps (uv)"
	@echo "make doctor  check every dependency and credential; non-zero on FAIL"
	@echo "make corpus  re-select the 10-video corpus (streams annotations only)"
	@echo "make corpus-check  assert the committed manifest reproduces byte-for-byte"
	@echo "make corpus-pointers  assert all 10 videos still resolve at their source url"
	@echo "make test    unit tests"
	@echo "make gate    run every phase gate in tests/gates/"
	@echo "make demo    run the thing end to end"

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

test:
	uv run pytest tests -q --ignore=tests/gates

gate:
	@test -n "$$(ls tests/gates/*.py 2>/dev/null)" || { echo "no gates written yet — see tests/gates/README.md"; exit 1; }
	uv run pytest tests/gates -q

demo:
	@echo "not implemented yet. make demo must run the system end to end from a clean clone."
	@exit 1

clean:
	rm -rf .venv .pytest_cache **/__pycache__
