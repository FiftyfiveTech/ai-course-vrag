# syntax=docker/dockerfile:1.7
#
# The toolbox image — VRAG.
#
# It runs the pipeline, the gates *and* the API, and that is a deliberate choice rather than
# a lazy one. This repo's discipline is that a phase is done when its number is computed and
# printed, and that a supervisor re-runs the gate command and compares output (CLAUDE.md).
# A slim image that can only serve /ask cannot be handed to a supervisor, so the image that
# ships is the image the gates run in.
#
# What it must provide, taken from src/doctor.py, which is the canonical list:
#
#   python >= 3.11   pyproject requires it (tomllib in the stdlib, for config.toml)
#   uv               every Makefile target shells through `uv run`
#   ffmpeg/ffprobe   audio extraction and media metadata (src/ingest.py)
#   make             the Makefile is the interface; `make gate` is what a supervisor runs
#
# What it deliberately does NOT provide is the ollama binary. Ollama is a *service* here,
# reached over OLLAMA_HOST (see compose.yaml) — the python client reads that variable, and
# so does src/doctor.py. Installing a second copy of the daemon in this image would give the
# app a local model store that nothing pulls into and that silently disagrees with the one
# the compose stack maintains.

# ---------------------------------------------------------------------------- dependencies
FROM python:3.12-slim AS deps

# uv from its own distroless image rather than a curl|sh in a layer: it is a pinned,
# checksummed artefact this way, and the install script is a moving target.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

ENV UV_PYTHON=python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Only the two files that decide the dependency set, so a source edit does not re-resolve
# and re-download every wheel. `--frozen` is the point of the layer: it installs exactly
# what uv.lock pins and fails if the lock is missing or stale, rather than quietly resolving
# something new inside a build nobody watches.
#
# uv.lock must therefore be committed. It was in .gitignore until this change; a lockfile
# that is not in the repo cannot pin anything for anyone else.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ---------------------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime

# ffmpeg brings ffprobe with it; both are required and src/doctor.py FAILs without either.
# This is the dependency that makes the image worth having: ffmpeg being installed but off
# PATH is what silently skips ~37 ingest tests and leaves a green suite that tested nothing.
# In here it is on PATH by construction.
#
# `make` because the Makefile is this project's interface. `ca-certificates` because the
# hosted arms (Groq, Hugging Face) are HTTPS and a slim image ships no root store.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        make \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

ENV UV_PYTHON=python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# UV_NO_SYNC: uv would otherwise re-resolve on every `uv run`. The venv is already built and
# correct, and a container that mutates its own dependency set at run time is not the thing
# that was tested.
ENV UV_NO_SYNC=1

# Where the daemon lives on the compose network. Overridden by the environment on any host
# that puts it somewhere else; src/doctor.py:_ollama_host reads exactly this variable.
ENV OLLAMA_HOST=http://ollama:11434

WORKDIR /app

COPY --from=deps /app/.venv /app/.venv

# The project itself. Everything the pipeline reads by relative path — config.toml, prompts/,
# schemas/, evals/, data/corpus/, web/, the Makefile — has to be here, because src/config.py
# reads Path("config.toml") and src/env.py reads Path(".env"), both relative to the working
# directory. WORKDIR is the repo root for that reason and must stay it.
COPY . .

# samples/, runs/ and chroma/ are volumes (compose.yaml) and are excluded from the build
# context by .dockerignore, so they do not exist yet. Create them empty and owned by the
# unprivileged user, so a bind mount or a named volume lands on a directory that is already
# writable rather than one docker creates as root.
RUN useradd --create-home --uid 1000 vrag \
    && mkdir -p samples runs chroma \
    && chown -R vrag:vrag /app

USER vrag

# The doctor is the honest healthcheck: it verifies ffmpeg, ffprobe, the ollama daemon over
# OLLAMA_HOST, the embedding model's tag and every credential, and it exits non-zero on any
# required failure. Nothing else in this repo answers "is this container actually able to
# work" in one command.
#
# start-period is generous because the ollama service pulls ~274 MB of weights on a cold
# volume before it can report the model, and a container marked unhealthy during that pull
# would be restarted into the same pull forever.
HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=3 \
    CMD ["python", "-m", "src.doctor"]

EXPOSE 8000

# Bind 0.0.0.0 rather than the api.host in config.toml, which is 127.0.0.1. In a container
# loopback means "this container and nothing else", so the published port would reach a
# server that is listening on an address nobody can route to.
#
# This is not a quiet reversal of that lever. config.toml argues 127.0.0.1 because
# serve_media = true range-serves corpus media off the disk and the licence forbids
# redistributing it. The container keeps that promise a different way: samples/ is a volume
# you have to mount deliberately, and config.deploy.toml — the config any *public* host must
# run with — sets serve_media = false. Reaching this port still means reaching one host.
CMD ["make", "api", "HOST=0.0.0.0"]
