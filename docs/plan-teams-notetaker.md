# Teams notetaker — containerize, then build, then integrate

Written 2026-08-28. The roadmap for the pivot, and the record of what the first pass
actually landed. Companion to `plan-whole-video-questions.md`, which this plan's Tier 2.2
executes and extends.

## Context

The track is pivoting from *video RAG over a Video-MME corpus* to a **meeting notetaker for
Microsoft Teams**. The capture target is a real-time meeting bot, but there is currently **no
M365 tenant, no admin who can grant consent, and no Azure budget** — so every Microsoft-facing
line of code is blocked on approvals nobody has requested yet.

This plan is therefore ordered by *what is actually unblocked*, in three tiers:

1. **Infra** — containerize this repo so the pipeline runs the same everywhere.
2. **Code** — everything that can be built with zero Microsoft dependency.
3. **Microsoft** — the Teams integration, gated on access.

> **Build scope so far: Tiers 1 and 2.** Tier 3 is documented for sequencing but is **not
> built** — it is blocked on tenant access nobody has yet. The only Tier 3 action available
> now is sending the free access requests, which is paperwork, not code.

### What already works (verified this session, not assumed)

The pipeline has already processed a real meeting end to end:
`runs/vector7-21aug-client-meeting/` holds a 91-minute (5454.7 s) client recording → 1274
whisper segments → 321 chunks in Chroma. Teams-shaped audio in, cited answers out, is proven.

Reusable unchanged: [src/ingest.py](src/ingest.py) (ffmpeg), [src/transcript.py](src/transcript.py)
(whisper-large-v3-turbo on Groq), [src/chunk.py](src/chunk.py), [src/embed.py](src/embed.py) +
[src/index.py](src/index.py), [src/retrieve.py](src/retrieve.py), [src/answer.py](src/answer.py)
(grounded citations + abstain), [src/api.py](src/api.py), [src/telemetry.py](src/telemetry.py) +
[src/latency.py](src/latency.py), and the gate discipline in [tests/gates/](tests/gates/).

### There is paused work in the tree that changes the plan

Uncommitted on this branch: `schemas/overview.py`, `prompts/overview_v1.md`,
`prompts/overview_answer_v1.md`, `src/overview.py`, a `[overview]` config section, and
`docs/plan-whole-video-questions.md` describing all of it.

**This is the notetaker's backbone, already 60% built.** `Overview{abstract, people[], topics[]}`
is whole-transcript *synthesis* with every claim carrying a citable span — exactly the shape
minutes need. `schemas/overview.py` even encodes the speaker problem honestly:
`speaker_labels: Literal[False]`, with the docstring explaining that `people` is *who the
transcript names*, not *who is speaking*, because nothing here diarizes.

So the notetaker **extends `Overview`**, it does not get a parallel `src/notes.py`.

### The three gaps that remain

1. **No speaker attribution.** The live Chroma metadata keys are exactly
   `video_id, t_start, t_end`. `grep -riE 'speaker|diariz|pyannote' src/` finds nothing.
2. **No decisions or action items.** `Overview` has `abstract`/`people`/`topics`; minutes need
   `decisions[]` and `action_items[]`.
3. **Nothing Microsoft.** No `teams|graph|msal|azure` anywhere in the repo.

### Two conflicts to settle with the supervisor, not in code

- **The PRD contradicts the pivot twice.** [docs/VRAG-Video-Rag.md](docs/VRAG-Video-Rag.md) §2
  puts *real-time ingest* out of scope for v1 and forbids *client-recorded sessions* — and a
  client recording is already on disk.
- **There are no notetaker cards on the board.** CLAUDE.md makes the board the source of truth
  for what to do next. Cards must exist before this work is legitimate; proposed names are given
  per phase below.

---

# Tier 1 — Containerize the infra in this repo

**Goal:** `docker compose up` gives a working pipeline on any machine, and the same image
deploys to Railway. Chosen shape: **app + ollama as separate compose services**, one **toolbox
image** that can run the pipeline, the gates *and* the API.

### Why Ollama must be a service, not an optional extra

`src/embed.py:118` and `src/retrieve.py:164` both call `ollama.embed`. **There is no hosted
embedding arm.** Nothing retrieves without a reachable Ollama daemon, so it is a hard service
dependency, not a local-dev convenience. The Python `ollama` client reads `OLLAMA_HOST` from the
environment, and [src/doctor.py:103](src/doctor.py#L103) already honours it — so pointing the app
at another container needs **no code change**, only an env var.

### Files to add

| File | Holds |
|---|---|
| `Dockerfile` | Multi-stage toolbox image. `python:3.12-slim` base, `apt-get install -y ffmpeg` (brings `ffprobe`), `uv` copied from `ghcr.io/astral-sh/uv`. `WORKDIR /app`. |
| `compose.yaml` | Two services (`app`, `ollama`) + four named volumes. |
| `.dockerignore` | The licence boundary — see below. |
| `docker/ollama-init.sh` | Pulls the embedding model with **the `:F16` tag**. |
| `config.deploy.toml` | The deployed levers: `api.host = "0.0.0.0"`, `api.serve_media = false`. |

### The four things that will bite, and what to do about them

**1. `uv.lock` is gitignored and untracked** — `.gitignore:6`, confirmed with
`git ls-files --error-unmatch uv.lock` (fails). A reproducible image needs `uv sync --frozen`,
and a clean clone has no lockfile. **Commit `uv.lock` and drop that `.gitignore` line.** This is
also the honest reading of CLAUDE.md's "`make setup` must work from a clean clone".

**2. The `:F16` tag is not optional.** `config.toml` records the measurement: an untagged
`ollama pull hf.co/<repo>` fetches the smallest file, here `Q2_K`, and recall@5 went
**0.9167 → 0.4167** on 2-bit weights. The init script must pull
`hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F16` exactly — this is the single most important line
in the compose stack and earns a comment saying so.

**3. `make doctor` will fail inside the app container, wrongly.**
[src/doctor.py:153](src/doctor.py#L153) runs `check_binary("ollama", ...)`, but the app container
has no `ollama` binary — only a reachable daemon in the sibling service. **Small code change:
demote the binary check to WARN when `check_ollama_daemon` passes.** `doctor` is otherwise the
perfect container healthcheck: it already verifies ffmpeg, ffprobe, the daemon, the model tag and
every credential, and exits non-zero.

**4. Nothing licence-bearing may enter the image.** `.dockerignore` must exclude `.git/`,
`.venv/`, `__pycache__/`, `.pytest_cache/`, `chroma/`, `runs/`, `samples/`, `*.mp4`, `*.wav`,
and **`.env`**. Media is pointers-not-copies (`data/corpus/PROVENANCE.md`), and CLAUDE.md puts
secrets in `~/.config/`, never the repo. Secrets reach the container as environment variables —
[src/env.py:66](src/env.py#L66) already gives the process environment highest precedence, so this
works with no code change.

### Configuration, without breaking the "levers live in config.toml" rule

`config.toml` sets `api.host = "127.0.0.1"` and `serve_media = true`, and its own comments say
both are **licence decisions, not conveniences**. A container must bind `0.0.0.0`, and a deployed
host must not serve corpus media.

Do **not** make these environment overrides — that would put a lever outside the config
fingerprint that every run is attributed by. Instead ship `config.deploy.toml` and select it with
the `CONFIG` variable the [Makefile](Makefile) already threads through every target
(`CONFIG ?= config.toml`). The run then records *which* config produced it, which is the point of
`Config.fingerprint()`.

### Makefile targets to add

`docker-build`, `docker-up`, `docker-down`, `docker-doctor`, `docker-test`, `docker-gate` —
each a thin wrapper, written in the existing house style where the comment says what the target
needs and what it refuses to do without it.

`docker-gate` matters most: it makes a supervisor's gate re-run reproducible, which is the
repo's whole discipline.

### Volumes

`chroma-data` (expensive to rebuild), `runs-data` (ingest output + telemetry), `samples-data`
(media; never in the image), `ollama-models` (so the F16 pull happens once).

---

# Tier 2 — Code items (zero Microsoft dependency)

Ordered so each step unblocks the next. Proposed cards **NOTE-001 … NOTE-006**.

### 2.1 Diagnose the Groq fallback, and make it fail loudly — NOTE-001

`docs/plan-whole-video-questions.md` records that the first real overview build silently ran on
the local 3B model and produced garbage (`people` contained "Baroque period" and "Rome"; every
span clustered in the last 300 s of a 1800 s video). Its stated next step is to find out why.

**First hypothesis to test:** [src/answer.py](src/answer.py) `_ask` falls back **only** on
`_GroqRateLimitError`. The overview build sends ~50k chars asking for 4000 tokens — on Groq's
free tier that reads like a **tokens-per-minute cap**, not a bad key. Re-run and read the *head*
of stderr (the recorded run had it cut off by a `tail`).

Then add an **`answer.fallback` lever** to `config.toml`, default true locally and **false in
`config.deploy.toml`**. A container that silently degrades from `gpt-oss-120b` to a 3B model is a
quality collapse nobody sees — the overview run is the proof. This is the tie between Tier 1 and
Tier 2 and should land with the Docker work.

### 2.2 Finish the overview feature as specified — NOTE-002

Items 4, 5, 7, 8, 9 and 10 of `docs/plan-whole-video-questions.md` are unstarted and **no tests
exist for what is done**: the `index_video` hook, `make overview`, `mode` on `AskRequest` /
`Provenance`, the `mention._label` fallback, the frontend control, and
`evals/overview/` + `tests/gates/gate_overview.py`.

That document is already a good plan. Execute it rather than re-deciding it — and heed its own
warning: **the new eval file must not go in `evals/dev/`**, because `recall_at_k` and
`leakage.load_split` glob `evals/dev/*.jsonl` and a file dropped there silently moves three
recorded gate numbers.

### 2.3 Extend Overview into Minutes — NOTE-003

Add to `schemas/overview.py`, beside `people` and `topics`:

```
Decision   {text, evidence: Span}
ActionItem {task, owner: str|null, due: str|null, evidence: Span}
```

Extend `prompts/overview_v1.md` to extract them. Reuse everything already there: the strict
`json_schema()` (which already reuses `schemas.answer._inline_refs` / `_strictify`),
`overview.as_chunks` → `answer.ground`, and the existing citation/seek path. `owner` is drawn
from `people[]` — someone the transcript *names* — and `speaker_labels: Literal[False]` already
states in the contract why that is not the same as who spoke.

### 2.4 The speaker contract — NOTE-004

Add `speaker: str | None` to `Chunk` ([src/chunk.py:67](src/chunk.py#L67)), `Chunk`
([src/embed.py:45](src/embed.py#L45)), the `metadatas` dict in `_upsert`
([src/embed.py:174](src/embed.py#L174)), `to_embed_chunks`
([src/index.py:41](src/index.py#L41)), and `RetrievedChunk`
([src/retrieve.py:56](src/retrieve.py#L56)). Populated `None` by the whisper arm.

Add **`src/diarize.py`** with the two-arms-behind-one-interface shape `src/transcript.py` uses
and a `[diarize] arm` lever: ship `"none"` implemented, `"teams"` and `"pyannote"` stubbed with
the reason. `transcript.py`'s comments are the model for documenting an arm that *cannot* work.

**Do the field before the re-index, not after.** Getting real speakers later (Tier 3) then costs
nothing. **Do not rename `video_id` → `meeting_id`** — it is an opaque source id, the rename
touches every module, both eval sets, four gates and the README, and moves recorded numbers for
no product value.

### 2.5 Meetings as first-class sources — NOTE-005

`mention._label` ([src/mention.py:206](src/mention.py#L206)) returns `""` for anything without a
`data/corpus/manifest.json` record, which is why `bob-video` and
`vector7-21aug-client-meeting` show blank in `/videos` and the `@` picker. A meeting registry
replaces the Video-MME manifest as the source of truth for what is indexed.

**Also clear the index while here.** The VRAG-018 standup records `./chroma` holding 908 points —
346 corpus chunks plus two unmanifested videos — and that pollution alone dropped
`make gate-phase1` from 0.9167 to 0.8333.

### 2.6 A sealed minutes eval set and a real gate — NOTE-006 (Evaluator-owned)

`evals/MINUTES_SPEC.md` (what a correct action item is: task + owner + a span within ±30 s of
where it was committed), `evals/minutes/dev_v1.jsonl`, and a sealed
`evals/minutes/heldout_v1.jsonl` tagged `heldout-minutes-v1`. Extend
[src/leakage.py](src/leakage.py) to cover them. The blind-labelling rule applies unchanged.

`tests/gates/gate_minutes.py` — named `gate_*.py` so `pyproject.toml`'s `python_files` collects
it. Three numbers, printed before asserted, in the style of
[tests/gates/gate_phase2a.py](tests/gates/gate_phase2a.py): schema-valid at 1.00, action-item
recall on dev (threshold set from the **first measured run**, not a hopeful number), and
ungrounded-citation rate at 0. Size thresholds with margin — `gate_phase2a` documents that Groq
is not reproducible at temperature 0.0.

---

# Tier 3 — Microsoft-dependent items

### The access track — start today, it is the long pole

Requests to other people, not code. **Items 1, 2, 4, 5 cost nothing.**

1. **An M365 tenant** you can record and transcribe test meetings in.
2. **Tenant admin consent** for `OnlineMeetingTranscript.Read.All`,
   `OnlineMeetingRecording.Read.All`, `CallRecords.Read.All`. Admin-consent-only, and some Teams
   meeting APIs additionally need Microsoft's protected-API approval — **verify which against
   current docs before promising a date.** This is the real critical path.
3. **Azure spend — only if you commit to N4b below.** Defer until Tier 2 has produced a
   minutes-quality number worth building live capture for.
4. **Licensing** — transcription requires it enabled on the meeting and the right M365 licences.
5. **Legal** — recording consent, retention, and whether *client* meetings are in scope at all
   (the PRD says no, and one is already on disk). No media is committed, but a 200 MB client
   recording and its 175 MB wav sit in the working tree.

### N4a — post-meeting fetch (runs on Railway, zero Azure spend)

Pull the transcript/recording after the meeting via Graph
(`/onlineMeetings/{id}/transcripts`, `/recordings`), triggered by a change-notification webhook.
Ordinary authenticated HTTPS — the Tier 1 container hosts it as-is. It also delivers **real
speaker names**, which makes `[diarize] arm = "teams"` real and retires the `None` from 2.4.

**A notetaker that posts minutes two minutes after the call is a complete product. Ship this and
find out whether N4b is needed at all.**

### N4b — real-time media bot (Azure-only; verified, not assumed)

Confirmed against [Microsoft's requirements doc](https://learn.microsoft.com/en-us/microsoftteams/platform/bots/calls-and-meetings/requirements-considerations-application-hosted-media-bots)
(updated 2026-07-22). This is a platform rule, not a budget line, and **no container plan can
route around it**:

- *"The bot must be developed using C# and deployed on Microsoft Azure."* Raw call audio needs
  the `Microsoft.Graph.Communications.Calls.Media` **.NET** library. The doc states that C++ and
  Node.js cannot access real-time media; there is **no Python SDK at all**.
- *"Production application-hosted media bots must be deployed on a Windows Server guest OS in
  Azure"* — Cloud Service, Service Fabric + VMSS, IaaS VM, or AKS only. **"The bot can't be
  deployed as an Azure web app."** If Azure's own PaaS is excluded, Railway and a Linux container
  are categorically out.
- Each instance needs an **instance-level public IP** plus an instance-mapped public port, and
  ≥2 CPU cores. Railway's TCP proxy gives one port per service, no custom certificate, no UDP.
- Still labelled **developer preview**.

**Shape if you go here:** a thin C#/.NET media bot on an Azure Windows VM that does nothing but
join the call and stream audio into this Python pipeline. Second language, second deployment,
and the only part of the system that needs Azure money.

### N5 — delivery into Teams

File export (`minutes.md` / `.json`) is the v1 surface. Posting into a channel needs
`ChannelMessage.Send`, and the PRD's "no auto-publishing, reviewer approval required" rule puts a
human gate in front of it regardless.

---

## Build order (Tiers 1 and 2)

1. **Commit `uv.lock`** and drop it from `.gitignore` — Tier 1 cannot be reproducible without it.
2. **Tier 1**: `Dockerfile`, `compose.yaml`, `.dockerignore`, `docker/ollama-init.sh`,
   `config.deploy.toml`, the `doctor` ollama-binary fix, and the `docker-*` Makefile targets.
3. **2.1** — diagnose the Groq fallback and land the `answer.fallback` lever with the Docker work.
4. **2.2** — finish the overview feature's unstarted items, with tests.
5. **2.3 → 2.5** — Minutes schema, the speaker contract, meetings as first-class sources.
6. **2.6** — hand to the Evaluator to label in parallel; the Builder must not see the held-out set.

In parallel, off the critical path and not code: send access-track items 1, 2, 4, 5, get
**NOTE-001 … NOTE-006** onto the board, and get the PRD amended.

---

## Verification

Nothing counts until a number is printed with the command that produced it (CLAUDE.md).

**Tier 1 — the container is real when the gates pass inside it:**
```bash
docker compose up -d
docker compose exec app make doctor        # must PASS: ffmpeg, ffprobe, daemon, F16 model, keys
docker compose exec app make test          # baseline today: 645 passed, 1 skipped
docker compose exec app make gate          # leakage first, then every phase gate
docker compose exec app make index-dev INDEX_FLAGS=--reset
docker compose exec app make gate-phase1   # recall@5 must still be 0.9167, threshold 0.80
```
The claim to disprove: a **clean clone on a machine with no ffmpeg, no Ollama and no venv**
reaches a passing `make gate` with only `docker compose up`.

**Tier 2:**
```bash
uv run python -m src.overview 611 --config config.toml --refresh   # 2.1: read the HEAD of stderr
make overview VIDEO=samples/vector7-21aug-client-meeting.mp4       # 2.2
make leakage-check                                                 # PASS, overlap 0, before any gate
uv run pytest tests/gates/gate_overview.py -q -s                   # 2.2
uv run pytest tests/gates/gate_minutes.py -q -s                    # 2.6
make latency                                                       # $/meeting-hour, per phase

# regressions — these numbers are recorded and must not move
uv run pytest tests/gates/gate_phase0.py tests/gates/gate_phase1.py \
              tests/gates/gate_phase2a.py tests/gates/test_no_leakage.py -q
```

Note: `gate_phase2.py::test_score_at_least_70_percent` is **already red on a clean tree**
(verified by stashing, per `docs/plan-whole-video-questions.md`). It is a live-model gate,
unrelated to this work, and must not be read as a regression.

**The manual check that matters more than any of the above:** read the minutes for
`vector7-21aug-client-meeting` against the recording. Every action item must name something a
person actually committed to, at a timestamp where they actually said it. An invented commitment
attributed to a named colleague is the worst output this system can produce — which is why
`ground()` is reused rather than rewritten, and why `speaker_labels` is `False` in the contract.

---

# Implementation status — 2026-08-28

## Tier 1 — done, except the build itself

Added: `Dockerfile` (multi-stage toolbox: python:3.12-slim + ffmpeg + make + uv, non-root,
`make doctor` as HEALTHCHECK), `compose.yaml` (`app` + `ollama` + one-shot `ollama-init`,
four volumes), `.dockerignore` (licence boundary: no media, no `.env`), `docker/ollama-init.sh`
(pulls `:F16` and refuses to drift from the `[embed]` lever), six `docker-*` Makefile targets,
`OLLAMA_HOST` in `.env.example`, and `uv.lock` un-ignored in `.gitignore`.

Code change: `src/doctor.py` gained `check_ollama_binary` — a missing `ollama` CLI is a WARN,
not a FAIL, when the daemon answers over `OLLAMA_HOST`. Without it `make doctor` is red inside
the app container, which is exactly where it is the healthcheck. Three tests.

**`docker compose config` validates (exit 0). The image has NOT been built** — Docker Desktop's
daemon is not running on this machine (`npipe:////./pipe/dockerDesktopLinuxEngine` not found).
Start Docker Desktop, then `make docker-build && make docker-up && make docker-doctor`.

**Still to do:** `config.deploy.toml`. `src/config.py` has no layering — `load()` reads one
file and refuses defaults — so a deploy config is either a 900-line copy that drifts, or a
small overlay that needs a `load(path, overlay=)` addition. Not attempted rather than done
badly. It is only needed for a *public* host (`serve_media = false`); local compose is correct
already, because the container binds `0.0.0.0` through the `HOST` override the Makefile
already had.

## Tier 2.1 — done, and the diagnosis was wrong in the plan

The plan guessed "a tokens-per-minute cap". It is a **413, not a 429**, and the difference is
the whole finding:

    413  {'code': 'rate_limit_exceeded', 'type': 'tokens'}
    "Request too large ... on tokens per minute (TPM): Limit 8000, Requested 17152"
    x-ratelimit-remaining-tokens: 8000      <- a FULL bucket
    x-should-retry: false                   <- the provider's own verdict

Groq reports both capacity failures under `rate_limit_exceeded`, so the old string match on
`"rate_limit"` read a permanent condition as throttling and fell back. The bucket was never
empty; the request simply cannot fit. That is why *every* overview build ran on the 3B model.

Landed in `src/answer.py`: `_GroqRequestTooLargeError` separate from `_GroqRateLimitError`,
`_classify_groq_error` switching on HTTP status rather than message text, a 413 that **never**
falls back whatever the lever says, `_groq_retry_after_s` + a bounded 429 retry honouring the
provider's `retry-after`, and the `answer.fallback` lever. Ten tests.

## Tier 2.2 — partly done, and it needed more than "finish as specified"

The overview feature **could not work at all** on this tier, and the fix was the map-reduce
fold its own plan deferred as "future work". Three measured levers, each found by a failed run:

| lever | was | now | why |
|---|---|---|---|
| `overview.max_context_chars` | 180 000 | 10 000 | 180 000 measured the model's *context window*; the binding limit is the tier's *throughput*. Measured 3.531 chars/token on transcript, 2 528 chars fixed overhead. |
| `overview.window_max_tokens` | — (new) | 4 000 | A window is a quarter of a video, but the cap is charged whether used or not. At 2 500 the reply was cut off before `topics` and strict mode rejected it. |
| `answer.reasoning_effort` | — (new) | `"low"` | gpt-oss reasoning tokens are charged against the completion cap. The window that failed finished in 1 539 tokens at `"low"`. |

Added to `src/overview.py`: `windows()` (contiguous, loses no chunk), `_build_one()`, and
`_merge()` — **people and topics merged in code, never by a model**, so no span can be
invented; one small call for the abstract only. New `prompts/overview_merge_v1.md` and
`overview.merge_prompt`. `main()` now catches `AnswerError` and reconfigures stdio to UTF-8
like every other CLI here. New `make overview` target. Thirteen tests in
`tests/unit/test_overview.py`.

**Measured result** — `make overview VIDEO=611 OVERVIEW_FLAGS=--refresh`, 6 windows, 3m16s,
on `openai/gpt-oss-120b`:

    people   24  evidence not an exact chunk range: 0
    topics   72  start not a real chunk start, or end past 1804.8s: 0

against the previous 3B output, whose `people` held "Baroque period", "Victorian England" and
"Rome", every span crammed into the last 300 s of a 1 805 s video.

**Still to do in 2.2:** items 4, 7, 8, 9, 10 of `docs/plan-whole-video-questions.md` — the
`index_video` hook, `mode` on `AskRequest`/`Provenance`, the `mention._label` fallback, the
frontend control, and `evals/overview/` + `tests/gates/gate_overview.py`.

## Tiers 2.3 – 2.6 — not started

Minutes schema, the speaker contract, meetings as first-class sources, and the sealed minutes
eval set.

## Numbers, with the commands

    uv run pytest tests/unit -q                  671 passed, 1 skipped   (was 645, 1)
    make leakage-check                           PASS, overlap 0
    uv run pytest tests/gates/gate_phase0.py tests/gates/gate_phase1.py -q     17 passed
    uv run pytest tests/gates/gate_phase2a.py -q -s                            10 passed

`gate_phase2a` was re-run specifically because `answer.reasoning_effort` is a **global** lever
and the extractive path is what that gate scores. It did not move: schema-valid 1.0000 (15/15),
abstentions 3/3, abstention rate 0.1667 on 12 answerable against a 0.25 ceiling, all 10
citations grounded with 0 repairs, on `openai/gpt-oss-120b`, $0.0000.

`docker compose config` exits 0. No container has been built or run.
