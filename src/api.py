"""The HTTP API — the same demo, over the wire.

`src.ask` answers a question and writes a static page you open by double-clicking it. That is
the right demo for a supervisor re-running a gate command and the wrong one for a frontend:
there is no page to open, the media reference is a `file://` path a browser will not follow
from an app served over HTTP, and nothing can be polled or retried. This module puts the same
pipeline behind four endpoints and changes none of it.

    make api                       # http://127.0.0.1:8000, /docs for the OpenAPI page
    make api PORT=9000 HOST=0.0.0.0

    uv run python -m src.api --port 8000 --reload

    curl -s localhost:8000/health | python -m json.tool
    curl -s localhost:8000/ask -H 'content-type: application/json' \
        -d '{"question": "what two tools do I need to cut paper?"}'

Endpoints
---------
    GET  /                the frontend (web/), or a redirect to /docs without it
    GET  /health          is there an index, which arm, which config bytes
    POST /ask             question in, answer + citations + provenance out
    GET  /videos          which videos are indexed, and where each can be watched
    GET  /media/{id}      the local media file, range-served so a player can seek

Nothing here decides anything either. Retrieval is VRAG-016, generation and grounding are
VRAG-019, and the padded seek and the source-url fallback are VRAG-020 — this calls
`src.ask.ask(..., write=False)` and translates its result into `schemas.api`. The division is
worth keeping sharp because it says where a bug lives: a wrong answer is VRAG-019, a citation
that points at the wrong second is VRAG-020, and a wrong *status code* or a url a browser
cannot play is this file.

Why the handlers are `def` and not `async def`
---------------------------------------------
Answering makes two blocking calls — an Ollama embed and a hosted completion — and takes
seconds. An `async def` handler that blocks does it on the event loop, which stalls every
other connection including `/health`; a plain `def` handler is run by Starlette in a
threadpool, so a slow answer holds a thread instead of the whole server. The pipeline is
synchronous by design and this is the honest way to serve it, not a stopgap.

Serving the media is a licensing decision, not a convenience
------------------------------------------------------------
`data/corpus/PROVENANCE.md`: no video is in this repo and none can be — Video-MME's terms
forbid redistributing it, so the corpus is pointers. `GET /media/{video_id}` reads files that
`make sample-real` fetched onto *this* machine, which is fine on localhost and is
redistribution the moment the host is public. So it is a lever (`api.serve_media`) rather
than always-on, and `api.cors_origins` is a list you have to write rather than `*`. With the
media off, a citation still carries `source_url` — the manifest url with the timestamp on it,
which resolves against the original upload and is the only copy this project may point at.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Top level, not inside `create_app`, and that is load-bearing rather than tidiness. This
# module uses `from __future__ import annotations`, so every annotation is a string that
# FastAPI resolves against the *module* globals when it builds a route. With `Request`
# imported inside the factory, `def health(request: Request)` resolved to nothing, FastAPI
# fell back to treating an unknown annotation as a query parameter, and /health answered 422
# "query.request: Field required" — a route that is wrong in a way no reading of the handler
# shows. fastapi is a declared dependency (pyproject.toml), so importing it here is a
# promise, not an assumption.
from fastapi import Body, FastAPI, Request
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from schemas.api import (
    AskRequest,
    AskResponse,
    CitationOut,
    Health,
    IndexStatus,
    Problem,
    Provenance,
    Spend,
    Video,
)
from src.answer import AnswerError, AnswerRun, effective_model
from src.ask import (
    SAMPLES,
    AskError,
    Cite,
    ask,
    deep_link,
    load_manifest,
    prompt_fingerprint,
)
from src.config import Config, ConfigError
from src.config import load as load_config
from src.index import local_file
from src.retrieve import RetrieveError
from src.telemetry import Meter

# A corpus video_id is a decimal string in the manifest ("611"). This is asserted rather
# than assumed because the id reaches the filesystem: `local_file` globs `samples/<id>_*`,
# and a `..` or a separator in there is a path-traversal read of anything the process can
# open. Rejecting the id is the fix; sanitising it would leave the question of what it
# sanitised to.
VIDEO_ID = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")

# The frontend: static files, served by this same app. Same origin is the load-bearing part
# and not a shortcut — a page served from here needs no `api.cors_origins` entry to call
# /ask, and a citation's root-relative `stream_url` ("/media/611") resolves against the
# origin it got the JSON from, which is the address already known to work. Served from the
# repo rather than bundled, so editing web/app.js and reloading is the whole edit loop and
# there is no build step to forget. Absent directory is not an error: it is a checkout that
# does not have it, and `/` falls back to the docs the way it did before there was a UI.
WEB = Path(__file__).resolve().parent.parent / "web"


class ApiError(Exception):
    """A request cannot be served. `status` is what the client is told, `hint` what to do."""

    def __init__(
        self,
        status: int,
        message: str,
        hint: str | None = None,
        retry_after_s: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.hint = hint
        self.retry_after_s = retry_after_s


# Groq's free tier is a daily token budget, and the day it runs out every question fails —
# this was not hypothetical, it is how the first live call through this API came back. That
# is a *transient* failure with a known wait, which is the one thing 429 means and 503 does
# not, so it gets its own status and a Retry-After a client can honour instead of a retry
# loop that cannot succeed. The wait is in the provider's own message; the pattern reads it
# rather than guessing, and falls back to no Retry-After when the wording changes.
_RATE_LIMITED = re.compile(r"rate.?limit|429|too many requests", re.IGNORECASE)
_RETRY_IN = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s", re.IGNORECASE)


def _retry_after_s(message: str) -> int | None:
    """Seconds to wait, out of a provider message that says so. None when it does not."""
    match = _RETRY_IN.search(message)
    if not match:
        return None
    minutes = int(match.group(1) or 0)
    # Ceil: a Retry-After that lands a fraction of a second early gets rate-limited again.
    return minutes * 60 + int(float(match.group(2))) + 1


# ---------------------------------------------------------------------------
# What the index holds
# ---------------------------------------------------------------------------


def index_status(cfg: Config) -> IndexStatus:
    """Read the collection without creating it.

    `src.embed._get_collection` is the writer's door: it mkdirs the path and
    get_or_create's the collection, which would make a health check *cause* the empty index
    it is reporting on. This one only ever reads, and an absent store is `ready=False` with
    zero chunks rather than an exception — that is the state a clean clone is in and it is
    not an error, it is the thing the frontend needs to be told.
    """
    name = str(cfg.get("embed.collection"))
    path = Path(cfg.get("embed.chroma_path"))
    empty = IndexStatus(ready=False, collection=name, path=path.as_posix(), chunks=0)

    if not path.exists():
        return empty
    try:
        import chromadb
    except ImportError:
        return empty
    try:
        client = chromadb.PersistentClient(path=str(path))
        collection = client.get_collection(name)
        count = collection.count()
    except Exception:
        # No such collection yet. Same state as no store at all, as far as a client cares.
        return empty
    if count == 0:
        return empty

    return IndexStatus(
        ready=True,
        collection=name,
        path=path.as_posix(),
        chunks=count,
        videos=_indexed_videos(collection),
    )


def _indexed_videos(collection) -> list[str]:
    """The distinct video_ids in the collection.

    Chroma has no DISTINCT, so this reads every row's metadata — O(chunks), which is a few
    hundred rows for the four dev videos and is why it is acceptable here. If the corpus
    grows by an order of magnitude this becomes the thing to cache, not to page through.
    """
    try:
        rows = collection.get(include=["metadatas"])
    except Exception:
        return []
    ids = {
        str(m.get("video_id"))
        for m in (rows.get("metadatas") or [])
        if m and m.get("video_id") is not None
    }
    return sorted(ids)


# ---------------------------------------------------------------------------
# Translating a run into the wire shape
# ---------------------------------------------------------------------------


def media_url(video_id: str, cfg: Config, samples: Path | None = None) -> str | None:
    """`/media/<id>` when this host has the file and is willing to serve it.

    Root-relative and not absolute: an absolute url would bake in the host and scheme this
    process happens to see, which is wrong the first time the API sits behind a reverse
    proxy or is reached over a tunnel. A browser resolves `/media/611` against wherever it
    got the JSON from, which is the address that is known to work.
    """
    if not bool(cfg.get("api.serve_media")):
        return None
    return f"/media/{video_id}" if local_file(video_id, samples or SAMPLES) else None


def to_citation(cite: Cite, cfg: Config) -> CitationOut:
    """One `src.ask.Cite` as the wire object, with both kinds of url resolved."""
    return CitationOut(
        n=cite.n,
        video_id=cite.video_id,
        t_start=cite.t_start,
        t_end=cite.t_end,
        seek_s=cite.seek_s,
        label=cite.label,
        passage=cite.passage,
        stream_url=media_url(cite.video_id, cfg),
        # The padded seek, not t_start: the deep link should land where the player would.
        source_url=(
            deep_link(cite.source.url, cite.seek_s) if cite.source.url else None
        ),
    )


def to_response(
    question: str, run: AnswerRun, cites: list[Cite], cfg: Config, meter: Meter
) -> AskResponse:
    """The whole answer as one JSON object.

    A reply that did not validate is `schema_valid=False` with the reason in `error`, and it
    is still a 200: the request was well formed and the server did its job. Returning 5xx
    would tell a client to retry a question that will fail the same way, and would hide the
    one number VRAG-019 is measured on behind a status code.
    """
    answered = run.answer is not None
    calls = meter._calls
    return AskResponse(
        question=question,
        answer=run.answer.answer if answered else "",
        abstain=run.answer.abstain if answered else False,
        schema_valid=answered,
        error=run.error,
        citations=[to_citation(c, cfg) for c in cites],
        repairs=list(run.repairs),
        spend=Spend(
            calls=len(calls),
            latency_s=round(sum(c.latency_s for c in calls), 3),
            cost_usd=round(sum(c.cost_usd for c in calls), 6),
        ),
        provenance=provenance(run, cfg),
    )


def provenance(run: AnswerRun, cfg: Config) -> Provenance:
    """Which prompt, which config bytes, which models — the CLI's footer as fields."""
    fingerprint = cfg.fingerprint()
    prompt_path, prompt_sha = prompt_fingerprint(cfg)
    return Provenance(
        arm=str(cfg.get("answer.arm")),
        answer_model=run.model or effective_model(cfg),
        embed_model=str(cfg.get("embed.model")),
        top_k=int(cfg.get("retrieve.top_k")),
        retrieved=len(run.hits),
        prompt=prompt_path.as_posix(),
        prompt_sha256=prompt_sha,
        config=fingerprint["path"],
        config_sha256=fingerprint["sha256"],
    )


def videos(cfg: Config) -> list[Video]:
    """Every video the manifest knows or the index holds, and where each can be watched.

    The union rather than either one: the manifest can name a video that was never indexed
    (a clean clone, or the held-out side), and the index can hold one the manifest no longer
    lists if the corpus was re-selected after indexing. A frontend building a video picker
    needs to see both cases, and `indexed` is which is which.
    """
    records = load_manifest()
    status = index_status(cfg)
    indexed = set(status.videos)
    out = []
    for video_id in sorted(set(records) | indexed, key=_numeric):
        record = records.get(video_id) or {}
        url = record.get("url") or None
        out.append(
            Video(
                video_id=video_id,
                split=record.get("split"),
                indexed=video_id in indexed,
                stream_url=media_url(video_id, cfg),
                source_url=url,
            )
        )
    return out


def _numeric(video_id: str) -> tuple[int, str]:
    """Sort '9' before '10'. Corpus ids are decimal strings, so lexical order misleads."""
    return (int(video_id), "") if video_id.isdigit() else (1 << 30, video_id)


def resolve_media(video_id: str, cfg: Config, samples: Path | None = None) -> Path:
    """The file `GET /media/{video_id}` serves, or an `ApiError` saying why there is none.

    `samples=None` resolves to the module-level `SAMPLES` at *call* time rather than binding
    it as a default at import time, which is what lets a test point the whole app at a
    tmp_path of fake media. Binding the default eagerly made the range-serving behaviour
    testable only against real corpus video — i.e. skipped on a clean clone, which is the
    "green suite that ran nothing" this repo has already been bitten by once.
    """
    samples = samples or SAMPLES
    if not bool(cfg.get("api.serve_media")):
        raise ApiError(
            403,
            "this server does not serve corpus media (api.serve_media = false)",
            "use the citation's source_url, which opens the original upload at the same second",
        )
    if not VIDEO_ID.fullmatch(video_id):
        raise ApiError(400, f"{video_id!r} is not a video_id")

    path = local_file(video_id, samples)
    if path is None:
        raise ApiError(
            404,
            f"video {video_id} is not in {samples.as_posix()}/ on this host — the corpus is "
            f"pointers, not copies (data/corpus/PROVENANCE.md)",
            f"make sample-real VIDEO_ID={video_id}",
        )

    # Belt and braces over the id check above: whatever the glob returned has to be inside
    # samples/. A symlink in samples/ pointing out of the tree would pass VIDEO_ID and
    # still read an arbitrary file, and this is the check that catches that.
    root = samples.resolve()
    resolved = path.resolve()
    if root not in resolved.parents:
        raise ApiError(404, f"video {video_id} does not resolve inside {samples.as_posix()}/")
    return resolved


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------

_DESCRIPTION = """\nAsk a question about a corpus of indexed videos. The answer comes back with the video and the
seconds it was taken from, or it comes back declining.

**Three outcomes, all of them 200.** `abstain: true` is the system saying the corpus does not
cover the question — a correct answer, not an error. `schema_valid: false` is the model having
produced something that was not an answer at all, with the reason in `error`. Anything else is
an answer with at least one citation. The 4xx/5xx cases are a malformed request, no index, no
embedding server, no API key, and the free tier's daily token cap (429, with `Retry-After`).

**Playing a citation.** Each one carries `seek_s` — the second to seek to, already padded back
from the chunk boundary so a viewer hears the run-up instead of landing mid-word — and up to
two urls. `stream_url` is media on this host, range-served, so an in-page `<video>` can seek
it. `source_url` is the original upload with the timestamp on it, and it is what a citation has
when the media was never fetched here: this corpus is pointers, not copies. Either can be
null; a citation with both null is still a real citation with nowhere to play it.

**Levers are not request parameters.** Retrieval depth, the model, the prompt and the
temperature live in `config.toml` and are reported back in `provenance` with a sha256 of the
exact bytes, so any answer can be re-run. `/ask` rejects unknown fields rather than ignoring
them, so a client that tries to override one is told so.
"""


def create_app(cfg: Config | None = None):
    """Build the ASGI app. A factory, not a module-level `app`, for two reasons.

    A test needs to point the app at a `Config` built in a tmp_path — one with the media off,
    or a different collection — and a module-level app fixes the config at import time. And
    `uvicorn --reload` re-imports the module per reload, so a module-level app re-reads
    `config.toml` on a schedule nobody chose; a factory re-reads it exactly when asked.
    """
    cfg = cfg or load_config()

    app = FastAPI(
        title="VRAG — video RAG with cited timestamps",
        version="0.1.0",
        summary="Ask a question about the indexed videos; get an answer and the seconds it came from.",
        # Written for whoever is reading /docs, which is not the same audience as the module
        # docstring: that one explains to the next person in this repo why the handlers are
        # synchronous and where a bug lives, and none of that helps someone wiring up a fetch
        # call. Piping the docstring in was the first version and it put four screens of
        # internal rationale at the top of the page.
        description=_DESCRIPTION,
    )
    app.state.cfg = cfg

    origins = [str(o) for o in cfg.get("api.cors_origins")]
    if origins:
        # Explicit origins, never "*". A frontend on a known port is the whole use case, and
        # a wildcard on a server that range-serves corpus media (see the module docstring)
        # lets any page on the internet read it.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["content-type"],
        )

    @app.exception_handler(ApiError)
    def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        """One error shape for everything this app refuses — `schemas.api.Problem`."""
        headers = (
            {"Retry-After": str(exc.retry_after_s)}
            if exc.retry_after_s is not None
            else None
        )
        return JSONResponse(
            status_code=exc.status,
            content=Problem(error=str(exc), hint=exc.hint).model_dump(),
            headers=headers,
        )

    index = WEB / "index.html"

    if index.is_file():
        # StaticFiles, not a route per file: it answers HEAD, sends ETag/Last-Modified and
        # honours If-None-Match, so a reload re-fetches the stylesheet only when it changed.
        # html=False because there is nothing to serve as a directory index here — / is the
        # route below, and a second way to reach the same page is a second thing to keep true.
        app.mount("/static", StaticFiles(directory=str(WEB), html=False), name="static")

        @app.get("/", include_in_schema=False)
        def root() -> FileResponse:
            """The UI. `/docs` is still the API's own page and is linked from the header."""
            return FileResponse(index, media_type="text/html")

    else:

        @app.get("/", include_in_schema=False)
        def root() -> RedirectResponse:
            """No web/ in this checkout — send a browser somewhere useful anyway."""
            return RedirectResponse("/docs")

    @app.get("/health", response_model=Health, tags=["status"])
    def health(request: Request) -> Health:
        """Can this server answer a question right now, and if not, what to run."""
        cfg = request.app.state.cfg
        status = index_status(cfg)
        return Health(
            ready=status.ready,
            detail=(
                "ready"
                if status.ready
                else f"collection {status.collection!r} at {status.path} is empty or absent — "
                "run `make index-dev`. Until then every question would abstain, which looks "
                "like a working demo of nothing."
            ),
            index=status,
            arm=str(cfg.get("answer.arm")),
            answer_model=effective_model(cfg),
            embed_model=str(cfg.get("embed.model")),
            media_served=bool(cfg.get("api.serve_media")),
            config=cfg.fingerprint()["path"],
            config_sha256=cfg.fingerprint()["sha256"],
        )

    @app.post(
        "/ask",
        response_model=AskResponse,
        tags=["ask"],
        responses={
            429: {
                "model": Problem,
                "description": "the free tier's daily budget — carries Retry-After",
            },
            500: {"model": Problem},
            503: {"model": Problem, "description": "no index, no embedder, or no API key"},
        },
    )
    def ask_endpoint(
        request: Request,
        body: AskRequest = Body(
            examples=[{"question": "What two tools do I need to make my first paper cut?"}]
        ),
    ) -> AskResponse:
        """Answer one question with citations, or decline.

        `abstain: true` is a correct outcome and comes back 200. So does
        `schema_valid: false`. The 5xx cases are the ones a client can do nothing about and
        an operator can: no index, no embedding server, no API key.
        """
        cfg = request.app.state.cfg
        question = " ".join(body.question.split())
        if not question:
            raise ApiError(422, "the question is blank")

        meter = Meter()
        try:
            run, cites, _ = ask(question, cfg, meter, write=False)
        except AskError as exc:
            # Chiefly the empty index. `src.ask` refuses rather than answering from nothing.
            raise ApiError(503, str(exc), "make index-dev") from exc
        except RetrieveError as exc:
            raise ApiError(
                503, str(exc), "is ollama running? `ollama serve`, then `make doctor`"
            ) from exc
        except AnswerError as exc:
            # Its message already carries the fix (missing key, unpulled model, bad arm) —
            # except for the free tier's daily cap, which is not a fix but a wait.
            if _RATE_LIMITED.search(str(exc)):
                raise ApiError(
                    429,
                    str(exc),
                    'the free tier is out of tokens for today. Wait, or run the local arm: '
                    'answer.arm = "ollama" in config.toml — no key, no network, no limit.',
                    retry_after_s=_retry_after_s(str(exc)),
                ) from exc
            raise ApiError(503, str(exc), "make doctor") from exc
        except ConfigError as exc:
            raise ApiError(500, str(exc)) from exc

        return to_response(question, run, cites, cfg, meter)

    @app.get("/videos", response_model=list[Video], tags=["status"])
    def videos_endpoint(request: Request) -> list[Video]:
        """Which videos are indexed, and where each one can be watched."""
        return videos(request.app.state.cfg)

    @app.get(
        "/media/{video_id}",
        tags=["media"],
        response_class=FileResponse,
        responses={
            200: {"content": {"video/mp4": {}}, "description": "the whole file"},
            206: {"description": "a byte range — what a seeking player actually asks for"},
            403: {"model": Problem},
            404: {"model": Problem},
        },
    )
    def media(
        request: Request,
        video_id: str = PathParam(description="corpus video_id, e.g. '611'"),
    ) -> FileResponse:
        """The local media file for one video, range-served.

        The range part is the whole point and it is not FastAPI boilerplate: a `<video>`
        element seeking to 7:12 issues `Range: bytes=…` and needs a 206 with a
        `Content-Range` back. A handler that returned the file whole would appear to work —
        the video plays from 0:00 — and the seek would silently do nothing, which is exactly
        the class of failure this project keeps writing tests about. Starlette's
        `FileResponse` implements the 206, `Accept-Ranges` and 416 for a range past the end.
        """
        path = resolve_media(video_id, request.app.state.cfg)
        # `inline`, and that is not a detail: FileResponse's `filename=` alone sends
        # `Content-Disposition: attachment`, and a browser opening /media/611 then downloads
        # 98 MB of video instead of playing it. A `<video>` element ignores the header, so
        # the in-page player worked and only the link was wrong — the sort of half-working
        # that gets found by a person clicking, which is what this comment is instead of.
        return FileResponse(path, filename=path.name, content_disposition_type="inline")

    return app


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--host", default=None, help="default: api.host in config.toml")
    parser.add_argument("--port", type=int, default=None, help="default: api.port")
    parser.add_argument(
        "--reload", action="store_true", help="restart on a source change (development)"
    )
    parser.add_argument(
        "--print-openapi",
        action="store_true",
        help="write the OpenAPI document to stdout and exit — no server, no network",
    )
    args = parser.parse_args(argv)

    # Same reason src/answer.py and src/ask.py do it: a startup banner with an em dash in it
    # must not exit non-zero on a cp1252 console.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"FAIL - {exc}", file=sys.stderr)
        return 1

    if args.print_openapi:
        import json

        print(json.dumps(create_app(cfg).openapi(), indent=2))
        return 0

    host = args.host or str(cfg.get("api.host"))
    port = args.port or int(cfg.get("api.port"))

    status = index_status(cfg)
    ui = "UI at /" if (WEB / "index.html").is_file() else "no web/ — / redirects to /docs"
    print(f"VRAG API on http://{host}:{port}  ({ui}, docs at /docs)")
    print(
        f"  index   {status.chunks} chunk(s) over {len(status.videos)} video(s) "
        f"in {status.collection!r}"
        if status.ready
        else f"  index   EMPTY — {status.collection!r} at {status.path}. Run `make index-dev`; "
        "until then every question abstains."
    )
    print(f"  answer  {cfg.get('answer.arm')} · {effective_model(cfg)}")
    print(f"  media   {'served from samples/' if cfg.get('api.serve_media') else 'off'}")
    print(f"  cors    {list(cfg.get('api.cors_origins')) or 'no browser origin allowed'}")

    try:
        import uvicorn
    except ImportError:
        print("FAIL - uvicorn not installed — run `uv sync`", file=sys.stderr)
        return 1

    if args.reload:
        # --reload needs an import string, not an object: the reloader re-imports in a
        # fresh process. `create_app` takes no argument there, so it re-reads config.toml,
        # which is what you want from a reloader and why --config is ignored under it.
        if Path(args.config) != Path("config.toml"):
            print(
                f"note: --reload re-reads config.toml on each restart, so --config "
                f"{args.config} is not in effect",
                file=sys.stderr,
            )
        uvicorn.run("src.api:create_app", host=host, port=port, reload=True, factory=True)
    else:
        uvicorn.run(create_app(cfg), host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
