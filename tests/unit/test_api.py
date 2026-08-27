"""src/api.py — the HTTP surface. No network, no model, no index, no real media.

`src.api` is a translation layer: it calls `src.ask.ask(..., write=False)` and turns the
`AnswerRun` it gets back into `schemas.api.AskResponse`. So everything below stubs that one
call and drives the translation, exactly the way `tests/test_ask.py` stubs it and drives the
rendering. Whether the answer is *good* is `tests/gates/gate_phase2a.py`; whether it is
*correctly served* is here.

The failures this file exists to catch are the ones a browser sees and a unit test of
`src.answer` never would:

* **a status code that lies.** An abstention is a correct outcome (QA_SPEC §4) and must not
  arrive as a 5xx, or a frontend reports a bug that is not there. Conversely an empty index
  must not arrive as a cheerful 200 full of abstentions, and the free tier's daily cap must
  arrive as a 429 with a wait rather than a 503 a client retries forever.
* **a url a player cannot use.** A seeking `<video>` sends `Range:` and needs a 206 with a
  `Content-Range` back; a handler that returns the whole file every time *looks* like it
  works — the video plays from 0:00 — and the seek silently does nothing.
* **serving what may not be served.** `data/corpus/PROVENANCE.md`: the corpus is pointers,
  not copies. With `api.serve_media = false` no response may carry a stream url and
  `/media/…` must refuse, and a video_id must never reach the filesystem unvalidated.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from schemas.answer import Answer
from schemas.api import IndexStatus
from src.answer import AnswerError, AnswerRun
from src.ask import Cite, Source, prompt_fingerprint
from src.config import load as load_config
from src.index import local_file
from src.retrieve import RetrieveError, RetrievedChunk
from src.api import (
    WEB,
    ApiError,
    _numeric,
    _retry_after_s,
    create_app,
    index_status,
    media_url,
    resolve_media,
    to_citation,
)

PROMPT = "prompts/answer_v1.md"


# ---------------------------------------------------------------------------
# Fixtures — a config in tmp_path, and the pieces src.ask would have produced
# ---------------------------------------------------------------------------


def write_config(
    tmp_path: Path,
    *,
    serve_media: bool = True,
    cors_origins: list[str] | None = None,
    chroma_path: str | None = None,
    collection: str = "vrag-test",
):
    """A whole config, because `Config.get` has no defaults for a lever — by design.

    Every key the API path touches has to be here or the request fails with a ConfigError,
    which is the guarantee `src/config.py` exists to make and the reason this helper is
    long rather than a two-line stub.
    """
    origins = cors_origins if cors_origins is not None else []
    path = tmp_path / "config.toml"
    path.write_text(
        "[embed]\n"
        f'model = "nomic-ai/nomic-embed-text-v1.5-GGUF:F16"\n'
        f'collection = "{collection}"\n'
        f'chroma_path = "{chroma_path or (tmp_path / "chroma").as_posix()}"\n'
        "\n[retrieve]\ntop_k = 5\n"
        "\n[answer]\n"
        'arm = "groq"\n'
        'model = "openai/gpt-oss-120b"\n'
        f'prompt = "{PROMPT}"\n'
        "temperature = 0.0\n"
        "max_tokens = 1200\n"
        "\n[ask]\npad_s = 5.0\n"
        "\n[api]\n"
        'host = "127.0.0.1"\n'
        "port = 8000\n"
        f"cors_origins = {origins!r}\n"
        f"serve_media = {str(serve_media).lower()}\n",
        encoding="utf-8",
    )
    return load_config(path)


def client(cfg) -> TestClient:
    return TestClient(create_app(cfg))


def chunk(video_id="611", t_start=20.0, t_end=45.0, text="Bernini was eight."):
    return RetrievedChunk(
        video_id=video_id, t_start=t_start, t_end=t_end, text=text, score=0.3
    )


def a_cite(
    n=1,
    video_id="611",
    t_start=20.0,
    t_end=45.0,
    seek_s=15.0,
    passage="Bernini was eight.",
    local: Path | None = None,
    url: str | None = "https://www.youtube.com/watch?v=abc",
) -> Cite:
    return Cite(
        n=n,
        video_id=video_id,
        t_start=t_start,
        t_end=t_end,
        seek_s=seek_s,
        passage=passage,
        source=Source(video_id=video_id, local=local, url=url),
    )


def a_run(answer_obj: Answer | None, hits=None, error=None, repairs=None) -> AnswerRun:
    return AnswerRun(
        question="q",
        hits=hits if hits is not None else [chunk()],
        raw="{}",
        answer=answer_obj,
        error=error,
        repairs=repairs or [],
        tokens=10,
    )


def stub_ask(monkeypatch, *, run: AnswerRun, cites: list[Cite] | None = None):
    """Replace the one call src.api makes into the pipeline."""

    def fake(question, cfg, meter, *, out_dir=None, write=True):
        assert write is False, "the API must not write a page to disk"
        return run, list(cites or []), None

    monkeypatch.setattr("src.api.ask", fake)


def stub_ask_raising(monkeypatch, exc: Exception):
    def fake(question, cfg, meter, *, out_dir=None, write=True):
        raise exc

    monkeypatch.setattr("src.api.ask", fake)


def fake_media(tmp_path: Path, video_id="611", size=8192) -> Path:
    """A file shaped like one `make sample-real` fetched: `samples/<id>_<ytid>.<ext>`."""
    samples = tmp_path / "samples"
    samples.mkdir(exist_ok=True)
    path = samples / f"{video_id}_abcdefgh.mp4"
    path.write_bytes(bytes(range(256)) * (size // 256))
    return path


def answered(text="He was eight years old.", citations=None) -> Answer:
    return Answer(
        answer=text,
        citations=citations if citations is not None else [
            {"video_id": "611", "t_start": 20.0, "t_end": 45.0}
        ],
        abstain=False,
    )


# ---------------------------------------------------------------------------
# The request contract
# ---------------------------------------------------------------------------


def test_a_question_is_required(tmp_path):
    assert client(write_config(tmp_path)).post("/ask", json={}).status_code == 422


def test_an_empty_question_is_refused(tmp_path):
    r = client(write_config(tmp_path)).post("/ask", json={"question": ""})
    assert r.status_code == 422


def test_a_whitespace_question_is_refused_before_any_model_call(tmp_path, monkeypatch):
    # min_length=1 is satisfied by " ", so this is the app's own check and it has to run
    # before the pipeline: a blank question would otherwise cost a retrieval and a completion.
    stub_ask_raising(monkeypatch, AssertionError("the pipeline must not be reached"))
    r = client(write_config(tmp_path)).post("/ask", json={"question": "   "})
    assert r.status_code == 422
    assert "blank" in r.json()["error"]


def test_a_per_request_lever_is_refused_rather_than_ignored(tmp_path):
    # The whole point of extra="forbid" on AskRequest: a client that thinks it can override
    # top_k is told it cannot, instead of watching the field do nothing.
    r = client(write_config(tmp_path)).post("/ask", json={"question": "q", "top_k": 9})
    assert r.status_code == 422


def test_a_question_cannot_be_unbounded(tmp_path):
    r = client(write_config(tmp_path)).post("/ask", json={"question": "x" * 1001})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# The three outcomes, and their status codes
# ---------------------------------------------------------------------------


def test_an_answer_comes_back_with_its_citations(tmp_path, monkeypatch):
    stub_ask(monkeypatch, run=a_run(answered()), cites=[a_cite(), a_cite(n=2, t_start=90.0)])
    r = client(write_config(tmp_path)).post("/ask", json={"question": "how old?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "He was eight years old."
    assert body["abstain"] is False
    assert body["schema_valid"] is True
    assert body["error"] is None
    assert [c["n"] for c in body["citations"]] == [1, 2]


def test_an_abstention_is_a_200_and_not_an_error(tmp_path, monkeypatch):
    # QA_SPEC §4: declining is a correct outcome. A 5xx here would have a frontend reporting
    # a server fault every time the corpus honestly does not cover the question.
    stub_ask(monkeypatch, run=a_run(Answer.abstention()), cites=[])
    r = client(write_config(tmp_path)).post("/ask", json={"question": "who won in 2027?"})
    assert r.status_code == 200
    body = r.json()
    assert body["abstain"] is True
    assert body["citations"] == []
    assert body["schema_valid"] is True


def test_a_schema_invalid_reply_is_a_200_that_says_so(tmp_path, monkeypatch):
    stub_ask(monkeypatch, run=a_run(None, error="citations.0.t_end: not a number"), cites=[])
    r = client(write_config(tmp_path)).post("/ask", json={"question": "q"})
    assert r.status_code == 200
    body = r.json()
    assert body["schema_valid"] is False
    assert body["error"] == "citations.0.t_end: not a number"
    assert body["answer"] == ""
    assert body["abstain"] is False


def test_grounding_repairs_are_reported_rather_than_hidden(tmp_path, monkeypatch):
    stub_ask(
        monkeypatch,
        run=a_run(answered(), repairs=["dropped video 999 12.0s-13.0s - no retrieved passage"]),
        cites=[a_cite()],
    )
    body = client(write_config(tmp_path)).post("/ask", json={"question": "q"}).json()
    assert body["repairs"] == ["dropped video 999 12.0s-13.0s - no retrieved passage"]


# ---------------------------------------------------------------------------
# Citations: the two kinds of url, and neither
# ---------------------------------------------------------------------------


def test_a_fetched_video_gets_a_stream_url(tmp_path, monkeypatch):
    fake_media(tmp_path)
    monkeypatch.setattr("src.api.SAMPLES", tmp_path / "samples")
    cite = to_citation(a_cite(), write_config(tmp_path))
    assert cite.stream_url == "/media/611"


def test_an_unfetched_video_gets_only_the_source_url(tmp_path, monkeypatch):
    monkeypatch.setattr("src.api.SAMPLES", tmp_path / "samples")
    cite = to_citation(a_cite(), write_config(tmp_path))
    assert cite.stream_url is None
    assert cite.source_url is not None


def test_the_source_url_opens_at_the_padded_second(tmp_path, monkeypatch):
    # Not t_start. A citation names a chunk boundary; the padded seek is where a viewer
    # should land, and the deep link has to agree with the in-page player about which.
    monkeypatch.setattr("src.api.SAMPLES", tmp_path / "samples")
    cite = to_citation(a_cite(t_start=20.0, seek_s=15.0), write_config(tmp_path))
    assert cite.source_url is not None and cite.source_url.endswith("t=15s")


def test_a_citation_with_nowhere_to_go_has_both_urls_null(tmp_path, monkeypatch):
    monkeypatch.setattr("src.api.SAMPLES", tmp_path / "samples")
    cite = to_citation(a_cite(url=None), write_config(tmp_path))
    assert cite.stream_url is None and cite.source_url is None


def test_the_stream_url_is_root_relative_so_a_proxy_cannot_break_it(tmp_path, monkeypatch):
    fake_media(tmp_path)
    monkeypatch.setattr("src.api.SAMPLES", tmp_path / "samples")
    url = media_url("611", write_config(tmp_path))
    assert url == "/media/611"
    assert not url.startswith("http")


def test_media_off_strips_every_stream_url(tmp_path, monkeypatch):
    fake_media(tmp_path)
    monkeypatch.setattr("src.api.SAMPLES", tmp_path / "samples")
    cfg = write_config(tmp_path, serve_media=False)
    assert media_url("611", cfg) is None
    assert to_citation(a_cite(), cfg).stream_url is None


def test_the_citation_carries_the_passage_and_the_clock_label(tmp_path, monkeypatch):
    monkeypatch.setattr("src.api.SAMPLES", tmp_path / "samples")
    cite = to_citation(a_cite(t_start=79.5, t_end=85.0), write_config(tmp_path))
    assert cite.passage == "Bernini was eight."
    assert cite.label == "video 611 · 1:19–1:25"


# ---------------------------------------------------------------------------
# The failures an operator can fix, and the one that is only a wait
# ---------------------------------------------------------------------------


def test_an_empty_index_is_a_503_that_names_the_command(tmp_path, monkeypatch):
    from src.ask import AskError

    stub_ask_raising(monkeypatch, AskError("retrieval returned no passages"))
    r = client(write_config(tmp_path)).post("/ask", json={"question": "q"})
    assert r.status_code == 503
    assert r.json()["hint"] == "make index-dev"


def test_no_embedding_server_is_a_503_saying_so(tmp_path, monkeypatch):
    stub_ask_raising(monkeypatch, RetrieveError("ollama embed failed: connection refused"))
    r = client(write_config(tmp_path)).post("/ask", json={"question": "q"})
    assert r.status_code == 503
    assert "ollama" in r.json()["hint"]


def test_a_missing_key_is_a_503_and_not_a_500(tmp_path, monkeypatch):
    stub_ask_raising(monkeypatch, AnswerError("GROQ_API_KEY is not set."))
    r = client(write_config(tmp_path)).post("/ask", json={"question": "q"})
    assert r.status_code == 503


def test_the_free_tiers_daily_cap_is_a_429_with_a_retry_after(tmp_path, monkeypatch):
    # The real message, from the first live call made through this API.
    stub_ask_raising(
        monkeypatch,
        AnswerError(
            "groq arm failed (openai/gpt-oss-120b): Error code: 429 - {'error': {'message': "
            "'Rate limit reached for model `openai/gpt-oss-120b` ... on tokens per day (TPD): "
            "Limit 200000, Used 198971, Requested 1758. Please try again in 5m14.928s.'}}"
        ),
    )
    r = client(write_config(tmp_path)).post("/ask", json={"question": "q"})
    assert r.status_code == 429
    assert r.headers["retry-after"] == "315"
    assert 'answer.arm = "ollama"' in r.json()["hint"]


def test_a_rate_limit_with_no_stated_wait_still_answers_429(tmp_path, monkeypatch):
    stub_ask_raising(monkeypatch, AnswerError("groq arm failed: 429 Too Many Requests"))
    r = client(write_config(tmp_path)).post("/ask", json={"question": "q"})
    assert r.status_code == 429
    assert "retry-after" not in r.headers


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Please try again in 5m14.928s.", 315),
        ("try again in 12.5s", 13),
        ("try again in 2m0.0s", 121),
        ("no wait is stated here", None),
    ],
)
def test_the_wait_is_read_from_the_providers_message(message, expected):
    assert _retry_after_s(message) == expected


def test_every_refusal_has_the_same_body_shape(tmp_path, monkeypatch):
    # FastAPI's default is `{"detail": ...}`, sometimes a string and sometimes a list, so a
    # client ends up type-switching on it. Everything this app refuses is {error, hint}.
    stub_ask_raising(monkeypatch, AnswerError("GROQ_API_KEY is not set."))
    cfg = write_config(tmp_path)
    c = client(cfg)
    for response in (
        c.post("/ask", json={"question": "q"}),
        c.get("/media/999"),
        c.get("/media/not a video id"),
    ):
        assert response.status_code >= 400
        assert set(response.json()) == {"error", "hint"}, response.json()


# ---------------------------------------------------------------------------
# Media: the range request is the feature
# ---------------------------------------------------------------------------


def test_a_seeking_player_gets_a_206_and_a_content_range(tmp_path, monkeypatch):
    # This is the test the endpoint exists for. A handler that ignored Range would pass a
    # naive "the video loads" check and make every citation seek to 0:00.
    media = fake_media(tmp_path)
    monkeypatch.setattr("src.api.SAMPLES", tmp_path / "samples")
    r = client(write_config(tmp_path)).get("/media/611", headers={"Range": "bytes=100-199"})
    assert r.status_code == 206
    assert r.headers["content-range"] == f"bytes 100-199/{media.stat().st_size}"
    assert r.headers["accept-ranges"] == "bytes"
    assert len(r.content) == 100


def test_a_range_past_the_end_is_a_416(tmp_path, monkeypatch):
    media = fake_media(tmp_path)
    monkeypatch.setattr("src.api.SAMPLES", tmp_path / "samples")
    r = client(write_config(tmp_path)).get(
        "/media/611", headers={"Range": f"bytes={media.stat().st_size + 10}-"}
    )
    assert r.status_code == 416


def test_no_range_serves_the_whole_file_as_video(tmp_path, monkeypatch):
    media = fake_media(tmp_path)
    monkeypatch.setattr("src.api.SAMPLES", tmp_path / "samples")
    r = client(write_config(tmp_path)).get("/media/611")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert int(r.headers["content-length"]) == media.stat().st_size


def test_the_media_is_served_to_be_played_and_not_downloaded(tmp_path, monkeypatch):
    # FileResponse(filename=...) alone sends Content-Disposition: attachment, and then
    # opening the url downloads the video instead of playing it. A <video> element ignores
    # the header, so this is invisible from the in-page player and obvious from the link.
    fake_media(tmp_path)
    monkeypatch.setattr("src.api.SAMPLES", tmp_path / "samples")
    r = client(write_config(tmp_path)).get("/media/611")
    assert r.headers["content-disposition"].startswith("inline")


def test_a_video_that_was_never_fetched_is_a_404_with_the_fetch_command(tmp_path, monkeypatch):
    monkeypatch.setattr("src.api.SAMPLES", tmp_path / "samples")
    r = client(write_config(tmp_path)).get("/media/611")
    assert r.status_code == 404
    assert r.json()["hint"] == "make sample-real VIDEO_ID=611"


def test_media_off_refuses_and_points_at_the_source_url(tmp_path, monkeypatch):
    fake_media(tmp_path)
    monkeypatch.setattr("src.api.SAMPLES", tmp_path / "samples")
    r = client(write_config(tmp_path, serve_media=False)).get("/media/611")
    assert r.status_code == 403
    assert "source_url" in r.json()["hint"]


@pytest.mark.parametrize(
    "video_id",
    ["../config.toml", "..%2F..%2Fconfig.toml", "611 611", "6*", "", "x" * 65],
)
def test_a_video_id_that_is_not_one_never_reaches_the_filesystem(video_id, tmp_path):
    # `local_file` globs samples/<id>_*, so an unvalidated id is a path-traversal read.
    # Whether the router 404s the path or resolve_media 400s the id, what must not happen is
    # a 200 with a file in it.
    cfg = write_config(tmp_path)
    assert client(cfg).get(f"/media/{video_id}").status_code in (400, 404, 405)
    if video_id:
        with pytest.raises(ApiError):
            resolve_media(video_id, cfg, samples=tmp_path / "samples")


def test_a_fetched_file_outside_samples_is_refused(tmp_path, monkeypatch):
    # The id check passes and the glob still returns something outside the tree — a symlink
    # in samples/, or a samples/ that is itself one. The resolved-path check is what catches
    # it, and it is only a duplicate of the id check until the day it is not.
    samples = tmp_path / "samples"
    samples.mkdir()
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"x")
    try:
        (samples / "611_link.mp4").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need a privileged account or developer mode on Windows")
    with pytest.raises(ApiError):
        resolve_media("611", write_config(tmp_path), samples=samples)


def test_the_endpoint_and_the_citation_agree_on_which_file(tmp_path, monkeypatch):
    media = fake_media(tmp_path)
    monkeypatch.setattr("src.api.SAMPLES", tmp_path / "samples")
    cfg = write_config(tmp_path)
    url = media_url("611", cfg)
    assert url is not None
    assert client(cfg).get(url).status_code == 200
    assert resolve_media("611", cfg, samples=tmp_path / "samples") == media.resolve()


# ---------------------------------------------------------------------------
# /health — what a frontend asks before it shows a question box
# ---------------------------------------------------------------------------


def test_health_says_not_ready_and_what_to_run_when_there_is_no_index(tmp_path):
    body = client(write_config(tmp_path)).get("/health").json()
    assert body["ready"] is False
    assert body["index"]["chunks"] == 0
    assert "make index-dev" in body["detail"]


def test_a_health_check_does_not_create_the_store_it_reports_on(tmp_path):
    # `src.embed._get_collection` mkdirs and get_or_creates. Reusing it here would have a
    # health check bring into being the empty index it is reporting, and the second call
    # would report a different (still useless) state than the first.
    chroma = tmp_path / "chroma"
    cfg = write_config(tmp_path, chroma_path=chroma.as_posix())
    assert client(cfg).get("/health").json()["ready"] is False
    assert not chroma.exists()
    assert index_status(cfg).chunks == 0


def test_health_reports_the_arm_the_models_and_the_config_bytes(tmp_path):
    cfg = write_config(tmp_path)
    body = client(cfg).get("/health").json()
    assert body["arm"] == "groq"
    assert body["answer_model"] == "openai/gpt-oss-120b"
    assert body["embed_model"] == "nomic-ai/nomic-embed-text-v1.5-GGUF:F16"
    assert body["config_sha256"] == cfg.fingerprint()["sha256"]


def test_health_says_whether_media_is_served(tmp_path):
    assert client(write_config(tmp_path)).get("/health").json()["media_served"] is True
    assert (
        client(write_config(tmp_path, serve_media=False)).get("/health").json()["media_served"]
        is False
    )


# ---------------------------------------------------------------------------
# /videos — the union of the manifest and the index
# ---------------------------------------------------------------------------


def fake_index(monkeypatch, videos: list[str], collection="vrag-test"):
    monkeypatch.setattr(
        "src.api.index_status",
        lambda cfg: IndexStatus(
            ready=bool(videos),
            collection=collection,
            path="chroma",
            chunks=len(videos) * 10,
            videos=sorted(videos),
        ),
    )


def test_a_manifest_video_that_is_not_indexed_is_listed_as_such(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.api.load_manifest",
        lambda: {"611": {"video_id": "611", "url": "https://y/611", "split": "dev"}},
    )
    fake_index(monkeypatch, [])
    monkeypatch.setattr("src.api.SAMPLES", tmp_path / "samples")
    body = client(write_config(tmp_path)).get("/videos").json()
    assert body == [
        {
            "video_id": "611",
            "split": "dev",
            "indexed": False,
            "stream_url": None,
            "source_url": "https://y/611",
        }
    ]


def test_an_indexed_video_the_manifest_forgot_is_still_listed(tmp_path, monkeypatch):
    # The corpus can be re-selected after an index was built, and then the index holds a
    # video the manifest no longer names. Listing only the manifest would hide it, and it is
    # the one that can actually be cited.
    monkeypatch.setattr("src.api.load_manifest", lambda: {})
    fake_index(monkeypatch, ["701"])
    monkeypatch.setattr("src.api.SAMPLES", tmp_path / "samples")
    body = client(write_config(tmp_path)).get("/videos").json()
    assert [v["video_id"] for v in body] == ["701"]
    assert body[0]["indexed"] is True
    assert body[0]["split"] is None
    assert body[0]["source_url"] is None


@pytest.mark.parametrize(
    "ids, expected",
    [
        # Zero-padded ids are how the manifest spells them, and "091" sorts as 91.
        (["10", "9", "091"], ["9", "10", "091"]),
        (["611", "bob-video", "181"], ["181", "611", "bob-video"]),
    ],
)
def test_video_ids_sort_numerically_not_lexically(ids, expected):
    assert sorted(ids, key=_numeric) == expected


# ---------------------------------------------------------------------------
# Provenance — the same lines the CLI prints
# ---------------------------------------------------------------------------


def test_the_response_records_what_produced_it(tmp_path, monkeypatch):
    stub_ask(monkeypatch, run=a_run(answered(), hits=[chunk(), chunk()]), cites=[a_cite()])
    cfg = write_config(tmp_path)
    body = client(cfg).post("/ask", json={"question": "q"}).json()
    p = body["provenance"]
    assert p["arm"] == "groq"
    assert p["answer_model"] == "openai/gpt-oss-120b"
    assert p["top_k"] == 5
    assert p["retrieved"] == 2
    assert p["config_sha256"] == cfg.fingerprint()["sha256"]


def test_the_api_and_the_demo_page_name_the_same_prompt(tmp_path, monkeypatch):
    # Both read it through src.ask.prompt_fingerprint. Computing the digest in two places is
    # how a footer and a JSON response end up disagreeing about which prompt ran.
    stub_ask(monkeypatch, run=a_run(answered()), cites=[a_cite()])
    cfg = write_config(tmp_path)
    body = client(cfg).post("/ask", json={"question": "q"}).json()
    path, sha = prompt_fingerprint(cfg)
    assert body["provenance"]["prompt"] == path.as_posix() == PROMPT
    assert body["provenance"]["prompt_sha256"] == sha != "unreadable"


def test_the_spend_is_this_request_and_not_the_process(tmp_path, monkeypatch):
    # One Meter per request. A process-wide meter would make the second question look like
    # it cost the first one too, and the $/question line on a frontend would only ever rise.
    # Asserted field by field rather than against a whole dict: this test used to pin the
    # exact shape of `spend` and broke the day `wall_s` was added, which told nobody anything.
    stub_ask(monkeypatch, run=a_run(answered()), cites=[a_cite()])
    c = client(write_config(tmp_path))
    first = c.post("/ask", json={"question": "q"}).json()["spend"]
    second = c.post("/ask", json={"question": "q2"}).json()["spend"]
    assert first["calls"] == second["calls"] == 0
    assert first["cost_usd"] == second["cost_usd"] == 0.0
    assert first["latency_s"] == second["latency_s"]


def test_the_response_separates_model_time_from_request_time(tmp_path, monkeypatch):
    # `latency_s` is the sum of model-call latency and is NOT the request duration — on a
    # cold request it understated the wall clock by 60%, because Chroma client construction
    # is not a model call. Both numbers are reported so a client can show the honest one.
    stub_ask(monkeypatch, run=a_run(answered()), cites=[a_cite()])
    spend = client(write_config(tmp_path)).post("/ask", json={"question": "q"}).json()["spend"]
    assert set(spend) == {"calls", "latency_s", "wall_s", "cost_usd", "phases"}
    assert spend["wall_s"] >= 0.0


def test_the_response_ranks_its_phases_slowest_first(tmp_path, monkeypatch):
    """The per-request half of `make latency`, so a frontend needs no log file."""
    from src.telemetry import Meter

    def fake(question, cfg, meter, *, out_dir=None, write=True):
        meter.log("openai/gpt-oss-120b", 1.5, tokens=10, phase="answer.generate")
        with meter.stage("retrieve.query"):
            pass
        meter.log("nomic-ai/nomic-embed-text-v1.5-GGUF:F16", 0.1, tokens=5,
                  phase="retrieve.embed")
        return a_run(answered()), [a_cite()], None

    monkeypatch.setattr("src.api.ask", fake)
    phases = client(write_config(tmp_path)).post("/ask", json={"question": "q"}).json()[
        "spend"
    ]["phases"]
    assert [p["phase"] for p in phases][:2] == ["answer.generate", "retrieve.embed"]
    assert phases[0]["model"] == "openai/gpt-oss-120b"
    # A stage makes no model call, so it reports no model rather than an empty string.
    assert next(p for p in phases if p["phase"] == "retrieve.query")["model"] is None


# ---------------------------------------------------------------------------
# CORS — the frontend's actual first obstacle
# ---------------------------------------------------------------------------


def test_a_configured_origin_is_allowed(tmp_path):
    cfg = write_config(tmp_path, cors_origins=["http://localhost:3000"])
    r = client(cfg).get("/health", headers={"Origin": "http://localhost:3000"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_an_unconfigured_origin_is_not_allowed(tmp_path):
    cfg = write_config(tmp_path, cors_origins=["http://localhost:3000"])
    r = client(cfg).get("/health", headers={"Origin": "https://somewhere.example"})
    assert "access-control-allow-origin" not in r.headers


def test_no_configured_origins_means_no_browser_origin_is_allowed(tmp_path):
    # curl still works — CORS is a browser rule — which is why an empty list is a sensible
    # default for a server nobody has pointed a frontend at yet.
    cfg = write_config(tmp_path, cors_origins=[])
    r = client(cfg).get("/health", headers={"Origin": "http://localhost:3000"})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_the_shipped_config_declares_the_api_levers():
    cfg = load_config("config.toml")
    assert isinstance(cfg.get("api.port"), int)
    assert isinstance(cfg.get("api.cors_origins"), list)
    assert isinstance(cfg.get("api.serve_media"), bool)
    # Loopback, because serve_media true means this process hands out corpus video.
    assert cfg.get("api.host") == "127.0.0.1"
    assert "*" not in cfg.get("api.cors_origins")


# ---------------------------------------------------------------------------
# The document a frontend codes against
# ---------------------------------------------------------------------------


def test_the_openapi_document_describes_every_endpoint(tmp_path):
    paths = create_app(write_config(tmp_path)).openapi()["paths"]
    assert set(paths) >= {"/health", "/ask", "/videos", "/media/{video_id}"}
    assert "429" in paths["/ask"]["post"]["responses"]
    assert "206" in paths["/media/{video_id}"]["get"]["responses"]


# ---------------------------------------------------------------------------
# The frontend — web/, served by this same app
# ---------------------------------------------------------------------------
#
# The UI is static files and its behaviour is a browser's business, not pytest's. What is
# testable here — and what actually breaks — is the wiring: that / serves the page rather
# than the docs redirect, that the stylesheet and script it asks for exist at the urls the
# HTML names, and that a checkout without web/ still lands a browser somewhere useful. A
# renamed file under web/ is a blank page and no error anywhere, which is exactly the kind
# of half-working this file exists to catch.


def test_the_root_serves_the_frontend(tmp_path):
    r = client(write_config(tmp_path)).get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<title>VRAG" in r.text


def test_the_page_asks_for_static_files_that_exist(tmp_path):
    c = client(write_config(tmp_path))
    page = c.get("/").text
    for asset in re.findall(r'/static/[A-Za-z0-9_.-]+', page):
        assert c.get(asset).status_code == 200, asset


def test_the_frontend_calls_the_endpoints_this_app_serves(tmp_path):
    # The client fetches by string literal; nothing type-checks those against the routes.
    script = (WEB / "app.js").read_text(encoding="utf-8")
    assert "'/health'" in script
    assert "'/ask'" in script


def test_the_root_falls_back_to_the_docs_without_a_web_directory(tmp_path, monkeypatch):
    # A checkout that does not ship web/ is not broken, it just has no UI.
    monkeypatch.setattr("src.api.WEB", tmp_path / "absent")
    r = client(write_config(tmp_path)).get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/docs"


def test_the_frontend_is_not_in_the_openapi_document(tmp_path):
    # /docs describes the API a client codes against. The page and its assets are not that.
    paths = create_app(write_config(tmp_path)).openapi()["paths"]
    assert "/" not in paths
    assert not [p for p in paths if p.startswith("/static")]


def test_the_api_never_writes_a_page_to_disk(tmp_path, monkeypatch):
    # stub_ask asserts write is False. This is the test that says why: `make ask` writing a
    # file per question is the demo; a server doing it per request is a disk leak.
    stub_ask(monkeypatch, run=a_run(answered()), cites=[a_cite()])
    before = set(Path("runs").glob("ask/*")) if Path("runs/ask").exists() else set()
    client(write_config(tmp_path)).post("/ask", json={"question": "q"})
    after = set(Path("runs").glob("ask/*")) if Path("runs/ask").exists() else set()
    assert before == after
