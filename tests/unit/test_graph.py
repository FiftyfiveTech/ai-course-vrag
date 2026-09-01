"""src/graph.py — Microsoft Graph, app-only. No network, no tenant, no credentials.

Everything here runs offline. `urllib.request.urlopen` is replaced per test, so what is
under test is the code that builds a request and reads a reply — never Microsoft.

Two of these tests are load-bearing beyond the usual, and both are about a thing that looks
fine when it is wrong:

* **the voice tag.** `<v Priya Nair>` is the entire reason this module exists: it is the only
  free source of *attributed* text in the project, because whisper cannot recover a name from
  audio and diarisation only ever yields "Speaker 1". A parser that quietly drops the name
  produces a perfectly valid transcript that is worth no more than the one whisper already
  gives us, and nothing downstream would notice. So the parser is tested on the shapes Teams
  actually emits — unpadded timestamps, cue identifiers, tag classes, CRLF — and on the case
  where the tenant has stripped attribution, which is a *valid* transcript with no names in
  it and has to be reported as a failure rather than a success.
* **no secret over the socket.** `/graph/health` is unauthenticated and `api.host` is a lever
  somebody can move off loopback. The CLI prints a `len=/sha256:` fingerprint per credential,
  which is right on an operator's terminal and wrong in an HTTP body, so the endpoint is
  asserted to emit neither the value nor anything derived from it.

`load_env` is monkeypatched in every test that reads credentials. Without that these tests
would pass or fail depending on whether the person running them happens to have a tenant —
`src.graph.read_credentials` goes through `src/env.py`, which reads the real `.env` and the
real `~/.config/`.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.config import load as load_config
from src.graph import (
    CREDENTIALS,
    FAIL,
    PASS,
    SKIP,
    WARN,
    Finding,
    GraphClient,
    GraphError,
    Token,
    VttCue,
    _graph_error,
    _token_error,
    check,
    decode_jwt_claims,
    main,
    parse_timestamp,
    parse_vtt,
    read_credentials,
    report,
    speakers,
    summarise,
    to_segments,
)

TENANT = "11111111-1111-1111-1111-111111111111"
CLIENT = "22222222-2222-2222-2222-222222222222"
SECRET = "a-client-secret-value-that-is-long-enough"
ROLES = ["OnlineMeetingTranscript.Read.All", "OnlineMeetings.Read.All"]


# ---------------------------------------------------------------------------
# Fixtures — a config in tmp_path, a fake urlopen, a fake JWT
# ---------------------------------------------------------------------------


def write_config(tmp_path: Path, *, max_pages: int = 20, required=None, optional=None) -> Path:
    """A config with the [graph] levers. Long because `Config.get` has no defaults."""
    required = ROLES if required is None else required
    optional = ["User.Read.All"] if optional is None else optional
    path = tmp_path / "config.toml"
    path.write_text(
        "[api]\n"
        "host = \"127.0.0.1\"\n"
        "port = 8000\n"
        "cors_origins = []\n"
        "serve_media = false\n"
        "\n[graph]\n"
        'authority = "https://login.microsoftonline.test"\n'
        'base_url = "https://graph.microsoft.test"\n'
        'api_version = "v1.0"\n'
        'scope = "https://graph.microsoft.test/.default"\n'
        "timeout_s = 5.0\n"
        "token_leeway_s = 300.0\n"
        f"max_pages = {max_pages}\n"
        f"required_roles = {json.dumps(required)}\n"
        f"optional_roles = {json.dumps(optional)}\n",
        encoding="utf-8",
    )
    return path


def make_jwt(**claims) -> str:
    """A JWT-shaped string. Unsigned — `decode_jwt_claims` never verifies one.

    Padding is stripped, as a real JWT's is, which is the case the decoder has to restore.
    """
    def seg(obj) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    return f"{seg({'alg': 'RS256'})}.{seg(claims)}.not-a-signature"


def token_body(*, roles=ROLES, expires_in: int = 3599, exp_in: int | None = None, **extra) -> dict:
    """A token response. `exp_in` sets the claim independently of the `expires_in` field.

    Independently on purpose: the two disagreeing is the case worth testing, and a helper
    that derives one from the other makes that test unable to fail.
    """
    claims = {
        "tid": TENANT,
        "appid": CLIENT,
        "roles": roles,
        "exp": int(time.time()) + (expires_in if exp_in is None else exp_in),
        **extra,
    }
    return {"token_type": "Bearer", "expires_in": expires_in, "access_token": make_jwt(**claims)}


class _Response:
    """The three things `urllib.request.urlopen`'s return value is used for."""

    def __init__(self, payload: bytes, headers: dict | None = None) -> None:
        self._payload = payload
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code: int, body: dict | str, url: str = "https://graph.microsoft.test/x"):
    raw = body if isinstance(body, str) else json.dumps(body)
    return urllib.error.HTTPError(
        url, code, "error", {}, io.BytesIO(raw.encode("utf-8"))
    )


@pytest.fixture
def fake_http(monkeypatch):
    """Install a request handler and record every request that reaches it.

    The handler is called with the `urllib.request.Request` and returns bytes, a dict (JSON
    encoded for you) or an exception to raise. `calls` is the record, which is how "the token
    was cached" is asserted — by there being one POST and not four.
    """
    calls: list[urllib.request.Request] = []
    state: dict = {"handler": None}

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        result = state["handler"](request)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, (dict, list)):
            return _Response(json.dumps(result).encode("utf-8"))
        if isinstance(result, str):
            return _Response(result.encode("utf-8"))
        return _Response(result)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    def install(handler):
        state["handler"] = handler
        return calls

    install.calls = calls  # type: ignore[attr-defined]
    return install


@pytest.fixture
def creds():
    return {name: (value, ".env") for name, value in zip(CREDENTIALS, (TENANT, CLIENT, SECRET))}


def client_for(tmp_path, creds, **kwargs) -> GraphClient:
    return GraphClient(cfg=load_config(write_config(tmp_path, **kwargs)), credentials=creds)


# ---------------------------------------------------------------------------
# parse_timestamp — Teams is not consistent about padding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stamp,seconds",
    [
        ("00:00:05.000", 5.0),
        ("0:0:5.0", 5.0),  # Teams really does emit this
        ("01:02:03.500", 3723.5),
        ("1:30.250", 90.25),  # MM:SS — legal WebVTT with hours omitted
        ("00:00:05,000", 5.0),  # SRT-style comma decimal
        ("100:00:00.000", 360000.0),  # three-digit hours
    ],
)
def test_parse_timestamp_shapes(stamp, seconds):
    assert parse_timestamp(stamp) == pytest.approx(seconds)


@pytest.mark.parametrize("stamp", ["", "5", "aa:bb:cc.ddd", "1:2:3:4.5"])
def test_parse_timestamp_refuses_nonsense(stamp):
    with pytest.raises(ValueError):
        parse_timestamp(stamp)


# ---------------------------------------------------------------------------
# parse_vtt — the voice tag is the whole point
# ---------------------------------------------------------------------------

TEAMS_VTT = """WEBVTT

d7d4c0f6-0000-0000-0000-000000000001/1-0
00:00:05.120 --> 00:00:09.400
<v Priya Nair>Let's start with the migration.</v>

d7d4c0f6-0000-0000-0000-000000000001/2-0
0:0:9.4 --> 0:0:14.0
<v Arun Kumar>I'll have the schema diff by Friday.</v>

d7d4c0f6-0000-0000-0000-000000000001/3-0
00:00:14.000 --> 00:00:18.750
<v Priya Nair>Thanks Arun.</v>
"""


def test_parse_vtt_keeps_the_speaker():
    cues = parse_vtt(TEAMS_VTT)
    assert [c.speaker for c in cues] == ["Priya Nair", "Arun Kumar", "Priya Nair"]
    assert cues[0].text == "Let's start with the migration."
    assert cues[1].t_start == pytest.approx(9.4)
    assert cues[1].t_end == pytest.approx(14.0)


def test_parse_vtt_self_assignment_is_only_attributable_via_the_voice_tag():
    """The sentence that makes the graph arm worth having.

    "I'll have the schema diff by Friday" carries no owner in its text. Most real action
    items are phrased exactly like this, so without the voice tag the owner is not merely
    hard to extract — it is absent from the data, and any owner assigned to it is invented.
    """
    cues = parse_vtt(TEAMS_VTT)
    self_assigned = next(c for c in cues if "I'll have" in c.text)
    assert "Arun" not in self_assigned.text
    assert self_assigned.speaker == "Arun Kumar"


def test_parse_vtt_handles_crlf_notes_and_tag_classes():
    text = (
        "WEBVTT\r\n"
        "\r\n"
        "NOTE this block is a comment and 00:00:01.000 --> 00:00:02.000 inside it\r\n"
        "\r\n"
        "1\r\n"
        "00:00:01.000 --> 00:00:02.000\r\n"
        "<v.loud.first Priya Nair>Morning<c.italic> all</c>.</v>\r\n"
    )
    cues = parse_vtt(text)
    assert len(cues) == 1
    assert cues[0].speaker == "Priya Nair"
    # Every tag is stripped, not just the voice span.
    assert cues[0].text == "Morning all."


def test_parse_vtt_multiline_payload_is_one_cue():
    text = (
        "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\n"
        "<v Priya Nair>First line\nsecond line</v>\n"
    )
    cues = parse_vtt(text)
    assert len(cues) == 1
    assert "First line" in cues[0].text and "second line" in cues[0].text


def test_parse_vtt_unattributed_is_none_not_a_guess():
    """A cue with no voice tag gets `speaker=None` and does NOT inherit the last speaker.

    Carrying the previous name forward would be a reasonable-sounding heuristic that
    produces a wrong name indistinguishable from a right one. In minutes, a commitment
    attributed to a colleague who never made it is the worst output this system can produce,
    so the nullable field is the honest representation of "the platform did not say".
    """
    text = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.000\n<v Priya Nair>Named.</v>\n\n"
        "00:00:02.000 --> 00:00:03.000\nNo voice tag here.\n"
    )
    cues = parse_vtt(text)
    assert cues[0].speaker == "Priya Nair"
    assert cues[1].speaker is None


def test_parse_vtt_skips_a_malformed_cue_without_losing_the_meeting():
    text = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.000\n<v A>one</v>\n\n"
        "not-a-timing-line\nstray payload\n\n"
        "99:99 --> broken\npayload\n\n"
        "00:00:03.000 --> 00:00:04.000\n<v B>two</v>\n"
    )
    cues = parse_vtt(text)
    assert [c.text for c in cues] == ["one", "two"]


def test_parse_vtt_empty_input_is_no_cues_not_an_error():
    assert parse_vtt("") == []
    assert parse_vtt("WEBVTT\n") == []


def test_speakers_counts_and_names_the_unattributed():
    cues = [
        VttCue(0.0, 1.0, "a", "Priya Nair"),
        VttCue(1.0, 2.0, "b", "Priya Nair"),
        VttCue(2.0, 3.0, "c", None),
    ]
    assert speakers(cues) == {"Priya Nair": 2, "(unattributed)": 1}


def test_to_segments_passes_the_speaker():
    """VRAG-026: speaker now survives the VttCue → Segment boundary.

    Previously the name was dropped here because Segment had no field for it.
    Now it rides as Segment.speaker and must survive to every downstream hop.
    """
    from src.transcript import Segment

    segments = to_segments(parse_vtt(TEAMS_VTT))
    assert all(isinstance(s, Segment) for s in segments)
    assert [s.text for s in segments][0] == "Let's start with the migration."
    assert segments[0].speaker == "Priya Nair"


# ---------------------------------------------------------------------------
# The token
# ---------------------------------------------------------------------------


def test_decode_jwt_claims_restores_stripped_padding():
    claims = decode_jwt_claims(make_jwt(tid=TENANT, roles=ROLES))
    assert claims["tid"] == TENANT
    assert claims["roles"] == ROLES


@pytest.mark.parametrize("bad", ["", "not-a-jwt", "a.b", "a.!!!!.c", "a." + base64.urlsafe_b64encode(b"[1,2]").decode() + ".c"])
def test_decode_jwt_claims_returns_empty_rather_than_raising(bad):
    """An undecodable payload must not fail the check: the token may still work."""
    assert decode_jwt_claims(bad) == {}


def test_token_reads_roles_tenant_and_expiry():
    token = Token(
        value="x",
        acquired_at=time.time(),
        expires_at=time.time() + 3600,
        claims={"tid": TENANT, "appid": CLIENT, "roles": ROLES},
    )
    assert token.roles == tuple(ROLES)
    assert token.tenant_id == TENANT
    assert token.app_id == CLIENT
    assert token.is_usable(leeway_s=300.0)


def test_token_with_no_roles_claim_is_the_no_consent_case():
    token = Token(value="x", acquired_at=0.0, expires_at=0.0, claims={"tid": TENANT})
    assert token.roles == ()


def test_token_inside_the_leeway_is_not_usable():
    token = Token(value="x", acquired_at=0.0, expires_at=time.time() + 60, claims={})
    assert not token.is_usable(leeway_s=300.0)
    assert token.is_usable(leeway_s=30.0)


def test_token_is_acquired_once_and_reused(tmp_path, creds, fake_http):
    """Caching is not an optimisation: Entra throttles token requests per app."""

    def handler(request):
        if request.method == "POST":
            return token_body()
        return {"value": [{"id": "org", "displayName": "Contoso"}]}

    calls = fake_http(handler)
    client = client_for(tmp_path, creds)
    client.organization()
    client.organization()
    client.organization()
    assert sum(1 for c in calls if c.method == "POST") == 1
    assert sum(1 for c in calls if c.method == "GET") == 3


def test_token_prefers_the_exp_claim_over_expires_in(tmp_path, creds, fake_http):
    """`exp` is what the resource server enforces; `expires_in` is advisory."""
    fake_http(lambda request: token_body(expires_in=99999, exp_in=3599))
    token = client_for(tmp_path, creds).token()
    assert token.expires_in_s() < 4000


def test_token_sends_the_default_scope_and_the_grant_type(tmp_path, creds, fake_http):
    calls = fake_http(lambda request: token_body())
    client_for(tmp_path, creds).token()
    body = calls[0].data.decode()
    assert "grant_type=client_credentials" in body
    assert ".default" in body
    assert calls[0].full_url.startswith("https://login.microsoftonline.test/")
    assert TENANT in calls[0].full_url


def test_missing_credentials_names_them_and_does_not_call(tmp_path, fake_http):
    fake_http(lambda request: pytest.fail("nothing should be sent with no credentials"))
    client = client_for(tmp_path, {"GRAPH_TENANT_ID": (TENANT, ".env")})
    with pytest.raises(GraphError) as exc:
        client.token()
    assert exc.value.code == "NotConfigured"
    assert "GRAPH_CLIENT_ID" in str(exc.value)
    assert "GRAPH_CLIENT_SECRET" in str(exc.value)
    # The rule from CLAUDE.md, in the error the developer actually reads.
    assert "never in the repo" in str(exc.value)


def test_no_credential_value_appears_in_an_error(tmp_path, creds, fake_http):
    fake_http(lambda request: http_error(401, {"error": "invalid_client", "error_description": "AADSTS7000215: bad"}))
    with pytest.raises(GraphError) as exc:
        client_for(tmp_path, creds).token()
    assert SECRET not in str(exc.value)
    assert SECRET not in exc.value.hint


# ---------------------------------------------------------------------------
# Error classification — branch on the code, never on the message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "description,code,hint_contains",
    [
        ("AADSTS7000215: Invalid client secret provided.", "AADSTS7000215", "Secret ID"),
        ("AADSTS7000222: The provided client secret keys are expired.", "AADSTS7000222", "expired"),
        ("AADSTS700016: Application not found in the directory.", "AADSTS700016", "different directory"),
        ("AADSTS90002: Tenant not found.", "AADSTS90002", "Directory (tenant) ID"),
    ],
)
def test_token_error_lifts_the_aadsts_code(description, code, hint_contains):
    error = _token_error(http_error(401, {"error": "invalid_client", "error_description": description}))
    assert error.code == code
    assert hint_contains in error.hint
    assert error.status == 401


def test_token_error_with_no_aadsts_falls_back_to_the_error_field():
    error = _token_error(http_error(400, {"error": "unsupported_grant_type"}))
    assert error.code == "unsupported_grant_type"


def test_token_error_survives_a_non_json_body():
    """A proxy's HTML error page must produce a GraphError, not a JSONDecodeError."""
    error = _token_error(http_error(502, "<html>Bad Gateway</html>"))
    assert error.status == 502
    assert isinstance(error, GraphError)


def test_graph_error_prefers_inner_error_code():
    """The outer code is a coarse HTTP label; the inner one names the reason."""
    error = _graph_error(
        http_error(
            403,
            {
                "error": {
                    "code": "Forbidden",
                    "message": "Attribution is disabled.",
                    "innerError": {"code": "SpeakerAttributionNotAllowed"},
                }
            },
        ),
        "https://graph.microsoft.test/v1.0/x",
    )
    assert error.code == "SpeakerAttributionNotAllowed"
    assert "whisper already gives us for free" in error.hint


def test_graph_error_403_hint_names_the_second_grant():
    """The 403 that costs a day: the app role is necessary and not sufficient."""
    error = _graph_error(
        http_error(403, {"error": {"code": "Forbidden", "message": "Forbidden"}}),
        "https://graph.microsoft.test/v1.0/x",
    )
    assert error.code == "Forbidden"
    assert "New-CsApplicationAccessPolicy" in error.hint


def test_graph_error_on_an_unknown_code_has_no_invented_hint():
    error = _graph_error(
        http_error(500, {"error": {"code": "SomethingNew", "message": "boom"}}),
        "https://graph.microsoft.test/v1.0/x",
    )
    assert error.code == "SomethingNew"
    assert error.hint == ""


def test_unreachable_host_is_its_own_code(tmp_path, creds, fake_http):
    fake_http(lambda request: urllib.error.URLError("no route to host"))
    with pytest.raises(GraphError) as exc:
        client_for(tmp_path, creds).token()
    assert exc.value.code == "Unreachable"


def test_a_200_with_no_access_token_is_an_error(tmp_path, creds, fake_http):
    fake_http(lambda request: {"token_type": "Bearer", "expires_in": 3599})
    with pytest.raises(GraphError) as exc:
        client_for(tmp_path, creds).token()
    assert exc.value.code == "NoAccessToken"


# ---------------------------------------------------------------------------
# Requests, urls and paging
# ---------------------------------------------------------------------------


def test_url_for_builds_from_the_levers_and_passes_absolute_through(tmp_path, creds):
    client = client_for(tmp_path, creds)
    assert client.url_for("organization") == "https://graph.microsoft.test/v1.0/organization"
    assert client.url_for("/organization") == "https://graph.microsoft.test/v1.0/organization"
    absolute = "https://graph.microsoft.test/v1.0/users?$skiptoken=OPAQUE"
    assert client.url_for(absolute) == absolute


def test_get_sends_the_bearer_token(tmp_path, creds, fake_http):
    calls = fake_http(lambda r: token_body() if r.method == "POST" else {"value": []})
    client_for(tmp_path, creds).get("organization")
    get = next(c for c in calls if c.method == "GET")
    assert get.get_header("Authorization", "").startswith("Bearer ")


def test_paged_follows_next_link(tmp_path, creds, fake_http):
    pages = {
        "https://graph.microsoft.test/v1.0/users": {
            "value": [{"id": "1"}],
            "@odata.nextLink": "https://graph.microsoft.test/v1.0/users?$skiptoken=OPAQUE",
        },
        "https://graph.microsoft.test/v1.0/users?$skiptoken=OPAQUE": {"value": [{"id": "2"}]},
    }

    def handler(request):
        if request.method == "POST":
            return token_body()
        return pages[request.full_url]

    fake_http(handler)
    got = list(client_for(tmp_path, creds).paged("users"))
    assert [item["id"] for item in got] == ["1", "2"]


def test_paged_stops_at_max_pages_rather_than_looping_forever(tmp_path, creds, fake_http):
    """A paginator that misreads nextLink requests page one forever.

    Unbounded, that is a silent hang against somebody's tenant. The cap turns it into an
    error with a number in it.
    """

    def handler(request):
        if request.method == "POST":
            return token_body()
        return {"value": [{"id": "x"}], "@odata.nextLink": "https://graph.microsoft.test/v1.0/users"}

    fake_http(handler)
    client = client_for(tmp_path, creds, max_pages=3)
    with pytest.raises(GraphError) as exc:
        list(client.paged("users"))
    assert exc.value.code == "TooManyPages"
    assert "max_pages = 3" in str(exc.value)


def test_non_json_from_graph_is_a_clean_error(tmp_path, creds, fake_http):
    fake_http(lambda r: token_body() if r.method == "POST" else "<html>nope</html>")
    with pytest.raises(GraphError) as exc:
        client_for(tmp_path, creds).get("organization")
    assert exc.value.code == "BadResponse"


def test_find_online_meeting_doubles_a_quote_in_the_odata_filter(tmp_path, creds, fake_http):
    """OData escapes a single quote by doubling it. Not doing so breaks the filter."""
    calls = fake_http(lambda r: token_body() if r.method == "POST" else {"value": [{"id": "m1"}]})
    client = client_for(tmp_path, creds)
    client.find_online_meeting("user-1", "https://teams.test/l/it's-a-url")
    get = next(c for c in calls if c.method == "GET")
    assert "it''s-a-url" in urllib.parse.unquote(get.full_url)


def test_find_online_meeting_with_no_match_says_which_two_things_to_check(tmp_path, creds, fake_http):
    fake_http(lambda r: token_body() if r.method == "POST" else {"value": []})
    with pytest.raises(GraphError) as exc:
        client_for(tmp_path, creds).find_online_meeting("user-1", "https://teams.test/x")
    assert exc.value.code == "NotFound"
    assert "organiser" in str(exc.value)


def test_transcript_vtt_asks_for_vtt_and_decodes_a_bom(tmp_path, creds, fake_http):
    def handler(request):
        if request.method == "POST":
            return token_body()
        return b"\xef\xbb\xbfWEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v A>hi</v>\n"

    calls = fake_http(handler)
    text = client_for(tmp_path, creds).transcript_vtt("u", "m", "t")
    assert text.startswith("WEBVTT")  # the BOM is gone
    get = next(c for c in calls if c.method == "GET")
    assert "text%2Fvtt" in get.full_url or "text/vtt" in get.full_url
    assert parse_vtt(text)[0].speaker == "A"


# ---------------------------------------------------------------------------
# check() / summarise() — the verdict
# ---------------------------------------------------------------------------


@pytest.fixture
def no_env(monkeypatch):
    """No credentials anywhere. Pins the tests off the developer's real tenant."""
    monkeypatch.setattr("src.graph.load_env", lambda *a, **k: {})


@pytest.fixture
def env_with_creds(monkeypatch):
    monkeypatch.setattr(
        "src.graph.load_env",
        lambda *a, **k: {
            "GRAPH_TENANT_ID": (TENANT, ".env"),
            "GRAPH_CLIENT_ID": (CLIENT, ".env"),
            "GRAPH_CLIENT_SECRET": (SECRET, str(Path.home() / ".config" / "ai-course-vrag.env")),
        },
    )


def test_read_credentials_reports_where_each_one_came_from(env_with_creds):
    found = read_credentials()
    assert set(found) == set(CREDENTIALS)
    assert found["GRAPH_CLIENT_SECRET"][1].endswith("ai-course-vrag.env")


def test_read_credentials_treats_blank_as_unset(monkeypatch):
    """`.env.example` is copied to `.env` by `make setup`, so every name exists and is empty."""
    monkeypatch.setattr(
        "src.graph.load_env",
        lambda *a, **k: {name: ("   ", ".env") for name in CREDENTIALS},
    )
    assert read_credentials() == {}


def test_check_with_no_credentials_fails_and_skips_the_probe(tmp_path, no_env, fake_http):
    fake_http(lambda request: pytest.fail("nothing should be sent"))
    findings = check(load_config(write_config(tmp_path)))
    creds = [f for f in findings if f.section == "credentials"]
    assert len(creds) == 3 and all(f.status == FAIL for f in creds)
    assert any(f.section == "token" and f.status == SKIP for f in findings)


def test_no_probe_sends_nothing(tmp_path, env_with_creds, fake_http):
    fake_http(lambda request: pytest.fail("--no-probe must send nothing"))
    findings = check(load_config(write_config(tmp_path)), probe=False)
    assert all(f.status == PASS for f in findings if f.section == "credentials")
    token = next(f for f in findings if f.section == "token")
    assert token.status == SKIP


def test_check_reports_a_granted_role_and_names_a_missing_one(tmp_path, env_with_creds, fake_http):
    fake_http(
        lambda r: token_body(roles=["OnlineMeetingTranscript.Read.All"])
        if r.method == "POST"
        else {"value": [{"displayName": "Contoso"}]}
    )
    findings = check(load_config(write_config(tmp_path)))
    by_name = {f.name: f for f in findings if f.section == "roles"}
    assert by_name["OnlineMeetingTranscript.Read.All"].status == PASS
    assert by_name["OnlineMeetings.Read.All"].status == FAIL
    assert "admin consent" in by_name["OnlineMeetings.Read.All"].detail


def test_no_roles_claim_is_reported_as_no_consent_at_all(tmp_path, env_with_creds, fake_http):
    fake_http(
        lambda r: token_body(roles=[]) if r.method == "POST" else {"value": [{"displayName": "C"}]}
    )
    findings = check(load_config(write_config(tmp_path)))
    granted = next(f for f in findings if f.section == "roles" and f.name == "granted")
    assert granted.status == FAIL
    assert "no admin has consented" in granted.detail


def test_a_403_on_organization_is_a_warn_because_the_token_was_accepted(
    tmp_path, env_with_creds, fake_http
):
    """Reachability asks whether Graph parsed the token. A 403 proves it did.

    Reporting it as a failure of reachability points at the network when the answer is a
    missing grant — and the roles section above already said which one.
    """

    def handler(request):
        if request.method == "POST":
            return token_body()
        return http_error(403, {"error": {"code": "Forbidden", "message": "no"}})

    fake_http(handler)
    findings = check(load_config(write_config(tmp_path)))
    reach = next(f for f in findings if f.section == "reachability")
    assert reach.status == WARN
    assert "the token was accepted" in reach.detail


def test_check_without_a_meeting_says_the_green_table_is_not_the_answer(
    tmp_path, env_with_creds, fake_http
):
    fake_http(
        lambda r: token_body() if r.method == "POST" else {"value": [{"displayName": "Contoso"}]}
    )
    findings = check(load_config(write_config(tmp_path)))
    speaker = next(f for f in findings if f.name == "speaker names")
    assert speaker.status == SKIP
    assert "--meeting" in speaker.detail


def test_a_meeting_without_a_user_is_refused_before_any_call(tmp_path, env_with_creds, fake_http):
    """A meeting hangs off its organiser's calendar; there is no tenant-wide lookup."""
    fake_http(lambda r: token_body() if r.method == "POST" else {"value": [{"displayName": "C"}]})
    findings = check(load_config(write_config(tmp_path)), meeting="some-meeting-id")
    user = next(f for f in findings if f.name == "user")
    assert user.status == FAIL
    assert "ORGANISER" in user.detail


def _meeting_handler(vtt: str, transcripts=None):
    transcripts = [{"id": "t1"}] if transcripts is None else transcripts

    def handler(request):
        url = request.full_url
        if request.method == "POST":
            return token_body()
        if url.endswith("/organization") or "organization" in url:
            return {"value": [{"displayName": "Contoso"}]}
        if "/transcripts/" in url:
            return vtt.encode("utf-8")
        if url.endswith("/transcripts"):
            return {"value": transcripts}
        if "/users/" in url and "/onlineMeetings" not in url:
            return {"id": "user-1", "displayName": "Priya Nair"}
        return {"value": []}

    return handler


def test_a_real_transcript_reports_who_was_speaking(tmp_path, env_with_creds, fake_http):
    fake_http(_meeting_handler(TEAMS_VTT))
    findings = check(
        load_config(write_config(tmp_path)), user="priya@contoso.test", meeting="meeting-1"
    )
    speaker = next(f for f in findings if f.name == "speaker names")
    assert speaker.status == PASS
    assert "Priya Nair" in speaker.detail
    assert "Arun Kumar" in speaker.detail


def test_an_unattributed_transcript_is_a_FAIL_not_a_success(tmp_path, env_with_creds, fake_http):
    """The failure mode that looks like success.

    A tenant can disable speaker attribution. The result is a perfectly valid transcript
    with every voice tag removed — a 200, well-formed VTT, real text. It is also worth
    nothing for minutes, because it is exactly what whisper already produces for free. So
    this has to be red.
    """
    stripped = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nLet's start with the migration.\n"
    fake_http(_meeting_handler(stripped))
    findings = check(
        load_config(write_config(tmp_path)), user="priya@contoso.test", meeting="meeting-1"
    )
    speaker = next(f for f in findings if f.name == "speaker names")
    assert speaker.status == FAIL
    assert "unattributed" in speaker.detail
    assert "whisper already produces" in speaker.detail


def test_a_meeting_with_no_transcript_is_a_warn_with_the_reason(tmp_path, env_with_creds, fake_http):
    fake_http(_meeting_handler(TEAMS_VTT, transcripts=[]))
    findings = check(
        load_config(write_config(tmp_path)), user="priya@contoso.test", meeting="meeting-1"
    )
    listed = next(f for f in findings if f.section == "transcript" and f.name == "list")
    assert listed.status == WARN
    assert "Start transcription" in listed.detail


def test_partial_attribution_reports_the_unattributed_share(tmp_path, env_with_creds, fake_http):
    mixed = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.000\n<v Priya Nair>named</v>\n\n"
        "00:00:02.000 --> 00:00:03.000\nunnamed\n\n"
        "00:00:03.000 --> 00:00:04.000\nunnamed too\n"
    )
    fake_http(_meeting_handler(mixed))
    findings = check(
        load_config(write_config(tmp_path)), user="priya@contoso.test", meeting="meeting-1"
    )
    unattributed = next(f for f in findings if f.name == "unattributed")
    assert unattributed.status == WARN
    assert "2 of 3" in unattributed.detail


def test_summarise_is_not_ready_without_credentials(tmp_path, no_env, fake_http):
    fake_http(lambda request: pytest.fail("nothing should be sent"))
    summary = summarise(load_config(write_config(tmp_path)))
    assert summary.ready is False
    assert summary.configured is False
    assert summary.probed is False
    assert set(summary.missing) == set(CREDENTIALS)
    assert "never in the repo" in summary.detail


def test_summarise_is_ready_with_every_required_role(tmp_path, env_with_creds, fake_http):
    fake_http(
        lambda r: token_body() if r.method == "POST" else {"value": [{"displayName": "Contoso"}]}
    )
    summary = summarise(load_config(write_config(tmp_path)))
    assert summary.ready is True
    assert summary.tenant_id == TENANT
    assert set(summary.granted_roles) == set(ROLES)
    assert summary.missing_roles == ()
    # Ready is not the same as "the first real meeting will work".
    assert "application access policy" in summary.detail


def test_summarise_names_the_missing_role(tmp_path, env_with_creds, fake_http):
    fake_http(
        lambda r: token_body(roles=["OnlineMeetings.Read.All"])
        if r.method == "POST"
        else {"value": [{"displayName": "C"}]}
    )
    summary = summarise(load_config(write_config(tmp_path)))
    assert summary.ready is False
    assert summary.missing_roles == ("OnlineMeetingTranscript.Read.All",)
    assert "OnlineMeetingTranscript.Read.All" in summary.detail


def test_summarise_mints_one_token_for_both_the_check_and_the_claims(
    tmp_path, env_with_creds, fake_http
):
    """`summarise` shares a client with `check` so the claims it reports cost no second call."""
    calls = fake_http(
        lambda r: token_body() if r.method == "POST" else {"value": [{"displayName": "C"}]}
    )
    summarise(load_config(write_config(tmp_path)))
    assert sum(1 for c in calls if c.method == "POST") == 1


def test_summarise_is_not_ready_when_the_token_is_refused(tmp_path, env_with_creds, fake_http):
    fake_http(
        lambda r: http_error(401, {"error": "invalid_client", "error_description": "AADSTS7000222: expired"})
    )
    summary = summarise(load_config(write_config(tmp_path)))
    assert summary.ready is False
    assert summary.configured is True
    assert summary.probed is True
    assert summary.tenant_id is None
    assert any(f.status == WARN and "expired" in f.detail for f in summary.findings)


# ---------------------------------------------------------------------------
# report() and the CLI
# ---------------------------------------------------------------------------


def test_report_exits_non_zero_on_a_fail():
    buffer = io.StringIO()
    code = report([Finding("credentials", "GRAPH_TENANT_ID", FAIL, "not set")], out=buffer)
    out = buffer.getvalue()
    assert code == 1
    assert "GRAPH_TENANT_ID" in out
    assert "1 check(s) failed" in out


def test_report_exits_zero_when_only_warns_and_skips():
    """A WARN and a SKIP are not failures: a 403 on /organization and an un-probed meeting
    are both states a correctly configured deployment reports."""
    buffer = io.StringIO()
    code = report(
        [
            Finding("credentials", "GRAPH_TENANT_ID", PASS, "set"),
            Finding("reachability", "GET /organization", WARN, "403"),
            Finding("transcript", "speaker names", SKIP, "pass --meeting"),
        ],
        out=buffer,
    )
    assert code == 0
    assert "PASS - Graph is reachable" in buffer.getvalue()


def test_cli_vtt_mode_needs_no_credentials_and_prints_the_speakers(tmp_path, capsys, fake_http):
    """The half of this module that can be checked with no tenant at all."""
    fake_http(lambda request: pytest.fail("--vtt must make no request"))
    path = tmp_path / "meeting.vtt"
    path.write_text(TEAMS_VTT, encoding="utf-8")
    code = main(["--vtt", str(path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Priya Nair" in out and "Arun Kumar" in out
    assert "3 cues" in out


def test_cli_vtt_mode_fails_on_a_transcript_with_no_names(tmp_path, capsys):
    path = tmp_path / "stripped.vtt"
    path.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nno tag\n", encoding="utf-8")
    assert main(["--vtt", str(path)]) == 1
    assert "(unattributed)" in capsys.readouterr().out


def test_cli_vtt_mode_on_a_missing_file_is_an_error_not_a_traceback(tmp_path, capsys):
    assert main(["--vtt", str(tmp_path / "nope.vtt")]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_cli_no_probe_reports_and_exits_non_zero_without_credentials(
    tmp_path, no_env, capsys, fake_http
):
    """Missing credentials outrank --no-probe in the report, and both mean nothing is sent.

    The token line says "no credentials to try" rather than "nothing was sent to Microsoft",
    and that precedence is right: the flag describes a choice, the empty credentials describe
    a blocker, and the first FAIL to fix is the blocker.
    """
    fake_http(lambda request: pytest.fail("nothing may be sent"))
    code = main(["--config", str(write_config(tmp_path)), "--no-probe"])
    out = capsys.readouterr().out
    assert code == 1
    assert "GRAPH_TENANT_ID" in out
    assert "no credentials to try" in out


def test_cli_no_probe_with_credentials_says_it_sent_nothing(tmp_path, env_with_creds, capsys, fake_http):
    fake_http(lambda request: pytest.fail("--no-probe must send nothing"))
    code = main(["--config", str(write_config(tmp_path)), "--no-probe"])
    out = capsys.readouterr().out
    assert code == 0
    assert "nothing was sent to Microsoft" in out


# ---------------------------------------------------------------------------
# GET /graph/health
# ---------------------------------------------------------------------------


def test_graph_health_is_offline_by_default(tmp_path, env_with_creds, fake_http):
    """Default probe=false: a poll of this endpoint must not authenticate to Entra."""
    fake_http(lambda request: pytest.fail("probe defaults to false — nothing may be sent"))
    app = create_app(load_config(write_config(tmp_path)))
    body = TestClient(app).get("/graph/health").json()
    assert body["probed"] is False
    assert body["configured"] is True
    assert body["ready"] is False
    assert body["tenant_id"] is None
    assert body["required_roles"] == ROLES


def test_graph_health_probe_true_reports_the_roles(tmp_path, env_with_creds, fake_http):
    fake_http(
        lambda r: token_body() if r.method == "POST" else {"value": [{"displayName": "Contoso"}]}
    )
    app = create_app(load_config(write_config(tmp_path)))
    body = TestClient(app).get("/graph/health?probe=true").json()
    assert body["ready"] is True
    assert body["probed"] is True
    assert body["tenant_id"] == TENANT
    assert set(body["granted_roles"]) == set(ROLES)
    assert body["missing_roles"] == []


def test_graph_health_never_emits_a_credential_or_a_digest_of_one(
    tmp_path, env_with_creds, fake_http
):
    """The CLI's `len=/sha256:` fingerprint is right on a terminal and wrong in an HTTP body.

    `api.host` is a lever and this endpoint takes no authentication, so a byte derived from a
    secret has no business crossing the socket to earn nothing: the caller's question is
    whether the value is set.
    """
    fake_http(
        lambda r: token_body() if r.method == "POST" else {"value": [{"displayName": "Contoso"}]}
    )
    app = create_app(load_config(write_config(tmp_path)))
    raw = TestClient(app).get("/graph/health?probe=true").text
    assert SECRET not in raw
    assert "sha256:" not in raw
    assert "len=" not in raw
    # The access token, in any form.
    assert "eyJ" not in raw and "Bearer" not in raw
    credential_lines = [
        f for f in json.loads(raw)["findings"] if f["section"] == "credentials"
    ]
    assert {f["detail"] for f in credential_lines} == {"set"}


def test_graph_health_reports_the_missing_names_when_unconfigured(tmp_path, no_env, fake_http):
    fake_http(lambda request: pytest.fail("nothing should be sent"))
    app = create_app(load_config(write_config(tmp_path)))
    body = TestClient(app).get("/graph/health?probe=true").json()
    assert body["ready"] is False
    assert body["configured"] is False
    assert set(body["missing"]) == set(CREDENTIALS)
    # Asked to probe, but there was nothing to probe with.
    assert body["probed"] is False


def test_graph_health_exposes_no_meeting_parameter(tmp_path, env_with_creds, fake_http):
    """A transcript's speaker list is colleagues' names. That stays on the CLI.

    An unknown query parameter is ignored by FastAPI rather than rejected, so the assertion
    that matters is the OpenAPI document: `user` and `meeting` are not part of the contract.
    """
    fake_http(lambda request: pytest.fail("nothing should be sent"))
    app = create_app(load_config(write_config(tmp_path)))
    schema = TestClient(app).get("/openapi.json").json()
    params = schema["paths"]["/graph/health"]["get"].get("parameters", [])
    assert [p["name"] for p in params] == ["probe"]


def test_graph_health_on_a_config_with_no_graph_section_is_503_not_500(tmp_path):
    """Nothing is broken — a lever is absent, and the response names which."""
    path = tmp_path / "config.toml"
    path.write_text(
        "[api]\nhost = \"127.0.0.1\"\nport = 8000\ncors_origins = []\nserve_media = false\n",
        encoding="utf-8",
    )
    app = create_app(load_config(path))
    response = TestClient(app).get("/graph/health")
    assert response.status_code == 503
    assert "graph" in response.json()["error"].lower()
    assert "config.toml" in response.json()["hint"]


def test_graph_health_does_not_affect_the_answering_health_check(tmp_path, no_env):
    """Graph is not a dependency of answering a question, so /health is untouched by it."""
    app = create_app(load_config(write_config(tmp_path)))
    # /health needs the answer levers, which this config does not carry; what matters is
    # that /graph/health being unready never turns into a 5xx on the ask path.
    assert TestClient(app).get("/graph/health").status_code == 200
