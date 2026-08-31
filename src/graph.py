"""Microsoft Graph client - the front door of the `graph` capture arm.

The pipeline's seam is `list[Segment]` (src/transcript.py). Today the only producer is
whisper over an audio file. Teams already transcribed its own meetings, so for a meeting
held in *our* tenant there is a second producer that costs nothing and knows something
whisper can never know: **who was speaking**. Teams writes `<v Priya Nair>` voice tags
into the VTT it stores, and no amount of diarisation over an audio file recovers a name.

That is the whole reason this module exists, and the reason its check command prints
speaker names rather than a green tick: the question worth answering before any capture
code is written is "does Graph actually hand us attributed text for this meeting", and
that question has a yes/no answer that a command can produce.

What this module is NOT: it does not ingest, it does not write to `runs/`, and nothing in
the pipeline imports it yet. `[capture] arm` does not exist. Wiring a `graph` producer into
`transcribe()` needs `Segment` to carry a speaker first, and that is a separate task with
its own gate.

Auth is app-only (OAuth2 client credentials) — a daemon reading meetings, not a user
signing in. Three values, from the environment or the env files src/env.py reads:

    GRAPH_TENANT_ID      the directory (a GUID, or a verified domain)
    GRAPH_CLIENT_ID      the app registration's Application (client) ID
    GRAPH_CLIENT_SECRET  a client secret from that registration

Plain `urllib`, no SDK. The client-credentials flow is one form POST and Graph is one
`Authorization: Bearer` header, so `msal` would be a new dependency for two request shapes
this file can state in full — and `make setup` from a clean clone is a promise (CLAUDE.md).

Zero spend: Graph is included with the tenant's own licences and this module can reach no
metered endpoint. No model is called here, so nothing is priced; latency is recorded
through `Meter.stage`, which is what it is for.

    make graph-check                     credentials, token, roles, reachability
    make graph-check GRAPH_FLAGS="--user amy@contoso.com --meeting '<join url>'"
    uv run python -m src.graph --vtt path/to/transcript.vtt      offline, no credentials
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.config import Config, ConfigError
from src.config import load as load_config
from src.env import fingerprint, load_env
from src.telemetry import Meter

# The three names `.env.example` lists. Kept here rather than inline so src/doctor.py and
# the check below agree on what "configured" means.
CREDENTIALS = ("GRAPH_TENANT_ID", "GRAPH_CLIENT_ID", "GRAPH_CLIENT_SECRET")

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


class GraphError(Exception):
    """A Graph or token request could not be served.

    `code` is the machine-readable identifier the service returned — an `AADSTS#####` for
    the token endpoint, `error.code` for Graph — and `hint` is what to do about it.

    Callers branch on `code`, never on `message`. Message text is prose written for a human
    and Microsoft rewrites it; the code is the contract. This is the same rule
    `src/answer.py:_classify_groq_error` follows for Groq's 413, and for the same reason: a
    fallback that triggers on a substring silently stops triggering when the wording moves.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str = "",
        hint: str = "",
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.hint = hint


# --------------------------------------------------------------------------- token errors
#
# The AADSTS codes worth naming. Every one of these is a configuration mistake with a
# different fix, and the token endpoint's own prose does not distinguish "you sent the
# wrong client id" from "your app is in another tenant" in a way that reads at a glance.
#
# 7000222 is the one to expect eventually rather than immediately: a client secret has a
# maximum lifetime, so a pipeline that worked for months stops on a Tuesday. Naming it here
# means the failure says "the secret expired" instead of "invalid_client".
AADSTS_HINTS: dict[str, str] = {
    "AADSTS7000215": (
        "the client secret is wrong. Check GRAPH_CLIENT_SECRET is the secret *value* and "
        "not the Secret ID - the portal shows both and only shows the value once."
    ),
    "AADSTS7000222": (
        "the client secret has expired. Azure portal -> App registrations -> your app -> "
        "Certificates & secrets -> New client secret, then update GRAPH_CLIENT_SECRET."
    ),
    "AADSTS700016": (
        "no application with this client id exists in this tenant. Either GRAPH_CLIENT_ID "
        "is wrong, or GRAPH_TENANT_ID names a different directory than the one the app was "
        "registered in."
    ),
    "AADSTS90002": (
        "the tenant in GRAPH_TENANT_ID does not exist. It is the Directory (tenant) ID "
        "GUID from the app registration's Overview blade, or a verified domain."
    ),
    "AADSTS900023": (
        "the tenant in GRAPH_TENANT_ID is not a recognised directory name. Use the "
        "Directory (tenant) ID GUID."
    ),
    "AADSTS1002012": (
        "the scope is not accepted for the client credentials flow. graph.scope in "
        "config.toml must be the '/.default' form - per-permission scopes are user flows."
    ),
}

# Graph's own `error.code` values that mean something specific for meeting transcripts.
#
# The 403 row is the one that costs a day if it is not written down. App-only access to a
# meeting needs TWO grants, not one: the app role (admin consent, visible in the token's
# `roles` claim) *and* a Teams application access policy authorising the app to act on that
# user's meetings. The role alone yields a 403 that looks exactly like a missing role, and
# the token says the role is there — so the token proves the wrong half.
GRAPH_HINTS: dict[str, str] = {
    "Forbidden": (
        "the token was accepted and the operation was refused. For onlineMeetings and "
        "transcripts app-only, the app role alone is not enough: a Teams administrator must "
        "also grant an application access policy for the meeting organiser -\n"
        "    New-CsApplicationAccessPolicy -Identity vrag-read -AppIds '<client id>' "
        "-Description 'VRAG transcript read'\n"
        "    Grant-CsApplicationAccessPolicy -PolicyName vrag-read -Identity '<organiser "
        "object id>'\n"
        "Allow ~30 minutes for the policy to propagate before re-running this."
    ),
    "Unauthorized": (
        "the token was rejected. Usually a token minted for a different tenant than the "
        "resource, or a clock far enough off that the token is not yet valid."
    ),
    "ResourceNotFound": (
        "the tenant is right and the object is not there. A meeting id is per-organiser: "
        "the same meeting read under the wrong --user is a 404, not a 403."
    ),
    "NotFound": (
        "no such object under this user. Check --user is the meeting *organiser*: a "
        "meeting lives under the organiser's calendar and nobody else's."
    ),
    "SpeakerAttributionNotAllowed": (
        "the transcript exists and the tenant forbids handing over WHO said each line. "
        "This is the tenant switch that makes the graph arm useless for minutes: without "
        "voice tags the VTT is unattributed text, which is what whisper already gives us "
        "for free. Ask the Teams admin whether attribution can be allowed for this app."
    ),
    "TooManyRequests": (
        "Graph throttled this app. Back off for the Retry-After interval; Graph's limits "
        "are per-app-per-tenant, so a second process sharing GRAPH_CLIENT_ID shares them."
    ),
}


# ------------------------------------------------------------------------------- the token


@dataclass(frozen=True)
class Token:
    """One access token, with the claims it carries.

    `claims` is decoded from the JWT payload WITHOUT verifying the signature, and it is used
    only to report and to pre-flight — never to decide that something is permitted. That
    distinction matters: an unverified token is attacker-controlled data in the general case,
    so treating a `roles` claim as authorisation would be a real vulnerability. Here the
    token came from a TLS connection to the authority we configured, and the only consumer is
    a printed table, so reading it is reporting, not trust.

    Which is exactly the thing worth reporting. "Did IT actually grant the permission?" is
    otherwise answered by making a call and reading a 403, and a 403 has two causes (see
    GRAPH_HINTS["Forbidden"]). The `roles` claim separates them: role absent means consent
    was never granted, role present with a 403 means the application access policy is
    missing.
    """

    value: str
    acquired_at: float
    expires_at: float
    claims: dict[str, Any] = field(default_factory=dict)

    @property
    def roles(self) -> tuple[str, ...]:
        """App roles admin consent actually granted. Empty is the interesting case."""
        raw = self.claims.get("roles") or []
        return tuple(str(r) for r in raw) if isinstance(raw, list) else ()

    @property
    def tenant_id(self) -> str:
        return str(self.claims.get("tid") or "")

    @property
    def app_id(self) -> str:
        return str(self.claims.get("appid") or self.claims.get("azp") or "")

    def expires_in_s(self, now: float | None = None) -> float:
        return self.expires_at - (time.time() if now is None else now)

    def is_usable(self, leeway_s: float, now: float | None = None) -> bool:
        """Still valid with room to spare. Leeway covers clock skew and the call itself."""
        return self.expires_in_s(now) > leeway_s


def decode_jwt_claims(token: str) -> dict[str, Any]:
    """The payload segment of a JWT, base64url-decoded. Signature NOT verified.

    Returns {} for anything unparseable rather than raising: a token whose claims cannot be
    read is still a token that may work, and failing the whole check on an undecodable
    payload would report a working configuration as broken.
    """
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    segment = parts[1]
    # JWT strips base64 padding; `binascii` insists on it.
    segment += "=" * (-len(segment) % 4)
    try:
        raw = base64.urlsafe_b64decode(segment)
        claims = json.loads(raw.decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return {}
    return claims if isinstance(claims, dict) else {}


def read_credentials() -> dict[str, tuple[str, str]]:
    """The three Graph credentials as {name: (value, where it came from)}.

    Uses src/env.py, so the process environment, `.env` and `~/.config/*.env` all work and
    every value knows which one it came from — CLAUDE.md puts secrets in `~/.config/` and
    `make doctor` has always reported the source rather than the value.
    """
    env = load_env()
    found: dict[str, tuple[str, str]] = {}
    for name in CREDENTIALS:
        value, source = env.get(name, ("", ""))
        if value.strip():
            found[name] = (value.strip(), source)
    return found


# ------------------------------------------------------------------------------ the client


@dataclass
class GraphClient:
    """App-only Graph access. One per run; it caches the token it acquires.

    Synchronous and un-pooled, matching src/caption.py's NIM arm: this is a handful of
    requests per meeting, not a hot path, and `urllib` is what the rest of the repo reaches
    for when there is no SDK worth the dependency.
    """

    cfg: Config
    credentials: dict[str, tuple[str, str]] = field(default_factory=read_credentials)
    meter: Meter | None = None
    _token: Token | None = field(default=None, repr=False)

    # ---------------------------------------------------------------- levers, from config
    @property
    def authority(self) -> str:
        return str(self.cfg.get("graph.authority")).rstrip("/")

    @property
    def base_url(self) -> str:
        return str(self.cfg.get("graph.base_url")).rstrip("/")

    @property
    def api_version(self) -> str:
        return str(self.cfg.get("graph.api_version"))

    @property
    def timeout_s(self) -> float:
        return float(self.cfg.get("graph.timeout_s"))

    @property
    def cached_token(self) -> Token | None:
        """The token this client already holds, if any. Does not acquire one.

        Exists so `summarise()` can report the `roles` claim of the token `check()` already
        acquired. Reading `_token` from outside would work and would also mean two callers
        that both want the claims mint two tokens, which is the rate-limit incident this
        client caches to avoid.
        """
        return self._token

    # ------------------------------------------------------------------------------ token
    def _require_credentials(self) -> tuple[str, str, str]:
        missing = [n for n in CREDENTIALS if n not in self.credentials]
        if missing:
            raise GraphError(
                f"Graph is not configured: {', '.join(missing)} not set. The names are in "
                f".env.example; the values belong in ~/.config/ai-course-vrag.env, never in "
                f"the repo. See docs/ for what to ask a tenant administrator for.",
                code="NotConfigured",
                hint="uv run python -m src.graph --check reports each one and where it was found",
            )
        return tuple(self.credentials[n][0] for n in CREDENTIALS)  # type: ignore[return-value]

    def token(self, *, force: bool = False) -> Token:
        """A usable access token, from cache when one is still good.

        Caching is not an optimisation here. Graph throttles token requests per app, and a
        naive client that re-authenticates per request turns one meeting's handful of reads
        into a rate-limit incident that looks like a Graph outage.
        """
        leeway = float(self.cfg.get("graph.token_leeway_s"))
        if not force and self._token is not None and self._token.is_usable(leeway):
            return self._token

        tenant, client_id, secret = self._require_credentials()
        url = f"{self.authority}/{urllib.parse.quote(tenant, safe='')}/oauth2/v2.0/token"
        form = urllib.parse.urlencode(
            {
                "client_id": client_id,
                "client_secret": secret,
                "scope": str(self.cfg.get("graph.scope")),
                "grant_type": "client_credentials",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=form,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            method="POST",
        )

        t0 = time.time()
        try:
            with self._timed("graph.token"):
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise _token_error(exc) from exc
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise GraphError(
                f"the token endpoint at {self.authority} could not be reached: {exc}",
                code="Unreachable",
                hint="a proxy or an offline machine looks exactly like this",
            ) from exc

        access = str(body.get("access_token") or "")
        if not access:
            raise GraphError(
                f"the token endpoint returned 200 with no access_token: {sorted(body)}",
                code="NoAccessToken",
            )
        # expires_in is seconds and Microsoft sends ~3599. Treated as advisory: the claims
        # carry `exp`, and when both are present the claim wins because it is what the
        # resource server will actually enforce.
        expires_in = float(body.get("expires_in") or 0.0)
        claims = decode_jwt_claims(access)
        exp = float(claims.get("exp") or 0.0) or (t0 + expires_in)
        self._token = Token(value=access, acquired_at=t0, expires_at=exp, claims=claims)
        return self._token

    # ----------------------------------------------------------------------------- requests
    def url_for(self, path: str) -> str:
        """Absolute Graph url for a resource path. Passes an absolute url through.

        Absolute-passthrough is what makes `paged()` work: Graph's `@odata.nextLink` is a
        fully-formed url carrying an opaque skiptoken, and re-deriving it from parts is how
        a paginator ends up looping on page one forever.
        """
        if path.startswith(("http://", "https://")):
            return path
        return f"{self.base_url}/{self.api_version}/{path.lstrip('/')}"

    def _open(
        self,
        path: str,
        *,
        params: dict[str, str] | None = None,
        accept: str = "application/json",
        phase: str = "graph.get",
    ) -> tuple[bytes, dict[str, str]]:
        url = self.url_for(path)
        if params:
            # safe="$'" keeps OData readable in a log line: `$filter` and the single quotes
            # around a string literal are syntax, not data, and percent-encoding them is
            # legal but turns every filter into an unreadable smear.
            query = urllib.parse.urlencode(params, safe="$' ()")
            url = f"{url}{'&' if '?' in url else '?'}{query}"
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.token().value}",
                "Accept": accept,
            },
            method="GET",
        )
        try:
            with self._timed(phase):
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return response.read(), dict(response.headers)
        except urllib.error.HTTPError as exc:
            raise _graph_error(exc, url) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise GraphError(
                f"GET {url} could not be reached: {exc}",
                code="Unreachable",
            ) from exc

    def get(self, path: str, *, params: dict[str, str] | None = None) -> dict[str, Any]:
        """One Graph GET, decoded as JSON."""
        raw, _ = self._open(path, params=params)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise GraphError(
                f"GET {self.url_for(path)} returned {len(raw)} bytes that are not JSON",
                code="BadResponse",
            ) from exc
        if not isinstance(body, dict):
            raise GraphError(
                f"GET {self.url_for(path)} returned {type(body).__name__}, not an object",
                code="BadResponse",
            )
        return body

    def get_text(
        self, path: str, *, params: dict[str, str] | None = None, accept: str
    ) -> str:
        """One Graph GET whose body is not JSON — a transcript's VTT bytes."""
        raw, _ = self._open(path, params=params, accept=accept, phase="graph.content")
        return raw.decode("utf-8-sig", errors="replace")

    def paged(
        self, path: str, *, params: dict[str, str] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Every item of a collection, following `@odata.nextLink`.

        Bounded by `graph.max_pages`. A cap and not a while-true: a paginator that misreads
        nextLink requests the same page forever, and unbounded that is a silent hang against
        someone's tenant rather than an error with a number in it.
        """
        max_pages = int(self.cfg.get("graph.max_pages"))
        next_path: str | None = path
        next_params = params
        for page in range(max_pages):
            body = self.get(next_path or "", params=next_params)
            for item in body.get("value") or []:
                if isinstance(item, dict):
                    yield item
            next_path = body.get("@odata.nextLink")
            next_params = None  # nextLink already carries the query
            if not next_path:
                return
        raise GraphError(
            f"stopped after graph.max_pages = {max_pages} pages of {path}. Either the "
            f"collection is larger than expected - narrow it with $filter - or nextLink is "
            f"not advancing.",
            code="TooManyPages",
        )

    # ------------------------------------------------------------------------ meeting reads
    def organization(self) -> dict[str, Any]:
        """The tenant this token belongs to. The cheapest proof Graph accepts the token."""
        body = self.get("organization", params={"$select": "id,displayName"})
        values = body.get("value") or []
        return values[0] if values and isinstance(values[0], dict) else {}

    def find_user(self, upn: str) -> dict[str, Any]:
        """Resolve a UPN or email to a directory object. Needs `User.Read.All`.

        Wanted because a meeting hangs off the organiser's object id and people give you an
        email address.
        """
        return self.get(
            f"users/{urllib.parse.quote(upn, safe='')}",
            params={"$select": "id,displayName,userPrincipalName,mail"},
        )

    def find_online_meeting(self, user_id: str, join_web_url: str) -> dict[str, Any]:
        """The meeting behind a Teams join link, as its organiser sees it.

        A join url is what a person can actually copy out of an invitation; the meeting id
        is not in the Teams UI anywhere. So this filter is the only practical entry point,
        and it is a filter rather than a lookup because Graph exposes no by-url route.

        `user_id` must be the ORGANISER. The same meeting read under an attendee is a 404 —
        a meeting is a child of one calendar, not a tenant-wide object.
        """
        quoted = join_web_url.replace("'", "''")  # OData escapes a quote by doubling it
        items = list(
            self.paged(
                f"users/{urllib.parse.quote(user_id, safe='')}/onlineMeetings",
                params={"$filter": f"joinWebUrl eq '{quoted}'"},
            )
        )
        if not items:
            raise GraphError(
                f"no meeting under {user_id} has this joinWebUrl. Either --user is not the "
                f"organiser, or the link is a channel-meeting url, which resolves through "
                f"the channel and not through /onlineMeetings.",
                code="NotFound",
            )
        return items[0]

    def list_transcripts(self, user_id: str, meeting_id: str) -> list[dict[str, Any]]:
        """Transcript metadata for one meeting. Needs `OnlineMeetingTranscript.Read.All`.

        An empty list is the common and uninteresting answer: Teams stores a transcript only
        when somebody pressed Start transcription, and nothing about having a recording
        implies one exists.
        """
        return list(
            self.paged(
                f"users/{urllib.parse.quote(user_id, safe='')}"
                f"/onlineMeetings/{urllib.parse.quote(meeting_id, safe='')}/transcripts"
            )
        )

    def transcript_vtt(self, user_id: str, meeting_id: str, transcript_id: str) -> str:
        """One transcript's VTT text — the bytes that carry the speaker names."""
        return self.get_text(
            f"users/{urllib.parse.quote(user_id, safe='')}"
            f"/onlineMeetings/{urllib.parse.quote(meeting_id, safe='')}"
            f"/transcripts/{urllib.parse.quote(transcript_id, safe='')}/content",
            params={"$format": "text/vtt"},
            accept="text/vtt",
        )

    # ------------------------------------------------------------------------------ helpers
    def _timed(self, phase: str):
        """`meter.stage(phase)` when there is a meter, otherwise a no-op context."""
        if self.meter is not None:
            return self.meter.stage(phase)

        from contextlib import nullcontext

        return nullcontext()


def _token_error(exc: urllib.error.HTTPError) -> GraphError:
    """Turn a token-endpoint HTTPError into a GraphError carrying its AADSTS code."""
    detail = _read_error_body(exc)
    payload = _as_json(detail)
    code = ""
    description = str(payload.get("error_description") or "")
    # The AADSTS number is in the description, not in a field of its own — which is why
    # this is a regex and not a lookup. `error_codes` is an int list and does not carry
    # the AADSTS prefix, so matching the description is the form that maps to the table.
    match = re.search(r"AADSTS\d+", description)
    if match:
        code = match.group(0)
    elif payload.get("error"):
        code = str(payload["error"])
    hint = AADSTS_HINTS.get(code, "")
    first_line = description.splitlines()[0] if description else (detail[:200] or exc.reason)
    return GraphError(
        f"the token endpoint returned {exc.code} ({code or 'no code'}): {first_line}",
        status=exc.code,
        code=code,
        hint=hint,
    )


def _graph_error(exc: urllib.error.HTTPError, url: str) -> GraphError:
    """Turn a Graph HTTPError into a GraphError carrying `error.code`.

    Prefers `innerError.code` over `error.code` when it is there: the outer code is a coarse
    HTTP-ish label ("Forbidden") and the inner one names the actual reason, which is what a
    caller can usefully branch on.
    """
    detail = _read_error_body(exc)
    payload = _as_json(detail).get("error")
    payload = payload if isinstance(payload, dict) else {}
    inner = payload.get("innerError")
    inner = inner if isinstance(inner, dict) else {}
    code = str(inner.get("code") or payload.get("code") or "")
    message = str(payload.get("message") or "").splitlines()
    hint = GRAPH_HINTS.get(code) or GRAPH_HINTS.get(str(payload.get("code") or "")) or ""
    return GraphError(
        f"GET {url} returned {exc.code} ({code or 'no code'})"
        + (f": {message[0]}" if message else ""),
        status=exc.code,
        code=code,
        hint=hint,
    )


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:2000]
    except Exception:  # noqa: BLE001 — the error body is a nicety, never the failure
        return ""


def _as_json(text: str) -> dict[str, Any]:
    try:
        body = json.loads(text)
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


# --------------------------------------------------------------------------------- the VTT
#
# Teams writes WebVTT with a voice span per cue:
#
#     WEBVTT
#
#     d7d4c0f6-.../1-0
#     00:00:05.120 --> 00:00:09.400
#     <v Priya Nair>Let's start with the migration.</v>
#
# The voice tag is the payload of this whole module. `Segment` (src/transcript.py) has no
# speaker field yet, so `VttCue` is deliberately richer than the seam: parsing is offline,
# testable and cheap, and throwing the name away at the parser would mean the name is
# unavailable to the task that adds the field.

_TIMING = re.compile(
    r"(?P<start>\d{1,3}:\d{1,2}:\d{1,2}[.,]\d{1,3}|\d{1,2}:\d{1,2}[.,]\d{1,3})"
    r"\s*-->\s*"
    r"(?P<end>\d{1,3}:\d{1,2}:\d{1,2}[.,]\d{1,3}|\d{1,2}:\d{1,2}[.,]\d{1,3})"
)
# `<v Priya Nair>` and also `<v.loud Priya Nair>` — WebVTT allows classes on the tag.
_VOICE = re.compile(r"<v(?P<classes>(?:\.[^\s.>]+)*)\s+(?P<name>[^>]*)>", re.IGNORECASE)
_TAG = re.compile(r"</?[^>]+>")


@dataclass(frozen=True)
class VttCue:
    """One cue: when, who, what.

    `speaker` is None when the cue carried no voice tag, and that is a real state rather
    than a gap to paper over. An unattributed line is honest; a line attributed to the
    previous speaker because they were probably still talking is a guess that reads
    identically to a fact, and this pipeline's worst possible output is a commitment
    attributed to a named colleague who never made it.
    """

    t_start: float
    t_end: float
    text: str
    speaker: str | None = None


def parse_timestamp(stamp: str) -> float:
    """`HH:MM:SS.mmm`, `H:M:S.m` or `MM:SS.mmm` to seconds.

    Teams is not consistent about padding — real transcripts carry both `0:0:5.0` and
    `00:00:05.000` — so this splits on `:` rather than matching a fixed width.
    """
    parts = stamp.replace(",", ".").split(":")
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"not a WebVTT timestamp: {stamp!r}")
    try:
        numbers = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"not a WebVTT timestamp: {stamp!r}") from exc
    if len(parts) == 2:  # MM:SS.mmm — legal WebVTT, hours omitted
        minutes, seconds = numbers
        return minutes * 60.0 + seconds
    hours, minutes, seconds = numbers
    return hours * 3600.0 + minutes * 60.0 + seconds


def parse_vtt(text: str) -> list[VttCue]:
    """WebVTT text to cues, keeping the speaker.

    Offline and total: it takes bytes off disk as happily as bytes off Graph, which is what
    makes `--vtt <file>` a real test of the interesting half with no tenant at all.

    Malformed cues are skipped rather than fatal. A transcript is data from a service we do
    not control, and one unparseable cue in a 90-minute meeting should cost that cue, not
    the meeting.
    """
    cues: list[VttCue] = []
    # Blocks are separated by a blank line. Splitting on the timing line instead would join
    # a cue to the identifier of the next one.
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n")):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        header = lines[0].strip().upper()
        if header.startswith("WEBVTT") or header.startswith("NOTE") or header.startswith("STYLE"):
            continue
        timing_at = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if timing_at is None:
            continue
        match = _TIMING.search(lines[timing_at])
        if not match:
            continue
        try:
            t_start = parse_timestamp(match.group("start"))
            t_end = parse_timestamp(match.group("end"))
        except ValueError:
            continue
        payload = "\n".join(lines[timing_at + 1 :]).strip()
        if not payload:
            continue
        speaker: str | None = None
        voice = _VOICE.search(payload)
        if voice:
            speaker = voice.group("name").strip() or None
        # Strip every tag, not just the voice span: cues carry <c>, <i>, <b> and ruby too,
        # and a name is the only thing worth lifting out before they go.
        body = _TAG.sub("", payload).strip()
        if not body:
            continue
        cues.append(VttCue(t_start=t_start, t_end=t_end, text=body, speaker=speaker))
    return cues


def speakers(cues: list[VttCue]) -> dict[str, int]:
    """How many cues each speaker got, plus `None` as the unattributed count.

    The number the whole graph arm is judged on. Attribution can be silently disabled at
    the tenant (GRAPH_HINTS["SpeakerAttributionNotAllowed"]), and the symptom is not an
    error — it is a perfectly valid transcript in which every cue is unattributed. So the
    check command prints this tally rather than "transcript fetched: ok".
    """
    tally: dict[str, int] = {}
    for cue in cues:
        key = cue.speaker or "(unattributed)"
        tally[key] = tally.get(key, 0) + 1
    return tally


def to_segments(cues: list[VttCue]) -> list[Any]:
    """Cues as `list[Segment]` — the pipeline's seam.

    **This drops the speaker**, because `Segment` has no field for one. That loss is the
    entire argument for the task that adds it: everything downstream of here consumes
    `Segment`, so until the field exists, the one thing Graph knows and whisper cannot is
    thrown away at the boundary.

    Provided anyway so the arm is reachable from today's pipeline, and so the diff that adds
    the field has one obvious place to change.
    """
    from src.transcript import Segment

    return [Segment(t_start=c.t_start, t_end=c.t_end, text=c.text) for c in cues]


# ------------------------------------------------------------------------------- the check


@dataclass
class Finding:
    """One line of the check table. Mirrors `src/doctor.Check` on purpose."""

    section: str
    name: str
    status: str
    detail: str


def check(
    cfg: Config,
    *,
    user: str | None = None,
    meeting: str | None = None,
    probe: bool = True,
    client: GraphClient | None = None,
) -> list[Finding]:
    """Everything that can be established about Graph access, cheapest claim first.

    Ordered so that the first FAIL is the one to fix. Credentials before token before roles
    before reachability: a missing role reported against an unusable secret is noise, and a
    check that runs everything and prints five reds hides which one is the cause.

    Pass `client` to share one token with a caller that also wants to read its claims —
    `summarise()` does. Without one, a client is built here.
    """
    findings: list[Finding] = []
    creds = read_credentials()

    for name in CREDENTIALS:
        if name in creds:
            value, source = creds[name]
            findings.append(
                Finding("credentials", name, PASS, f"{source:<28} {fingerprint(value)}")
            )
        else:
            findings.append(
                Finding(
                    "credentials",
                    name,
                    FAIL,
                    "not set - .env.example lists the name; the value belongs in "
                    "~/.config/ai-course-vrag.env",
                )
            )

    findings.append(
        Finding(
            "config",
            "endpoint",
            PASS,
            f"{cfg.get('graph.base_url')}/{cfg.get('graph.api_version')} "
            f"via {cfg.get('graph.authority')}",
        )
    )

    if len(creds) < len(CREDENTIALS):
        findings.append(
            Finding("token", "acquire", SKIP, "no credentials to try - fix the FAILs above")
        )
        return findings
    if not probe:
        findings.append(
            Finding(
                "token",
                "acquire",
                SKIP,
                "--no-probe: nothing was sent to Microsoft. Drop the flag to check for real.",
            )
        )
        return findings

    client = client or GraphClient(cfg=cfg, credentials=creds)
    try:
        token = client.token()
    except GraphError as exc:
        findings.append(Finding("token", "acquire", FAIL, str(exc)))
        if exc.hint:
            findings.append(Finding("token", "fix", WARN, exc.hint))
        return findings

    findings.append(
        Finding(
            "token",
            "acquire",
            PASS,
            f"expires in {token.expires_in_s() / 60:.0f} min",
        )
    )
    findings.append(Finding("token", "tenant (tid)", PASS, token.tenant_id or "(no tid claim)"))
    findings.append(Finding("token", "app (appid)", PASS, token.app_id or "(no appid claim)"))

    # The roles claim, checked against config rather than a constant: which permissions this
    # repo needs is a decision that changes with what the arm reads, and a decision belongs
    # in config.toml where it can be seen without reading Python.
    granted = set(token.roles)
    findings.append(
        Finding(
            "roles",
            "granted",
            PASS if granted else FAIL,
            ", ".join(sorted(granted))
            or "none. The token carries no `roles` claim, which means no admin has consented "
            "to any application permission for this app. Nothing app-only can be read until "
            "one does.",
        )
    )
    for role in (str(r) for r in cfg.get("graph.required_roles")):
        findings.append(
            Finding("roles", role, PASS, "granted")
            if role in granted
            else Finding(
                "roles",
                role,
                FAIL,
                "not in the token - an administrator has not granted admin consent for it",
            )
        )
    for role in (str(r) for r in cfg.get("graph.optional_roles")):
        findings.append(
            Finding("roles (optional)", role, PASS, "granted")
            if role in granted
            else Finding("roles (optional)", role, WARN, "not granted - see config.toml [graph]")
        )

    # Reachability. A 403 here is a PASS for the question being asked: Graph parsed the
    # token and refused the operation, which proves tenant, signature and audience are all
    # right. Reporting it as a failure of reachability would point at the network when the
    # answer is a missing grant, and the roles section above already said which.
    try:
        org = client.organization()
        findings.append(
            Finding(
                "reachability",
                "GET /organization",
                PASS,
                f"200 - {org.get('displayName') or org.get('id') or 'tenant reached'}",
            )
        )
    except GraphError as exc:
        reachable = exc.status is not None and exc.status not in (0,)
        findings.append(
            Finding(
                "reachability",
                "GET /organization",
                WARN if reachable else FAIL,
                f"{exc}"
                + (
                    "  (the token was accepted; this endpoint needs Organization.Read.All, "
                    "which the graph arm does not)"
                    if exc.status == 403
                    else ""
                ),
            )
        )

    if not meeting:
        findings.append(
            Finding(
                "transcript",
                "speaker names",
                SKIP,
                "pass --user <organiser upn> --meeting <join url|meeting id> to find out "
                "whether Teams hands this tenant attributed text. That is the question the "
                "graph arm lives or dies on and no amount of green above answers it.",
            )
        )
        return findings

    findings.extend(_check_meeting(client, user, meeting))
    return findings


def _check_meeting(
    client: GraphClient, user: str | None, meeting: str
) -> list[Finding]:
    """Resolve one meeting and report what its transcript actually contains."""
    findings: list[Finding] = []
    if not user:
        return [
            Finding(
                "transcript",
                "user",
                FAIL,
                "--meeting needs --user: a meeting hangs off its ORGANISER's calendar and "
                "cannot be read under anyone else's id.",
            )
        ]

    try:
        who = client.find_user(user)
        user_id = str(who.get("id") or "")
        findings.append(
            Finding(
                "transcript",
                "organiser",
                PASS,
                f"{who.get('displayName') or user} - {user_id}",
            )
        )
    except GraphError as exc:
        # A UPN works in the path as well as an object id, so a failure to resolve the
        # display name is not fatal — carry on with what was passed in.
        user_id = user
        findings.append(
            Finding("transcript", "organiser", WARN, f"{exc}  (continuing with {user!r})")
        )

    meeting_id = meeting
    if meeting.startswith(("http://", "https://")):
        try:
            found = client.find_online_meeting(user_id, meeting)
            meeting_id = str(found.get("id") or "")
            findings.append(
                Finding("transcript", "meeting", PASS, f"{found.get('subject') or ''} {meeting_id}".strip())
            )
        except GraphError as exc:
            findings.append(Finding("transcript", "meeting", FAIL, str(exc)))
            if exc.hint:
                findings.append(Finding("transcript", "fix", WARN, exc.hint))
            return findings

    try:
        items = client.list_transcripts(user_id, meeting_id)
    except GraphError as exc:
        findings.append(Finding("transcript", "list", FAIL, str(exc)))
        if exc.hint:
            findings.append(Finding("transcript", "fix", WARN, exc.hint))
        return findings

    if not items:
        findings.append(
            Finding(
                "transcript",
                "list",
                WARN,
                "the meeting exists and has no transcript. Teams stores one only when "
                "somebody pressed Start transcription - a recording does not imply one.",
            )
        )
        return findings

    findings.append(Finding("transcript", "list", PASS, f"{len(items)} transcript(s)"))
    newest = items[-1]
    transcript_id = str(newest.get("id") or "")
    try:
        vtt = client.transcript_vtt(user_id, meeting_id, transcript_id)
    except GraphError as exc:
        findings.append(Finding("transcript", "content", FAIL, str(exc)))
        if exc.hint:
            findings.append(Finding("transcript", "fix", WARN, exc.hint))
        return findings

    cues = parse_vtt(vtt)
    findings.append(
        Finding("transcript", "content", PASS, f"{len(vtt)} bytes of VTT, {len(cues)} cues")
    )
    tally = speakers(cues)
    named = {k: v for k, v in tally.items() if k != "(unattributed)"}
    unattributed = tally.get("(unattributed)", 0)
    findings.append(
        Finding(
            "transcript",
            "speaker names",
            PASS if named else FAIL,
            ", ".join(f"{k} ({v})" for k, v in sorted(named.items(), key=lambda kv: -kv[1]))
            or "every cue is unattributed. The transcript is valid text with no voice tags, "
            "which is what whisper already produces - the graph arm buys nothing for minutes "
            "on this tenant. See SpeakerAttributionNotAllowed in src/graph.py.",
        )
    )
    if named and unattributed:
        findings.append(
            Finding(
                "transcript",
                "unattributed",
                WARN,
                f"{unattributed} of {len(cues)} cues carry no name "
                f"({unattributed / len(cues):.1%})",
            )
        )
    return findings


@dataclass(frozen=True)
class Summary:
    """`check()`'s findings plus the few facts a caller wants to switch on.

    The CLI prints the findings and the HTTP endpoint serialises this, so both come off one
    code path. The alternative was an endpoint that re-derives "is it ready" by scanning the
    printed table for the word FAIL, which is a parser over prose and drifts the first time
    a message is reworded.
    """

    ready: bool
    detail: str
    configured: bool
    missing: tuple[str, ...]
    probed: bool
    tenant_id: str | None
    required_roles: tuple[str, ...]
    granted_roles: tuple[str, ...]
    missing_roles: tuple[str, ...]
    findings: tuple[Finding, ...]


def summarise(
    cfg: Config,
    *,
    user: str | None = None,
    meeting: str | None = None,
    probe: bool = True,
) -> Summary:
    """Run the check and reduce it to a verdict.

    `ready` means a token was acquired and every `graph.required_roles` entry is in its
    `roles` claim. Deliberately NOT "no FAIL anywhere": the transcript section can fail for
    reasons that are nothing to do with configuration — a meeting with no transcript because
    nobody pressed record — and a deployment is not misconfigured because of somebody else's
    meeting.
    """
    creds = read_credentials()
    missing = tuple(n for n in CREDENTIALS if n not in creds)
    required = tuple(str(r) for r in cfg.get("graph.required_roles"))
    probed = bool(probe and not missing)

    # One client, so the token minted for the role check is the token the summary reports.
    client = GraphClient(cfg=cfg, credentials=creds) if not missing else None
    findings = check(cfg, user=user, meeting=meeting, probe=probe, client=client)

    token = client.cached_token if client is not None else None
    granted = tuple(sorted(token.roles)) if token is not None else ()
    missing_roles = tuple(r for r in required if r not in granted)
    ready = token is not None and not missing_roles

    if missing:
        detail = (
            f"not configured: {', '.join(missing)} unset. The names are in .env.example; the "
            f"values belong in ~/.config/ai-course-vrag.env, never in the repo."
        )
    elif not probed:
        detail = "configured, and nothing was sent to Microsoft - pass probe=true to check."
    elif token is None:
        detail = "credentials are set and no token could be acquired. See the token findings."
    elif missing_roles:
        detail = (
            f"a token was acquired and admin consent is incomplete: {', '.join(missing_roles)} "
            f"missing. An administrator grants these on the app registration's API "
            f"permissions blade."
        )
    else:
        detail = (
            "ready. Note that a granted role is still not sufficient on its own for "
            "onlineMeetings app-only: a Teams application access policy has to name this app "
            "for the meeting organiser. The first 403 on a real meeting is that, not this."
        )

    return Summary(
        ready=ready,
        detail=detail,
        configured=not missing,
        missing=missing,
        probed=probed,
        tenant_id=(token.tenant_id or None) if token is not None else None,
        required_roles=required,
        granted_roles=granted,
        missing_roles=missing_roles,
        findings=tuple(findings),
    )


def report(findings: list[Finding], out=None) -> int:
    """Print the table. Non-zero exit if anything FAILed — same contract as `make doctor`.

    `out=None` and resolved here rather than `out=sys.stdout` in the signature: a default
    argument is evaluated once, at import, so the function would keep writing to whatever
    stream existed then and ignore any later reassignment of `sys.stdout`. That is invisible
    in normal use and it is why the first version of this was untestable under pytest, which
    replaces the stream after import.
    """
    out = sys.stdout if out is None else out
    print("VRAG Graph check - Microsoft Graph, app-only (client credentials)", file=out)
    for section in dict.fromkeys(f.section for f in findings):
        print(f"\n{section}", file=out)
        for finding in (f for f in findings if f.section == section):
            print(f"  {finding.status:<5} {finding.name:<34} {finding.detail}", file=out)
    tally = {s: sum(1 for f in findings if f.status == s) for s in (PASS, WARN, FAIL, SKIP)}
    print(
        f"\n{tally[PASS]} PASS  {tally[WARN]} WARN  {tally[FAIL]} FAIL  {tally[SKIP]} SKIP",
        file=out,
    )
    if tally[FAIL]:
        print(
            f"FAIL - {tally[FAIL]} check(s) failed. Fix the first FAIL above; the rest are "
            f"often downstream of it.",
            file=out,
        )
        return 1
    print("PASS - Graph is reachable with this app's credentials.", file=out)
    return 0


def _report_vtt(path: Path, out=None) -> int:
    """`--vtt <file>`: the parser and the speaker tally, offline, no credentials.

    Here because it is the half of this module that can be checked without a tenant, and
    because it is how a sample Teams transcript gets turned into a fixture.

    Exit code is the verdict, not just "did it parse": a transcript with no names in it is a
    non-zero exit, because an unattributed transcript is precisely what whisper already
    produces and the graph arm exists to do better.
    """
    out = sys.stdout if out is None else out
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return 1
    cues = parse_vtt(text)
    if not cues:
        print(f"{path}: no cues parsed out of {len(text)} bytes - is this WebVTT?", file=out)
        return 1
    tally = speakers(cues)
    print(f"{path}", file=out)
    print(f"  {len(cues)} cues, {cues[-1].t_end:.1f}s of meeting", file=out)
    print("  speakers:", file=out)
    for name, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>5}  {name}", file=out)
    print("\n  first 5 cues:", file=out)
    for cue in cues[:5]:
        who = cue.speaker or "(unattributed)"
        print(f"    [{cue.t_start:8.2f} {cue.t_end:8.2f}] {who}: {cue.text[:70]}", file=out)
    return 0 if any(c.speaker for c in cues) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--user",
        default=None,
        help="the meeting ORGANISER's UPN or object id - a meeting is on their calendar",
    )
    parser.add_argument(
        "--meeting",
        default=None,
        help="a Teams join url, or an onlineMeeting id. With --user, fetches the transcript "
        "and prints who Teams says was speaking.",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="report the configuration and send nothing to Microsoft",
    )
    parser.add_argument(
        "--vtt",
        default=None,
        type=Path,
        help="parse a WebVTT file on disk and print its speakers. Offline; needs no "
        "credentials and makes no request.",
    )
    args = parser.parse_args(argv)

    if args.vtt is not None:
        return _report_vtt(args.vtt)

    try:
        cfg = load_config(args.config)
        findings = check(
            cfg, user=args.user, meeting=args.meeting, probe=not args.no_probe
        )
    except ConfigError as exc:
        print(f"config: {exc}", file=sys.stderr)
        return 1
    return report(findings)


if __name__ == "__main__":
    raise SystemExit(main())
