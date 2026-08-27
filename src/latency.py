"""Latency — where did the time go?

Reads the session logs `src.telemetry` writes under `runs/telemetry/` and prints the phases
of one session ranked by the time they consumed. The default session is the most recent one,
because the question this exists to answer is always about the run that just happened:

    make latency                      # the last session
    make latency LATENCY_FLAGS=--list # which sessions exist
    make latency SESSION=20260827-104755-32440

    uv run python -m src.latency --all        # every session, one summary line each

Why this is a separate module from `src/telemetry.py`
----------------------------------------------------
Telemetry is on the hot path — every model call in the pipeline goes through a `Meter`, and
`src/telemetry.py` is deliberately small enough to read in one sitting. Report formatting,
argument parsing and file globbing have no business there. This module never writes, imports
nothing from the pipeline, and needs no index, no key and no network: it is a reader over
JSONL, so it works on a log copied off another machine.

What the numbers mean, and what they do not
-------------------------------------------
A `phase` row is the sum of the spans carrying that label. Spans are flat and
non-overlapping by construction (`Meter.stage`), so those sums add up and the percentages
mean something.

The row that matters most is **unattributed**: request wall time minus the spans inside it.
It is the honest version of "and the rest went somewhere nothing measures", and it is the
reason this module exists at all — the meter used to report 1.44s for a request that took
3.59s, because 60% of it was Chroma client construction and imports that no span covered.
A large unattributed number is not a rounding error, it is a to-do.

`×` counts are per session, so a table from a `make api` that served forty questions reads
as a mean-per-call table; one from a single `make ask` reads as that one question. `--list`
shows the request count so the two are not confused.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.telemetry import SESSION_DIR


class LatencyError(Exception):
    """There is no session log to read, or the one named does not exist."""


# ---------------------------------------------------------------------------
# Reading a session
# ---------------------------------------------------------------------------


@dataclass
class Span:
    phase: str
    model: str
    t: float
    s: float
    cost: float = 0.0

    @property
    def end(self) -> float:
        return self.t + self.s


@dataclass
class Request:
    what: str
    t: float
    s: float

    @property
    def end(self) -> float:
        return self.t + self.s


@dataclass
class Session:
    """One process's worth of spans."""

    id: str
    path: Path
    started_at: str = ""
    argv: list[str] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    requests: list[Request] = field(default_factory=list)
    bad_lines: int = 0

    @property
    def command(self) -> str:
        """The invocation, shortened — `python -m src.ask` reads better than a venv path."""
        if not self.argv:
            return "(unrecorded)"
        argv = list(self.argv)
        argv[0] = Path(argv[0]).name
        return " ".join(argv)

    @property
    def span_total_s(self) -> float:
        return sum(sp.s for sp in self.spans)

    @property
    def measured_wall_s(self) -> float:
        """Wall time this session can account for.

        The sum of the request brackets when there are any — that is a real end-to-end
        measurement. Otherwise the span timeline's extent (last end minus first start),
        which is what a gate run or an ingest has, since neither goes through an /ask.
        """
        if self.requests:
            return sum(r.s for r in self.requests)
        if not self.spans:
            return 0.0
        return max(sp.end for sp in self.spans) - min(sp.t for sp in self.spans)

    @property
    def unattributed_s(self) -> float:
        """Wall time inside no span. Never negative — see `_overlap` on why it could look it."""
        return max(0.0, self.measured_wall_s - self._spans_inside_requests())

    def _spans_inside_requests(self) -> float:
        """Span time that the wall figure actually covers.

        With request brackets, only the spans inside one count against them: a `make api`
        process embeds at import time and serves its first question later, and charging that
        warm-up to a request that had not started yet would show a negative remainder.
        """
        if not self.requests:
            return self.span_total_s
        total = 0.0
        for sp in self.spans:
            if any(r.t <= sp.t and sp.end <= r.end + 1e-6 for r in self.requests):
                total += sp.s
        return total

    def phases(self) -> list[PhaseStat]:
        """Per-phase totals, worst first — the answer to the question this module asks."""
        groups: dict[str, list[Span]] = {}
        for sp in self.spans:
            groups.setdefault(sp.phase or "unlabelled", []).append(sp)
        stats = [PhaseStat.of(phase, spans) for phase, spans in groups.items()]
        return sorted(stats, key=lambda p: p.total_s, reverse=True)

    def slowest(self) -> Span | None:
        return max(self.spans, key=lambda sp: sp.s) if self.spans else None


@dataclass
class PhaseStat:
    phase: str
    calls: int
    total_s: float
    mean_s: float
    max_s: float
    cost_usd: float
    models: list[str]

    @classmethod
    def of(cls, phase: str, spans: list[Span]) -> "PhaseStat":
        total = sum(sp.s for sp in spans)
        return cls(
            phase=phase,
            calls=len(spans),
            total_s=total,
            mean_s=total / len(spans),
            max_s=max(sp.s for sp in spans),
            cost_usd=sum(sp.cost for sp in spans),
            models=sorted({sp.model for sp in spans if sp.model}),
        )

    @property
    def model_label(self) -> str:
        """Which model served this phase. `-` for a stage that makes no model call."""
        if not self.models:
            return "-"
        if len(self.models) == 1:
            return _short_model(self.models[0])
        # Two models under one phase is the Groq→Ollama fallback having fired.
        return " + ".join(_short_model(m) for m in self.models)


def read_session(path: Path) -> Session:
    """Parse one session log. A truncated last line is counted, not fatal.

    The log is appended to while a server runs, so reading it live can catch a half-written
    line. That is a normal condition for this file rather than corruption, and losing one
    span is not a reason to refuse to report on the other four hundred.
    """
    session = Session(id=path.stem, path=path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LatencyError(f"cannot read {path}: {exc}") from exc

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            session.bad_lines += 1
            continue
        kind = record.get("kind")
        if kind == "session":
            session.started_at = str(record.get("started_at", ""))
            session.argv = [str(a) for a in record.get("argv", [])]
        elif kind == "span":
            session.spans.append(
                Span(
                    phase=str(record.get("phase", "")),
                    model=str(record.get("model", "")),
                    t=float(record.get("t", 0.0)),
                    s=float(record.get("s", 0.0)),
                    cost=float(record.get("cost", 0.0) or 0.0),
                )
            )
        elif kind == "request":
            session.requests.append(
                Request(
                    what=str(record.get("what", "")),
                    t=float(record.get("t", 0.0)),
                    s=float(record.get("s", 0.0)),
                )
            )
        else:
            session.bad_lines += 1
    return session


def session_paths(session_dir: Path = SESSION_DIR) -> list[Path]:
    """Every session log, newest first.

    Sorted by the filename and not by mtime: the name carries the start time, and a server
    that ran for an hour has a newer mtime than a `make ask` that started after it did.
    Ordering by mtime would call that server "the last session" for the rest of the day.
    """
    if not session_dir.is_dir():
        return []
    return sorted(session_dir.glob("*.jsonl"), key=lambda p: p.name, reverse=True)


def latest_session(session_dir: Path = SESSION_DIR) -> Session:
    """The most recent session that recorded at least one span.

    Empty logs are skipped rather than reported. Importing `src.telemetry` is enough to
    create a sink, so a session that made no model call and ran no stage leaves a header and
    nothing else — and "the last session" meaning that file, rather than the run the person
    is actually asking about, would be useless.
    """
    paths = session_paths(session_dir)
    if not paths:
        raise LatencyError(
            f"no session logs in {session_dir.as_posix()}/ — nothing has run yet, or "
            f"VRAG_TELEMETRY_LOG=0 was set. Run `make ask Q=\"...\"` or `make api` and a "
            f"question through it, then try again."
        )
    for path in paths:
        session = read_session(path)
        if session.spans:
            return session
    raise LatencyError(
        f"{len(paths)} session log(s) in {session_dir.as_posix()}/ but none recorded a span. "
        f"That happens when a process imported the pipeline and exited before doing any "
        f"work — try `make ask Q=\"...\"`."
    )


def find_session(session_id: str, session_dir: Path = SESSION_DIR) -> Session:
    path = session_dir / f"{session_id}.jsonl"
    if not path.is_file():
        have = ", ".join(p.stem for p in session_paths(session_dir)[:5]) or "none"
        raise LatencyError(f"no session {session_id!r} in {session_dir.as_posix()}/. Have: {have}")
    return read_session(path)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(session: Session, out=None) -> None:
    """The ranked table. This is the whole point of the module."""
    out = out or sys.stdout
    phases = session.phases()
    wall = session.measured_wall_s
    # Percentages are of measured wall time, not of the span total: against the span total
    # they would sum to 100% and hide the unattributed remainder, which is the one row a
    # reader most needs to see.
    denominator = wall if wall > 0 else session.span_total_s

    print(f"\nsession {session.id}   {session.path.as_posix()}", file=out)
    print(f"  started  {session.started_at or '(unrecorded)'}", file=out)
    print(f"  command  {session.command}", file=out)
    if session.requests:
        print(
            f"  requests {len(session.requests)} "
            f"({', '.join(sorted({r.what for r in session.requests}))})",
            file=out,
        )
    print(f"  wall     {wall:.3f}s accounted for, {session.span_total_s:.3f}s in spans", file=out)
    if session.bad_lines:
        print(f"  note     {session.bad_lines} unreadable line(s) skipped", file=out)

    if not phases:
        print("\n  no spans recorded.", file=out)
        return

    width = max(max(len(p.phase) for p in phases), len("unattributed"))
    print(
        f"\n  {'phase':<{width}}  {'n':>3}  {'total':>9}  {'share':>6}  "
        f"{'mean':>8}  {'max':>8}  model",
        file=out,
    )
    print(f"  {'-' * width}  ---  ---------  ------  --------  --------  " + "-" * 28, file=out)
    for p in phases:
        share = (p.total_s / denominator * 100) if denominator else 0.0
        print(
            f"  {p.phase:<{width}}  {p.calls:>3}  {p.total_s:>8.3f}s  {share:>5.1f}%  "
            f"{p.mean_s:>7.3f}s  {p.max_s:>7.3f}s  {p.model_label}",
            file=out,
        )

    unattributed = session.unattributed_s
    share = (unattributed / denominator * 100) if denominator else 0.0
    print(
        f"  {'unattributed':<{width}}  {'':>3}  {unattributed:>8.3f}s  {share:>5.1f}%  "
        f"{'':>8}  {'':>8}  (wall time inside no span)",
        file=out,
    )

    worst = phases[0]
    print(
        f"\n  slowest phase: {worst.phase} — {worst.total_s:.3f}s over {worst.calls} call(s), "
        f"{worst.total_s / denominator * 100:.1f}% of accounted wall time"
        if denominator
        else f"\n  slowest phase: {worst.phase}",
        file=out,
    )
    span = session.slowest()
    if span is not None:
        print(
            f"  slowest single span: {span.phase} {span.s:.3f}s"
            + (f" ({_short_model(span.model)})" if span.model else ""),
            file=out,
        )
    cost = sum(p.cost_usd for p in phases)
    print(f"  cost this session: ${cost:.6f}", file=out)


def list_sessions(session_dir: Path = SESSION_DIR, out=None, limit: int = 20) -> None:
    """One line per session, newest first, so a person can pick one to look at."""
    out = out or sys.stdout
    paths = session_paths(session_dir)
    if not paths:
        print(f"no session logs in {session_dir.as_posix()}/", file=out)
        return
    print(f"\n{'session':<26} {'wall':>9} {'spans':>6} {'reqs':>5}  slowest phase", file=out)
    print("-" * 26 + " " + "-" * 9 + " " + "-" * 6 + " " + "-" * 5 + "  " + "-" * 24, file=out)
    for path in paths[:limit]:
        session = read_session(path)
        phases = session.phases()
        worst = phases[0].phase if phases else "-"
        print(
            f"{session.id:<26} {session.measured_wall_s:>8.3f}s {len(session.spans):>6} "
            f"{len(session.requests):>5}  {worst}",
            file=out,
        )
    if len(paths) > limit:
        print(f"... and {len(paths) - limit} older session(s)", file=out)


def _short_model(model: str) -> str:
    """`nomic-ai/nomic-embed-text-v1.5-GGUF:F16` -> `nomic-embed-text-v1.5-GGUF:F16`.

    The org prefix is the same for every model in a given row and eats the column width that
    the distinguishing half needs. The full HF repo id is in the log.
    """
    return model.split("/", 1)[1] if "/" in model else model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "session",
        nargs="?",
        help="session id to report on (default: the most recent one with spans in it)",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_", help="one line per session, newest first"
    )
    parser.add_argument("--all", action="store_true", help="a full report for every session")
    parser.add_argument(
        "--dir", default=str(SESSION_DIR), help=f"where the logs are (default {SESSION_DIR})"
    )
    args = parser.parse_args(argv)

    # A phase name with an em dash in it must not exit non-zero on a cp1252 console — the
    # same reason src/answer.py, src/ask.py and src/api.py all do this.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    session_dir = Path(args.dir)
    try:
        if args.list_:
            list_sessions(session_dir)
        elif args.all:
            paths = session_paths(session_dir)
            if not paths:
                raise LatencyError(f"no session logs in {session_dir.as_posix()}/")
            for path in paths:
                report(read_session(path))
        elif args.session:
            report(find_session(args.session, session_dir))
        else:
            report(latest_session(session_dir))
    except LatencyError as exc:
        print(f"FAIL - {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
