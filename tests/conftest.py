"""Shared test setup.

One job so far: keep the test suite out of the real session log.

`src/telemetry.py` appends a span record per model call and per pipeline stage to
`runs/telemetry/<session>.jsonl`, and it does so as a side effect of using a `Meter` — which
is deliberate, because a dozen call sites should not each have to opt in to being measured.
The cost is that a test run creating hundreds of Meters would write sessions into the
directory `make latency` reads, and "the last session" would mean the last `pytest` rather
than the run someone is actually asking about.

Done in `pytest_configure` rather than in an autouse fixture, and that is not a style
preference — the fixture version did not work:

* `tests/gates/gate_phase2a.py` collects its results in a **session-scoped** fixture, which
  pytest instantiates before any function-scoped autouse fixture. All fifteen questions had
  already run, and been logged, before the fixture that was supposed to disable logging got
  a turn. `make gate` left four session files behind.
* resetting the sink per test meant the next write built a *new* sink with a new timestamp,
  so one `pytest tests/gates` fragmented into four separate "sessions".

`pytest_configure` runs once, before collection and therefore before any test module is even
imported, so nothing can have logged yet. `tests/test_latency.py` turns the sink back on
per-test against a tmp_path, which is the only place that wants one.
"""

from __future__ import annotations

from src import telemetry


def pytest_configure(config):
    """Disable the telemetry session log for the whole test run, before anything imports."""
    import os

    os.environ[telemetry.DISABLE_ENV] = "0"
    telemetry.reset_sink_for_tests()
