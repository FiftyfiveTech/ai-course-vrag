"""Ask — VRAG-020. The demo.

Question in, answer out, and every citation is a link that opens the video at the second it
came from:

    make ask Q="what two tools do I need to cut paper?"
    make ask Q="..." ASK_FLAGS=--open        # and open it in a browser

    uv run python -m src.ask "how old was Bernini when he met the Pope" --config config.toml

Nothing here decides anything. `src.answer` already retrieves, generates against
`schemas.answer.json_schema()`, and grounds every citation onto a passage that was really
retrieved; this module takes that `AnswerRun` and makes it *clickable*, which is the one
thing a person outside this repo can check without reading any of it. It is the last mile
and it is deliberately thin — a bug in the answer belongs to VRAG-019, a bug in the link
belongs here.

What comes out
--------------
Two things, and the card wants both:

* **stdout** — the answer, then one block per citation with the passage it came from and a
  `file://` url that opens the page already seeked to that moment.
* **a static HTML page** under `runs/ask/`. One file, no build step, no server, no network:
  the CSS and the JS are inline and the only external reference is the video itself. Open it
  by double-clicking it.

The page has a player per cited video, and clicking a citation seeks that player to
`t_start - ask.pad_s` and pauses it again at `t_end`. The pad is why you hear the run-up to
the sentence instead of landing mid-word, and the pause at `t_end` is what makes the cited
window visible as a window rather than as a starting point.

Pointers, not copies — and the page has to survive that
-------------------------------------------------------
`data/corpus/PROVENANCE.md` is why this is not as simple as `<video src=...>`. No video is in
the repo and none can be: Video-MME's terms forbid redistributing it, `.gitignore` blocks
`samples/`, and the manifest holds urls. So on a machine that has run `make index-dev` the
media is on disk and the page embeds it; on a clean clone that has only asked a question it
is not, and a page whose player is a broken file:// reference would be a worse demo than no
player at all.

Both cases render, and `resolve_source` is where the branch lives:

    local file present   ->  <video src="../../samples/521_….mp4#t=8.8">, seeks in place
    not fetched          ->  a link to the manifest url with &t=8s, opens at the same second

The second one is not a degraded mode to apologise for. The citation is a `(video_id,
t_start)` pair either way, and a YouTube deep link resolves it against the original upload —
which is the only copy this project is allowed to point at.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse

from schemas.answer import Answer
from src.answer import AnswerRun
from src.answer import answer as answer_question
from src.config import Config
from src.config import load as load_config
from src.retrieve import RetrievedChunk
from src.telemetry import Meter

MANIFEST = Path("data/corpus/manifest.json")
SAMPLES = Path("samples")
OUT_DIR = Path("runs/ask")


class AskError(Exception):
    """The demo could not be produced — message says which step and why."""


# ---------------------------------------------------------------------------
# Where a cited video can be watched
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """Everything the page needs to point at one cited video.

    `local` is a fetched media file if there is one; `url` is the manifest url, which is
    always there for a corpus id and is the only pointer a clean clone has. Both can be
    absent for a `video_id` that is not in the manifest and not on disk — the page still
    renders the citation, it just cannot make it clickable, and it says so rather than
    emitting a link that goes nowhere.
    """

    video_id: str
    local: Path | None = None
    url: str | None = None

    @property
    def playable(self) -> bool:
        return self.local is not None

    @property
    def linkable(self) -> bool:
        return self.local is not None or bool(self.url)


def load_manifest(manifest: Path = MANIFEST) -> dict[str, dict]:
    """video_id -> its manifest record. Empty when the manifest is missing.

    Missing is not an error here: `make ask` on a clone that has never run `make corpus`
    still has an index and can still answer, and the page falls back to "no pointer for this
    video" rather than refusing to render.
    """
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(v["video_id"]): v for v in data.get("videos", [])}


def resolve_source(
    video_id: str,
    records: dict[str, dict],
    samples: Path = SAMPLES,
) -> Source:
    """Find the media file and the source url for one cited video_id."""
    from src.index import local_file

    return Source(
        video_id=video_id,
        local=local_file(video_id, samples),
        url=(records.get(video_id) or {}).get("url") or None,
    )


def deep_link(url: str, t_start: float) -> str:
    """A source url that opens at a given second.

    YouTube spells it `&t=<seconds>s`, whole seconds only — a float in that parameter is
    ignored and the video opens at 0, which looks exactly like a broken citation. So the pad
    is applied first and the result is floored, never rounded: rounding up can land after the
    first word of the sentence being cited.
    """
    seconds = max(0, int(t_start))
    parts = urlparse(url)
    query = f"{parts.query}&" if parts.query else ""
    return urlunparse(parts._replace(query=query + urlencode({"t": f"{seconds}s"})))


# ---------------------------------------------------------------------------
# One citation, ready to render
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cite:
    """A citation with everything the page and the terminal need to show it."""

    n: int
    video_id: str
    t_start: float
    t_end: float
    seek_s: float
    passage: str
    source: Source

    @property
    def anchor(self) -> str:
        return f"c{self.n}"

    @property
    def label(self) -> str:
        return f"video {self.video_id} · {clock(self.t_start)}–{clock(self.t_end)}"

    @property
    def href(self) -> str | None:
        """Where clicking this citation goes when the page is not the one being clicked."""
        if self.source.url:
            return deep_link(self.source.url, self.seek_s)
        return None


def build_cites(run: AnswerRun, cfg: Config, records: dict[str, dict]) -> list[Cite]:
    """Turn a grounded `AnswerRun` into the citations the page renders.

    The passage text is looked up from `run.hits` rather than carried on the citation,
    because a citation is three numbers by design (`schemas/answer.py`) and the text it came
    from is what makes the demo checkable by eye. `src.answer.ground` has already snapped
    every citation onto a retrieved chunk, so this lookup hits.
    """
    if run.answer is None:
        return []
    pad = float(cfg.get("ask.pad_s"))
    cites = []
    for n, cite in enumerate(run.answer.citations, start=1):
        cites.append(
            Cite(
                n=n,
                video_id=cite.video_id,
                t_start=cite.t_start,
                t_end=cite.t_end,
                seek_s=max(0.0, cite.t_start - pad),
                passage=passage_for(cite.video_id, cite.t_start, run.hits),
                source=resolve_source(cite.video_id, records),
            )
        )
    return cites


def passage_for(
    video_id: str, t_start: float, hits: list[RetrievedChunk], tol: float = 0.05
) -> str:
    """The retrieved chunk a grounded citation points at, as one line of text."""
    for hit in hits:
        if hit.video_id == video_id and abs(hit.t_start - t_start) <= tol:
            return " ".join(hit.text.split())
    return ""


def clock(seconds: float) -> str:
    """Seconds as `m:ss` (or `h:mm:ss`), because 2847.0 is not a timestamp to a human."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #14161a; --muted: #5b6472; --line: #e2e6ec;
  --card: #f6f8fa; --accent: #1f6feb; --warn: #8a5a00; --warn-bg: #fff8e6;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e6e9ef; --muted: #9aa4b2; --line: #262b33;
    --card: #171a21; --accent: #6aa8ff; --warn: #f0c674; --warn-bg: #241f12;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
main { max-width: 54rem; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
.kicker {
  margin: 0 0 .35rem; text-transform: uppercase; letter-spacing: .09em;
  font-size: .72rem; color: var(--muted);
}
h1 { margin: 0 0 1.5rem; font-size: 1.5rem; line-height: 1.3; font-weight: 620; }
h2 {
  margin: 2.25rem 0 .75rem; font-size: .78rem; font-weight: 650; color: var(--muted);
  text-transform: uppercase; letter-spacing: .07em;
}
.answer {
  background: var(--card); border: 1px solid var(--line); border-left: 3px solid var(--accent);
  border-radius: 8px; padding: 1rem 1.15rem; font-size: 1.08rem;
}
.answer.abstain { border-left-color: var(--warn); background: var(--warn-bg); color: var(--warn); }
ol.cites { list-style: none; margin: 0; padding: 0; display: grid; gap: .6rem; }
ol.cites li {
  border: 1px solid var(--line); border-radius: 8px; padding: .7rem .85rem;
  background: var(--card); transition: border-color .15s;
}
ol.cites li.active { border-color: var(--accent); }
.seek {
  display: inline-flex; gap: .5rem; align-items: baseline; font: inherit; font-size: .93rem;
  font-weight: 600; color: var(--accent); background: none; border: 0; padding: 0;
  cursor: pointer; text-decoration: none; text-align: left;
}
.seek:hover { text-decoration: underline; }
.seek .n {
  color: var(--muted); font-variant-numeric: tabular-nums; font-weight: 500;
}
.seek.dead { color: var(--muted); cursor: default; }
.passage { margin: .45rem 0 0; color: var(--muted); font-size: .9rem; }
figure.player { margin: 0 0 1.25rem; }
figure.player figcaption {
  color: var(--muted); font-size: .8rem; margin-bottom: .4rem;
  font-variant-numeric: tabular-nums;
}
video { width: 100%; max-height: 27rem; background: #000; border-radius: 8px; display: block; }
.fallback {
  border: 1px dashed var(--line); border-radius: 8px; padding: .9rem 1rem;
  color: var(--muted); font-size: .9rem;
}
.fallback a { color: var(--accent); }
footer {
  margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--line);
  color: var(--muted); font-size: .78rem;
}
footer div { margin: .15rem 0; font-variant-numeric: tabular-nums; }
code { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: .95em; }
"""

# Seeking is two lines of the job and the rest of this is the parts that bite. `currentTime`
# is silently dropped while readyState is 0, so a click before the browser has any metadata
# does nothing at all and looks like a dead link — hence the loadedmetadata deferral. And the
# pause at t_end is armed per element rather than as a permanent listener, so scrubbing the
# player by hand afterwards is not fought by a stop that is still installed.
_SCRIPT = """
(function () {
  var stopAt = {};
  function seek(el, target, end, li) {
    stopAt[el.id] = end;
    var go = function () {
      try { el.currentTime = target; } catch (e) { return; }
      var p = el.play();
      if (p && p.catch) { p.catch(function () {}); }
    };
    if (el.readyState === 0) {
      el.addEventListener('loadedmetadata', go, { once: true });
      el.load();
    } else {
      go();
    }
    if (li) {
      var all = document.querySelectorAll('ol.cites li');
      for (var i = 0; i < all.length; i++) { all[i].classList.remove('active'); }
      li.classList.add('active');
    }
    el.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }
  function activate(btn) {
    var el = document.getElementById(btn.getAttribute('data-video'));
    if (!el) { return; }
    seek(el,
         parseFloat(btn.getAttribute('data-seek')),
         parseFloat(btn.getAttribute('data-end')),
         btn.closest ? btn.closest('li') : null);
  }
  var buttons = document.querySelectorAll('button.seek');
  for (var i = 0; i < buttons.length; i++) {
    (function (btn) {
      btn.addEventListener('click', function () { activate(btn); });
    })(buttons[i]);
  }
  var players = document.querySelectorAll('video');
  for (var j = 0; j < players.length; j++) {
    (function (el) {
      el.addEventListener('timeupdate', function () {
        var end = stopAt[el.id];
        if (typeof end === 'number' && isFinite(end) && el.currentTime >= end) {
          delete stopAt[el.id];
          el.pause();
        }
      });
      el.addEventListener('seeking', function () { delete stopAt[el.id]; });
    })(players[j]);
  }
  function fromHash() {
    var id = (location.hash || '').replace('#', '');
    if (!id) { return; }
    var li = document.getElementById(id);
    if (!li) { return; }
    var btn = li.querySelector('button.seek');
    if (btn) { activate(btn); }
  }
  window.addEventListener('hashchange', fromHash);
  fromHash();
})();
"""


def render_page(
    question: str,
    run: AnswerRun,
    cites: list[Cite],
    cfg: Config,
    out_path: Path,
    meter: Meter | None = None,
) -> str:
    """The whole demo as one self-contained HTML document.

    `out_path` is not written here — it is needed to make the `<video src>` relative, so that
    the page keeps working if the checkout is moved or copied to another machine that has the
    same `samples/` beside it. An absolute path would break on the first of those.
    """
    esc = html.escape
    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<title>{esc(_title(question))}</title>")
    parts.append(f"<style>{_STYLE}</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<main>")
    parts.append('<p class="kicker">VRAG · answer with citations</p>')
    parts.append(f"<h1>{esc(question)}</h1>")

    if run.answer is None:
        parts.append(
            f'<section class="answer abstain">The model\'s reply did not validate against '
            f"the answer schema: {esc(run.error or 'unknown')}</section>"
        )
    elif run.answer.abstain:
        parts.append(f'<section class="answer abstain">{esc(run.answer.answer)}</section>')
        parts.append(
            "<h2>Citations</h2><p class=\"passage\">None — the system declined to answer, "
            "and QA_SPEC §4 makes a citation on a declined question incorrect regardless of "
            "what it says.</p>"
        )
    else:
        parts.append(f'<section class="answer">{esc(run.answer.answer)}</section>')

    if cites:
        parts.append("<h2>Citations</h2>")
        parts.append('<ol class="cites">')
        for cite in cites:
            parts.append(f'<li id="{cite.anchor}">{_cite_html(cite)}')
            if cite.passage:
                parts.append(f'<p class="passage">“{esc(cite.passage)}”</p>')
            parts.append("</li>")
        parts.append("</ol>")

        parts.append("<h2>Player</h2>")
        for source in _distinct_sources(cites):
            parts.append(_player_html(source, cites, out_path))

    parts.append(_footer_html(question, run, cfg, meter))
    parts.append("</main>")
    parts.append(f"<script>{_SCRIPT}</script>")
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def _cite_html(cite: Cite) -> str:
    """One citation, as the control that plays it.

    Three renderings, one per kind of pointer, and the difference is not cosmetic — a
    `<button>` seeks the player on this page, an `<a>` leaves for the source url, and a
    `<span>` admits there is nowhere to go. Rendering the third as a dead `<a href="#">`
    would be the one outcome worth avoiding: a citation that looks clickable and is not.
    """
    esc = html.escape
    inner = f'<span class="n">[{cite.n}]</span><span>{esc(cite.label)}</span>'
    if cite.source.playable:
        return (
            f'<button type="button" class="seek" data-video="{esc(_dom_id(cite.video_id))}" '
            f'data-seek="{cite.seek_s:.2f}" data-end="{cite.t_end:.2f}">{inner}</button>'
        )
    href = cite.href
    if href:
        return (
            f'<a class="seek" href="{esc(href)}" target="_blank" rel="noopener">{inner}</a>'
        )
    return f'<span class="seek dead">{inner}</span>'


def _player_html(source: Source, cites: list[Cite], out_path: Path) -> str:
    """A player for one cited video, or an honest note about why there is not one."""
    esc = html.escape
    first = min(c.seek_s for c in cites if c.video_id == source.video_id)
    if source.playable:
        assert source.local is not None
        rel = relative_src(source.local, out_path)
        return (
            f'<figure class="player">'
            f"<figcaption>video {esc(source.video_id)} — "
            f"<code>{esc(source.local.as_posix())}</code></figcaption>"
            f'<video id="{esc(_dom_id(source.video_id))}" controls preload="metadata" '
            f'src="{esc(rel)}#t={first:.2f}"></video>'
            f"</figure>"
        )
    if source.url:
        return (
            f'<figure class="player"><div class="fallback">video '
            f"{esc(source.video_id)} is not in <code>samples/</code> on this machine — the "
            f"corpus is pointers, not copies (<code>data/corpus/PROVENANCE.md</code>). Run "
            f"<code>make sample-real VIDEO_ID={esc(source.video_id)}</code> to fetch it and "
            f"re-run <code>make ask</code>, or "
            f'<a href="{esc(deep_link(source.url, first))}" target="_blank" '
            f'rel="noopener">open it at {esc(clock(first))} at the source</a>.'
            f"</div></figure>"
        )
    return (
        f'<figure class="player"><div class="fallback">video {esc(source.video_id)} is '
        f"neither in <code>samples/</code> nor in <code>data/corpus/manifest.json</code>, so "
        f"there is nowhere to point. Re-run <code>make corpus</code> if the corpus was "
        f"re-selected after this index was built.</div></figure>"
    )


def _footer_html(
    question: str, run: AnswerRun, cfg: Config, meter: Meter | None
) -> str:
    """What produced this page. The same identity line every gate in this repo prints.

    A demo page with no provenance is a screenshot: it cannot be re-run and it cannot be
    disagreed with. These four lines say which prompt, which config bytes, which models and
    what it cost, so the page can be checked against `make ask` rather than believed.
    """
    esc = html.escape
    fingerprint = cfg.fingerprint()
    prompt_path, prompt_sha = prompt_fingerprint(cfg)
    rows = [
        f"answer: {cfg.get('answer.arm')} · {cfg.get('answer.model')} · "
        f"temperature {cfg.get('answer.temperature')}",
        f"retrieval: {cfg.get('embed.model')} · top_k {cfg.get('retrieve.top_k')} · "
        f"{len(run.hits)} passage(s) retrieved",
        f"prompt: {prompt_path.as_posix()} sha256:{prompt_sha[:16]} · "
        f"config: {fingerprint['path']} sha256:{fingerprint['sha256'][:16]}",
    ]
    if meter is not None:
        calls = meter._calls
        rows.append(
            f"cost: {len(calls)} model call(s), "
            f"{sum(c.latency_s for c in calls):.2f}s, "
            f"${sum(c.cost_usd for c in calls):.4f}"
        )
    rows.append(
        "Generated by `make ask` (VRAG-020). Citations are grounded — every one points at a "
        "passage that was actually retrieved (src.answer.ground)."
    )
    body = "".join(f"<div>{esc(r)}</div>" for r in rows)
    return f"<footer>{body}</footer>"


def prompt_fingerprint(cfg: Config) -> tuple[Path, str]:
    """The prompt file behind a run, and a sha256 of its bytes.

    Public because `src.api` puts the same pair in its `provenance` object: the page footer
    and the JSON response have to name the same prompt, and computing the digest twice is how
    they end up disagreeing about which one it was. `"unreadable"` rather than an exception —
    a missing prompt file has already failed the run in `src.answer.load_prompt`, and if it
    somehow has not, a provenance line is the wrong place to raise from.
    """
    path = Path(cfg.get("answer.prompt"))
    try:
        return path, hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return path, "unreadable"


def _distinct_sources(cites: list[Cite]) -> list[Source]:
    """One player per cited video, in the order the citations first mention it."""
    seen: dict[str, Source] = {}
    for cite in cites:
        seen.setdefault(cite.video_id, cite.source)
    return list(seen.values())


def _dom_id(video_id: str) -> str:
    """A DOM id from a corpus video_id — ids are numeric strings, which cannot start one."""
    return "v" + re.sub(r"[^A-Za-z0-9_-]", "-", video_id)


def relative_src(media: Path, out_path: Path) -> str:
    """`media` as a url path relative to the page that will reference it.

    Relative and not absolute so the checkout can be moved or copied; posix separators
    because a Windows backslash in an href is not a path separator to a browser, it is an
    escape, and the player silently shows nothing. `os.path.relpath` rather than
    `Path.relative_to` because the two are usually siblings (`runs/ask/` and `samples/`),
    not nested, and `relative_to` cannot express that.
    """
    try:
        rel = os.path.relpath(media.resolve(), out_path.resolve().parent)
    except ValueError:
        # Different drives on Windows — no relative path exists. Absolute is the only
        # honest answer, and it is still a working file:// reference on this machine.
        return media.resolve().as_uri()
    return Path(rel).as_posix()


def _title(question: str) -> str:
    q = " ".join(question.split())
    return f"VRAG · {q[:70]}" if len(q) <= 70 else f"VRAG · {q[:70].rstrip()}…"


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------


def page_path(question: str, out_dir: Path = OUT_DIR) -> Path:
    """A stable filename for a question: `<sha8>-<slug>.html`.

    Deterministic, so re-asking the same question overwrites its page instead of leaving a
    directory of near-identical files; the digest is on the full question so two questions
    that slug identically still get their own page.
    """
    q = " ".join(question.split())
    digest = hashlib.sha256(q.encode("utf-8")).hexdigest()[:8]
    slug = re.sub(r"[^a-z0-9]+", "-", q.lower()).strip("-")[:48].strip("-") or "question"
    return out_dir / f"{digest}-{slug}.html"


def ask(
    question: str,
    cfg: Config,
    meter: Meter,
    *,
    out_dir: Path = OUT_DIR,
    write: bool = True,
) -> tuple[AnswerRun, list[Cite], Path | None]:
    """Answer a question and build its page. Returns the run, its citations and the path."""
    question = " ".join(question.split())
    if not question:
        raise AskError("no question — `make ask Q=\"…\"`")

    run = answer_question(question, cfg, meter)
    if not run.hits:
        raise AskError(
            "retrieval returned no passages, so there is nothing to cite and nothing to "
            "play. The Chroma collection "
            f"{cfg.get('embed.collection')!r} at {cfg.get('embed.chroma_path')} is empty or "
            "absent — run `make index-dev` first. (An empty index makes every question "
            "abstain, which is a working demo of nothing.)"
        )

    cites = build_cites(run, cfg, load_manifest())

    if not write:
        return run, cites, None

    out_path = page_path(question, out_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_page(question, run, cites, cfg, out_path, meter), encoding="utf-8"
    )
    return run, cites, out_path


def report(
    question: str,
    run: AnswerRun,
    cites: list[Cite],
    out_path: Path | None,
    cfg: Config,
    meter: Meter,
    out=None,
) -> None:
    """The terminal half of the demo: the answer, and a url per citation."""
    out = out or sys.stdout
    print(f"\nQ: {question}\n", file=out)

    if run.answer is None:
        print(f"   SCHEMA-INVALID  {run.error}", file=out)
        print(f"   raw: {run.raw[:300]}", file=out)
    elif run.answer.abstain:
        print(f"   ABSTAIN  {run.answer.answer}", file=out)
    else:
        print(f"   A: {run.answer.answer}", file=out)

    for cite in cites:
        print(f"\n   [{cite.n}] {cite.label}", file=out)
        if out_path is not None:
            print(f"       play  {out_path.resolve().as_uri()}#{cite.anchor}", file=out)
        if cite.source.playable:
            assert cite.source.local is not None
            print(f"       file  {cite.source.local.as_posix()}", file=out)
        if cite.href:
            print(f"       source  {cite.href}", file=out)
        if not cite.source.linkable:
            print("       (no local file and no manifest url for this video)", file=out)
        if cite.passage:
            print(f"       “{cite.passage[:160]}”", file=out)

    for line in run.repairs:
        print(f"\n   grounding: {line}", file=out)

    print("", file=out)
    if out_path is not None:
        print(f"player: {out_path.as_posix()}", file=out)

    calls = meter._calls
    print(
        f"{len(calls)} model call(s), {sum(c.latency_s for c in calls):.2f}s, "
        f"${sum(c.cost_usd for c in calls):.4f}  "
        f"(answer.arm={cfg.get('answer.arm')} {cfg.get('answer.model')})",
        file=out,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="*", help="the question to answer")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--out-dir", default=str(OUT_DIR), help=f"where the page is written (default {OUT_DIR})"
    )
    parser.add_argument(
        "--open", action="store_true", dest="open_", help="open the page in a browser"
    )
    parser.add_argument(
        "--no-html", action="store_true", help="print the answer only; write no page"
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="print the answer as JSON too"
    )
    args = parser.parse_args(argv)

    # Same reason src/answer.py does this: a citation line rendered on a cp1252 console must
    # not exit non-zero for an encoding accident.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if not args.question:
        parser.error('give a question. e.g. make ask Q="what two tools do I need?"')

    question = " ".join(args.question)
    meter = Meter()

    try:
        cfg = load_config(args.config)
        run, cites, out_path = ask(
            question, cfg, meter, out_dir=Path(args.out_dir), write=not args.no_html
        )
    except Exception as exc:
        print(f"FAIL - {exc}", file=sys.stderr)
        return 1

    report(question, run, cites, out_path, cfg, meter)

    if args.as_json:
        payload: Answer | None = run.answer
        print("\n" + json.dumps(
            payload.model_dump() if payload else {"error": run.error}, indent=2
        ))

    if args.open_ and out_path is not None:
        webbrowser.open(out_path.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
