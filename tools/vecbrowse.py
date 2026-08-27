"""vecbrowse — a read-only browser for the Chroma collection.

    make browse            # then open http://127.0.0.1:8100

Why this exists as a separate tool rather than endpoints on src/api.py: that app's shape is
the reviewed API contract (`make openapi` prints it, a frontend generates a client from it),
and a debugging view has no business in it. Nothing here writes: no add, no update, no
delete_collection. The only model call is embedding a query string, and it goes through the
shared Meter like every other call in this repo.

Two things it shows that `make api` cannot:

  * the index as a table — every chunk, its video_id and time range, paged, filterable by
    video and by substring. `make api` only ever shows you the k chunks one question hit.
  * cosine distance per hit, and an arbitrary k. retrieve.retrieve() reads k from
    config.retrieve.top_k, which is the gate's lever and not a browsing control.

Concurrency: this opens its own Chroma PersistentClient on ./chroma. Reading alongside a
running `make api` is fine; running it against a live `make index` is not — that writes, and
you would be reading a half-written index.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import Config, load
from src.retrieve import RetrieveError, _embed_question, _query
from src.telemetry import Meter

# Not 8000: that is config.toml [api] port, and the point of a browser is to have it open
# next to the thing it is explaining.
DEFAULT_PORT = 8100


class BrowseError(Exception):
    """Reading the store failed — the message says which step."""


# ---------------------------------------------------------------------------
# Store access
# ---------------------------------------------------------------------------


def _collection(cfg: Config):
    """The configured collection, or a BrowseError naming what to run."""
    import chromadb

    chroma_path = Path(cfg.get("embed.chroma_path"))
    name = cfg.get("embed.collection")

    if not chroma_path.exists():
        raise BrowseError(
            f"no store at {chroma_path} — run `make index-dev` to build one"
        )

    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        return client.get_collection(name)
    except Exception as exc:
        raise BrowseError(
            f"collection {name!r} is not in {chroma_path} — run `make index-dev`"
        ) from exc


def info(cfg: Config) -> dict:
    """Collection identity, size, and the per-video tally.

    The tally is what tells you an index is lopsided — 800 chunks all from one video reads
    as a healthy count until it is broken out per video.
    """
    col = _collection(cfg)
    got = col.get(include=["metadatas"])
    tally: dict[str, int] = {}
    for md in got["metadatas"] or []:
        vid = str((md or {}).get("video_id", "—"))
        tally[vid] = tally.get(vid, 0) + 1

    return {
        "collection": cfg.get("embed.collection"),
        "path": str(Path(cfg.get("embed.chroma_path")).resolve()),
        "count": col.count(),
        "embed_model": cfg.get("embed.model"),
        "top_k": int(cfg.get("retrieve.top_k")),
        "videos": [
            {"video_id": v, "chunks": n}
            for v, n in sorted(tally.items(), key=lambda kv: -kv[1])
        ],
        "config": cfg.fingerprint(),
    }


def records(
    cfg: Config,
    *,
    offset: int = 0,
    limit: int = 50,
    video_id: str | None = None,
    contains: str | None = None,
) -> dict:
    """One page of the index, in video/time order.

    Chroma's get() has no order_by, so the sort is done here. That is honest for this
    corpus — a few thousand chunks — and would need paging in the store to stay honest at a
    hundred thousand.
    """
    col = _collection(cfg)

    where = {"video_id": video_id} if video_id else None
    where_document = {"$contains": contains} if contains else None

    got = col.get(
        where=where,
        where_document=where_document,
        include=["documents", "metadatas"],
    )

    rows = []
    for i, cid in enumerate(got["ids"]):
        md = (got["metadatas"] or [{}] * len(got["ids"]))[i] or {}
        doc = (got["documents"] or [""] * len(got["ids"]))[i] or ""
        rows.append(
            {
                "id": cid,
                "video_id": str(md.get("video_id", "")),
                "t_start": float(md.get("t_start", 0.0)),
                "t_end": float(md.get("t_end", 0.0)),
                "text": doc,
                "chars": len(doc),
            }
        )

    rows.sort(key=lambda r: (r["video_id"], r["t_start"]))
    total = len(rows)
    return {"total": total, "offset": offset, "limit": limit,
            "rows": rows[offset : offset + limit]}


def query(cfg: Config, text: str, k: int = 10) -> dict:
    """Semantic search: embed `text` via Ollama, return the k nearest chunks.

    Reuses src.retrieve's embedding path rather than calling ollama here, so the query is
    embedded exactly the way the indexed chunks were — same model id, same tag translation,
    same no-task-prefix decision recorded in config.toml [embed]. A browser that embedded
    queries even slightly differently would show ranks the pipeline never sees.
    """
    meter = Meter()
    vector = _embed_question(text, cfg.get("embed.model"), meter)
    hits = _query(
        vector,
        k,
        Path(cfg.get("embed.chroma_path")),
        cfg.get("embed.collection"),
    )
    return {
        "query": text,
        "k": k,
        "hits": [
            {
                "rank": i + 1,
                "video_id": h.video_id,
                "t_start": h.t_start,
                "t_end": h.t_end,
                "text": h.text,
                "distance": h.score,
            }
            for i, h in enumerate(hits)
        ],
        # Whether the query embedding was slow enough to notice, from the same meter the
        # pipeline uses. `make latency` reads the durable version of this.
        "elapsed_s": round(meter.elapsed_s, 3),
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def create_app(cfg: Config):
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    app = FastAPI(title="vecbrowse", description=__doc__, docs_url="/docs")
    app.state.cfg = cfg

    class QueryIn(BaseModel):
        text: str
        k: int = 10

    def _guard(fn, *args, **kwargs):
        """Store and embed failures are 400s with the message, not 500s with a traceback."""
        try:
            return fn(*args, **kwargs)
        except (BrowseError, RetrieveError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/", include_in_schema=False)
    def root() -> HTMLResponse:
        return HTMLResponse(PAGE)

    @app.get("/api/info")
    def api_info() -> dict:
        return _guard(info, app.state.cfg)

    @app.get("/api/records")
    def api_records(
        offset: int = 0,
        limit: int = Query(50, ge=1, le=500),
        video_id: str | None = None,
        contains: str | None = None,
    ) -> dict:
        return _guard(
            records,
            app.state.cfg,
            offset=offset,
            limit=limit,
            video_id=video_id,
            contains=contains,
        )

    @app.post("/api/query")
    def api_query(body: QueryIn) -> dict:
        if not body.text.strip():
            raise HTTPException(status_code=400, detail="empty query")
        return _guard(query, app.state.cfg, body.text, body.k)

    return app


# The page. Inline and not in web/ because web/ is the demo frontend, a reviewed artefact
# with its own conventions; this is a tool. Same no-build-step rule though, and the same
# rule app.js states first: nothing from the store is written with innerHTML — every
# document string is a model transcript and textContent is the only safe version.
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>vecbrowse — the VRAG index</title>
<style>
  :root{--bg:#fff;--fg:#14161a;--dim:#666e7a;--line:#e3e6ea;--card:#f7f8fa;--accent:#2b5cd9}
  @media(prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8eaed;--dim:#98a1ad;--line:#2a2f37;--card:#1c2027;--accent:#7aa2ff}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
  header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;
         gap:16px;align-items:baseline;flex-wrap:wrap}
  h1{font-size:15px;margin:0;font-weight:650}
  .meta{color:var(--dim);font-size:12px}
  main{padding:20px;max-width:1180px}
  .tabs{display:flex;gap:6px;margin-bottom:16px}
  .tabs button{padding:7px 14px;border:1px solid var(--line);background:transparent;
               color:var(--fg);border-radius:7px;cursor:pointer;font:inherit}
  .tabs button[aria-selected=true]{background:var(--accent);color:#fff;border-color:var(--accent)}
  .controls{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;align-items:center}
  input,select{padding:7px 10px;border:1px solid var(--line);border-radius:7px;
               background:var(--bg);color:var(--fg);font:inherit}
  input[type=text]{min-width:260px}
  button.go{padding:7px 14px;border:0;border-radius:7px;background:var(--accent);
            color:#fff;cursor:pointer;font:inherit}
  table{border-collapse:collapse;width:100%;font-size:13px}
  th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);
        vertical-align:top}
  th{color:var(--dim);font-weight:600;font-size:12px;position:sticky;top:0;background:var(--bg)}
  td.mono{font-variant-numeric:tabular-nums;white-space:nowrap;color:var(--dim)}
  td.doc{white-space:pre-wrap}
  .pill{display:inline-block;padding:1px 7px;border-radius:99px;background:var(--card);
        border:1px solid var(--line);font-size:12px;margin-right:6px}
  .wrap{overflow-x:auto;border:1px solid var(--line);border-radius:9px}
  .note{color:var(--dim);margin:10px 0}
  .err{color:#c0392b;white-space:pre-wrap}
  @media(prefers-color-scheme:dark){.err{color:#ff8a7a}}
  .pager{display:flex;gap:10px;align-items:center;margin-top:12px;color:var(--dim)}
</style></head><body>

<header>
  <h1>vecbrowse</h1>
  <span class="meta" id="hdr">loading…</span>
</header>

<main>
  <div class="tabs" role="tablist">
    <button id="tab-browse" role="tab" aria-selected="true">Browse</button>
    <button id="tab-query"  role="tab" aria-selected="false">Query</button>
  </div>

  <section id="pane-browse">
    <div class="controls">
      <select id="f-video"><option value="">all videos</option></select>
      <input type="text" id="f-contains" placeholder="document contains…">
      <select id="f-limit">
        <option>25</option><option selected>50</option><option>100</option><option>250</option>
      </select>
      <button class="go" id="f-go">Filter</button>
    </div>
    <div id="tally" class="note"></div>
    <div class="wrap"><table>
      <thead><tr><th>video</th><th>start</th><th>end</th><th>chars</th><th>document</th></tr></thead>
      <tbody id="rows"></tbody>
    </table></div>
    <div class="pager">
      <button class="go" id="prev">‹ prev</button>
      <span id="page"></span>
      <button class="go" id="next">next ›</button>
    </div>
  </section>

  <section id="pane-query" hidden>
    <div class="controls">
      <input type="text" id="q" placeholder="ask the index — nearest chunks, by cosine distance">
      <select id="q-k"><option>5</option><option selected>10</option><option>25</option></select>
      <button class="go" id="q-go">Search</button>
    </div>
    <div id="q-note" class="note">Embeds via Ollama with the indexed model. Lower distance = closer.</div>
    <div class="wrap"><table>
      <thead><tr><th>#</th><th>dist</th><th>video</th><th>start</th><th>end</th><th>document</th></tr></thead>
      <tbody id="hits"></tbody>
    </table></div>
  </section>
</main>

<script>
(function(){
  'use strict';
  var offset = 0;

  function el(tag, cls, text){
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }
  function clock(s){
    s = Math.max(0, Number(s) || 0);
    var m = Math.floor(s/60), r = (s%60).toFixed(1).padStart(4,'0');
    return m + ':' + r;
  }
  function get(url){
    return fetch(url).then(function(r){
      return r.json().then(function(j){
        if (!r.ok) throw new Error(j.detail || ('HTTP ' + r.status));
        return j;
      });
    });
  }

  function fail(tbody, cols, msg){
    tbody.textContent = '';
    var td = el('td', 'err', msg);
    td.colSpan = cols;
    var tr = el('tr'); tr.appendChild(td); tbody.appendChild(tr);
  }

  // ---- header + video filter -------------------------------------------
  get('/api/info').then(function(i){
    document.getElementById('hdr').textContent =
      i.count + ' chunks · collection "' + i.collection + '" · ' + i.embed_model;
    var sel = document.getElementById('f-video');
    var tally = document.getElementById('tally');
    i.videos.forEach(function(v){
      var o = el('option', null, v.video_id + '  (' + v.chunks + ')');
      o.value = v.video_id;
      sel.appendChild(o);
      tally.appendChild(el('span', 'pill', v.video_id + ' · ' + v.chunks));
    });
  }).catch(function(e){
    document.getElementById('hdr').textContent = '';
    document.getElementById('hdr').appendChild(el('span','err', e.message));
  });

  // ---- browse -----------------------------------------------------------
  function load(){
    var v = document.getElementById('f-video').value;
    var c = document.getElementById('f-contains').value.trim();
    var lim = document.getElementById('f-limit').value;
    var u = '/api/records?offset=' + offset + '&limit=' + lim;
    if (v) u += '&video_id=' + encodeURIComponent(v);
    if (c) u += '&contains=' + encodeURIComponent(c);

    var tb = document.getElementById('rows');
    get(u).then(function(d){
      tb.textContent = '';
      if (!d.rows.length){
        var td = el('td','note','no chunks match'); td.colSpan = 5;
        var tr = el('tr'); tr.appendChild(td); tb.appendChild(tr);
      }
      d.rows.forEach(function(r){
        var tr = el('tr');
        tr.appendChild(el('td','mono', r.video_id));
        tr.appendChild(el('td','mono', clock(r.t_start)));
        tr.appendChild(el('td','mono', clock(r.t_end)));
        tr.appendChild(el('td','mono', r.chars));
        tr.appendChild(el('td','doc', r.text));
        tb.appendChild(tr);
      });
      var to = Math.min(d.offset + d.rows.length, d.total);
      document.getElementById('page').textContent =
        d.total ? (d.offset + 1) + '–' + to + ' of ' + d.total : '0 of 0';
      window.__total = d.total; window.__lim = Number(lim);
    }).catch(function(e){ fail(tb, 5, e.message); });
  }

  document.getElementById('f-go').onclick = function(){ offset = 0; load(); };
  document.getElementById('f-contains').addEventListener('keydown', function(e){
    if (e.key === 'Enter'){ offset = 0; load(); }
  });
  document.getElementById('prev').onclick = function(){
    offset = Math.max(0, offset - (window.__lim || 50)); load();
  };
  document.getElementById('next').onclick = function(){
    if (offset + (window.__lim || 50) < (window.__total || 0)){
      offset += (window.__lim || 50); load();
    }
  };
  load();

  // ---- query ------------------------------------------------------------
  function search(){
    var text = document.getElementById('q').value.trim();
    if (!text) return;
    var k = Number(document.getElementById('q-k').value);
    var tb = document.getElementById('hits');
    var note = document.getElementById('q-note');
    note.textContent = 'embedding…';
    fetch('/api/query', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text:text, k:k})
    }).then(function(r){
      return r.json().then(function(j){
        if (!r.ok) throw new Error(j.detail || ('HTTP ' + r.status));
        return j;
      });
    }).then(function(d){
      note.textContent = d.hits.length + ' hits in ' + d.elapsed_s + 's';
      tb.textContent = '';
      d.hits.forEach(function(h){
        var tr = el('tr');
        tr.appendChild(el('td','mono', h.rank));
        tr.appendChild(el('td','mono', h.distance.toFixed(4)));
        tr.appendChild(el('td','mono', h.video_id));
        tr.appendChild(el('td','mono', clock(h.t_start)));
        tr.appendChild(el('td','mono', clock(h.t_end)));
        tr.appendChild(el('td','doc', h.text));
        tb.appendChild(tr);
      });
    }).catch(function(e){ note.textContent = ''; fail(tb, 6, e.message); });
  }
  document.getElementById('q-go').onclick = search;
  document.getElementById('q').addEventListener('keydown', function(e){
    if (e.key === 'Enter') search();
  });

  // ---- tabs -------------------------------------------------------------
  function tab(which){
    var b = which === 'browse';
    document.getElementById('pane-browse').hidden = !b;
    document.getElementById('pane-query').hidden = b;
    document.getElementById('tab-browse').setAttribute('aria-selected', String(b));
    document.getElementById('tab-query').setAttribute('aria-selected', String(!b));
  }
  document.getElementById('tab-browse').onclick = function(){ tab('browse'); };
  document.getElementById('tab-query').onclick = function(){
    tab('query'); document.getElementById('q').focus();
  };
})();
</script>
</body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--print-info",
        action="store_true",
        help="print the collection summary and exit; no server, no browser",
    )
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    cfg = load(args.config)

    if args.print_info:
        try:
            i = info(cfg)
        except BrowseError as exc:
            print(f"FAIL - {exc}", file=sys.stderr)
            return 1
        print(f"collection  {i['collection']}")
        print(f"path        {i['path']}")
        print(f"count       {i['count']}")
        print(f"model       {i['embed_model']}")
        for v in i["videos"]:
            print(f"  {v['video_id']:>8}  {v['chunks']} chunk(s)")
        return 0

    import uvicorn

    print(f"vecbrowse  http://{args.host}:{args.port}   (read-only, Ctrl+C to stop)")
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
