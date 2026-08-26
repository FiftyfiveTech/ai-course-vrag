/* VRAG frontend — the whole client.
 *
 * Talks to src/api.py: GET /health on boot, POST /ask per question, GET /media/{id} through
 * whatever `stream_url` a citation carries. No framework and no build step, because
 * `make setup` has to work from a clean clone and a second package manager would make that
 * a lie.
 *
 * Three rules the rest of this file follows:
 *
 *   1. Nothing is built with innerHTML. Every string that reaches the page — the answer, a
 *      passage, an error from a provider — came from a model or from the network, and
 *      textContent is the only version of this that is not one bad passage away from script
 *      injection. The one exception is the SVG icon, which is a literal in this file.
 *
 *   2. The three outcomes stay distinguishable, because the API deliberately makes them all
 *      200 (schemas/api.py). `abstain` is the system declining and is a correct answer;
 *      `schema_valid: false` is the model having produced something that was not an answer;
 *      anything else is an answer. Collapsing them into "it worked" would report a bug that
 *      is not there, or hide one that is.
 *
 *   3. A citation is rendered as what it can actually do. stream_url -> a button that seeks
 *      the in-page player; source_url only -> a link that leaves; neither -> plain text that
 *      says so. A dead control that looks live is the failure worth avoiding.
 */

(function () {
  'use strict';

  var thread = document.getElementById('thread');
  var hero = document.getElementById('hero');
  var form = document.getElementById('composer');
  var input = document.getElementById('q');
  var send = document.getElementById('send');
  var statusEl = document.getElementById('status');
  var statusText = document.getElementById('status-text');
  var footConfig = document.getElementById('foot-config');
  var examplesEl = document.getElementById('examples');

  var EXAMPLES = [
    'What two tools do I need to make my first paper cut?',
    'What is demonstrated at the start of the video?',
    'What does the speaker say about the ingredients?'
  ];

  var busy = false;

  // ---------------------------------------------------------------- helpers

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  /** mm:ss, the way a video clock reads. Citations carry seconds. */
  function clock(seconds) {
    var s = Math.max(0, Math.floor(Number(seconds) || 0));
    var m = Math.floor(s / 60);
    return m + ':' + String(s % 60).padStart(2, '0');
  }

  /** A DOM id that is safe to reuse per turn — video_ids are decimal strings from the corpus. */
  function playerId(turn, videoId) {
    return 'v' + turn + '-' + String(videoId).replace(/[^A-Za-z0-9_-]/g, '');
  }

  /**
   * Bring a turn's *top* into view, not its bottom.
   *
   * `block: 'end'` is the obvious choice for a chat log and is wrong here: it parks the last
   * element of the turn against the bottom of the viewport, which is where the composer is
   * fixed, so the provenance chips end up underneath it. Scrolling to the top of the turn
   * puts the question and the answer where the eye already is and leaves the rest to be
   * scrolled to. `.turn` carries a `scroll-margin-top` so the sticky header does not land on
   * top of the question bubble.
   */
  function show(turn) {
    turn.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ------------------------------------------------------------- the bubbly

  /**
   * The answer, revealed a word at a time.
   *
   * The API does not stream — src.ask.ask() returns one whole object — so this is a reveal
   * of text that has fully arrived, not a fake token stream, and it is capped so a long
   * answer does not take longer to appear than it took to compute. Every word is in the DOM
   * from the first frame and only its opacity is animated, so with reduced motion on (or
   * with the stylesheet's animation disabled) the answer is simply there, immediately.
   */
  function reveal(node, text) {
    var words = String(text).split(/(\s+)/);
    var step = 0.028;                       // seconds between words
    var cap = 1.6;                          // ...but the last word is never later than this
    var shown = 0;
    words.forEach(function (word) {
      if (!word.trim()) { node.appendChild(document.createTextNode(word)); return; }
      var span = el('span', 'w', word);
      span.style.setProperty('--d', Math.min(shown * step, cap) + 's');
      node.appendChild(span);
      shown += 1;
    });
  }

  function thinkingBubble() {
    var bubble = el('div', 'bubble-a');
    var dots = el('div', 'thinking');
    dots.appendChild(el('span', 'd1'));
    dots.appendChild(el('span', 'd2'));
    dots.appendChild(el('span', 'd3'));
    bubble.appendChild(dots);
    return bubble;
  }

  // ------------------------------------------------------------------ boot

  function boot() {
    EXAMPLES.forEach(function (question) {
      var chip = el('button', null, question);
      chip.type = 'button';
      chip.addEventListener('click', function () {
        input.value = question;
        input.focus();
      });
      examplesEl.appendChild(chip);
    });

    fetch('/health')
      .then(function (r) { return r.json(); })
      .then(function (h) {
        // An empty index is not an error and must not be shown as one: the server is up and
        // every question would abstain, which is a working demo of nothing. `detail` names
        // the command to run, so it goes in the tooltip rather than being invented here.
        var ready = !!h.ready;
        statusEl.dataset.state = ready ? 'ready' : 'empty';
        statusText.textContent = ready
          ? (h.index.chunks + ' chunks · ' + h.index.videos.length + ' videos · ' + h.arm)
          : 'index empty';
        statusEl.title = h.detail + '\n' + h.answer_model + '\n' + h.embed_model;
        // Basename, not the path: --config can be an absolute one and the bar is not the
        // place for it. The full path and both digests are in the tooltip.
        footConfig.textContent = String(h.config).split(/[\\/]/).pop() + ' · ' + String(h.config_sha256).slice(0, 12);
        footConfig.title = h.config + '\n' + h.config_sha256;
      })
      .catch(function () {
        statusEl.dataset.state = 'down';
        statusText.textContent = 'api unreachable';
        statusEl.title = 'is the server running? `make api`';
      });

    // /?q=... asks on load, so a question is a link someone can send. It is also the only
    // way this page is reachable by anything that is not a pair of hands — a headless
    // browser can open one url and see the answered state, which is what makes the render
    // checkable at all rather than checkable by someone clicking.
    var asked = new URLSearchParams(location.search).get('q');
    if (asked && asked.trim()) { ask(asked.trim()); }
  }

  // ------------------------------------------------------------------- ask

  form.addEventListener('submit', function (event) {
    event.preventDefault();
    var question = input.value.trim();
    if (!question || busy) { return; }
    input.value = '';
    ask(question);
  });

  function ask(question) {
    busy = true;
    send.disabled = true;
    hero.classList.add('gone');

    var turn = el('section', 'turn');
    turn.appendChild(el('div', 'bubble-q', question));
    var pending = thinkingBubble();
    turn.appendChild(pending);
    thread.appendChild(turn);
    show(turn);

    var index = thread.children.length;

    fetch('/ask', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ question: question })
    })
      .then(function (response) {
        return response.json().then(function (body) {
          return { ok: response.ok, status: response.status, body: body, headers: response.headers };
        });
      })
      .then(function (result) {
        pending.remove();
        if (result.ok) {
          renderAnswer(turn, result.body, index);
        } else {
          renderProblem(turn, result);
        }
      })
      .catch(function (err) {
        pending.remove();
        renderProblem(turn, { status: 0, body: { error: String(err && err.message || err), hint: 'is the server running? `make api`' } });
      })
      .then(function () {
        busy = false;
        send.disabled = false;
        input.focus();
        show(turn);
      });
  }

  // --------------------------------------------------------------- render

  function renderAnswer(turn, data, index) {
    var bubble = el('div', 'bubble-a');

    if (!data.schema_valid) {
      // Not an outage and not an abstention: the model replied and the reply was not an
      // Answer. It is the number VRAG-019 is measured on, so it says so instead of hiding.
      bubble.classList.add('broken');
      reveal(bubble, 'The reply did not validate against the answer schema.');
      var why = el('p', 'why');
      why.appendChild(el('code', null, data.error || 'unknown'));
      bubble.appendChild(why);
    } else {
      if (data.abstain) { bubble.classList.add('abstain'); }
      reveal(bubble, data.answer);
    }
    turn.appendChild(bubble);

    if (data.abstain) {
      // QA_SPEC §4: a citation on a declined question is incorrect regardless of what it
      // says. The server sends none; this explains the absence rather than leaving a gap.
      turn.appendChild(el('p', 'note',
        'Declined — the corpus does not cover this, so there is nothing to cite.'));
    }

    if (data.citations && data.citations.length) {
      renderCitations(turn, data.citations, index);
    }

    turn.appendChild(meta(data));
  }

  function renderCitations(turn, citations, index) {
    var head = el('div', 'cite-head', 'Citations');
    turn.appendChild(head);

    var list = el('ol', 'cites');
    var buttons = [];

    citations.forEach(function (c) {
      var item = document.createElement('li');
      var node = citationNode(c, index, buttons);
      item.appendChild(node);
      list.appendChild(item);
    });
    turn.appendChild(list);

    // One player per distinct cited video that is actually on this host. A cited video with
    // only a source_url gets a note, not a player: the media was never fetched here — the
    // corpus is pointers, not copies (data/corpus/PROVENANCE.md).
    var seen = {};
    citations.forEach(function (c) {
      if (seen[c.video_id]) { return; }
      seen[c.video_id] = true;
      if (c.stream_url) {
        turn.appendChild(playerFor(c, index));
      } else if (!c.source_url) {
        turn.appendChild(el('p', 'note',
          'video ' + c.video_id + ' is not on this host and has no source url — the citation ' +
          'stands, but there is nowhere to play it. Try `make sample-real VIDEO_ID=' + c.video_id + '`.'));
      }
    });

    // Deep-linkable: /#2 on a fresh load is meaningless, but clicking one citation and
    // having the player follow is the entire point of the page.
    if (buttons.length) { buttons[0].click(); }
  }

  /**
   * The timestamp, as its own object rather than a run of text inside the label.
   *
   * The API sends `label` already assembled — "video 611 · 1:19–1:25" — and rendering that
   * string is what the first version did. It reads as a sentence, which is wrong for what it
   * is: the times are the citation's coordinates and the thing a reader scans a list of
   * citations *for*. So this builds from t_start/t_end instead and gives each part its own
   * box, tabular figures and its own tint, and the assembled `label` becomes the tooltip.
   *
   * Tabular figures matter more than they look: with proportional digits "0:13" and "1:04"
   * are different widths, so a column of timestamps does not line up and the eye cannot run
   * down it. `font-variant-numeric: tabular-nums` in the stylesheet is what fixes that.
   */
  function stamp(c) {
    var box = el('span', 'stamp');
    box.title = c.label || '';
    box.appendChild(icon('clock'));
    box.appendChild(el('span', 'from', clock(c.t_start)));
    box.appendChild(el('span', 'arrow', '→'));
    box.appendChild(el('span', 'to', clock(c.t_end)));
    var span = Math.max(0, Math.round(c.t_end - c.t_start));
    if (span) { box.appendChild(el('span', 'dur', span + 's')); }
    return box;
  }

  /** Inline SVG, built node by node — the one place markup is assembled, and it is literal. */
  function icon(name) {
    var d = {
      clock: 'M12 7v5l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z',
      play: 'M8 5.5v13l11-6.5z',
      ext: 'M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5'
    }[name];
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('class', 'ico ico-' + name);
    var path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', d);
    path.setAttribute('fill', name === 'play' ? 'currentColor' : 'none');
    path.setAttribute('stroke', name === 'play' ? 'none' : 'currentColor');
    path.setAttribute('stroke-width', '1.8');
    path.setAttribute('stroke-linecap', 'round');
    path.setAttribute('stroke-linejoin', 'round');
    svg.appendChild(path);
    return svg;
  }

  function citationNode(c, index, buttons) {
    var node;

    if (c.stream_url) {
      node = el('button', 'cite');
      node.type = 'button';
      node.addEventListener('click', function () {
        var video = document.getElementById(playerId(index, c.video_id));
        if (!video) { return; }
        document.querySelectorAll('.cite.active').forEach(function (n) { n.classList.remove('active'); });
        node.classList.add('active');
        // seek_s, not t_start — the API already padded it back off the chunk boundary so a
        // viewer hears the run-up instead of landing mid-word (src/ask.py, VRAG-020).
        seek(video, c.seek_s);
      });
      buttons.push(node);
    } else if (c.source_url) {
      node = el('a', 'cite');
      node.href = c.source_url;
      node.target = '_blank';
      node.rel = 'noopener';
    } else {
      node = el('div', 'cite dead');
    }

    node.appendChild(el('span', 'n', c.n));

    // The head row: where it came from on the left, when on the right, and the control on the
    // far right saying what clicking does. Three separate things, so they get three boxes.
    var head = el('div', 'cite-row');
    head.appendChild(el('span', 'vid', 'video ' + c.video_id));
    head.appendChild(stamp(c));

    var action = el('span', 'action');
    if (c.stream_url) {
      action.appendChild(icon('play'));
      action.appendChild(el('span', 'act-text', 'jump to ' + clock(c.seek_s)));
    } else if (c.source_url) {
      action.appendChild(icon('ext'));
      action.appendChild(el('span', 'act-text', 'open source'));
    } else {
      action.appendChild(el('span', 'act-text', 'no playable copy'));
    }
    head.appendChild(action);
    node.appendChild(head);

    if (c.passage) {
      node.appendChild(el('p', 'passage', '“' + c.passage + '”'));
    }
    return node;
  }

  /**
   * Seek a player, waiting for its metadata first if it has not arrived.
   *
   * Assigning `currentTime` before `readyState >= HAVE_METADATA` is silently a no-op: the
   * element does not yet know its duration, so it has nothing to seek within. The failure
   * is invisible — the video sits at 0:00 and plays, so it looks like a working player and
   * the citation's timestamp is the thing that quietly did not happen. It is a race, which
   * is why it showed up on one screenshot and not the next: it only bites when a citation is
   * activated before the first bytes of `/media/{id}` come back, which is exactly what the
   * auto-activated first citation does on a cold load.
   */
  function seek(video, seconds) {
    function go() {
      video.currentTime = seconds;
      video.play().catch(function () { /* autoplay policy; the seek still landed */ });
    }
    if (video.readyState >= 1) { go(); }
    else { video.addEventListener('loadedmetadata', go, { once: true }); }
  }

  function playerFor(c, index) {
    var figure = el('figure', 'player');
    var video = document.createElement('video');
    video.id = playerId(index, c.video_id);
    video.src = c.stream_url;
    video.controls = true;
    video.preload = 'metadata';
    figure.appendChild(video);
    figure.appendChild(el('figcaption', null, 'video ' + c.video_id + ' — click a citation to jump'));
    return figure;
  }

  /** The provenance line: what produced this answer, and what it cost. */
  function meta(data) {
    var row = el('div', 'meta');
    var p = data.provenance || {};
    var s = data.spend || {};

    function chip(key, value, cls) {
      var node = el('span', 'chip' + (cls ? ' ' + cls : ''));
      node.appendChild(el('b', null, key + ' '));
      node.appendChild(document.createTextNode(String(value)));
      row.appendChild(node);
      return node;
    }

    if (p.arm) { chip('arm', p.arm + ' · ' + p.answer_model); }
    if (p.top_k !== undefined) { chip('retrieved', p.retrieved + '/' + p.top_k); }
    if (s.latency_s !== undefined) { chip('latency', s.latency_s + 's'); }
    if (s.cost_usd !== undefined) { chip('cost', '$' + Number(s.cost_usd).toFixed(6)); }
    if (p.config_sha256) {
      chip('config', String(p.config_sha256).slice(0, 12), 'mono').title =
        p.config + '  ' + p.config_sha256 + '\n' + p.prompt + '  ' + p.prompt_sha256;
    }
    // What grounding had to fix. Silence here would make a repaired answer indistinguishable
    // from a clean one, and the repairs are the thing worth seeing.
    (data.repairs || []).forEach(function (r) { chip('repair', r, 'repair'); });
    return row;
  }

  function renderProblem(turn, result) {
    var body = result.body || {};
    var bubble = el('div', 'bubble-a error');
    reveal(bubble, body.error || ('the request failed (' + result.status + ')'));

    var hint = body.hint;
    if (result.status === 429 && result.headers) {
      var wait = result.headers.get('Retry-After');
      if (wait) { hint = 'retry in ' + wait + 's. ' + (hint || ''); }
    }
    if (hint) {
      var why = el('p', 'why');
      why.appendChild(el('code', null, hint));
      bubble.appendChild(why);
    }
    turn.appendChild(bubble);
  }

  boot();
})();
