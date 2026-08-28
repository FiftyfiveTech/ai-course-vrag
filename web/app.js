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
 *   3. An @ tag is a filter, not a hint. The menu offers only sources GET /videos says
 *      are indexed, because a handle this page suggests and /ask then refuses with a 422
 *      is a control that looks live and is not — the same failure as rule 4 below, moved
 *      one step earlier. What the answer was scoped to is then shown on the answer, not
 *      only in the question the user typed: a scoped answer cites one video and looks
 *      exactly like an unscoped answer that happened to.
 *
 *   4. A citation is rendered as what it can actually do. stream_url -> a button that seeks
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
  var mentionsEl = document.getElementById('mentions');
  var highlightEl = document.getElementById('qhl');

  var EXAMPLES = [
    'What two tools do I need to make my first paper cut?',
    'What is demonstrated at the start of the video?',
    'What does the speaker say about the ingredients?'
  ];

  var busy = false;

  /* Sources that can be tagged with @, from GET /videos. Indexed ones only: the menu is
     a promise that what it offers can be asked. */
  var SOURCES = [];
  var menu = { open: false, items: [], active: 0 };

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

  // ------------------------------------------------------- the @ source picker

  /**
   * The tag being typed at the caret, or null.
   *
   * Anchored to the caret and not to the whole string, so editing the front of a question
   * that already carries a tag does not reopen the menu on it. The leading class mirrors
   * src/mention.py's MENTION lookbehind — a tag starts the value or follows whitespace or an
   * opening bracket — which is what keeps an email address out of the picker and, more to
   * the point, keeps this page and the server agreeing on what a tag is.
   */
  var AT_CARET = /(?:^|[\s(\[{])@([A-Za-z0-9][A-Za-z0-9_-]*|)$/;

  /* The same rule over the whole value rather than at the caret — what the highlighter
     paints. Kept beside AT_CARET so the two cannot drift: a tag the picker completes and
     the highlighter does not draw would be the two halves disagreeing on screen. Both
     mirror src/mention.py's MENTION, including that a handle starts with a letter or a
     digit — which is why `@-dfvdKf-KR0` is not a tag anywhere. */
  var MENTION_ALL = /(^|[\s(\[{])@([A-Za-z0-9][A-Za-z0-9_-]*)/g;

  function tokenAtCaret() {
    var pos = input.selectionStart;
    if (pos === null || pos === undefined) { return null; }
    var found = AT_CARET.exec(input.value.slice(0, pos));
    if (!found) { return null; }
    return { query: found[1], at: pos - found[1].length - 1, caret: pos };
  }

  /**
   * Sources matching what has been typed, best first. -1 means no match.
   *
   * Ranked and not merely filtered, because the top row is what Enter takes and a plain
   * filter puts whatever the catalogue happened to list first there. Measured: with a flat
   * `indexOf` over handle, label and aliases, typing `@5` offered video 181 above 521 — 181
   * matched on the `5` buried in its youtube id `8np5YKYx3sU`, and Enter then scoped the
   * question to a video the user had not begun to type.
   *
   * So an alias has to match from its *start* rather than anywhere inside it. An alias is a
   * name you type from the front (a youtube id, a filename); a substring hit in the middle
   * of one is a coincidence, and the two ranks above it exist so that a coincidence can
   * never outrank the handle someone is actually spelling out.
   */
  function rank(source, q) {
    var handle = source.handle.toLowerCase();
    if (handle.indexOf(q) === 0) { return 0; }
    if (handle.indexOf(q) > 0) { return 1; }
    var aliases = source.aliases || [];
    for (var i = 0; i < aliases.length; i++) {
      if (aliases[i].toLowerCase().indexOf(q) === 0) { return 2; }
    }
    if ((source.label || '').toLowerCase().indexOf(q) >= 0) { return 3; }
    return -1;
  }

  function matches(query) {
    var q = query.toLowerCase();
    if (!q) { return SOURCES.slice(0, 8); }
    return SOURCES
      .map(function (s) { return { s: s, r: rank(s, q) }; })
      .filter(function (m) { return m.r >= 0; })
      // Stable on rank: within one rank the catalogue's own order stands, which is video_id
      // ascending, so the menu does not reshuffle between keystrokes.
      .sort(function (a, b) { return a.r - b.r; })
      .map(function (m) { return m.s; })
      .slice(0, 8);
  }

  // --------------------------------------------------- painting the tags

  /**
   * Is this handle one `/ask` will accept? SOURCES holds the indexed sources and nothing
   * else, so membership here is exactly the set src.mention.resolve() will not refuse.
   * Trailing `-`/`_` is trimmed first because the server trims it too.
   */
  function known(token) {
    var t = token.replace(/[-_]+$/, '').toLowerCase();
    if (!t) { return false; }
    return SOURCES.some(function (s) {
      if (s.handle.toLowerCase() === t) { return true; }
      return (s.aliases || []).some(function (a) { return a.toLowerCase() === t; });
    });
  }

  /**
   * Redraw the mirror under the input: the same characters, with each @tag in a pill.
   *
   * Built node by node, never innerHTML — this is the question someone typed and rule 1 at
   * the top of this file applies to it more than to anything else on the page.
   *
   * The invariant worth stating: the mirror's text is character-for-character the input's
   * value. It is what the reader actually sees, so a dropped or duplicated character here is
   * not a cosmetic bug, it is the page lying about what is in the box. `matchIndex` walking
   * with `lastIndex` is what keeps the untagged runs whole.
   */
  function renderHighlight() {
    if (!highlightEl) { return; }
    var value = input.value;
    highlightEl.textContent = '';

    var at = 0;
    var found;
    MENTION_ALL.lastIndex = 0;
    while ((found = MENTION_ALL.exec(value)) !== null) {
      // found[1] is the character before the @ (or ''), and belongs to the plain run.
      var start = found.index + found[1].length;
      if (start > at) {
        highlightEl.appendChild(document.createTextNode(value.slice(at, start)));
      }
      var text = value.slice(start, MENTION_ALL.lastIndex);
      // Nothing loaded yet is not the same as "no such source", so it gets its own state
      // instead of being drawn as a mistake the user has not made.
      var state = !SOURCES.length ? ' pending' : (known(found[2]) ? '' : ' unknown');
      highlightEl.appendChild(el('span', 'tag' + state, text));
      at = MENTION_ALL.lastIndex;
    }
    if (at < value.length) {
      highlightEl.appendChild(document.createTextNode(value.slice(at)));
    }

    // The mirror does not scroll itself: it is one line of `pre` in an overflow-hidden box,
    // so it has to be dragged to wherever the input has scrolled to or the two part company
    // as soon as the question is longer than the pill.
    highlightEl.scrollLeft = input.scrollLeft;
  }

  function syncScroll() {
    if (highlightEl) { highlightEl.scrollLeft = input.scrollLeft; }
  }

  function closeMenu() {
    menu.open = false;
    menu.items = [];
    menu.active = 0;
    if (!mentionsEl) { return; }
    mentionsEl.hidden = true;
    mentionsEl.textContent = '';
    input.setAttribute('aria-expanded', 'false');
  }

  function refreshMenu() {
    // No #mentions in the DOM means this page was served before the picker existed — a tab
    // left open across a server restart, which is exactly how this was first reported as
    // "typing shows no list". Without the guard every keystroke throws a TypeError out of
    // the input handler, which is a worse version of the same symptom and harder to read.
    if (!mentionsEl) { return; }
    var token = tokenAtCaret();
    if (!token || !SOURCES.length) { closeMenu(); return; }
    menu.items = matches(token.query);
    menu.active = 0;
    menu.open = true;
    input.setAttribute('aria-expanded', 'true');
    drawMenu(token);
  }

  function drawMenu(token) {
    mentionsEl.textContent = '';
    mentionsEl.hidden = false;

    if (!menu.items.length) {
      // Said, not silently closed. A menu that vanishes as you type reads as "this feature is
      // broken"; this reads as "that is not one of the sources", which is the true thing.
      mentionsEl.appendChild(el('div', 'mention-empty',
        'no indexed source matches @' + token.query));
      return;
    }

    menu.items.forEach(function (source, i) {
      var row = el('button', 'mention' + (i === menu.active ? ' active' : ''));
      row.type = 'button';
      row.setAttribute('role', 'option');
      row.setAttribute('aria-selected', i === menu.active ? 'true' : 'false');
      row.appendChild(el('span', 'at', '@'));
      row.appendChild(el('span', 'mhandle', source.handle));
      row.appendChild(el('span', 'mlabel', source.label || 'indexed on this host'));
      if (source.split) { row.appendChild(el('span', 'msplit', source.split)); }
      // mousedown, not click, for the preventDefault: the click that picks a source must not
      // blur the input first, or the caret position this insert is measured against is gone.
      row.addEventListener('mousedown', function (event) { event.preventDefault(); });
      row.addEventListener('click', function () { choose(source); });
      mentionsEl.appendChild(row);
    });
  }

  function move(delta) {
    if (!menu.items.length) { return; }
    menu.active = (menu.active + delta + menu.items.length) % menu.items.length;
    var rows = mentionsEl.querySelectorAll('.mention');
    for (var i = 0; i < rows.length; i++) {
      rows[i].classList.toggle('active', i === menu.active);
      rows[i].setAttribute('aria-selected', i === menu.active ? 'true' : 'false');
    }
    if (rows[menu.active]) { rows[menu.active].scrollIntoView({ block: 'nearest' }); }
  }

  function choose(source) {
    var token = tokenAtCaret();
    if (!token) { closeMenu(); return; }
    var insert = '@' + source.handle + ' ';
    var before = input.value.slice(0, token.at);
    var after = input.value.slice(token.caret);
    input.value = before + insert + after;
    var caret = before.length + insert.length;
    input.setSelectionRange(caret, caret);
    closeMenu();
    // Assigning .value fires no input event, so the mirror would keep painting the
    // half-typed handle the menu was opened on.
    renderHighlight();
    input.focus();
    syncScroll();
  }

  input.addEventListener('input', function () { renderHighlight(); refreshMenu(); });
  input.addEventListener('click', function () { syncScroll(); refreshMenu(); });
  input.addEventListener('scroll', syncScroll);
  // keyup and not keydown: the caret has moved by then, so the mirror follows an arrow
  // key that scrolled the input without changing a character.
  input.addEventListener('keyup', syncScroll);
  input.addEventListener('blur', closeMenu);

  input.addEventListener('keydown', function (event) {
    if (!menu.open) { return; }
    if (event.key === 'Escape') { closeMenu(); event.preventDefault(); return; }
    if (event.key === 'ArrowDown') { move(1); event.preventDefault(); return; }
    if (event.key === 'ArrowUp') { move(-1); event.preventDefault(); return; }
    // Enter picks the highlighted source instead of submitting. That is the one keystroke
    // worth being careful about: with the menu open, Enter meaning "ask" would send a
    // half-typed handle — "@6" — which is a 422 the user did not type on purpose.
    if ((event.key === 'Enter' || event.key === 'Tab') && menu.items.length) {
      choose(menu.items[menu.active]);
      event.preventDefault();
    }
  });

  function loadSources() {
    return fetch('/videos')
      .then(function (r) { return r.json(); })
      .then(function (list) {
        // `v.handle` is required, not assumed. An API older than the picker answers /videos
        // without it, and `rank()` would then call .toLowerCase() on undefined and throw out
        // of the input handler on the first keystroke — a dead menu with the cause buried in
        // the console. Dropping the row instead degrades to "nothing to offer", which is
        // true of an API that cannot tell this page what to type.
        SOURCES = (list || []).filter(function (v) { return v && v.indexed && v.handle; });
        // A tag typed (or arriving in ?q=) before this resolved was drawn `pending`.
        renderHighlight();
      })
      .catch(function () { SOURCES = []; });
  }

  // ------------------------------------------------------------------ boot

  function boot() {
    // The class, and therefore the transparent input text, is added only once the mirror
    // is known to be on the page. A checkout served before the highlighter existed keeps
    // an ordinary visible input rather than an empty-looking one.
    if (highlightEl) {
      input.classList.add('mirrored');
      renderHighlight();
    }

    EXAMPLES.forEach(function (question) {
      var chip = el('button', null, question);
      chip.type = 'button';
      chip.addEventListener('click', function () {
        input.value = question;
        input.focus();
      });
      examplesEl.appendChild(chip);
    });

    loadSources();

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
    renderHighlight();
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

    var scope = (data.provenance && data.provenance.scope) || [];
    if (scope.length) {
      // On the answer and not only in the question bubble. A scoped answer cites one video
      // and is indistinguishable by eye from an unscoped answer that happened to cite one
      // — and the difference is whether the others were ever eligible.
      turn.appendChild(el('p', 'note scoped',
        'Answered from ' + scope.map(function (v) { return 'video ' + v; }).join(', ') +
        ' only — the rest of the index was not searched.'));
    }

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
   * Seek a player, waiting for its metadata first if it has not arrived, and start it.
   *
   * Assigning `currentTime` before `readyState >= HAVE_METADATA` is silently a no-op: the
   * element does not yet know its duration, so it has nothing to seek within. The failure
   * is invisible — the video sits at 0:00 and plays, so it looks like a working player and
   * the citation's timestamp is the thing that quietly did not happen. It is a race, which
   * is why it showed up on one screenshot and not the next: it only bites when a citation is
   * activated before the first bytes of `/media/{id}` come back, which is exactly what the
   * auto-activated first citation does on a cold load.
   *
   * So the seek waits and `play()` does not, and the split is the point. A gesture only
   * authorises playback for as long as the browser considers it live: Safari wants the
   * `play()` inside the handler the click ran, and a `play()` posted from `loadedmetadata`
   * a few hundred milliseconds later is a different task with no gesture behind it. Calling
   * it here keeps it in the click, and the deferred `currentTime` still lands — playback
   * has not begun by the time metadata arrives, so there is nothing to jump.
   */
  function seek(video, seconds) {
    function land() { video.currentTime = seconds; }
    if (video.readyState >= 1) { land(); }
    else { video.addEventListener('loadedmetadata', land, { once: true }); }
    // Rejects on a page the browser thinks nobody asked anything of — a `/?q=…` deep link
    // answers and auto-activates its first citation with no click anywhere in the document,
    // and no amount of code makes that autoplay. The seek still landed; the viewer presses
    // play. Anything louder here would be reporting a policy as a bug.
    var started = video.play();
    if (started && started.catch) { started.catch(function () {}); }
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
    if (p.scope && p.scope.length) { chip('scope', p.scope.join(' '), 'scope'); }
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
