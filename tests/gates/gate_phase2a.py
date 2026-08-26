"""GATE VRAG-019 - answering with citations.

    make gate-phase2a
    uv run pytest tests/gates/gate_phase2a.py -v -s

The card's criterion, verbatim: **schema-valid on 100% of dev; abstains on a planted
unanswerable question.** Both halves are computed here over all 15 pairs in `evals/dev`, and
both are printed before they are asserted.

This is **not** the Phase 2 exit gate. That is VRAG-021, it scores QA_SPEC section 5
accuracy on `evals/heldout`, and it belongs to the Evaluator. Nothing here reads
`evals/heldout` and nothing here reports an accuracy number - the two things this gate
measures are that the contract holds and that the refusal works, which are the
preconditions for that gate meaning anything.

The three numbers
-----------------
**schema-valid, threshold 1.00.** Every reply parses as `schemas.answer.Answer`. It is
measured on the model's raw output, before `src.answer.ground` touches it, so a reply that
only validated after repair counts as a failure. Generation is already constrained by
`schemas.answer.json_schema()`, which makes 100% the right threshold rather than a hopeful
one: anything under it means the schema handed to the model and the schema it is checked
against have come apart, and that is a bug in one declaration, not a tuning problem.

**abstentions on the planted unanswerable pairs, threshold: all of them.** `d013`-`d015` ask
for a sales figure, a commission price and a person's name that the corpus never states.
All three are about videos that *are* indexed, so retrieval returns five confident on-topic
passages for each - the refusal has to come from reading them, not from an empty result. And
`openai/gpt-oss-120b` has read about Bernini and has seen *Suits*, so the corpus is not the
only place it could get an answer from. This is the number that says it did not.

**the abstention rate over answerable pairs, ceiling 0.25.** Without a third number the gate
is trivially passable: a module that abstains on everything scores 100% schema-valid and 3/3
abstentions and looks excellent. This is the guard against that, and it is worth being exact
about what it is not. It is not an accuracy measure - accuracy is QA_SPEC section 5, scored on
`evals/heldout` by VRAG-021. It measures *selectivity*: that the refusal fires on the
questions the corpus cannot answer and not on the ones it can. 1.00 on the unanswerable pairs
against <=0.25 on the answerable ones is a separation of at least 0.75; an abstain-everything
module scores 1.00 on both and fails here.

Why a rate, and not a per-pair assertion
----------------------------------------
It was a per-pair assertion first - zero false abstentions among the pairs whose ground-truth
moment was retrieved - and it passed, then failed the next run with nothing changed. The pair
was `d001`, and the reason is worth the space because it is two separate findings.

**The provider is not reproducible at temperature 0.0.** Six identical calls for `d001`, same
code, same config, same prompt: 4 answers, 2 abstentions, three different answer texts. The
six lines are recorded in config.toml next to the lever. A gate the supervisor re-runs and
compares needs margin against that, and a threshold of exactly 0 on a 12-pair denominator has
none.

**And `d001` is a pair no answerer can win.** The line asked about is "I'm gonna graduate" at
t_ref=30.0 s, and no retrieved passage contains it - the chunk that does is outside the top 5.
What satisfies the QA_SPEC section 2 hit rule instead is a chunk from video 181 starting at
0.0 s whose entire text is "." , plus a lyric chunk at 51.0 s; both are inside the +/-30 s
tolerance, so the rule is met by passages that do not hold the answer. `make gate-phase1`
duly scores d001 a HIT at dt=30.0 s, exactly on the boundary. Both of the answerer's
behaviours are therefore defensible - abstaining is honest, and answering would be a guess
that section 2 would score correct anyway - so a per-pair rule demanding an answer would be
demanding the guess. Flagged on the card: the empty-text chunk is a chunker finding
(VRAG-014/017) and is deliberately not fixed here, because fixing it moves a recorded Phase 1
number that belongs to another gate.

`d010` is the other pair worth knowing about, and the one Phase 1 scores a MISS (rank 5,
dt=98.8 s): its five passages genuinely do not contain "troubled situations". Both pairs are
printed with the reason rather than excused quietly.

Measured spread, three passes of the same code, config and prompt over all 15 dev pairs:

    pass          schema-valid   abstain(unanswerable)   abstain(answerable)
    1             15/15          3/3                     1/12
    2             15/15          3/3                     1/12
    3             15/15          3/3                     2/12

Per pair, `d001` is the only one that moves. `d010` abstains in all three - it is the
retrieval miss. Everything else answers in all three. So the two asserted numbers above are
stable and the third has a one-pair wobble, which is what the 0.25 ceiling (3 of 12) is sized
for: one pair of margin over the worst pass observed.

What it costs
-------------
15 hosted chat completions plus 15 local embeddings, a minute and a half, $0.00 on Groq's
free tier - two orders of magnitude slower than every other gate in this directory.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

# Project root so imports resolve without an editable install.
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from schemas.answer import Answer, json_schema
from src.answer import AnswerRun, answer, load_prompt
from src.config import load as load_config
from src.leakage import collisions, load_split
from src.retrieve import CITATION_TOLERANCE_S
from src.telemetry import Meter

# THE THRESHOLDS - the VRAG-019 criterion.
SCHEMA_VALID_THRESHOLD = 1.00

# The ceiling on the abstention rate over ANSWERABLE dev pairs. Deliberately not an accuracy
# threshold: it is the guard that stops a module which abstains on everything from passing the
# two checks above, and nothing more. 0.25 of 12 pairs is 3. The docstring above says why this
# is a rate and not a per-pair assertion, and records the measured spread.
ANSWERABLE_ABSTENTION_CEILING = 0.25

DEV_DIR = ROOT / "evals" / "dev"
HELDOUT_DIR = ROOT / "evals" / "heldout"
MANIFEST = ROOT / "data" / "corpus" / "manifest.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dev_pairs() -> list[dict]:
    pairs = []
    for path in sorted(DEV_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def dev_video_ids() -> list[str]:
    videos = json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"]
    return sorted(str(v["video_id"]) for v in videos if v["split"] == "dev")


def indexed_video_ids(cfg) -> tuple[set[str], int]:
    """Distinct video_ids in the Chroma collection, and the chunk count."""
    import chromadb

    chroma_path = Path(cfg.get("embed.chroma_path"))
    if not chroma_path.is_absolute():
        chroma_path = ROOT / chroma_path
    if not chroma_path.exists():
        return set(), 0

    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        collection = client.get_collection(cfg.get("embed.collection"))
    except Exception:
        return set(), 0
    if collection.count() == 0:
        return set(), 0

    got = collection.get(include=["metadatas"])
    ids = {str((m or {}).get("video_id", "")) for m in (got.get("metadatas") or [])}
    return {i for i in ids if i}, collection.count()


def evidence_was_retrieved(pair: dict, run: AnswerRun) -> bool:
    """Did the ground-truth moment reach the model at all?

    The QA_SPEC section 2 hit rule, applied to the passages this run was given rather than
    to what it replied. This is what separates "the answerer refused to answer" from "there
    was nothing to answer from".
    """
    target, t_ref = str(pair["video_id"]), float(pair["t_ref"])
    return any(
        h.video_id == target and abs(h.t_start - t_ref) <= CITATION_TOLERANCE_S
        for h in run.hits
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "config.toml")


@pytest.fixture(scope="module")
def pairs() -> list[dict]:
    return _dev_pairs()


@pytest.fixture(scope="module")
def runs(cfg, pairs) -> dict:
    """Answer every dev pair once. One model call per pair, reused by every test below."""
    if not pairs:
        pytest.skip("no dev pairs - test_dev_set_has_both_kinds_of_pair reports this")

    meter = Meter()
    rows = [(pair, answer(pair["question"], cfg, meter)) for pair in pairs]
    return {"rows": rows, "meter": meter}


# ---------------------------------------------------------------------------
# Preconditions - none of these are the number, all of them can void it
# ---------------------------------------------------------------------------


def test_no_leakage_before_scoring():
    """dev intersect heldout = the empty set. tests/gates/README: nothing counts until this holds."""
    found = collisions(load_split(DEV_DIR), load_split(HELDOUT_DIR))
    assert not found, (
        f"{len(found)} dev/held-out collision(s). Stop, do not tune. "
        f"First: {found[0] if found else ''}. See `make leakage-check`."
    )


def test_dev_set_has_both_kinds_of_pair(pairs):
    """A planted unanswerable question is half the criterion - there has to be one."""
    unanswerable = [p for p in pairs if p.get("unanswerable")]
    answerable = [p for p in pairs if not p.get("unanswerable")]
    print(
        f"\ndev pairs: {len(pairs)} total, {len(answerable)} answerable, "
        f"{len(unanswerable)} planted unanswerable, from {DEV_DIR.as_posix()}"
    )
    assert answerable, f"{DEV_DIR.as_posix()} holds no answerable pairs"
    assert unanswerable, (
        f"{DEV_DIR.as_posix()} holds no unanswerable pair, so 'abstains on a planted "
        f"unanswerable question' cannot be measured. QA_SPEC section 3 says how to write one."
    )


def test_index_covers_all_four_dev_videos(cfg):
    """Abstention has to be earned by reading passages, not handed over by an empty index.

    With nothing indexed, retrieval returns nothing, the model abstains on all 15 pairs, and
    this gate would report 3/3 abstentions on a pipeline that answers nothing at all.
    """
    expected = dev_video_ids()
    indexed, count = indexed_video_ids(cfg)
    print(f"\nindex: {count} chunk(s), video_ids {sorted(indexed) or '{}'}")
    assert count > 0, (
        f"the Chroma collection {cfg.get('embed.collection')!r} at "
        f"{cfg.get('embed.chroma_path')} is empty or absent. Run `make index-dev` first - "
        f"every abstention would otherwise be free."
    )
    missing = [v for v in expected if v not in indexed]
    assert not missing, f"dev video(s) {missing} are not indexed. Run `make index-dev`."


def test_the_prompt_is_a_file_on_disk_and_its_identity_is_printed(cfg):
    """Which prompt produced the number, by digest. README: prompts are versioned, never inlined."""
    path = Path(cfg.get("answer.prompt"))
    if not path.is_absolute():
        path = ROOT / path
    assert path.is_file(), (
        f"answer.prompt is {cfg.get('answer.prompt')!r} and there is no such file. The "
        f"prompt lives in prompts/, versioned; it is not inlined in src/answer.py."
    )
    system, user = load_prompt(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(
        f"\nprompt:  {cfg.get('answer.prompt')}  sha256:{digest[:16]}  "
        f"system {len(system)} chars, user template {len(user)} chars"
    )
    print(f"config:  {cfg.fingerprint()['path']}  sha256:{cfg.fingerprint()['sha256'][:16]}")
    print(f"arm:     {cfg.get('answer.arm')}  {cfg.get('answer.model')}  "
          f"temperature={cfg.get('answer.temperature')}")
    assert system.strip() and user.strip()


def test_the_schema_handed_to_the_model_is_the_schema_it_is_checked_against():
    """One declaration, two uses. If these come apart, 'schema-valid' measures nothing.

    Generation is constrained by `json_schema()` and the reply is validated by
    `Answer.model_validate`. Both come off the same Pydantic model, and this asserts the
    rendering did not lose the three fields the card names on the way out.
    """
    schema = json_schema()
    print(f"\nwire schema fields: {sorted(schema['properties'])}, "
          f"required {sorted(schema['required'])}")
    assert set(schema["properties"]) == {"answer", "citations", "abstain"}
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    assert "$ref" not in json.dumps(schema), "strict mode does not follow $ref"


# ---------------------------------------------------------------------------
# THE NUMBERS
# ---------------------------------------------------------------------------


def test_every_dev_reply_is_schema_valid(runs, pairs):
    """schema-valid on 100% of dev. Measured on the model's own output, before grounding."""
    rows = runs["rows"]
    invalid = [(p, r) for p, r in rows if not r.valid]

    for pair, run in invalid:
        print(f"\n  INVALID {pair.get('id')}  {run.error}")
        print(f"          q: {pair['question']}")
        print(f"          raw: {run.raw[:300]}")

    rate = (len(rows) - len(invalid)) / len(rows)
    print(
        f"\nschema-valid = {rate:.4f}  ({len(rows) - len(invalid)}/{len(rows)} dev pairs)  "
        f"threshold {SCHEMA_VALID_THRESHOLD:.2f}"
    )
    sys.stdout.flush()

    assert rate >= SCHEMA_VALID_THRESHOLD, (
        f"{len(invalid)} of {len(rows)} replies did not validate. Generation is already "
        f"constrained by schemas.answer.json_schema(), so this is not a prompt problem: the "
        f"schema handed to the model and schemas.answer.Answer have come apart, or "
        f"answer.max_tokens is clipping the JSON. First: {invalid[0][1].error}"
    )


def test_it_abstains_on_every_planted_unanswerable_question(runs):
    """The refusal. QA_SPEC section 4: any citation here is incorrect regardless of content."""
    unanswerable = [(p, r) for p, r in runs["rows"] if p.get("unanswerable")]

    print("\nplanted unanswerable pairs:")
    for pair, run in unanswerable:
        got = "ABSTAIN" if run.abstained else "ANSWERED"
        videos = sorted({h.video_id for h in run.hits})
        print(f"  {got:8} {pair.get('id')}  {len(run.hits)} passage(s) from {videos}")
        print(f"           q: {pair['question']}")
        if not run.abstained and run.answer is not None:
            print(f"           a: {run.answer.answer}")
            for cite in run.answer.citations:
                print(f"           HALLUCINATED CITATION: {cite}")

    abstained = sum(1 for _, r in unanswerable if r.abstained)
    print(f"\nabstentions = {abstained}/{len(unanswerable)} planted unanswerable pairs  "
          f"threshold {len(unanswerable)}/{len(unanswerable)}")
    sys.stdout.flush()

    assert abstained == len(unanswerable), (
        f"{len(unanswerable) - abstained} unanswerable pair(s) got an answer with a "
        f"citation. Every one of these retrieves five on-topic passages from an indexed "
        f"video, so the answer came from the model's own knowledge of the subject, not from "
        f"the corpus. Tune prompts/answer_v1.md on evals/dev - three attempts, then "
        f"escalate (CLAUDE.md). Do not weaken this threshold."
    )


def test_the_abstention_is_selective(runs):
    """The refusal has to fire on the unanswerable pairs and not on the answerable ones.

    A rate with a ceiling rather than a per-pair assertion - the module docstring says why,
    and the short version is that the provider is not reproducible at temperature 0.0 and one
    dev pair has no answer in its retrieved passages either way.

    Every abstention on an answerable pair is still printed, with whether the ground-truth
    moment was among the passages, because that is the line that says whether the refusal
    fired wrongly or retrieval never gave it a chance.
    """
    rows = runs["rows"]
    answerable = [(p, r) for p, r in rows if not p.get("unanswerable")]
    unanswerable = [(p, r) for p, r in rows if p.get("unanswerable")]

    abstained = [(p, r) for p, r in answerable if r.abstained]
    for pair, run in abstained:
        had = evidence_was_retrieved(pair, run)
        print(
            f"\n  ABSTAINED on answerable {pair.get('id')}  video {pair['video_id']} "
            f"t_ref={float(pair['t_ref']):.1f}s  "
            f"ground-truth moment retrieved: {'yes' if had else 'NO'}"
        )
        print(f"           q: {pair['question']}")
        print(f"           note: {pair.get('answer_note', '')}")
        if not had:
            print("           a retrieval miss, not a refusal - see `make gate-phase1`")

    answerable_rate = len(abstained) / len(answerable)
    unanswerable_rate = (
        sum(1 for _, r in unanswerable if r.abstained) / len(unanswerable)
        if unanswerable
        else 0.0
    )

    print(
        f"\nabstention rate: {unanswerable_rate:.4f} on {len(unanswerable)} unanswerable "
        f"pairs, {answerable_rate:.4f} on {len(answerable)} answerable "
        f"({len(abstained)}/{len(answerable)})  ceiling {ANSWERABLE_ABSTENTION_CEILING:.2f}"
    )
    print(
        f"selectivity = {unanswerable_rate - answerable_rate:.4f}  "
        f"(an abstain-everything module scores 0.0000)"
    )
    sys.stdout.flush()

    assert answerable_rate <= ANSWERABLE_ABSTENTION_CEILING, (
        f"{len(abstained)} of {len(answerable)} answerable pairs were refused "
        f"({answerable_rate:.4f} > {ANSWERABLE_ABSTENTION_CEILING:.2f}). The abstention rule "
        f"is firing too widely and it costs accuracy directly at VRAG-021: rule 3 in "
        f"prompts/answer_v1.md is the one to loosen, not this ceiling. Read the "
        f"'ground-truth moment retrieved: NO' lines first - those are retrieval misses and "
        f"they belong to `make gate-phase1`."
    )
    assert unanswerable_rate > answerable_rate, (
        f"the refusal is not selective: {unanswerable_rate:.4f} on unanswerable pairs vs "
        f"{answerable_rate:.4f} on answerable. A module that abstains uniformly satisfies "
        f"every other check in this file, and this is the one that rejects it."
    )


def test_no_citation_points_outside_the_retrieved_passages(runs):
    """The safety property `src.answer.ground` exists for, asserted rather than assumed.

    A citation the user can click has to name a passage that was actually retrieved. This
    holds by construction after grounding, which is exactly why it is worth an assertion -
    a future edit that reorders validate/ground would break it silently.
    """
    offenders = []
    repaired = 0
    for pair, run in runs["rows"]:
        if run.repairs:
            repaired += 1
        if run.answer is None:
            continue
        allowed = {(h.video_id, round(h.t_start, 3)) for h in run.hits}
        for cite in run.answer.citations:
            if (cite.video_id, round(cite.t_start, 3)) not in allowed:
                offenders.append((pair.get("id"), str(cite), sorted({h.video_id for h in run.hits})))

    total = len(runs["rows"])
    cited = sum(
        len(r.answer.citations) for _, r in runs["rows"] if r.answer is not None
    )
    print(f"\ncitations: {cited} across {total} pairs, all grounded in a retrieved passage; "
          f"ground() repaired {repaired}/{total}")
    for row in offenders:
        print(f"  UNGROUNDED {row[0]}  {row[1]}  retrieved videos {row[2]}")
    sys.stdout.flush()

    assert not offenders, (
        f"{len(offenders)} citation(s) point at a passage that was never retrieved, so "
        f"src.answer.ground did not run or ran before validation. First: {offenders[0]}"
    )


def test_cost_of_the_gate_is_printed(runs, cfg):
    """What the measurement cost. Free tier is $0.00 - printed, not assumed."""
    meter: Meter = runs["meter"]
    calls = meter._calls
    tokens = sum(r.tokens for _, r in runs["rows"])
    generation = [c for c in calls if c.model == cfg.get("answer.model")]
    print(
        f"\ngate cost: {len(runs['rows'])} question(s), {len(generation)} generation call(s) "
        f"({cfg.get('answer.model')}) + {len(calls) - len(generation)} embed call(s) "
        f"({cfg.get('embed.model')}), {tokens} tokens, "
        f"{sum(c.latency_s for c in calls):.2f}s, ${sum(c.cost_usd for c in calls):.4f}"
    )
    assert generation, (
        "no call was logged against answer.model, so nothing went through the shared "
        "cost/latency logger (CLAUDE.md: every model call does)"
    )
    assert tokens > 0, "no token volume recorded - the meter cannot show a tier change"
