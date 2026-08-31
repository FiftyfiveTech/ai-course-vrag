"""The primer's numbers, checked against the sweep that produced them — VRAG-018.

    uv run pytest tests/unit/test_primer_numbers.py -v

`docs/learning/primer-chunking-embeddings.md` is prose, and prose about measurements rots in
a way code does not: `make sweep` is re-run after an ASR arm or an embedding model moves, the
JSON changes, and every figure quoted in the primer silently becomes last month's. Nothing
catches that — a stale sentence still reads as a true one.

So this parses the tables back out of the markdown and asserts each cell against
`docs/learning/data/chunking_sweep.json`. It fails naming the row, the column, the value in
the document and the value in the data, which is the message that says which sentence to
rewrite.

The value of this is not hypothetical: the first draft of the primer had five hand-rounded
word counts in it (17,700 against a real 17,249, 46,900 against 47,397). That is exactly the
drift this file exists to make impossible, and it was found by writing the check.

Scope. This checks the three measured tables and the handful of figures quoted in the running
text. It does not check the argument — that a window is a constraint rather than a tuning
lever is a claim about what the numbers mean, and no test can hold that true.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
PRIMER = ROOT / "docs" / "learning" / "primer-chunking-embeddings.md"
SWEEP = ROOT / "docs" / "learning" / "data" / "chunking_sweep.json"


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def primer() -> str:
    if not PRIMER.is_file():
        pytest.fail(f"{PRIMER.relative_to(ROOT)} is missing — VRAG-018 committed it")
    return PRIMER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def sweep() -> dict:
    if not SWEEP.is_file():
        pytest.fail(
            f"{SWEEP.relative_to(ROOT)} is missing — run `make sweep` (~22 min, $0.00)"
        )
    data = json.loads(SWEEP.read_text(encoding="utf-8"))
    if data.get("dry_run"):
        pytest.fail("the committed sweep is a --dry-run: it has no recall to check against")
    return data


@pytest.fixture(scope="module")
def points(sweep) -> dict[tuple[float, float], dict]:
    """Grid points keyed by (window_s, overlap_s)."""
    return {(p["levers"]["window_s"], p["levers"]["overlap_s"]): p for p in sweep["points"]}


# --------------------------------------------------------------------------- parsing


def cells(row: str) -> list[str]:
    """`| a | b |` → ['a', 'b'], with bold markers and unit suffixes stripped."""
    parts = [c.strip() for c in row.strip().strip("|").split("|")]
    out = []
    for c in parts:
        c = c.replace("**", "").strip()
        c = re.sub(r"\s*(s|MB|×)$", "", c)          # "25 s", "3.0 MB", "1.79×"
        out.append(c.replace(",", "").strip())      # "17,249" → "17249"
    return out


def table_rows(md: str, header: str) -> list[list[str]]:
    """Every data row of the markdown table whose header line starts with `header`.

    Rows are returned in document order, without the header or the `|---|` separator.
    """
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(header):
            break
    else:
        pytest.fail(f"no table in the primer with a header starting {header!r}")

    rows = []
    for line in lines[i + 2 :]:                      # skip header and separator
        if not line.startswith("|"):
            break
        rows.append(cells(line))
    if not rows:
        pytest.fail(f"the table headed {header!r} has no rows")
    return rows


def num(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        pytest.fail(f"expected a number in the primer, got {text!r}")


# --------------------------------------------------------------------------- the tables

WINDOW_HEADER = "| window | hop | chunks | duplication | longest chunk |"
OVERLAP_HEADER = "| overlap | hop | chunks | duplication | words indexed |"
DUP_HEADER = "| | window ÷ hop | measured | excess |"


def check_row(row: list[str], point: dict, columns: list[tuple[int, str, float]]) -> None:
    """Assert the named columns of one markdown row against one sweep point."""
    lv = point["levers"]
    where = f"window {lv['window_s']:g} / overlap {lv['overlap_s']:g}"
    for index, name, expected in columns:
        got = num(row[index])
        assert got == pytest.approx(expected, abs=0.051), (
            f"{where}: the primer says {name} = {row[index]}, "
            f"the sweep says {expected}. Re-run `make sweep` and update the primer."
        )


def test_window_table_matches_the_sweep(primer, points, sweep):
    """§3 — the window cut, every cell."""
    rows = table_rows(primer, WINDOW_HEADER)
    overlap = sweep["shipped"]["overlap_s"]

    swept = sorted(w for (w, o) in points if o == overlap)
    assert [num(r[0]) for r in rows] == swept, (
        "the primer's window table does not list the windows that were swept"
    )

    for row in rows:
        p = points[(num(row[0]), overlap)]
        check_row(
            row,
            p,
            [
                (1, "hop", p["levers"]["hop_s"]),
                (2, "chunks", p["counts"]["chunks"]),
                (3, "duplication", round(p["counts"]["duplication"], 2)),
                (4, "longest chunk", round(p["shape"]["max_chunk_s"], 1)),
                (5, ">tol", p["shape"]["over_tolerance"]),
                (6, "recall@1", round(p["score"]["recall_at_1"], 2)),
                (7, "recall@5", round(p["score"]["recall_at_5"], 4)),
                (8, "embed", round(p["index"]["embed_wall_s"], 1)),
                (9, "store", round(p["index"]["store_bytes"] / 1e6, 1)),
            ],
        )


def test_overlap_table_matches_the_sweep(primer, points, sweep):
    """§5 — the overlap cut, every cell."""
    rows = table_rows(primer, OVERLAP_HEADER)
    window = sweep["shipped"]["window_s"]

    swept = sorted(o for (w, o) in points if w == window)
    assert [num(r[0]) for r in rows] == swept, (
        "the primer's overlap table does not list the overlaps that were swept"
    )

    for row in rows:
        p = points[(window, num(row[0]))]
        check_row(
            row,
            p,
            [
                (1, "hop", p["levers"]["hop_s"]),
                (2, "chunks", p["counts"]["chunks"]),
                (3, "duplication", round(p["counts"]["duplication"], 2)),
                (4, "words indexed", p["counts"]["words_indexed"]),
                (5, "recall@1", round(p["score"]["recall_at_1"], 2)),
                (6, "recall@5", round(p["score"]["recall_at_5"], 4)),
                (7, "embed", round(p["index"]["embed_wall_s"], 1)),
                (8, "store", round(p["index"]["store_bytes"] / 1e6, 1)),
            ],
        )


def test_duplication_table_is_arithmetic_on_the_sweep(primer, points, sweep):
    """§5 — `window ÷ hop` against measured duplication, and the excess between them.

    This one is a derivation, not a copy, so the test re-derives it rather than looking it up:
    a transcription error and an arithmetic error should both fail here.
    """
    rows = table_rows(primer, DUP_HEADER)
    window = sweep["shipped"]["window_s"]

    for row in rows:
        overlap = num(row[0].replace("overlap", "").strip())
        p = points[(window, overlap)]
        predicted = p["levers"]["window_s"] / p["levers"]["hop_s"]
        measured = p["counts"]["duplication"]

        assert num(row[1]) == pytest.approx(predicted, abs=0.005), (
            f"overlap {overlap:g}: primer says window÷hop = {row[1]}, "
            f"arithmetic says {predicted:.2f}"
        )
        assert num(row[2]) == pytest.approx(measured, abs=0.005), (
            f"overlap {overlap:g}: primer says measured = {row[2]}, "
            f"the sweep says {measured:.2f}"
        )
        assert num(row[3]) == pytest.approx(measured / predicted, abs=0.005), (
            f"overlap {overlap:g}: primer says excess = {row[3]}, "
            f"measured ÷ predicted = {measured / predicted:.2f}"
        )


# ------------------------------------------------------------------- figures in the prose


def test_headline_recall_is_the_one_the_sweep_measured(primer, points, sweep):
    """The tie the primer opens on: six windows, one recall, and 60 s alone below it."""
    overlap = sweep["shipped"]["overlap_s"]
    cut = {w: p for (w, o), p in points.items() if o == overlap}
    top = max(p["score"]["recall_at_5"] for p in cut.values())
    tied = sorted(w for w, p in cut.items() if p["score"]["recall_at_5"] == top)

    quoted = re.search(
        r"Six of the seven window widths\s*\n?>?\s*measured — ([^—]+) — return the \*\*identical\*\* "
        r"recall@5 of (\d+\.\d+)",
        primer,
    )
    assert quoted, "the primer's headline sentence has been reworded — re-check it by hand"

    listed = [float(x) for x in re.findall(r"[\d.]+", quoted.group(1))]
    assert listed == tied, (
        f"the primer lists {listed} as tied at the top; the sweep says {tied}"
    )
    assert float(quoted.group(2)) == pytest.approx(top, abs=1e-4), (
        f"the primer quotes {quoted.group(2)}; the sweep says {top}"
    )


def test_shipped_point_figures_in_the_prose(primer, points, sweep):
    """The shipped setting is quoted in four places; all four are the same number."""
    p = points[(sweep["shipped"]["window_s"], sweep["shipped"]["overlap_s"])]

    assert f"the sweep produces {p['counts']['chunks']} chunks" in primer
    assert f"recall@5 = {p['score']['recall_at_5']:.4f} — the number `make gate-phase1`" in primer
    assert f"{p['shape']['over_tolerance']} of {p['counts']['chunks']} chunks" in primer
    assert f"longest is {p['shape']['max_chunk_s']:.1f} s" in primer


def test_the_index_composition_finding(primer, points, sweep):
    """§6 contrasts the gate's number with the sweep's. The sweep's half is checkable here.

    The gate's half (0.8333 on an index holding two non-corpus videos) is a snapshot of one
    working copy: `./chroma` is a gitignored build artefact, so asserting it would fail on a
    clean clone. The primer says so where it makes the claim.
    """
    p = points[(sweep["shipped"]["window_s"], sweep["shipped"]["overlap_s"])]
    s = p["score"]
    quoted = re.search(
        r"tools/sweep_chunking\.py\s+\S+\s+recall@5 = (\d+\.\d+)\s+\((\d+)/(\d+)\)", primer
    )
    assert quoted, "§6's side-by-side block no longer quotes the sweep's own number"
    assert float(quoted.group(1)) == pytest.approx(s["recall_at_5"], abs=1e-4)
    assert (int(quoted.group(2)), int(quoted.group(3))) == (s["hits_at_5"], s["pairs"])
    assert f"{p['counts']['chunks']} chunks" in primer


def test_cost_and_scale_figures(primer, sweep):
    """The provenance block and the 'where the dollars are' paragraph."""
    total_chunks = sum(p["counts"]["chunks"] for p in sweep["points"])
    assert f"{sweep['wall_s']:g}s · $0.0000" in primer, "the cost line does not match the run"
    assert f"{total_chunks:,} chunks embedded" in primer, (
        f"the primer should say {total_chunks:,} chunks embedded across the grid"
    )
    assert sweep["total_cost_usd"] == 0.0, (
        "the sweep cost money — the primer's zero-spend claim is no longer true"
    )
    assert f"{sweep['answerable_pairs']} answerable pairs" in primer


def test_the_pair_that_never_hits(primer, sweep):
    """`d010` misses at all twelve settings — the primer's claim, checked against the rows."""
    always_missing = None
    for p in sweep["points"]:
        missed = {r["id"] for r in p["score"]["rows"] if r["hit_rank"] is None}
        always_missing = missed if always_missing is None else always_missing & missed

    assert always_missing == {"d010"}, (
        f"the primer says d010 alone misses everywhere; the sweep says {always_missing}"
    )
    assert "`d010` misses at all twelve settings" in primer, (
        "§6's claim about d010 has been reworded — re-check it against the rows"
    )


def test_the_boundary_miss_overlap_buys_off(primer, points, sweep):
    """§5's `d003` at overlap 0: the miss is real and the primer quotes its true distance."""
    zero = points[(sweep["shipped"]["window_s"], 0.0)]
    misses = {r["id"]: r["nearest_dt_s"] for r in zero["score"]["rows"] if r["hit_rank"] is None}

    assert "d003" in misses, (
        "overlap 0 no longer misses d003 — §5's worked example has to be rewritten"
    )
    dt = misses["d003"]
    assert f"**{dt:.1f} s**" in primer, (
        f"the primer quotes a distance for d003 that is not {dt:.1f} s"
    )
    tolerance = sweep["citation_tolerance_s"]
    assert dt > tolerance, "d003 is inside the tolerance, so it is not a boundary miss"
    # The primer spells the gap out in words, so the check spells the number out too rather
    # than accepting any sentence that happens to contain "over the line".
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
    gap = round(dt - tolerance)
    assert f"Over the line by {words.get(gap, gap)} seconds" in primer, (
        f"d003 is {dt - tolerance:.1f} s past the tolerance; the primer says something else"
    )


def test_per_video_longest_segments_table(primer):
    """§4's segment table is measured off the cached transcripts, so re-measure it."""
    import sys

    sys.path.insert(0, str(ROOT))
    from src.chunk import load_transcript

    manifest = json.loads((ROOT / "data" / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    dev = sorted(str(v["video_id"]) for v in manifest["videos"] if v["split"] == "dev")

    rows = table_rows(primer, "| video | segments | longest segment |")
    quoted = {r[0]: (int(r[1]), float(r[2])) for r in rows}
    assert sorted(quoted) == dev, f"the primer's table lists {sorted(quoted)}, dev is {dev}"

    for video_id, (n_quoted, longest_quoted) in quoted.items():
        runs = sorted(
            d for d in (ROOT / "runs").glob(f"{video_id}_*") if (d / "transcript.json").is_file()
        )
        if not runs:
            pytest.skip(f"no cached transcript for {video_id} — run `make index-dev`")
        segments, _ = load_transcript(runs[0] / "transcript.json")
        longest = max(s.t_end - s.t_start for s in segments)

        assert len(segments) == n_quoted, (
            f"video {video_id}: primer says {n_quoted} segments, the transcript has {len(segments)}"
        )
        assert longest == pytest.approx(longest_quoted, abs=0.005), (
            f"video {video_id}: primer says the longest segment is {longest_quoted} s, "
            f"the transcript says {longest:.2f} s"
        )


def test_the_coach_page_was_built_from_this_sweep(primer):
    """The primer sends the reader to coach.html, so it has to exist and be current."""
    coach = ROOT / "docs" / "learning" / "coach.html"
    assert coach.is_file(), "docs/learning/coach.html is missing — run `make coach`"
    assert "coach.html" in primer

    html = coach.read_text(encoding="utf-8")
    inlined = json.loads(
        html.split('id="sweep-data">')[1].split("</script>")[0].replace("<\\/", "</")
    )
    live = json.loads(SWEEP.read_text(encoding="utf-8"))
    assert inlined["points"] == live["points"], (
        "coach.html was built from a different sweep than the one committed — run `make coach`"
    )
