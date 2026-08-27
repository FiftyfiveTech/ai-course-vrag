"""Tests for the corpus manifest — VRAG-004.

Offline by design: these read the committed manifest and exercise the selection
algorithm on synthetic rows. Nothing here touches the network, so `make test` stays
runnable without HF_TOKEN. The network path is covered by `make corpus-check`, which
re-streams the dataset and asserts the manifest reproduces byte-for-byte.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.corpus import (
    DEV_QUOTA,
    DURATION_ORDER,
    DURATION_QUOTA,
    Video,
    assign_splits,
    build_manifest,
    pick_bucket,
)

MANIFEST_PATH = Path("data/corpus/manifest.json")


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not MANIFEST_PATH.exists():
        pytest.fail(f"{MANIFEST_PATH} is missing — run `make corpus`")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def videos(manifest: dict) -> list[dict]:
    return manifest["videos"]


# --- the acceptance criterion: 10 ids, 4 dev / 6 held-out ---------------------------


def test_ten_videos(videos):
    assert len(videos) == 10


def test_split_is_four_dev_six_heldout(videos):
    counts = {s: sum(1 for v in videos if v["split"] == s) for s in ("dev", "heldout")}
    assert counts == {"dev": 4, "heldout": 6}


def test_every_video_is_assigned_to_a_split(videos):
    assert all(v["split"] in ("dev", "heldout") for v in videos)


def test_ids_are_unique(videos):
    for key in ("video_id", "youtube_id"):
        ids = [v[key] for v in videos]
        assert len(set(ids)) == len(ids), f"duplicate {key}"


def test_every_video_has_a_resolvable_pointer(videos):
    for v in videos:
        assert v["youtube_id"], v
        assert v["url"].startswith("https://www.youtube.com/watch?v="), v["url"]
        # The url must point at the id we recorded, or provenance is broken.
        assert v["url"].endswith(v["youtube_id"]), v


# --- "10 varied videos" ------------------------------------------------------------


def test_duration_quota_is_met(videos):
    got = {d: sum(1 for v in videos if v["duration"] == d) for d in DURATION_ORDER}
    assert got == DURATION_QUOTA


def test_all_six_domains_are_covered(videos):
    # Video-MME has 6 domains; with 10 slots there is no excuse for missing one.
    assert len({v["domain"] for v in videos}) == 6


def test_sub_categories_are_all_distinct(videos):
    subs = [v["sub_category"] for v in videos]
    assert len(set(subs)) == len(subs)


@pytest.mark.parametrize("split", ["dev", "heldout"])
def test_each_half_spans_every_duration(videos, split):
    """A held-out set with no long videos would not test what is most likely to break."""
    side = [v for v in videos if v["split"] == split]
    assert {v["duration"] for v in side} == set(DURATION_ORDER)


@pytest.mark.parametrize(("split", "least"), [("dev", 4), ("heldout", 6)])
def test_each_half_is_domain_diverse(videos, split, least):
    side = [v for v in videos if v["split"] == split]
    assert len({v["domain"] for v in side}) >= least


def test_dev_quota_matches_the_manifest(videos):
    for duration, quota in DEV_QUOTA.items():
        dev = [v for v in videos if v["duration"] == duration and v["split"] == "dev"]
        assert len(dev) == quota, duration


# --- licence and provenance --------------------------------------------------------


def test_licence_is_recorded_with_its_source(manifest):
    licence = manifest["licence"]
    assert "academic research" in licence["effective_terms"]
    assert "in whole or in part" in licence["effective_terms"]
    assert licence["source"].startswith("http")
    # The HF card genuinely declares nothing; record that rather than inventing an SPDX id.
    assert licence["declared_on_hf"] is None


def test_provenance_pins_a_revision(manifest):
    d = manifest["dataset"]
    assert d["requested_repo_id"] == "lmms-lab/Video-MME"
    assert len(d["revision"]) == 40, "want a full commit sha, not a branch name"
    assert d["read_via"].startswith("datasets.load_dataset")


def test_provenance_doc_exists():
    doc = MANIFEST_PATH.parent / "PROVENANCE.md"
    assert doc.exists(), "the licence + provenance write-up is part of the deliverable"


# --- nothing bulk-downloaded -------------------------------------------------------


def test_manifest_records_the_media_it_skipped(manifest):
    skipped = manifest["dataset"]["media_files_skipped"]
    assert skipped["count"] == 20
    assert skipped["bytes"] > 50e9, "the archives we are declining to fetch"


def test_no_media_is_committed():
    """The licence forbids redistributing Video-MME, so no media may live in the repo."""
    media_suffixes = {".mp4", ".mkv", ".webm", ".wav", ".m4a", ".zip"}
    stray = [
        p
        for p in Path("data").rglob("*")
        if p.is_file() and p.suffix.lower() in media_suffixes
    ]
    assert stray == [], f"media under data/: {stray}"


def test_manifest_carries_no_benchmark_qa(videos):
    """Copying Video-MME's questions in would be copying the benchmark 'in part'."""
    forbidden = {"question", "answer", "options", "question_id", "task_type"}
    for v in videos:
        assert not forbidden & set(v), f"benchmark Q&A leaked into the manifest: {v}"


# --- the selection algorithm, on synthetic rows -------------------------------------


def _fake(n: int, duration: str, domains: list[str]) -> list[Video]:
    """n videos in one duration bucket, cycling through `domains`."""
    return [
        Video(
            video_id=f"{i:03d}",
            youtube_id=f"yt{i:03d}",
            url=f"https://www.youtube.com/watch?v=yt{i:03d}",
            duration=duration,
            domain=domains[i % len(domains)],
            sub_category=f"sub{i}",
        )
        for i in range(n)
    ]


def test_pick_bucket_respects_the_quota():
    picked = pick_bucket(_fake(50, "short", ["a", "b", "c"]), 3, set(), {})
    assert len(picked) == 3


def test_pick_bucket_spreads_across_domains():
    picked = pick_bucket(_fake(60, "short", ["a", "b", "c", "d"]), 4, set(), {})
    assert len({v.domain for v in picked}) == 4


def test_pick_bucket_prefers_domains_not_used_by_earlier_buckets():
    """The bug this guards: alphabetical ordering re-picked the same domains every bucket."""
    domain_use = {"a": 1, "b": 1, "c": 0}
    picked = pick_bucket(_fake(30, "long", ["a", "b", "c"]), 1, set(), domain_use)
    assert picked[0].domain == "c"


def test_pick_bucket_avoids_repeating_a_sub_category():
    rows = _fake(30, "short", ["a"])
    seen = {rows[0].sub_category}
    picked = pick_bucket(rows, 1, seen, {})
    assert picked[0].sub_category not in {rows[0].sub_category}


def test_pick_bucket_is_deterministic():
    args = lambda: (_fake(40, "medium", ["a", "b", "c"]), 3, set(), {})  # noqa: E731
    first = [v.youtube_id for v in pick_bucket(*args())]
    second = [v.youtube_id for v in pick_bucket(*args())]
    assert first == second


def test_assign_splits_prefers_domains_missing_from_dev():
    chosen = [
        Video("001", "a", "u", "long", "AlreadyInDev", "s1"),
        Video("002", "b", "u", "long", "Fresh", "s2"),
    ]
    out = assign_splits(chosen, 1, {"AlreadyInDev": 1})
    dev = [v for v in out if v.split == "dev"]
    assert [v.domain for v in dev] == ["Fresh"]


def test_assign_splits_labels_everything():
    chosen = _fake(4, "long", ["a", "b"])
    out = assign_splits(chosen, 2, {})
    assert sum(v.split == "dev" for v in out) == 2
    assert sum(v.split == "heldout" for v in out) == 2


def test_build_manifest_counts_match_its_videos():
    picks = assign_splits(_fake(4, "long", ["a", "b", "c", "d"]), 2, {})
    m = build_manifest(picks, {"media_files_skipped": {"count": 0, "bytes": 0}})
    assert m["counts"]["total"] == 4
    assert m["counts"]["dev"] == 2
    assert m["counts"]["heldout"] == 2
    assert m["counts"]["per_split"]["dev"]["videos"] == 2
