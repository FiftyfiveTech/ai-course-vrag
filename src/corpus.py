"""Corpus selection — VRAG-004.

Streams the Video-MME annotation table and picks 10 varied videos, 4 dev / 6 held-out.

Nothing is bulk-downloaded, and that is a *licence* requirement rather than a disk-space
preference. Video-MME's terms (upstream README, MME-Benchmarks/Video-MME) say:

    Without prior approval, you cannot distribute, publish, copy, disseminate, or
    modify Video-MME in whole or in part.

Two consequences shape this module:

* the media never lands here. The manifest stores *pointers* — the YouTube id and url
  each video came from — so the repo carries no copy of the benchmark's video files.
  The 20 `videos_chunked_*.zip` archives (~100 GB) are never requested.
* the benchmark's own questions and answers are not copied either. Only video-level
  facts are kept (duration bucket, domain, sub-category), which is what "10 varied
  videos" has to be justified against. Our Q&A pairs are written from scratch in
  VRAG-011/012.

The selection is deterministic: no seed, no timestamp, sorted tie-breaks. Re-running
`make corpus` on the same dataset revision rewrites a byte-identical manifest, so the
supervisor can re-derive the split instead of trusting it. `--check` asserts exactly that.

    make corpus
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from src.env import load_env

DATASET_REPO = "lmms-lab/Video-MME"
CONFIG = "videomme"
SPLIT = "test"

MANIFEST = Path("data/corpus/manifest.json")

# 10 videos, spread over all three duration buckets. Long videos get the extra slot:
# they are where chunking and retrieval actually get hard, so the corpus should not be
# mostly short clips.
DURATION_QUOTA = {"short": 3, "medium": 3, "long": 4}

# Of those, this many per bucket go to evals/dev. The rest are held out. Both sides
# cover all three durations — a held-out set with no long videos would not test the
# thing most likely to break.
DEV_QUOTA = {"short": 1, "medium": 1, "long": 2}

DURATION_ORDER = ("short", "medium", "long")

# Everything under this prefix is the redistributable media we must not fetch.
MEDIA_PREFIX = "videos_chunked"

LICENCE_TEXT = (
    "Video-MME is only used for academic research. Commercial use in any form is "
    "prohibited.\n"
    "The copyright of all videos belongs to the video owners.\n"
    "If there is any infringement in Video-MME, please email videomme2024@gmail.com "
    "and we will remove it immediately.\n"
    "Without prior approval, you cannot distribute, publish, copy, disseminate, or "
    "modify Video-MME in whole or in part.\n"
    "You must strictly comply with the above restrictions."
)
LICENCE_SOURCE = (
    "https://github.com/MME-Benchmarks/Video-MME#-dataset "
    "(stated in the README; neither that repo nor the HF dataset ships a LICENSE file)"
)


@dataclass(frozen=True)
class Video:
    """One video's pointer and taxonomy. Deliberately carries no question or answer."""

    video_id: str        # Video-MME's own index, e.g. "001"
    youtube_id: str      # the durable pointer; media is fetched from here, not from HF
    url: str
    duration: str        # short | medium | long
    domain: str
    sub_category: str
    split: str = ""      # dev | heldout, filled in by select()


def _export_hf_token() -> str:
    """Put HF_TOKEN where huggingface_hub looks for it. Returns its source, for printing.

    Secrets live in .env / ~/.config (CLAUDE.md), which are read but not exported, so
    without this the stream falls back to unauthenticated and gets rate-limited.
    """
    if os.environ.get("HF_TOKEN"):
        return "environment"
    found = load_env().get("HF_TOKEN")
    if not found or not found[0]:
        return "not set (unauthenticated; rate limits apply)"
    os.environ["HF_TOKEN"] = found[0]
    return found[1]


def dataset_provenance() -> dict:
    """Resolve what we are actually reading: canonical id, revision, and media we skip."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(DATASET_REPO, files_metadata=True)
    media = [f for f in info.siblings if f.rfilename.startswith(MEDIA_PREFIX)]
    return {
        "requested_repo_id": DATASET_REPO,
        "canonical_repo_id": info.id,
        "config": CONFIG,
        "split": SPLIT,
        "revision": info.sha,
        "gated": bool(info.gated),
        "read_via": "datasets.load_dataset(..., streaming=True)",
        "annotation_files_read": sorted(
            f.rfilename for f in info.siblings if f.rfilename.startswith(CONFIG + "/")
        ),
        "media_files_skipped": {
            "count": len(media),
            "bytes": sum(f.size or 0 for f in media),
            "note": "the benchmark's video archives — never requested; see licence",
        },
    }


def stream_index() -> dict[str, Video]:
    """Stream the annotation table and collapse 2700 QA rows to one record per video.

    `streaming=True` reads the 405 KB parquet over range requests. It cannot touch the
    video zips: those are not part of any split.
    """
    from datasets import load_dataset

    rows = load_dataset(DATASET_REPO, CONFIG, split=SPLIT, streaming=True)
    index: dict[str, Video] = {}
    for row in rows:
        key = row["videoID"]
        if key in index:
            continue
        index[key] = Video(
            video_id=row["video_id"],
            youtube_id=key,
            url=row["url"],
            duration=row["duration"],
            domain=row["domain"],
            sub_category=row["sub_category"],
        )
    return index


def pick_bucket(
    rows: list[Video],
    quota: int,
    seen_sub: set[str],
    domain_use: dict[str, int],
) -> list[Video]:
    """Take `quota` videos from one duration bucket, spread as widely as possible.

    Domains are visited least-used-first, counting usage across *all* buckets — an
    earlier version ordered them alphabetically per bucket, which kept re-picking the
    same first three domains and covered only 4 of the 6. Within a domain, a video whose
    sub-category is still unused wins; ties go to the lowest video_id. No randomness, so
    the result is reproducible.
    """
    by_domain: dict[str, list[Video]] = {}
    for row in sorted(rows, key=lambda r: r.video_id):
        by_domain.setdefault(row.domain, []).append(row)

    picked: list[Video] = []
    while len(picked) < quota:
        order = sorted(by_domain, key=lambda d: (domain_use.get(d, 0), d))
        progressed = False
        for domain in order:
            if len(picked) >= quota:
                break
            queue = by_domain[domain]
            if not queue:
                continue
            fresh = next(
                (i for i, r in enumerate(queue) if r.sub_category not in seen_sub), 0
            )
            row = queue.pop(fresh)
            picked.append(row)
            seen_sub.add(row.sub_category)
            domain_use[domain] = domain_use.get(domain, 0) + 1
            progressed = True
        if not progressed:  # bucket exhausted; cannot happen at 300 videos per bucket
            break
    return picked


def assign_splits(chosen: list[Video], dev_quota: int, dev_use: dict[str, int]) -> list[Video]:
    """Label one bucket's picks dev/held-out, keeping dev's domains diverse too.

    The dev slots go to the domains least represented in dev *so far*, so dev does not
    fill up with repeats of whichever domain happens to sort first.
    """
    ranked = sorted(
        range(len(chosen)), key=lambda i: (dev_use.get(chosen[i].domain, 0), i)
    )
    dev_idx = set(ranked[:dev_quota])
    for i in dev_idx:
        dev_use[chosen[i].domain] = dev_use.get(chosen[i].domain, 0) + 1
    return [
        Video(**{**asdict(v), "split": "dev" if i in dev_idx else "heldout"})
        for i, v in enumerate(chosen)
    ]


def select(index: dict[str, Video]) -> list[Video]:
    """Apply DURATION_QUOTA, then label each bucket's picks per DEV_QUOTA."""
    picks: list[Video] = []
    seen_sub: set[str] = set()
    domain_use: dict[str, int] = {}
    dev_use: dict[str, int] = {}
    for duration in DURATION_ORDER:
        bucket = [v for v in index.values() if v.duration == duration]
        chosen = pick_bucket(bucket, DURATION_QUOTA[duration], seen_sub, domain_use)
        if len(chosen) < DURATION_QUOTA[duration]:
            raise SystemExit(
                f"only {len(chosen)} '{duration}' videos available, "
                f"need {DURATION_QUOTA[duration]}"
            )
        picks.extend(assign_splits(chosen, DEV_QUOTA[duration], dev_use))
    return picks


def build_manifest(videos: list[Video], provenance: dict) -> dict:
    """The committed artifact. No timestamp — the dataset revision is the anchor."""
    return {
        "task": "VRAG-004",
        "what_this_is": (
            "The 10-video pilot corpus: pointers only, no media and no benchmark Q&A. "
            "Regenerate with `make corpus`; verify with `make corpus-check`."
        ),
        "dataset": provenance,
        "licence": {
            "declared_on_hf": None,
            "effective_terms": LICENCE_TEXT,
            "source": LICENCE_SOURCE,
            "our_use": (
                "non-commercial coursework; academic research use is permitted by the terms"
            ),
            "why_pointers_only": (
                "the terms forbid distributing or copying Video-MME in whole or in part "
                "without prior approval, so this repo stores ids and urls, not video "
                "files and not the benchmark's questions or answers"
            ),
            "video_copyright": (
                "each video's copyright belongs to its original YouTube uploader"
            ),
        },
        "selection": {
            "deterministic": True,
            "rule": (
                "one record per videoID from the streamed test split; per duration "
                "bucket, round-robin over domains least-used-first (counted across all "
                "buckets) preferring unused sub-categories, lowest video_id breaking "
                "ties; dev slots then go to the domains least represented in dev so far"
            ),
            "duration_quota": DURATION_QUOTA,
            "dev_quota": DEV_QUOTA,
        },
        "counts": {
            "total": len(videos),
            "dev": sum(1 for v in videos if v.split == "dev"),
            "heldout": sum(1 for v in videos if v.split == "heldout"),
            "durations": {
                d: sum(1 for v in videos if v.duration == d) for d in DURATION_ORDER
            },
            "domains": len({v.domain for v in videos}),
            "sub_categories": len({v.sub_category for v in videos}),
            # Per side, because a split is only useful if each half is varied on its own.
            "per_split": {
                split: {
                    "videos": len(side),
                    "domains": len({v.domain for v in side}),
                    "durations": sorted({v.duration for v in side}),
                }
                for split in ("dev", "heldout")
                for side in ([v for v in videos if v.split == split],)
            },
        },
        "videos": [asdict(v) for v in videos],
    }


def _dump(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


def report(manifest: dict, out=sys.stdout) -> None:
    """Print the table and the numbers. A number not printed is not a number."""
    d, c = manifest["dataset"], manifest["counts"]
    print(
        f"corpus — {d['requested_repo_id']} (canonical {d['canonical_repo_id']})", file=out
    )
    print(f"  revision {d['revision']}", file=out)
    print(f"  read     {', '.join(d['annotation_files_read'])} via streaming", file=out)
    skipped = d["media_files_skipped"]
    print(
        f"  skipped  {skipped['count']} media archive(s), "
        f"{skipped['bytes'] / 1e9:.1f} GB never requested",
        file=out,
    )
    header = (
        f"  {'split':<8} {'dur':<7} {'domain':<20} {'sub-category':<26} {'id':<5} youtube"
    )
    print("\n" + header, file=out)
    for v in manifest["videos"]:
        print(
            f"  {v['split']:<8} {v['duration']:<7} {v['domain']:<20} "
            f"{v['sub_category']:<26} {v['video_id']:<5} {v['youtube_id']}",
            file=out,
        )
    print(
        f"\n{c['total']} videos = {c['dev']} dev + {c['heldout']} held-out · "
        f"durations {c['durations']} · {c['domains']} domains · "
        f"{c['sub_categories']} sub-categories",
        file=out,
    )
    for split, s in c["per_split"].items():
        print(
            f"  {split:<8} {s['videos']} videos · {s['domains']} domains · "
            f"durations {'/'.join(s['durations'])}",
            file=out,
        )


def generate() -> dict:
    print(f"HF_TOKEN from {_export_hf_token()}", file=sys.stderr)
    return build_manifest(select(stream_index()), dataset_provenance())


def verify_pointers(manifest: dict, out=sys.stdout) -> int:
    """Confirm every recorded video is still fetchable from its source url.

    Video-MME was published in 2024 and we hold pointers, not copies, so a video can be
    deleted or made private out from under us — a dead pointer is not provenance. Ingest
    (VRAG-005) needs this to be true, so it stays re-runnable. Uses YouTube's oembed
    endpoint: no API key, no quota, no spend.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    dead = []
    for v in manifest["videos"]:
        query = urllib.parse.urlencode({"format": "json", "url": v["url"]})
        req = urllib.request.Request(
            "https://www.youtube.com/oembed?" + query, headers={"User-Agent": "Mozilla/5.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                title = json.loads(resp.read().decode())["title"]
            print(f"  OK    {v['split']:<8} {v['youtube_id']:<12} {title[:56]}", file=out)
        except Exception as exc:  # noqa: BLE001 — any failure means unusable
            reason = getattr(exc, "code", exc)
            print(f"  DEAD  {v['split']:<8} {v['youtube_id']:<12} {reason}", file=out)
            dead.append(v)

    total = len(manifest["videos"])
    print(f"\n{total - len(dead)}/{total} reachable, {len(dead)} dead", file=out)
    if dead:
        print(
            "FAIL — re-select the corpus; a pointer that does not resolve is not provenance.",
            file=out,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Select the VRAG pilot corpus (VRAG-004).")
    ap.add_argument(
        "--check",
        action="store_true",
        help="re-stream and confirm the committed manifest is byte-identical; writes nothing",
    )
    ap.add_argument(
        "--verify-pointers",
        action="store_true",
        help="confirm every video in the committed manifest still resolves; writes nothing",
    )
    args = ap.parse_args(argv)

    if args.verify_pointers:
        if not MANIFEST.exists():
            print(f"FAIL — {MANIFEST} does not exist; run `make corpus`", file=sys.stderr)
            return 1
        return verify_pointers(json.loads(MANIFEST.read_text(encoding="utf-8")))

    manifest = generate()
    rendered = _dump(manifest)

    if args.check:
        if not MANIFEST.exists():
            print(f"FAIL — {MANIFEST} does not exist; run `make corpus`", file=sys.stderr)
            return 1
        report(manifest)
        if MANIFEST.read_text(encoding="utf-8") != rendered:
            print(
                f"\nFAIL — {MANIFEST} does not match a fresh stream of revision "
                f"{manifest['dataset']['revision']}.",
                file=sys.stderr,
            )
            return 1
        print(
            f"\nPASS — {MANIFEST} reproduced byte-for-byte from revision "
            f"{manifest['dataset']['revision']}."
        )
        return 0

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(rendered, encoding="utf-8")
    report(manifest)
    print(f"\nwrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
