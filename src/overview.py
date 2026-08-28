"""Overview — what a whole video is.

    make overview VIDEO=samples/bob-video.mp4
    make overview VIDEO=samples/bob-video.mp4 OVERVIEW_FLAGS=--refresh

    from src.overview import build, load, as_chunks
    stored = load("bob-video")
    print(stored.overview.abstract)

Why this module exists
----------------------
Retrieval answers "what did they say about X" by finding the five chunks nearest to X. It
cannot answer "what is this video about?" or "who is taking part?", and not because it is
tuned badly: five chunks of 25 s is 1.5 % of a 56-minute meeting, and the question has no
semantic target to be near. `prompts/answer_v1.md` then declines — correctly, under its own
rules 2 and 3, which are the rules that make the abstention selective on `d013`-`d015`.

So the synthesis happens here instead, **once per video at index time**, over the whole
transcript, into `runs/<stem>/overview.json`. A question asked in overview mode is answered
against that document (~1 k tokens) rather than against the transcript (~12 k), which is why
the second model call is small even though the first one is not.

The chunks, not the transcript file
-----------------------------------
`build` reads the chunks back out of Chroma rather than reading `runs/<stem>/transcript.json`
off disk. Two reasons, and the second is the real one:

* the chunk boundaries are what a citation is allowed to name, so an overview built from them
  can only produce spans that `src.answer.ground` will accept;
* `runs/` can be ahead of the index. A video that has been re-chunked but not re-embedded has
  a transcript on disk that no question can reach. An overview built from that describes a
  corpus that is not the one being served, and nothing downstream would say so.

Citations come out the same door
--------------------------------
`as_chunks` projects an overview's spans into `RetrievedChunk`s, so an answer built from an
overview is grounded by `src.answer.ground` and rendered by `src.api.to_citation` — the same
two functions that handle a retrieved answer. There is no second citation path, which is why
the player seeks to an overview's citations without knowing they came from here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from schemas.overview import Overview, Person, StoredOverview, json_schema
from src.config import Config
from src.config import load as load_config
from src.retrieve import RetrievedChunk
from src.telemetry import Meter

RUNS = Path("runs")
SAMPLES = Path("samples")

# The instruction for the build pass, substituted into the prompt's `{{question}}`. It is a
# constant rather than a literal in the call so that `make overview` and the gate send the
# same string — a prompt whose two callers differ is two prompts.
BUILD_TASK = (
    "Describe this video as a whole: the abstract, everyone the transcript names, and the "
    "topics in order."
)

# The same, for the reduce pass of the fold. A separate constant because it is a genuinely
# different instruction — the input is a list of partial abstracts, not a transcript, and the
# only thing wanted back is one abstract of the whole.
MERGE_TASK = (
    "These are abstracts of consecutive parts of one video, in order. Write a single "
    "abstract of the whole video."
)


class OverviewError(Exception):
    """Building or loading an overview failed — the message says which and why."""


# ---------------------------------------------------------------------------
# Where one lives
# ---------------------------------------------------------------------------


def path_for(video_id: str, runs: Path = RUNS, samples: Path = SAMPLES) -> Path:
    """`runs/<stem>/overview.json` for a video_id.

    The run directory is named after the *media file's stem*, not the video_id
    (`src.chunk.build_chunks`: `out_dir = out_root / video.stem`), and the two differ for
    every corpus video — `611` lives in `runs/611_H8fGd3fCJbg/`. So the stem is recovered
    through `src.index.local_file`, which knows both naming layouts.

    When the media is no longer on disk the stem cannot be recovered that way, so the
    fallback reads each `runs/*/chunks.json` and matches on the `video_id` it records. That
    is a directory scan and it only runs in the case that needs it.
    """
    from src.index import local_file

    path = local_file(video_id, samples)
    if path is not None:
        return runs / path.stem / "overview.json"

    for chunks in sorted(runs.glob("*/chunks.json")):
        try:
            data = json.loads(chunks.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(data.get("video_id")) == str(video_id):
            return chunks.parent / "overview.json"

    # Nothing on disk names this video. Return the path it *would* have, so a caller that is
    # writing gets a sensible destination and a caller that is reading gets a miss rather
    # than an exception from a directory scan that found nothing.
    return runs / str(video_id) / "overview.json"


def load(video_id: str, runs: Path = RUNS, samples: Path = SAMPLES) -> StoredOverview | None:
    """The stored overview for a video, or None when there is not one yet.

    None rather than an exception: "no overview" is the normal state of a freshly indexed
    corpus, and the caller that cares (`src.answer`) turns it into a message naming the
    command to run. A file that exists but does not validate *is* an error — it means the
    schema moved under a stored document, and serving half of it would be worse than saying
    so.
    """
    path = path_for(video_id, runs, samples)
    if not path.is_file():
        return None
    try:
        return StoredOverview.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise OverviewError(
            f"{path}: not a valid overview document — {exc}\n"
            f"Rebuild it: make overview VIDEO=<the file in samples/>"
        ) from exc


def has_overview(video_id: str, runs: Path = RUNS, samples: Path = SAMPLES) -> bool:
    """Is there an overview for this video? Cheap enough for a per-row check in `/videos`."""
    return path_for(video_id, runs, samples).is_file()


# ---------------------------------------------------------------------------
# Reading the video back out of the index
# ---------------------------------------------------------------------------


def chunks_for(video_id: str, cfg: Config) -> list[RetrievedChunk]:
    """Every indexed chunk of one video, earliest first.

    Read-only on the store, the same way `src.retrieve.indexed_video_ids` is: `get_collection`
    rather than `src.embed._get_collection`, so describing a corpus never creates the empty
    one it is describing.
    """
    try:
        import chromadb
    except ImportError as exc:
        raise OverviewError("chromadb not installed — run `uv sync`") from exc

    chroma_path = Path(cfg.get("embed.chroma_path"))
    if not chroma_path.exists():
        return []
    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(str(cfg.get("embed.collection")))
        rows = collection.get(
            where={"video_id": str(video_id)}, include=["documents", "metadatas"]
        )
    except Exception:
        return []

    docs = rows.get("documents") or []
    metas = rows.get("metadatas") or []
    out = [
        RetrievedChunk(
            video_id=str((meta or {}).get("video_id", "")),
            t_start=float((meta or {}).get("t_start", 0.0)),
            t_end=float((meta or {}).get("t_end", 0.0)),
            text=doc or "",
            score=0.0,
        )
        for doc, meta in zip(docs, metas)
    ]
    # Chroma's `get` does not promise an order and the whole point of this list is that it
    # reads as the video does. Sorting here rather than trusting insertion order is the
    # difference between a timeline and a shuffled one.
    out.sort(key=lambda c: (c.t_start, c.t_end))
    return out


def render_transcript(chunks: list[RetrievedChunk]) -> str:
    """The whole video as numbered passages, with the seconds the model is told to copy.

    Deliberately the same header shape `src.answer.render_context` uses. The model sees one
    passage format across both prompts, and the one decimal place is the precision a span is
    checked at in `_validate_spans` and snapped to in `src.answer.ground`.
    """
    if not chunks:
        return "(this video has no indexed chunks)"
    blocks = []
    for n, chunk in enumerate(chunks, start=1):
        text = " ".join(chunk.text.split())
        blocks.append(
            f"[{n}] t_start={chunk.t_start:.1f}  t_end={chunk.t_end:.1f}\n{text}"
        )
    return "\n\n".join(blocks)


def render_overview(stored: StoredOverview) -> str:
    """The stored overview as the context for a question about the video.

    This is what makes the query-time call small: the transcript that produced this document
    was ~12 k tokens and this is ~1 k. Each line carries the seconds behind it, because the
    answer built from it has to cite something a player can seek to.
    """
    ov = stored.overview
    lines = [f"Video: {stored.video_id}", "", "What it is:", ov.abstract, ""]

    lines.append("People the transcript names:")
    if ov.people:
        for p in ov.people:
            said = f" — {p.described_as}" if p.described_as.strip() else ""
            lines.append(
                f"- {p.name}{said}  (named at t_start={p.evidence.t_start:.1f} "
                f"t_end={p.evidence.t_end:.1f})"
            )
    else:
        lines.append("- (the transcript names nobody)")

    lines.extend(["", "What happens, in order:"])
    for t in ov.topics:
        lines.append(f"- t_start={t.t_start:.1f}  t_end={t.t_end:.1f}  {t.topic}")

    lines.extend(
        [
            "",
            "This video has no speaker labels: the transcript records what was said, not "
            "who said it.",
        ]
    )
    return "\n".join(lines)


def as_chunks(stored: StoredOverview) -> list[RetrievedChunk]:
    """The overview's spans as retrieved chunks, for `src.answer.ground`.

    Grounding drops a citation whose video has no chunk and snaps the rest onto the nearest
    real span. Handing it these means an overview answer is checked by exactly the code a
    retrieved answer is checked by — and that a citation the model invented out of the middle
    of the video is snapped back onto a moment the overview actually names.
    """
    out = []
    for person in stored.overview.people:
        out.append(
            RetrievedChunk(
                video_id=stored.video_id,
                t_start=person.evidence.t_start,
                t_end=person.evidence.t_end,
                text=f"{person.name}: {person.described_as}".strip(" :"),
                score=0.0,
            )
        )
    for topic in stored.overview.topics:
        out.append(
            RetrievedChunk(
                video_id=stored.video_id,
                t_start=topic.t_start,
                t_end=topic.t_end,
                text=topic.topic,
                score=0.0,
            )
        )
    out.sort(key=lambda c: (c.t_start, c.t_end))
    return out


# ---------------------------------------------------------------------------
# Building one
# ---------------------------------------------------------------------------


def build(
    video_id: str,
    cfg: Config,
    meter: Meter,
    *,
    refresh: bool = False,
    source_sha256: str = "",
    runs: Path = RUNS,
    samples: Path = SAMPLES,
) -> StoredOverview:
    """Synthesise one video's overview from its whole transcript and write it to disk.

    Skips the model call when an overview is already stored for the same `source_sha256`,
    which is what makes re-running `make index` free. `refresh=True` rebuilds regardless.
    An empty `source_sha256` means the caller does not know it, and then a stored overview is
    reused on existence alone — the alternative, rebuilding whenever the digest is unknown,
    would charge `make index` for an overview on every run of a video it cannot check.
    """
    from src.answer import build_messages, _ask

    path = path_for(video_id, runs, samples)
    if not refresh and path.is_file():
        existing = load(video_id, runs, samples)
        if existing is not None and (
            not source_sha256 or existing.source_sha256 == source_sha256
        ):
            return existing

    chunks = chunks_for(video_id, cfg)
    if not chunks:
        raise OverviewError(
            f"video {video_id} has no chunks in collection "
            f"{cfg.get('embed.collection')!r} — index it first: make index VIDEO=<file>"
        )

    ceiling = int(cfg.get("overview.max_context_chars"))
    groups = windows(chunks, ceiling)
    prompt_path = Path(cfg.get("overview.prompt"))

    if len(groups) == 1:
        # One window means this is the whole video, so it gets the whole-video cap.
        overview, used = _build_one(
            video_id, groups[0], cfg, meter, prompt_path,
            int(cfg.get("overview.max_tokens")),
        )
    else:
        # The fold. No real transcript fits one call on the free tier — see the arithmetic on
        # overview.max_context_chars in config.toml — so each window is summarised on its own
        # and the partials are merged. Progress goes to stderr because the windows are paced
        # by the provider's per-minute bucket and this is otherwise a silent multi-minute wait.
        partials = []
        used = ""
        for n, group in enumerate(groups, start=1):
            print(
                f"  overview {video_id}: window {n}/{len(groups)} "
                f"({len(group)} chunks, {group[0].t_start:.0f}s-{group[-1].t_end:.0f}s)",
                file=sys.stderr,
            )
            part, used = _build_one(
                video_id, group, cfg, meter, prompt_path,
                int(cfg.get("overview.window_max_tokens")),
            )
            partials.append(part)
        overview = _merge(video_id, partials, cfg, meter)

    overview = _validate_spans(video_id, overview, chunks)

    stored = StoredOverview(
        video_id=str(video_id),
        source_sha256=source_sha256,
        model=used,
        prompt=prompt_path.as_posix(),
        prompt_sha256=_sha256(prompt_path),
        chunks=len(chunks),
        overview=overview,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(stored.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return stored


def windows(chunks: list[RetrievedChunk], ceiling: int) -> list[list[RetrievedChunk]]:
    """Split the transcript into consecutive groups that each fit one model call.

    Greedy and in time order: a window is closed when adding the next chunk would put the
    *rendered* context over the ceiling. Rendered, not raw, because the header
    `render_transcript` writes for every passage is part of what the tier charges for.

    Time order is not an implementation convenience. A window is handed to the build prompt
    as though it were a whole transcript, and the topics it returns are "the video in order,
    as stretches" — which is only true if the window is a contiguous stretch of the video.

    One chunk larger than the ceiling on its own still gets its own window rather than being
    dropped: losing a stretch of the video silently is worse than the 413 that follows, and
    the 413 names the numbers.
    """
    if not chunks:
        return []
    groups: list[list[RetrievedChunk]] = []
    current: list[RetrievedChunk] = []
    for chunk in chunks:
        candidate = current + [chunk]
        if current and len(render_transcript(candidate)) > ceiling:
            groups.append(current)
            current = [chunk]
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def _build_one(
    video_id: str,
    chunks: list[RetrievedChunk],
    cfg: Config,
    meter: Meter,
    prompt_path: Path,
    max_tokens: int,
) -> tuple[Overview, str]:
    """One model call: a window of transcript in, a validated `Overview` out."""
    from src.answer import build_messages, _ask

    system, user = build_messages(
        BUILD_TASK, [], cfg, prompt=prompt_path, context=render_transcript(chunks)
    )
    raw, _tokens, used = _ask(
        system,
        user,
        cfg,
        meter,
        schema=json_schema(),
        schema_name="overview",
        # Its own cap, and which one depends on how much video this call is describing.
        # The cap is charged against the per-minute budget whether it is used or not, so
        # reserving a whole-video 4000 for a quarter-video answer is what took the first
        # folded run to 8315 tokens against a limit of 8000.
        max_tokens=max_tokens,
    )
    try:
        return Overview.model_validate_json(raw), used
    except (ValidationError, ValueError) as exc:
        raise OverviewError(
            f"video {video_id}: the model's overview did not validate against "
            f"schemas.overview.Overview — {exc}"
        ) from exc


def _merge(
    video_id: str, partials: list[Overview], cfg: Config, meter: Meter
) -> Overview:
    """Fold per-window overviews into one document.

    `people` and `topics` are merged **in code, not by a model**, and that is the important
    decision here. Every span in this document has to be a span off a real chunk — that is
    the promise `schemas/overview.py` makes and the thing `_validate_spans` and
    `src.answer.ground` both rely on. A model asked to merge two documents will happily
    adjust a timestamp to make two entries line up, and a merged span that no chunk backs is
    exactly the failure the whole citation path exists to prevent. Concatenating cannot
    invent one.

    `abstract` is the one field that genuinely needs synthesis: eight window abstracts
    stapled together is not an abstract of the video. That is one small call — a few hundred
    tokens of input, well inside the per-minute budget even on the free tier — and it is
    given no timestamps at all, so it has none to get wrong.
    """
    from src.answer import build_messages, _ask

    # People: earliest evidence wins, matched on a normalised name so "Bernini" and
    # "bernini " do not both survive. Earliest because the first time a video names someone
    # is the moment a viewer wants to jump to.
    people: dict[str, Person] = {}
    for part in partials:
        for person in part.people:
            key = " ".join(person.name.split()).casefold()
            if key not in people or person.evidence.t_start < people[key].evidence.t_start:
                people[key] = person

    # Topics: already contiguous per window and the windows are in time order, so ordering by
    # start time is a sort over what is nearly sorted already, not a re-interpretation.
    topics = sorted(
        (topic for part in partials for topic in part.topics),
        key=lambda t: t.t_start,
    )

    merge_prompt = Path(cfg.get("overview.merge_prompt"))
    joined = "\n\n".join(
        f"[part {n} of {len(partials)}] {part.abstract}"
        for n, part in enumerate(partials, start=1)
    )
    system, user = build_messages(
        MERGE_TASK, [], cfg, prompt=merge_prompt, context=joined
    )
    raw, _tokens, _used = _ask(
        system,
        user,
        cfg,
        meter,
        schema=_ABSTRACT_SCHEMA,
        schema_name="merged_abstract",
        # Small on purpose: this returns one paragraph and nothing else. Keeping it small is
        # also what keeps the reduce call inside the per-minute budget the map calls have
        # just spent most of.
        max_tokens=800,
    )
    try:
        abstract = str(json.loads(raw)["abstract"]).strip()
    except (ValueError, KeyError, TypeError) as exc:
        raise OverviewError(
            f"video {video_id}: the merged abstract did not come back as "
            f'{{"abstract": "..."}} — {exc}'
        ) from exc
    if not abstract:
        raise OverviewError(f"video {video_id}: the merged abstract came back empty")

    return Overview(abstract=abstract, people=list(people.values()), topics=topics)


# The reduce step's contract, inline because it is one field and exists only here. Same
# strict-mode treatment schemas.answer.json_schema() applies: every object closed, every
# property required, no $ref.
_ABSTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "abstract": {
            "type": "string",
            "description": "three to five sentences describing the whole video",
        }
    },
    "required": ["abstract"],
    "additionalProperties": False,
}


def _validate_spans(
    video_id: str, overview: Overview, chunks: list[RetrievedChunk]
) -> Overview:
    """Snap every span in the document onto a chunk that really exists.

    The same repair `src.answer.ground` performs on a citation, done once at build time so a
    stored overview cannot hold a moment the video does not have. Dropping instead of
    snapping would delete a person the transcript really names because the model copied their
    timestamp badly; snapping keeps the finding and fixes the pointer, which is the truthful
    version of what the model was claiming.
    """
    if not chunks:
        return overview
    starts = sorted(chunks, key=lambda c: c.t_start)

    def snap(t_start: float) -> RetrievedChunk:
        return min(starts, key=lambda c: abs(c.t_start - t_start))

    people = []
    for person in overview.people:
        near = snap(person.evidence.t_start)
        people.append(
            person.model_copy(
                update={
                    "evidence": person.evidence.model_copy(
                        update={"t_start": near.t_start, "t_end": near.t_end}
                    )
                }
            )
        )

    last = max(c.t_end for c in chunks)
    topics = []
    for topic in overview.topics:
        start = snap(topic.t_start).t_start
        end = max(start, min(float(topic.t_end), last))
        topics.append(topic.model_copy(update={"t_start": start, "t_end": end}))

    return overview.model_copy(update={"people": people, "topics": topics})


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def report(stored: StoredOverview, out=None) -> None:
    out = out or sys.stdout
    ov = stored.overview
    print(f"video {stored.video_id}  ({stored.chunks} chunks, {stored.model})", file=out)
    print(f"\n{ov.abstract}\n", file=out)
    print(f"people named: {len(ov.people)}", file=out)
    for p in ov.people:
        print(f"  {p.evidence}  {p.name}" + (f" — {p.described_as}" if p.described_as else ""), file=out)
    print(f"topics: {len(ov.topics)}", file=out)
    for t in ov.topics:
        print(f"  {t.t_start:7.1f}s  {t.topic}", file=out)
    print(f"\nwrote {path_for(stored.video_id).as_posix()}", file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "video",
        help="the media file in samples/, or a bare video_id that is already indexed",
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="rebuild even when an overview is already stored (costs a model call)",
    )
    args = parser.parse_args(argv)

    # Same reason src/answer.py, src/ask.py and src/api.py do it: an abstract that came back
    # with a non-breaking hyphen in it must not take the process down on a cp1252 console.
    # This one was found the hard way - six windows and a merge, then UnicodeEncodeError in
    # report() with the document already safely on disk.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    cfg = load_config(Path(args.config))
    video = Path(args.video)
    if video.is_file():
        from src.chunk import resolve_video_id

        video_id, _ = resolve_video_id(video)
    else:
        video_id = args.video

    # Lazily, like build() does: src.answer imports this module back (for the OVERVIEW
    # answering mode), so neither may reach for the other at import time.
    from src.answer import AnswerError

    meter = Meter()
    try:
        stored = build(video_id, cfg, meter, refresh=args.refresh)
    except (OverviewError, AnswerError) as exc:
        # AnswerError too, and _GroqRequestTooLargeError especially: a 413 here is a normal,
        # expected outcome for any transcript over the tier's per-minute budget, and it
        # arrives with the numbers and the levers in its message. A traceback would bury
        # that under thirty frames of the groq client.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    report(stored)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
