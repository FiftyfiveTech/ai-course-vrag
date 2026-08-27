"""`@source` — scoping a question to one video.

    make ask Q="@611 what two tools do I need to make my first paper cut?"
    make answer Q="@611 @521 what do both presenters demonstrate?"
    echo "@701 what is on the table" | uv run python -m src.probe -

A question that names a source with `@` is answered from that source alone: the `@` tokens
are resolved to `video_id`s, retrieval is filtered to them in Chroma, and the tokens are
taken out of the text before it is embedded or shown to the model.

Why it is a filter and not a hint
---------------------------------
The obvious cheap version is to leave `@611` in the question text and hope the embedder
pulls chunks from video 611. That is not what the user asked for — it changes which chunks
*score* well, not which chunks are *eligible*, so a strongly-worded passage from another
video still outranks the right one and the answer cites a source the question excluded. The
filter goes to the store (`where={"video_id": {"$in": [...]}}`), so a chunk from an unnamed
video cannot be retrieved, cannot be cited, and cannot be grounded onto. "Only consider that
file" is then a property of the query rather than a tendency of the ranking.

Leaving the token in the text is a second, smaller wrong: `@611` is not a word the corpus
ever says, so embedding it moves the query vector for no reason. `Scope.text` is what
reaches the embedder and the prompt; `Scope.raw` is what the person typed and is what gets
echoed back to them.

An unresolvable tag is refused, not ignored
-------------------------------------------
`@bernini` matching nothing could silently mean "search everything", and that is the one
outcome worth avoiding: the answer would look scoped, cite whatever it liked, and there
would be nothing on screen saying the scope was dropped. So it raises, and the message
lists the handles that do resolve. Same for a source that exists in the corpus manifest but
has no chunk in the index — scoping to it would return nothing and read as "the corpus does
not cover this", which is a different claim from "that video was never indexed".

What can be typed after the `@`
-------------------------------
The corpus `video_id` (`@611`), the youtube id from the manifest (`@H8fGd3fCJbg`), or the
stem of the file `make sample-real` fetched (`@611_H8fGd3fCJbg`) — that last one because
the thing being tagged is, to the person typing, a file. Matching is case-insensitive.
Non-corpus ids are first-class here: `src.index --video-id` will happily index
`bob-video`, and `@bob-video` has to work or the feature only covers Video-MME.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.config import Config

MANIFEST = Path("data/corpus/manifest.json")
SAMPLES = Path("samples")

# `@` then the handle. The lookbehind is what keeps `ritika@fiftyfivetech.io` an email
# address instead of a mention of a source called `fiftyfivetech`: a mention starts the
# string or follows whitespace or an opening bracket, and nothing else.
#
# The handle charset is the one `src.api.VIDEO_ID` already enforces on anything that
# reaches the filesystem, so a token that parses here is a token that is safe to compare
# against a video_id. A trailing `-`/`_` is trimmed because it comes from punctuation
# ("@611-ish" is not a handle anyone means) and no corpus id ends in one.
MENTION = re.compile(r"(?:\A|(?<=[\s(\[{]))@([A-Za-z0-9][A-Za-z0-9_-]{0,63})")


class MentionError(Exception):
    """A question tags a source that cannot be answered from. Message says which and why."""


@dataclass(frozen=True)
class Source:
    """One thing that can be tagged with `@`.

    The union of the corpus manifest and the index, because neither one alone is the set of
    things a person can ask about: the manifest names videos that were never fetched, and
    the index holds ids that were never in the manifest (`src.index --video-id`).
    """

    video_id: str
    label: str = ""  # "Artistic Performance / Stage Play", or "" when nothing describes it
    indexed: bool = False
    split: str | None = None  # 'dev' / 'heldout' per the manifest; None for a stray id
    aliases: tuple[str, ...] = ()  # youtube id, fetched-file stem

    @property
    def handle(self) -> str:
        """What you type after the `@`. The video_id — it is the one name that is unique."""
        return self.video_id

    def matches(self, token: str) -> bool:
        folded = token.casefold()
        return folded == self.video_id.casefold() or folded in {
            a.casefold() for a in self.aliases
        }

    @property
    def line(self) -> str:
        """One line for a terminal — the handle, what it is, and whether it can be asked."""
        where = self.split or "not in the manifest"
        state = "indexed" if self.indexed else "NOT indexed"
        return f"@{self.handle:<12} {self.label or '-':<38} ({where}, {state})"


@dataclass(frozen=True)
class Scope:
    """What one question turned out to be asking, and of what."""

    raw: str  # exactly what was typed, mentions and all
    text: str  # the question with the mentions removed — this is what gets embedded
    video_ids: tuple[str, ...] = ()  # empty means the whole index, which is the default

    @property
    def scoped(self) -> bool:
        return bool(self.video_ids)

    def describe(self) -> str:
        """`video 611` / `videos 611, 521` / `the whole index`. For a footer or a log line."""
        if not self.video_ids:
            return "the whole index"
        plural = "videos" if len(self.video_ids) > 1 else "video"
        return f"{plural} {', '.join(self.video_ids)}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def tokens(question: str) -> list[str]:
    """The handles tagged in a question, in the order they were typed, without duplicates."""
    found: list[str] = []
    for match in MENTION.finditer(question):
        token = match.group(1).rstrip("-_")
        if token and token.casefold() not in {t.casefold() for t in found}:
            found.append(token)
    return found


def strip(question: str) -> str:
    """The question with the `@` tags taken out and the whitespace closed up."""
    return " ".join(MENTION.sub(" ", question).split())


# ---------------------------------------------------------------------------
# What can be tagged
# ---------------------------------------------------------------------------


def catalogue(
    *,
    indexed: Sequence[str] | None = None,
    records: dict[str, dict] | None = None,
    manifest: Path = MANIFEST,
    samples: Path = SAMPLES,
) -> list[Source]:
    """Every taggable source, lowest video_id first.

    Both halves of the union are injectable, and for the same reason: a caller that has
    already paid for them must not pay again, and must not end up describing a *different*
    corpus than the one it is serving. `src.api.videos` has the manifest loaded and the
    collection open, and if this re-read the manifest off disk the endpoint would answer
    from one copy and resolve `@` tags against another.
    """
    records = _manifest_records(manifest) if records is None else records
    present = [str(v) for v in (indexed or [])]
    ids = sorted(set(records) | set(present), key=_numeric)

    sources = []
    for video_id in ids:
        record = records.get(video_id) or {}
        sources.append(
            Source(
                video_id=video_id,
                label=_label(record),
                indexed=video_id in present,
                split=record.get("split"),
                aliases=_aliases(video_id, record, samples),
            )
        )
    return sources


def from_config(cfg: Config, samples: Path = SAMPLES) -> list[Source]:
    """The catalogue, reading the index for itself. For callers with no collection open."""
    from src.retrieve import indexed_video_ids

    return catalogue(indexed=indexed_video_ids(cfg), samples=samples)


def _manifest_records(manifest: Path) -> dict[str, dict]:
    """video_id -> its manifest record. Missing manifest is empty, not an error.

    Same tolerance `src.ask.load_manifest` has and for the same reason: a clone that has an
    index but never ran `make corpus` can still be asked a question, and the tags it can
    resolve are then the ids in the index.
    """
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(v["video_id"]): v for v in data.get("videos", []) if v.get("video_id")}


def _label(record: dict) -> str:
    """What the video is, out of the manifest. Video-MME has no titles, only a taxonomy."""
    parts = [str(record[key]) for key in ("domain", "sub_category") if record.get(key)]
    return " / ".join(parts)


def _aliases(video_id: str, record: dict, samples: Path) -> tuple[str, ...]:
    """The other names for this source: the youtube id, and the fetched file's stem.

    Filtered through the tag grammar, and that is not belt and braces. A youtube id may start
    with a hyphen — dev video 701's is `-dfvdKf-KR0` — and `@-dfvdKf-KR0` does not parse as a
    tag at all, because a handle has to begin with a letter or a digit. Listing it anyway
    would put a name in `make sources` and in the frontend's picker that resolves to nothing
    when it is typed back: a control that looks live and is not, which is the one failure
    this repo keeps writing tests about.
    """
    from src.index import local_file

    candidates: list[str] = []
    youtube_id = record.get("youtube_id")
    if youtube_id:
        candidates.append(str(youtube_id))
    path = local_file(video_id, samples)
    if path is not None and path.stem != video_id:
        candidates.append(path.stem)

    return tuple(a for a in candidates if tokens("@" + a) == [a])


def _numeric(video_id: str) -> tuple[int, str]:
    """Sort '9' before '10'; anything non-decimal sorts after, by name. Same rule as the API."""
    return (int(video_id), "") if video_id.isdigit() else (1 << 30, video_id)


# ---------------------------------------------------------------------------
# Resolving
# ---------------------------------------------------------------------------


def resolve(question: str, sources: Sequence[Source]) -> Scope:
    """Turn a tagged question into a `Scope`, or say why it cannot be one."""
    handles = tokens(question)
    if not handles:
        return Scope(raw=question, text=" ".join(question.split()))

    video_ids: list[str] = []
    for token in handles:
        source = next((s for s in sources if s.matches(token)), None)
        if source is None:
            raise MentionError(_unknown(token, sources))
        if not source.indexed:
            raise MentionError(_not_indexed(token, source))
        if source.video_id not in video_ids:
            video_ids.append(source.video_id)

    text = strip(question)
    if not text:
        raise MentionError(
            f"'{question.strip()}' tags a source but asks nothing. Put the question after "
            f"the tag: @{video_ids[0]} what happens at the start?"
        )
    return Scope(raw=question, text=text, video_ids=tuple(video_ids))


def scope(
    question: str,
    cfg: Config,
    *,
    sources: Sequence[Source] | None = None,
) -> Scope:
    """`resolve`, but it only builds the catalogue when the question actually tags something.

    The short-circuit is the point: an untagged question is the common case and must not pay
    for a manifest read, a samples/ glob and a full scan of the collection's metadata to
    discover that there was nothing to resolve.
    """
    if "@" not in question or not tokens(question):
        return Scope(raw=question, text=" ".join(question.split()))
    return resolve(question, sources if sources is not None else from_config(cfg))


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def _unknown(token: str, sources: Sequence[Source]) -> str:
    askable = [s for s in sources if s.indexed]
    near = _near(token, sources)
    lines = [f"no source is tagged @{token}."]
    if near:
        lines.append(f"Did you mean @{near.handle}?")
    if askable:
        lines.append("Tag one of these:")
        lines.extend("  " + s.line for s in askable)
    else:
        lines.append(
            "Nothing is indexed on this host, so there is no source to tag — run "
            "`make index-dev`."
        )
    return "\n".join(lines)


def _not_indexed(token: str, source: Source) -> str:
    if source.split == "heldout":
        return (
            f"@{token} is video {source.video_id}, which is on the held-out side of the "
            f"corpus split and is deliberately not in the index (CLAUDE.md: the held-out "
            f"set is sealed and scored by the Evaluator). Tag a dev video instead."
        )
    return (
        f"@{token} is video {source.video_id}, which the corpus knows about but the index "
        f"holds no chunk of — scoping to it would retrieve nothing and read as 'the corpus "
        f"does not cover this', which is a different claim. Index it first: "
        f"`make sample-real VIDEO_ID={source.video_id}` then "
        f"`make index VIDEO=samples/{source.video_id}_*`, or `make index-dev` for the "
        f"whole dev split."
    )


def _near(token: str, sources: Sequence[Source]) -> Source | None:
    """The one source whose handle or alias contains what was typed. None if it is ambiguous.

    Substring rather than an edit distance: the misses this has to catch are a truncated id
    and a half-typed filename, and offering a suggestion that is merely *close* to a name
    the user did not mean is worse than offering none — they would take it.
    """
    folded = token.casefold()
    hits = [
        s
        for s in sources
        if folded in s.video_id.casefold()
        or any(folded in a.casefold() for a in s.aliases)
    ]
    return hits[0] if len(hits) == 1 else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """`make sources` — what `@` accepts on this host, and what it does not.

    Offline apart from reading the local store: no model call, no network. The un-taggable
    rows are printed too, because "why does @091 not work" is the question this answers and
    a list that silently omitted it would not.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args(argv)

    # Same reason every other CLI here does it: an em dash on a cp1252 console must not exit
    # non-zero for an encoding accident.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    from src.config import load as load_config

    try:
        sources = from_config(load_config(args.config))
    except Exception as exc:
        print(f"FAIL - {exc}", file=sys.stderr)
        return 1

    askable = [s for s in sources if s.indexed]
    print("Tag any of these in a question to answer from it alone:\n")
    for source in askable:
        print("  " + source.line)
        if source.aliases:
            print(f"      also: {', '.join('@' + a for a in source.aliases)}")
    if not askable:
        print("  (nothing is indexed - run `make index-dev`)")

    rest = [s for s in sources if not s.indexed]
    if rest:
        print("\nIn the corpus but not in the index, so not taggable:\n")
        for source in rest:
            print("  " + source.line)

    print('\n  make ask Q="@%s ..."' % (askable[0].handle if askable else "611"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
