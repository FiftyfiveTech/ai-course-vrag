"""Read the text off slide-heavy keyframes — VRAG-023, the stretch task.

    make captions VIDEO=samples/vector7-21aug-client-meeting.mp4
    make captions VIDEO=611 CAPTION_FLAGS="--arm ollama --limit 3"

    from src.caption import caption_video
    stored = caption_video("bob-video", cfg, meter)
    print(stored.text_yield(), stored.reduction())

Why this module exists
----------------------
Five of the seventeen answerable held-out pairs turn on something on *screen* rather than
something said, and one of them is on a video with no speech at all (STANDUP, VRAG-012). A
transcript-only index cannot reach those however good retrieval gets.

This measures what closing that gap would cost. It does **not** close it: nothing here is
embedded, `caption.index` ships `false`, and `src/index.py` is not touched. The acceptance
criterion for VRAG-023 is "2-arm cost table hosted vs local; **the gate is untouched**", and
an unindexed caption is how the second half of that stays true.

The two arms
------------
`arm = "nim"`    hosted — `meta-llama/Llama-3.2-11B-Vision-Instruct` on NVIDIA NIM's free tier
`arm = "ollama"` local  — `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M`

The hosted arm is **not Groq**, and that was measured rather than assumed. `client.models.list()`
on our key returns fourteen models and not one of them takes an image — which is the same fact
`config.toml`'s `[answer]` section already records from the other direction ("Groq serves
exactly two models with a real HF repo id and a chat endpoint"). NIM is the provider
`src/doctor.py` already calls "hosted fallback" and already checks a credential for, so the
hosted arm here is the one this repo was already set up to reach.

Neither arm gets a JSON schema, and `prompts/caption_v1.md` explains why at length: the arms
do not offer the same structured-output guarantees, and the deliverable is a table in which
the arm is supposed to be the only variable. The reply is text; `NO_TEXT` is the sentinel;
`parse_reply` normalises it.

What this module does not decide
-------------------------------
Which frames to caption. That is `src/keyframes.py`, which is pure and measured separately —
the selection ratio it produces (1091 frames -> 64 calls on the client meeting) is the number
that makes any per-call cost here meaningful.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from schemas.caption import Caption, StoredCaptions
from src.config import Config
from src.config import load as load_config
from src.telemetry import Meter

RUNS = Path("runs")
SAMPLES = Path("samples")

NIM = "nim"
OLLAMA = "ollama"
ARMS = (NIM, OLLAMA)

# The sentinel `prompts/caption_v1.md` rule 6 defines. A constant because two places depend on
# the exact string — the prompt that asks for it and the parser that recognises it — and a
# prompt whose sentinel has drifted from its parser reports 100% text yield on blank frames.
NO_TEXT = "NO_TEXT"

# The instruction substituted into the prompt's `{{question}}`. A constant so `make captions`
# and `tools/caption_arms.py` send the same string: a prompt whose two callers differ is two
# prompts, and the cost table would be comparing them.
CAPTION_TASK = "Transcribe the text visible in this frame, or reply NO_TEXT."

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_TIMEOUT_S = 120

# HF repo id -> the id NIM actually answers to. Explicit and checked rather than derived,
# because every id in this pipeline that could be derived turned out not to be: whisper's Groq
# id drops the `openai/` owner, nomic's needs a `-GGUF` suffix that only exists on a converted
# repo, and gpt-oss's happens to match — three different rules for three models. A lookup that
# raises on a miss is the only version of this that cannot silently send a wrong name.
NIM_WIRE_NAMES = {
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "meta/llama-3.2-11b-vision-instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct": "meta/llama-3.2-90b-vision-instruct",
    "microsoft/Phi-3-vision-128k-instruct": "microsoft/phi-3-vision-128k-instruct",
}


class CaptionError(Exception):
    """Captioning failed — the message says which frame, which arm, and why."""


# ---------------------------------------------------------------------------
# Where they live
# ---------------------------------------------------------------------------


def path_for(video_id: str, runs: Path = RUNS, samples: Path = SAMPLES) -> Path:
    """`runs/<stem>/captions.json` for a video_id.

    Same resolution `src.overview.path_for` does, and for the same reason: the run directory is
    named after the media file's stem, not the video_id, and the two differ for every corpus
    video (`611` lives in `runs/611_H8fGd3fCJbg/`).
    """
    from src.overview import path_for as overview_path

    return overview_path(video_id, runs, samples).with_name("captions.json")


def frames_record(video_id: str, runs: Path = RUNS, samples: Path = SAMPLES) -> dict[str, Any]:
    """The `frames` block ingest wrote to `media.json`: the dir, the fps, and every frame's t_s.

    Read from disk rather than recomputed. The frames on disk and their timestamps are ingest's
    output (VRAG-005) and re-deriving them here would be a second implementation of
    `frame_timestamps` that could disagree with the one that made the files.
    """
    media = path_for(video_id, runs, samples).with_name("media.json")
    if not media.is_file():
        raise CaptionError(
            f"{media} does not exist, so there are no sampled frames to caption. Ingest the "
            f"video first: make chunks VIDEO=<the file in samples/>"
        )
    try:
        payload = json.loads(media.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CaptionError(f"{media}: not readable as JSON — {exc}") from exc

    frames = payload.get("frames") or {}
    if not frames.get("frames"):
        raise CaptionError(
            f"{media} records no frames. `ingest.frames.fps` may have been 0 on that run, or "
            f"the video has no video stream."
        )
    return frames


# ---------------------------------------------------------------------------
# The reply
# ---------------------------------------------------------------------------


def parse_reply(raw: str) -> tuple[str, bool]:
    """A model's reply as `(text, has_text)`.

    The sentinel is matched on the *stripped whole reply*, case-sensitively, and only when it
    is the entire reply. A model that writes "NO_TEXT" as part of a sentence has not followed
    rule 7, and treating that as a clean no-text frame would hide a prompt that is drifting.

    Trailing code fences are stripped because both arms occasionally wrap a transcription in
    them despite rule 7, and a fence is unambiguously not text that was on the slide.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop the opening fence (with any language tag) and a closing one if present.
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text == NO_TEXT or not text:
        return "", False
    return text, True


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------


def _nim_wire_name(model: str) -> str:
    """The id NIM answers to, for an HF repo id. Raises rather than guessing."""
    try:
        return NIM_WIRE_NAMES[model]
    except KeyError:
        raise CaptionError(
            f"caption.model is {model!r}, and there is no NIM wire id recorded for it in "
            f"src.caption.NIM_WIRE_NAMES. The HF repo id is the name this repo uses "
            f"(CLAUDE.md), but the provider's id for it is a separate fact that has to be "
            f"looked up and written down — whisper's and nomic's both differ from theirs. "
            f"Known: {', '.join(sorted(NIM_WIRE_NAMES))}"
        ) from None


def encode_frame(path: Path, max_b64_bytes: int) -> str:
    """One frame as base64, or a refusal naming both numbers.

    NIM inlines an image only below its own size limit and wants an upload handle above it.
    Measured across the whole indexed corpus, the limit does not currently bite: the largest
    frame on disk is 85 291 bytes, which is 113 721 base64, against a 180 000 limit at
    `ingest.frames.width = 768` and `jpeg_quality = 4`. So this is a check rather than a
    re-encode path — raising the frame width or the quality is what would make it fire, and
    then the fix is a decision about those levers, not a silent downscale here.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CaptionError(f"cannot read the frame {path}: {exc}") from exc
    encoded = base64.b64encode(raw).decode("ascii")
    if len(encoded) > max_b64_bytes:
        raise CaptionError(
            f"{path.name} is {len(raw)} bytes, which is {len(encoded)} as base64 and over "
            f"caption.max_b64_bytes = {max_b64_bytes}. Lower ingest.frames.width or raise "
            f"ingest.frames.jpeg_quality (higher is worse) and re-ingest, rather than raising "
            f"the limit — the provider enforces its own."
        )
    return encoded


def _nim_arm(
    system: str, user: str, image_b64: str, model: str, cfg: Config
) -> tuple[str, int]:
    """One hosted vision call. Returns the reply text and the tokens it reported.

    Plain `urllib` rather than an SDK: the `groq` client cannot reach NIM, and NIM's interface
    is an OpenAI-compatible POST, so an SDK would be a new dependency for one request shape.
    Note that this is the opposite call from the one in `src/answer.py` — memory says Groq's
    REST endpoint 403s outside its SDK, but that is a Cloudflare rule on *Groq's* host, not a
    property of REST, and NIM documents this endpoint as its interface.
    """
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        raise CaptionError(
            "NVIDIA_API_KEY is not set, so the hosted arm cannot run. It is in .env.example "
            "and `make doctor` checks it. For a run with no key at all, use the local arm: "
            "make captions VIDEO=<file> CAPTION_FLAGS=--arm ollama"
        )
    payload = {
        "model": _nim_wire_name(model),
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ],
        "temperature": float(cfg.get("answer.temperature")),
        "max_tokens": int(cfg.get("caption.max_tokens")),
        "stream": False,
    }
    request = urllib.request.Request(
        NIM_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=NIM_TIMEOUT_S) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:  # noqa: BLE001 - the error body is a nicety, never the failure
            pass
        raise CaptionError(
            f"NIM returned {exc.code} for {model} — {detail or exc.reason}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise CaptionError(f"NIM call for {model} failed: {exc}") from exc

    try:
        text = body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise CaptionError(
            f"NIM's reply for {model} has no choices[0].message.content — {body}"
        ) from exc
    tokens = int((body.get("usage") or {}).get("total_tokens") or 0)
    return text, tokens


def _ollama_arm(
    system: str, user: str, image_b64: str, model: str, cfg: Config
) -> tuple[str, int]:
    """One local vision call.

    The model must be pulled with an explicit tag and from a repo that carries an `mmproj-*`
    projector — without one Ollama loads a vision model that cannot see the image and captions
    every frame from the text alone:

        ollama pull hf.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M

    Untagged is the trap `[embed]` in config.toml already documents: `ollama pull hf.co/<repo>`
    with no tag takes the repo's smallest file, which halved recall@5 when it happened to the
    embedder.
    """
    try:
        import ollama
    except ImportError as exc:
        raise CaptionError("ollama is not installed — run `uv sync`") from exc

    try:
        response = ollama.chat(
            model=f"hf.co/{model}",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user, "images": [image_b64]},
            ],
            options={
                "temperature": float(cfg.get("answer.temperature")),
                "num_predict": int(cfg.get("caption.max_tokens")),
            },
        )
    except Exception as exc:  # ollama raises ResponseError and httpx errors alike
        raise CaptionError(
            f"the local arm failed for {model}: {exc}\nPull it first: "
            f"ollama pull hf.co/{model}"
        ) from exc

    text = ((response or {}).get("message") or {}).get("content") or ""
    tokens = int((response or {}).get("eval_count") or 0) + int(
        (response or {}).get("prompt_eval_count") or 0
    )
    return text, tokens


def effective_model(cfg: Config, arm: str) -> str:
    """The HF repo id the named arm runs. Two models, two sets of measured numbers.

    Same reason `src.answer.effective_model` exists: reporting the hosted id while the local
    model answered labels a run with a number that was never measured on it — and here the
    whole deliverable is a table of those numbers side by side.
    """
    if arm == OLLAMA:
        return str(cfg.get("caption.ollama_model"))
    return str(cfg.get("caption.model"))


def _call_arm(
    arm: str, system: str, user: str, image_b64: str, model: str, cfg: Config
) -> tuple[str, int]:
    if arm == NIM:
        return _nim_arm(system, user, image_b64, model, cfg)
    if arm == OLLAMA:
        return _ollama_arm(system, user, image_b64, model, cfg)
    raise CaptionError(
        f"caption.arm is {arm!r}, which is not an arm. Use {NIM!r} (hosted) or "
        f"{OLLAMA!r} (local)."
    )


# ---------------------------------------------------------------------------
# Captioning one video
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def frame_context(video_id: str, keyframe: dict[str, Any]) -> str:
    """What the prompt's `{{context}}` says about the frame travelling beside it.

    The seconds are here so the reply can be attributed by a human reading the raw exchange,
    not so the model can copy them: unlike every other prompt in this repo, this one's output
    carries no timestamp at all. The span is attached in code from
    `src.keyframes.spans`, which measured it off ingest's own `t_s` values — so a caption's
    seconds cannot be a number a model produced.
    """
    return (
        f"This is a single still frame from video {video_id}, sampled at "
        f"t={keyframe['t_s']:.1f} seconds."
    )


def caption_video(
    video_id: str,
    cfg: Config,
    meter: Meter,
    *,
    arm: str | None = None,
    limit: int | None = None,
    refresh: bool = False,
    keyframes: list[dict[str, Any]] | None = None,
    runs_found: int | None = None,
    runs: Path = RUNS,
    samples: Path = SAMPLES,
    write: bool = True,
) -> StoredCaptions:
    """Caption one video's slide-heavy keyframes and write `runs/<stem>/captions.json`.

    `keyframes` is the seam `tools/caption_arms.py` needs: the two-arm table has to send the
    **same** frames through both arms, so it selects once and passes the selection in. When it
    is None the selection is made here, which is what `make captions` does.

    `runs_found` goes with it. A caller that passes a *truncated* selection still knows how many
    stretches the video really has, and that number is the denominator of the recorded
    reduction — inferring it from `len(keyframes)` would make a `--limit 10` run claim the
    video has ten slide-heavy stretches.

    `write=False` is for the same caller — the table runs two arms over one video and only one
    of them can own `captions.json`, so neither does.
    """
    from src.answer import build_messages

    arm = (arm or str(cfg.get("caption.arm"))).strip().lower()
    if arm not in ARMS:
        raise CaptionError(
            f"caption.arm is {arm!r}, which is not an arm. Use {NIM!r} (hosted) or "
            f"{OLLAMA!r} (local)."
        )

    path = path_for(video_id, runs, samples)
    if not refresh and keyframes is None and write and path.is_file():
        existing = load(video_id, runs, samples)
        if existing is not None and existing.arm == arm:
            return existing

    frames = frames_record(video_id, runs, samples)
    frames_dir = Path(frames["dir"])
    records = frames["frames"]
    threshold = float(cfg.get("caption.still_threshold"))
    min_run = int(cfg.get("caption.min_run_frames"))
    max_keyframes = int(cfg.get("caption.max_keyframes"))

    if keyframes is None:
        from src.keyframes import select

        step_s = 1.0 / float(frames["fps"])
        ceiling = max_keyframes if limit is None else min(max_keyframes, limit)
        keyframes, _stage, found = select(
            frames_dir,
            records,
            threshold=threshold,
            min_frames=min_run,
            step_s=step_s,
            fmt=str(frames["format"]),
            limit=ceiling,
        )
        if found > ceiling:
            print(
                f"  note: {found} still stretches found, captioning {ceiling} "
                f"(caption.max_keyframes={max_keyframes}"
                f"{f', --limit {limit}' if limit is not None else ''}). The recorded "
                f"reduction still describes all {found}.",
                file=sys.stderr,
            )
    else:
        found = len(keyframes) if runs_found is None else int(runs_found)

    model = effective_model(cfg, arm)
    prompt_path = Path(cfg.get("caption.prompt"))
    max_b64 = int(cfg.get("caption.max_b64_bytes"))

    captions: list[Caption] = []
    tokens_total = 0
    latency_total = 0.0
    for n, keyframe in enumerate(keyframes, start=1):
        image_b64 = encode_frame(frames_dir / keyframe["frame"], max_b64)
        system, user = build_messages(
            CAPTION_TASK,
            [],
            cfg,
            prompt=prompt_path,
            context=frame_context(video_id, keyframe),
        )
        print(
            f"  caption {video_id}: {n}/{len(keyframes)} {keyframe['frame']} "
            f"({keyframe['t_start']:.0f}s-{keyframe['t_end']:.0f}s, "
            f"stands for {keyframe['run_frames']} frames)",
            file=sys.stderr,
        )
        started = time.perf_counter()
        raw, tokens = _call_arm(arm, system, user, image_b64, model, cfg)
        elapsed = time.perf_counter() - started
        # Logged rather than wrapped in meter.span() because the arms return their own token
        # count and the cost is tokens x rate — span() would have to be told the tokens before
        # the call it is timing has reported them.
        meter.log(model, elapsed, tokens=tokens, phase="caption.vision")
        tokens_total += tokens
        latency_total += elapsed

        text, has_text = parse_reply(raw)
        captions.append(
            Caption(
                frame=keyframe["frame"],
                t_start=float(keyframe["t_start"]),
                t_end=float(keyframe["t_end"]),
                text=text,
                has_text=has_text,
                run_frames=int(keyframe["run_frames"]),
            )
        )

    stored = StoredCaptions(
        video_id=str(video_id),
        arm=arm,
        model=model,
        prompt=prompt_path.as_posix(),
        prompt_sha256=_sha256(prompt_path),
        threshold=threshold,
        min_run_frames=min_run,
        frames_considered=len(records),
        runs_found=found,
        tokens=tokens_total,
        latency_s=round(latency_total, 3),
        captions=captions,
    )
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(stored.model_dump(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return stored


def load(video_id: str, runs: Path = RUNS, samples: Path = SAMPLES) -> StoredCaptions | None:
    """The stored captions for a video, or None when there are not any yet.

    None rather than an exception: no captions is the normal state of every video, because
    VRAG-023 is a stretch task that nothing else in the pipeline depends on. A file that
    exists but does not validate *is* an error — the schema moved under a stored document.
    """
    path = path_for(video_id, runs, samples)
    if not path.is_file():
        return None
    try:
        return StoredCaptions.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise CaptionError(
            f"{path}: not a valid captions document — {exc}\n"
            f"Rebuild it: make captions VIDEO=<the file in samples/>"
        ) from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def report(stored: StoredCaptions, out=None) -> None:
    out = out or sys.stdout
    print(
        f"video {stored.video_id}  arm={stored.arm}  {stored.model}",
        file=out,
    )
    print(
        f"selection: {stored.frames_considered} frames -> {stored.runs_found} still "
        f"stretches ({stored.reduction():.1f}x fewer vision calls) at "
        f"threshold={stored.threshold} min_run_frames={stored.min_run_frames}",
        file=out,
    )
    print(
        f"captioned: {len(stored.captions)}  with text: "
        f"{sum(1 for c in stored.captions if c.has_text)} "
        f"({stored.text_yield():.1%})  {stored.tokens} tokens  {stored.latency_s:.2f}s",
        file=out,
    )
    for c in stored.captions:
        head = " ".join(c.text.split())[:96] if c.has_text else "(no text)"
        print(f"  {c.t_start:8.1f}s  {c.frame}  {head}", file=out)
    print(f"\nwrote {path_for(stored.video_id).as_posix()}", file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "video", help="the media file in samples/, or a bare video_id already ingested"
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--arm", choices=ARMS, default=None, help="override caption.arm for this run"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="caption at most this many keyframes"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="re-caption even when a document is stored"
    )
    args = parser.parse_args(argv)

    # Same reason src/overview.py does it, and more pressing here: a caption is *transcribed
    # text off a slide*, so a typographic quote or an em dash is the expected case rather than
    # the unlucky one, and cp1252 is still the default console encoding on Windows.
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

    from src.ingest import IngestError

    meter = Meter()
    try:
        stored = caption_video(
            video_id, cfg, meter, arm=args.arm, limit=args.limit, refresh=args.refresh
        )
    except (CaptionError, IngestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    report(stored)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
