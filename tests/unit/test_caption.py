"""src/caption.py and schemas/caption.py — no network, no model, no ffmpeg.

The two arms are the gate's business; what is testable cheaply is everything around them: the
wire-name lookup that must not guess, the size guard, the sentinel that turns a plain-text
reply into a countable number, and the contract that refuses a document whose yield and text
disagree.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from schemas.caption import Caption, StoredCaptions
from src import caption as mod


# --------------------------------------------------------------------------- parse_reply


def test_a_transcription_has_text():
    text, has_text = mod.parse_reply("Q3 revenue by region")
    assert (text, has_text) == ("Q3 revenue by region", True)


def test_the_sentinel_is_no_text():
    assert mod.parse_reply("NO_TEXT") == ("", False)


def test_the_sentinel_survives_surrounding_whitespace():
    assert mod.parse_reply("  NO_TEXT\n") == ("", False)


def test_an_empty_reply_is_no_text():
    """A model that returns nothing has found nothing, and must not count as yield."""
    assert mod.parse_reply("") == ("", False)
    assert mod.parse_reply("   ") == ("", False)


def test_the_sentinel_inside_a_sentence_is_not_the_sentinel():
    """Rule 7 says the reply is the transcription alone.

    Treating "There is NO_TEXT in this frame" as a clean no-text frame would hide a prompt
    that is drifting — so it stays text, and reading the captions shows the drift.
    """
    text, has_text = mod.parse_reply("There is NO_TEXT visible in this frame.")
    assert has_text is True
    assert "NO_TEXT" in text


def test_lowercase_is_not_the_sentinel():
    """Case-sensitive on purpose: the prompt asks for one exact string."""
    _text, has_text = mod.parse_reply("no_text")
    assert has_text is True


def test_a_fenced_transcription_is_unfenced():
    """Both arms occasionally wrap a transcription in a code fence despite rule 7.

    A fence is unambiguously not text that was on the slide, so it is stripped rather than
    stored as part of the caption.
    """
    text, has_text = mod.parse_reply("```\nAgenda\n1. Scope\n```")
    assert has_text is True
    assert text == "Agenda\n1. Scope"


def test_a_fence_with_a_language_tag_is_unfenced():
    text, _ = mod.parse_reply("```text\nAgenda\n```")
    assert text == "Agenda"


def test_a_fenced_sentinel_is_still_the_sentinel():
    assert mod.parse_reply("```\nNO_TEXT\n```") == ("", False)


def test_multiline_structure_is_preserved():
    """Rule 2: a slide's structure is part of what it says."""
    reply = "Roadmap\n- Q1 discovery\n- Q2 build"
    text, _ = mod.parse_reply(reply)
    assert text == reply


# --------------------------------------------------------------------------- wire names


def test_the_wire_name_is_looked_up_not_derived():
    """The HF repo id is the name this repo uses; the provider's id is a separate fact.

    Three models, three different rules — whisper's Groq id drops the owner, nomic's needs a
    -GGUF suffix, gpt-oss's happens to match. None was derivable.
    """
    assert (
        mod._nim_wire_name("meta-llama/Llama-3.2-11B-Vision-Instruct")
        == "meta/llama-3.2-11b-vision-instruct"
    )


def test_the_configured_model_has_a_wire_name():
    """config.toml's caption.model must be one this module can actually send."""
    from src.config import load as load_config

    cfg = load_config(Path("config.toml"))
    assert mod._nim_wire_name(str(cfg.get("caption.model")))


def test_an_unknown_model_raises_rather_than_guessing():
    with pytest.raises(mod.CaptionError, match="NIM_WIRE_NAMES"):
        mod._nim_wire_name("some-org/Some-Vision-Model")


def test_lowercasing_is_not_the_rule():
    """`meta-llama/...` -> `meta/...`: the owner changes, so no case transform would do it."""
    hf = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    assert mod._nim_wire_name(hf) != hf.lower()


# --------------------------------------------------------------------------- encode_frame


def test_a_frame_is_encoded_as_base64(tmp_path):
    frame = tmp_path / "frame_00001.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    encoded = mod.encode_frame(frame, 180_000)
    assert base64.b64decode(encoded) == frame.read_bytes()


def test_an_oversized_frame_names_both_numbers(tmp_path):
    """The message has to carry the size and the limit, because the fix is a lever decision.

    Raising ingest.frames.width or jpeg_quality is what makes this fire, and a silent
    downscale here would change what the cost table measured.
    """
    frame = tmp_path / "frame_00001.jpg"
    frame.write_bytes(b"x" * 1000)
    with pytest.raises(mod.CaptionError) as exc:
        mod.encode_frame(frame, 100)
    message = str(exc.value)
    assert "1000 bytes" in message
    assert "100" in message
    assert "ingest.frames.width" in message


def test_a_missing_frame_says_so(tmp_path):
    with pytest.raises(mod.CaptionError, match="cannot read the frame"):
        mod.encode_frame(tmp_path / "nope.jpg", 180_000)


def test_the_largest_real_frame_fits_the_configured_limit():
    """Measured, not assumed: the guard does not bite at the shipped ingest settings.

    Skips rather than fails when there are no frames on disk — a fresh clone has none, and
    this is a fact about the corpus, not about the code.
    """
    from src.config import load as load_config

    frames = sorted(Path("runs").glob("*/frames/*.jpg"))
    if not frames:
        pytest.skip("no ingested frames on disk to measure")
    cfg = load_config(Path("config.toml"))
    limit = int(cfg.get("caption.max_b64_bytes"))
    largest = max(frames, key=lambda p: p.stat().st_size)
    b64 = (largest.stat().st_size + 2) // 3 * 4
    assert b64 <= limit, f"{largest} is {b64} as base64, over caption.max_b64_bytes={limit}"


# --------------------------------------------------------------------------- the contract


def test_a_caption_with_text_must_say_so():
    with pytest.raises(ValidationError, match="has_text"):
        Caption(frame="frame_00001.jpg", t_start=0.0, t_end=5.0, text="Agenda", has_text=False)


def test_an_unrecognised_sentinel_cannot_be_stored_as_yield():
    """The failure this validator exists for: yield silently becoming 100%."""
    with pytest.raises(ValidationError, match="has_text"):
        Caption(frame="frame_00001.jpg", t_start=0.0, t_end=5.0, text="", has_text=True)


def test_a_span_must_run_forward():
    with pytest.raises(ValidationError, match="does not run forward"):
        Caption(frame="frame_00001.jpg", t_start=9.0, t_end=1.0, text="", has_text=False)


def caption(text: str) -> Caption:
    return Caption(
        frame="frame_00001.jpg",
        t_start=0.0,
        t_end=5.0,
        text=text,
        has_text=bool(text),
    )


def test_text_yield_counts_only_frames_that_had_text():
    stored = StoredCaptions(
        video_id="611",
        captions=[caption("Agenda"), caption(""), caption("Roadmap"), caption("")],
    )
    assert stored.text_yield() == 0.5


def test_text_yield_of_nothing_is_zero_not_an_error():
    """A video with no still stretches is a real outcome — it is what 'not slide-heavy' is."""
    assert StoredCaptions(video_id="611").text_yield() == 0.0


def test_reduction_is_frames_over_stretches():
    stored = StoredCaptions(video_id="611", frames_considered=1091, runs_found=64)
    assert round(stored.reduction(), 1) == 17.0


def test_reduction_without_a_selection_is_zero():
    assert StoredCaptions(video_id="611", frames_considered=1091).reduction() == 0.0


def test_a_stored_document_round_trips():
    stored = StoredCaptions(
        video_id="611",
        arm="nim",
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        frames_considered=361,
        runs_found=7,
        captions=[caption("Agenda")],
    )
    again = StoredCaptions.model_validate_json(json.dumps(stored.model_dump()))
    assert again == stored
    assert again.task == "VRAG-023"


# --------------------------------------------------------------------------- arm dispatch


def test_an_unknown_arm_names_the_two_valid_ones():
    with pytest.raises(mod.CaptionError) as exc:
        mod._call_arm("groq", "sys", "user", "", "some/model", None)
    assert "nim" in str(exc.value) and "ollama" in str(exc.value)


def test_effective_model_reports_the_arm_that_will_run():
    """Reporting the hosted id while the local model answered labels a run with a number that
    was never measured on it — and here the deliverable is those two numbers side by side."""
    from src.config import load as load_config

    cfg = load_config(Path("config.toml"))
    assert mod.effective_model(cfg, mod.NIM) == str(cfg.get("caption.model"))
    assert mod.effective_model(cfg, mod.OLLAMA) == str(cfg.get("caption.ollama_model"))


def test_the_local_arm_model_is_pinned_to_a_tag():
    """An untagged `ollama pull hf.co/<repo>` takes the smallest file in the repo.

    That is the measured trap [embed] documents: it was 2-bit weights, and it halved recall@5.
    """
    from src.config import load as load_config

    cfg = load_config(Path("config.toml"))
    assert ":" in str(cfg.get("caption.ollama_model")).rsplit("/", 1)[-1]


def test_captions_are_not_indexed_by_default():
    """The line that keeps VRAG-023's acceptance criterion true: 'the gate is untouched'.

    A caption in the `vrag` collection would move recall@5 and the Phase 2 abstention rates,
    which are recorded numbers scored once.
    """
    from src.config import load as load_config

    cfg = load_config(Path("config.toml"))
    assert cfg.get("caption.index") is False


def test_the_prompt_file_exists_and_parses():
    """Both placeholders, or build_messages would leave a literal {{token}} in the request."""
    from src.answer import load_prompt
    from src.config import load as load_config

    cfg = load_config(Path("config.toml"))
    system, user = load_prompt(Path(cfg.get("caption.prompt")))
    assert "NO_TEXT" in system, "the parser's sentinel has to be the one the prompt asks for"
    assert "{{context}}" in user and "{{question}}" in user


def test_the_task_string_is_shared():
    """`make captions` and tools/caption_arms.py must send the same instruction.

    A prompt whose two callers differ is two prompts, and the cost table would be comparing
    them rather than the arms.
    """
    assert mod.CAPTION_TASK and mod.NO_TEXT in mod.CAPTION_TASK
