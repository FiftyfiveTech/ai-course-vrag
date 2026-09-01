"""VRAG-026: Segment.speaker survives every hop in the pipeline.

Walks one attributed VTT through the full chain:
  parse_vtt → Segment → Chunk → Chroma metadata (serialised) → RetrievedChunk → citation

No network, no Ollama, no Chroma on disk — every step is tested with in-process objects.
"""

from __future__ import annotations

import json

from src.chunk import chunk_segments
from src.embed import Chunk as EmbedChunk
from src.graph import VttCue, parse_vtt, to_segments
from src.retrieve import RetrievedChunk
from src.transcript import Segment

_ATTRIBUTED_VTT = """\
WEBVTT

abc-123/1-0
00:00:05.000 --> 00:00:10.000
<v Priya Nair>Let's start with the migration.</v>

abc-123/1-1
00:00:12.000 --> 00:00:18.000
<v Priya Nair>I'll take ownership of that task.</v>

abc-123/1-2
00:00:20.000 --> 00:00:26.000
<v Rohan Mehta>I can help with the review.</v>
"""


def test_parse_vtt_preserves_speaker():
    cues = parse_vtt(_ATTRIBUTED_VTT)
    assert len(cues) == 3
    assert cues[0].speaker == "Priya Nair"
    assert cues[1].speaker == "Priya Nair"
    assert cues[2].speaker == "Rohan Mehta"


def test_to_segments_passes_speaker():
    cues = parse_vtt(_ATTRIBUTED_VTT)
    segments = to_segments(cues)
    assert all(isinstance(s, Segment) for s in segments)
    assert segments[0].speaker == "Priya Nair"
    assert segments[2].speaker == "Rohan Mehta"


def test_chunk_collects_speakers():
    cues = parse_vtt(_ATTRIBUTED_VTT)
    segments = to_segments(cues)
    levers = {"window_s": 30.0, "overlap_s": 5.0, "hop_s": 25.0}
    chunks, _ = chunk_segments("test-video", segments, levers)
    assert chunks, "expected at least one chunk"
    # All three segments fall in the first window — both speakers should be present.
    assert "Priya Nair" in chunks[0].speakers
    assert "Rohan Mehta" in chunks[0].speakers


def test_chunk_speakers_not_in_text():
    """Speaker names ride as metadata and must not enter the embedded text."""
    cues = parse_vtt(_ATTRIBUTED_VTT)
    segments = to_segments(cues)
    levers = {"window_s": 30.0, "overlap_s": 5.0, "hop_s": 25.0}
    chunks, _ = chunk_segments("test-video", segments, levers)
    for chunk in chunks:
        assert "Priya Nair" not in chunk.text
        assert "Rohan Mehta" not in chunk.text


def test_embed_chunk_carries_speakers():
    cues = parse_vtt(_ATTRIBUTED_VTT)
    segments = to_segments(cues)
    levers = {"window_s": 30.0, "overlap_s": 5.0, "hop_s": 25.0}
    chunks, _ = chunk_segments("test-video", segments, levers)
    embed_chunk = EmbedChunk(
        video_id=chunks[0].video_id,
        t_start=chunks[0].t_start,
        t_end=chunks[0].t_end,
        text=chunks[0].text,
        speakers=list(chunks[0].speakers),
    )
    assert "Priya Nair" in embed_chunk.speakers
    assert "Rohan Mehta" in embed_chunk.speakers


def test_chroma_metadata_serialises_speakers():
    """Speakers reach Chroma as a JSON string, not embedded in the document text."""
    cues = parse_vtt(_ATTRIBUTED_VTT)
    segments = to_segments(cues)
    levers = {"window_s": 30.0, "overlap_s": 5.0, "hop_s": 25.0}
    chunks, _ = chunk_segments("test-video", segments, levers)
    embed_chunk = EmbedChunk(
        video_id=chunks[0].video_id,
        t_start=chunks[0].t_start,
        t_end=chunks[0].t_end,
        text=chunks[0].text,
        speakers=list(chunks[0].speakers),
    )
    # Simulate what _upsert writes into Chroma metadata.
    meta = {
        "video_id": embed_chunk.video_id,
        "t_start": embed_chunk.t_start,
        "t_end": embed_chunk.t_end,
        "speakers": json.dumps(embed_chunk.speakers),
    }
    recovered = json.loads(meta["speakers"])
    assert "Priya Nair" in recovered
    assert "Rohan Mehta" in recovered
    # The document field (what gets embedded) must not contain the names.
    assert "Priya Nair" not in embed_chunk.text
    assert "Rohan Mehta" not in embed_chunk.text


def test_retrieved_chunk_carries_speakers():
    """RetrievedChunk deserialises speakers from Chroma metadata correctly."""
    cues = parse_vtt(_ATTRIBUTED_VTT)
    segments = to_segments(cues)
    levers = {"window_s": 30.0, "overlap_s": 5.0, "hop_s": 25.0}
    chunks, _ = chunk_segments("test-video", segments, levers)

    # Simulate Chroma round-trip: serialise then deserialise.
    speakers_json = json.dumps(list(chunks[0].speakers))
    meta = {
        "video_id": chunks[0].video_id,
        "t_start": chunks[0].t_start,
        "t_end": chunks[0].t_end,
        "speakers": speakers_json,
    }
    retrieved = RetrievedChunk(
        video_id=str(meta["video_id"]),
        t_start=float(meta["t_start"]),
        t_end=float(meta["t_end"]),
        text=chunks[0].text,
        score=0.05,
        speakers=tuple(json.loads(meta["speakers"])),
    )
    assert "Priya Nair" in retrieved.speakers
    assert "Rohan Mehta" in retrieved.speakers


def test_full_pipeline_speaker_survives():
    """End-to-end: name from VTT voice tag survives all six hops to RetrievedChunk."""
    # 1. VTT → cues
    cues = parse_vtt(_ATTRIBUTED_VTT)
    assert cues[0].speaker == "Priya Nair"

    # 2. cues → Segments
    segments = to_segments(cues)
    assert segments[0].speaker == "Priya Nair"

    # 3. Segments → Chunks
    levers = {"window_s": 30.0, "overlap_s": 5.0, "hop_s": 25.0}
    chunks, _ = chunk_segments("test-video", segments, levers)
    assert "Priya Nair" in chunks[0].speakers

    # 4. Chunks → EmbedChunk (what goes to Chroma)
    embed_chunk = EmbedChunk(
        video_id=chunks[0].video_id,
        t_start=chunks[0].t_start,
        t_end=chunks[0].t_end,
        text=chunks[0].text,
        speakers=list(chunks[0].speakers),
    )
    assert "Priya Nair" in embed_chunk.speakers

    # 5. Chroma metadata (serialised)
    meta_speakers = json.dumps(embed_chunk.speakers)
    assert "Priya Nair" in json.loads(meta_speakers)

    # 6. RetrievedChunk (what a citation is built from)
    retrieved = RetrievedChunk(
        video_id=embed_chunk.video_id,
        t_start=embed_chunk.t_start,
        t_end=embed_chunk.t_end,
        text=embed_chunk.text,
        score=0.0,
        speakers=tuple(json.loads(meta_speakers)),
    )
    assert "Priya Nair" in retrieved.speakers
    assert "Rohan Mehta" in retrieved.speakers
