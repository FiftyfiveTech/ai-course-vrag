# PRD — Track 3: VRAG (Enterprise Video RAG & Learning Intelligence)

**Project code:** `VRAG`
**Date:** 2026-07-28
**Status:** draft — **backlog** (Track 1 M2X is the active build)
**Owners:** TBD · **Shashank** (advisory)

← [[../context|Course Context]] · [[../meetings/2026-07-28-course-scope-alignment|scope call]] · active track: [[2026-07-28-track1-m2x-meeting-to-execution|M2X PRD]]

---

## 1. Product brief

**User.** L&D, engineering, product, sales enablement, support, onboarding — anyone sitting on
recorded knowledge nobody can search.

**Problem.** Training recordings, demos, and workshops are effectively write-only. Finding
"where was that deployment step explained?" means scrubbing an hour of video. Outdated
recordings keep teaching stale processes with no way to flag them.

**Inputs.** Training recordings, product demos, workshops, screen recordings, slides,
related documents.

**Outputs.** Semantic search + Q&A with **timestamp citations**; chapters; clips; quizzes;
learning playlists; freshness/conflict flags — all reviewable before publication.

**Workflow.**
`video upload → audio extract + transcribe → scene/slide/frame analysis → structured metadata
→ multimodal index → RAG with timestamps → learning tools → review + publish`

**Data boundaries.** Internal recordings only, permission-aware by collection. Consent and
retention rules defined before ingest. No client-recorded sessions in v1.

**Prohibited actions.** No auto-publishing and no auto-assigning learning content — every
generated artifact (clip, quiz, playlist, caption) requires reviewer approval. No claims
about individuals from video.

## 2. Scope decisions

**In scope** — a curated pilot corpus of 20–50 internal recordings from **one** domain
(recommend: onboarding or engineering workshops). Search, Q&A with timestamps, chapters,
clip generation, quiz generation.

**Out of scope (v1)** — video generation/dubbing/avatars, LMS integration, cross-org
permissions, real-time ingest, hundreds of hours of backlog.

**Infra note (constraint check).** Fits the local-only rule — transcription via hosted STT,
frame analysis via hosted vision, FFmpeg locally. **No GPU.** The real cost driver is
**storage + per-video-hour processing spend**, not hardware. Cap the pilot corpus.

## 3. Architecture (target)

```
video -> ffmpeg (audio + keyframes) -> transcribe + diarize
                                    -> scene/slide detect -> OCR
                          -> structured metadata (chapters, topics, entities)
                                    -> multimodal index
                    -> RAG w/ timestamp citations -> learning tools -> review -> publish
```

**Stack:** Python · FFmpeg · hosted STT + pyannote · hosted vision/OCR · Pydantic +
Instructor · Qdrant · sentence-transformers · RAGAS · LangGraph · Langfuse · Docker.

## 4. Phased plan (trimmed)

| Phase | Build | Exit gate | Est. |
|---|---|---|---|
| **0 Video literacy** | Upload one video → extract audio + frames → transcript → summary; hosted vs local text model | One command yields transcript, summary, frame samples, media metadata, latency, cost | 2–3 d |
| **1 Processing playground** | Compare segmentation, diarization, scene/slide detection, OCR, frame-sampling rates, chaptering | Stable chapters with aligned transcript ranges, speakers, representative frames across ≥5 varied videos | 4–5 d |
| **1B Metadata prompts + eval** | Versioned prompts extracting chapters, topics, procedures, entities, skill tags, learning objectives → validated schemas | Schema-valid on 100% of test set; ≥0.85 agreement with human-labelled topics/chapters on 20 segments | 4–5 d |
| **2 Multimodal video RAG** | Search + Q&A over transcript chunks, frame descriptions, slide OCR, speaker metadata | 40-question set: correct video + time range for ≥80%; faithfulness ≥0.80; permission filtering passes all access tests | 1 wk |
| **3 Learning tools** | Clip creation, chapter export, quiz generation, caption export — as MCP tools, async jobs | Chains ≥3 media/learning tools; artifacts reproducible; approval required before publish/assign | 1 wk |
| **4 Curation + freshness** | Planner-executor that researches across recordings, builds a curated learning brief, critiques coverage, flags outdated/conflicting instructions | ≥15% improvement in evidence coverage + conflict detection vs single-shot on 10 research tasks | 1 wk |
| **5 Publishing derivatives** *(optional)* | Narrated clips, translated captions from **approved** source only, with visible provenance | Derivatives pass a factuality/terminology/traceability rubric; none publishable without approval | 3–4 d |
| **6 Capstone** | Deployed role-based platform: ingest, search, Q&A, clips, review, analytics, retention, tracing | Passes the universal gate (see M2X §6) | 1 wk |

**Total: ~6–7 weeks part-time** (Phase 5 optional — drop first if trimming further).

## 5. Pilot

One internal video collection (20–50 recordings, single domain). Measure: search time saved
vs manual scrubbing, unanswered-question rate, cost per video hour, most-reused knowledge.

## 6. Open questions

1. Which corpus — onboarding, engineering workshops, or product demos?
2. Where do the videos live and who controls access?
3. Retention + consent policy for internal recordings — who signs off?
4. Keep Phase 5 (generated derivatives) or cut it from v1?
