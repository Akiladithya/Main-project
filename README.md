# Multimodal CS Lecture Memory System

A searchable multimodal memory engine for Computer Science lecture videos.
Students ask questions in natural language and receive grounded answers with lecture timestamps.

## Pipeline

```
Input Audio/Video
       │
       ├─── Audio Processing Service
       │         │
       │    Speech-to-Text (Whisper)
       │         │
       │    Speech Segments (timestamped)
       │         │
       │         └──────────────────────┐
       │                                │
       └─── Video Processing Service    │
                 │                      │
            Frame Sampler               │
                 │                      │
            Visual Change Detection     │
                 │                      │
            Important Frames            │
                 │                      │
            Visual Description Model    │
            (Gemini / Qwen2.5-VL)       │
                 │                      │
            Visual Events               │
                 │                      │
                 └──────────────────────┘
                                        │
                                        ↓
                           Cross-Modal Alignment
                                        │
                                        ↓
                           Multimodal Concepts
                                        │
                                        ↓
                           Adaptive Segmentation
                           (dynamic, content-aware)
                                        │
                                        ↓
                           Hierarchical Representation
                           (multi-level abstraction)
                                        │
                                        ↓
                           Vector Database (ChromaDB)
                                        │
                          ┌─────────────┘
                          │
                     User Query
                          │
                     Query Embedding
                          │
                          ↓
                   Multi-Stage Retrieval
                   ├── Stage 1: Topic Retrieval
                   ├── Stage 2: Concept Retrieval
                   └── Stage 3: Evidence Retrieval
                          │
                          ↓
                   Temporal Evidence Expansion
                          │
                          ↓
                   Cross-Modal Re-ranking
                          │
                          ↓
                   LLM Answer Generation
                          │
                          ↓
                   Evidence Verification
                          │
              ┌───────────┴───────────┐
              │                       │
         Supported               Not Supported
              │                       │
              ↓                  Re-retrieval
    Answer + Timestamp    (back to Multi-Stage Retrieval)
         + Summary
```

## Architecture

```
lecture_memory/
├── domain/          # Data contracts only — no logic
├── interfaces/      # Abstract contracts for every replaceable component
├── pipeline/        # One file per pipeline stage
├── orchestrator/    # Coordinates stages, tracks job state
├── adapters/        # Concrete implementations (all replaceable)
├── config/          # All configuration in one place
├── observability/   # Structured logging, stage timing
├── storage/         # File-based media store
├── app/             # Thin Streamlit UI
└── tests/           # Unit + integration tests
```

## Design Principles

- **Single Responsibility** — each module owns one concern
- **Dependency Inversion** — pipelines depend on interfaces, not adapters
- **Temporal Grounding** — every evidence unit carries start/end timestamps
- **Evidence First** — answers are traceable to stored speech/visual evidence
- **Adaptive Segmentation** — content boundaries are detected dynamically, not by fixed intervals
- **Hierarchical Memory** — concepts are indexed at topic, concept, and evidence granularity
- **Replaceable Models** — swap Gemini for Qwen2.5-VL by changing one adapter file
- **Idempotency** — re-running a stage on the same input is safe
- **Observability** — every stage logs duration, model, scores, failures

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Set GEMINI_API_KEY in .env
pip install -e .
```

## Run

```bash
streamlit run lecture_memory/app/streamlit_app.py
```

## Test

```bash
pytest lecture_memory/tests/
```

## Replacing a Component

| Component | Current Adapter | Replace With |
|---|---|---|
| Vision model | `adapters/gemini_vision.py` | `adapters/qwen_vision.py` |
| LLM | `adapters/gemini_generator.py` | any `IAnswerGenerator` impl |
| Vector store | `adapters/chroma_vector_store.py` | `adapters/qdrant_vector_store.py` |
| Embedder | `adapters/sentence_embedder.py` | any `IEmbedder` impl |
| Transcriber | `adapters/whisper_transcriber.py` | any `ISpeechTranscriber` impl |

No pipeline or orchestrator code changes required when swapping adapters.

## Key Invariants

- No semantic record loses its link to the original video and timestamp
- The vector database stores knowledge, not raw video bytes
- Retrieval is three-stage: topic → concept → evidence
- Temporal expansion and cross-modal re-ranking run before every generation call
- Unsupported claims trigger re-retrieval; the full retrieval-generation loop repeats
- Visual processing runs cheap frame selection before expensive model calls
