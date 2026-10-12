# Skills Index

This folder defines the modular engineering rules for the multimodal CS lecture memory system.

## Recommended Reading Order

1. `00-system-design-principles.md`
2. `01-domain-model.md`
3. `02-video-ingestion.md`
4. `03-audio-processing.md`
5. `04-visual-frame-selection.md`
6. `05-visual-understanding.md`
7. `06-visual-event-formation.md`
8. `07-cross-modal-alignment.md`
9. `08-multimodal-concept-memory.md`
10. `09-vector-store.md`
11. `10-temporal-rag.md`
12. `11-answer-generation.md`
13. `12-evidence-verification.md`
14. `13-pipeline-orchestration.md`
15. `14-evaluation.md`
16. `15-security-and-data-handling.md`

## Architectural Boundary

```text
                ┌──────────────────────────┐
                │       APPLICATION        │
                └────────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │       ORCHESTRATION      │
                └────────────┬─────────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼
   INGESTION              PROCESSING           QUESTION QA
       │                     │                     │
       │            ┌────────┴────────┐            │
       │            │                 │            │
       ▼            ▼                 ▼            ▼
     VIDEO        AUDIO            VISUAL       RETRIEVAL
       │            │                 │            │
       │            └───────┬─────────┘            │
       │                    ▼                      │
       │             MULTIMODAL MEMORY            │
       │                    │                      │
       └────────────────────┼──────────────────────┘
                            ▼
                     VECTOR MEMORY
```

## Non-Negotiable Architecture Rules

- No fine-tuning is required in the current design.
- Qwen2.5-VL is a replaceable visual-understanding component.
- The vector database stores searchable semantic memory, not raw video.
- Timestamps remain attached to evidence.
- Retrieval returns evidence before generation.
- Generation and verification remain separate.
- Unsupported claims trigger additional retrieval or are rejected.
- The system should optimize for efficient visual processing, multimodal memory, temporal retrieval, and grounded QA.
