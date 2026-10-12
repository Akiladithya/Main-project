# Pipeline Orchestration Skill

## Responsibility
Coordinate the complete processing and QA workflow without embedding domain logic into the orchestrator.

## Ingestion Pipeline

```text
Video
 |
 +--> Audio Processing
 |      -> Speech Segments
 |
 +--> Visual Processing
        -> Selected Frames
        -> Visual Evidence
        -> Visual Events
 |
 -> Cross-Modal Alignment
 -> Multimodal Concepts
 -> Embeddings
 -> Vector Memory
```

## Question Pipeline

```text
Question
 -> Query Embedding
 -> Retrieval
 -> Temporal Expansion
 -> Cross-Modal Re-ranking
 -> Answer Generation
 -> Claim Verification
      |
      +--> supported -> final response
      |
      +--> unsupported -> retrieve again
```

## Orchestrator Rules
- Coordinate; do not implement every algorithm.
- Support retries at stage boundaries.
- Preserve intermediate artifacts.
- Record stage status.
- Make processing resumable.
- Avoid recomputing successful stages.

## State Model

```text
PENDING
 -> RUNNING
 -> COMPLETED

RUNNING
 -> FAILED
 -> RETRYING
 -> COMPLETED
```

## Idempotency
Each stage should be safely rerunnable using stable identifiers and processing versions.
