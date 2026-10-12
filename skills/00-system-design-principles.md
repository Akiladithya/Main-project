# System Design Principles

## Purpose
Define the engineering principles that govern the multimodal CS lecture memory system.

## Core Principles

### 1. Single Responsibility
Each pipeline stage owns one concern:
- ingestion owns video intake and metadata
- audio processing owns timestamped speech
- visual processing owns frame selection
- visual understanding owns visual interpretation
- alignment owns speech-visual association
- memory owns persistence and indexing
- retrieval owns evidence discovery
- generation owns answer synthesis
- verification owns claim support checking

Avoid putting unrelated responsibilities into one module.

### 2. Separation of Concerns
Keep these layers independent:
1. raw media
2. extracted evidence
3. semantic representation
4. retrieval
5. answer generation
6. verification

A change in one layer should not require rewriting the others.

### 3. Pipeline Composition
Prefer small stages with explicit inputs and outputs.

```text
Video
 -> Audio Evidence
 -> Visual Evidence
 -> Multimodal Concepts
 -> Memory
 -> Retrieval
 -> Grounded Answer
 -> Verification
```

Each stage should be testable independently.

### 4. Dependency Inversion
Business logic must not depend directly on a particular model or database.

Use interfaces for:
- speech-to-text
- vision model
- embedding model
- vector store
- LLM
- verifier

Qwen2.5-VL is an implementation choice, not a system-level dependency.

### 5. Model Agnosticism
The architecture must allow pretrained models to be replaced without changing the data model or pipeline contracts.

### 6. Evidence First
Every generated answer should be traceable to stored evidence:
- speech
- visual description
- visual event
- timestamp
- video identifier

### 7. Temporal Grounding
Time is a first-class property, not optional metadata.

Every evidence unit should preserve:
- start time
- end time
- source video

### 8. Idempotency
Re-running a stage on the same input should not create uncontrolled duplicates.

Use stable identifiers and processing versions.

### 9. Observability
Record:
- processing duration
- model used
- model version
- confidence/relevance scores
- selected frames
- retrieval results
- verification outcomes
- failures

### 10. Fail Safely
If evidence is insufficient, do not invent an answer.
Return an uncertainty state or trigger additional retrieval.

## Design Goal
The system should behave as a reusable multimodal memory engine rather than a one-off video summarizer.
