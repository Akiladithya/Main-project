# Multimodal Concept Memory Skill

## Responsibility
Convert aligned evidence into a structured searchable representation.

## Hierarchy

```text
Subject
  -> Topic
      -> Concept
          -> Event
              -> Evidence
```

## Example

```text
Computer Networks
 -> Transport Layer
    -> TCP
       -> Three-Way Handshake
          -> Speech + Visual + Timestamp
```

## Canonical Record

```json
{
  "video_id": "CN_01",
  "subject": "Computer Networks",
  "topic": "Transport Layer",
  "concept": "TCP Three-Way Handshake",
  "start_time": 250,
  "end_time": 270,
  "speech_text": "...",
  "visual_description": "...",
  "visual_event": "...",
  "evidence_ids": [],
  "embedding": []
}
```

## Design Rules
- Store semantic knowledge, not raw video bytes.
- Keep source references.
- Preserve timestamps.
- Keep speech and visual evidence separately identifiable.
- Generate embeddings from a stable searchable representation.
- Version the schema.

## Central Memory
All processed CS lecture concepts may be stored in one central vector memory.

Filtering metadata should support:
- subject
- topic
- video
- concept
- time range
