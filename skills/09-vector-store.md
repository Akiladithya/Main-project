# Vector Memory Skill

## Responsibility
Persist and retrieve multimodal semantic representations.

## Storage Separation

```text
Original Video
 -> File/Object Storage

Processed Knowledge
 -> Vector Database
```

The vector database is not the video repository.

## Stored Information
A vector record should contain:
- embedding
- video ID
- subject
- topic
- concept
- speech text
- visual description
- visual event
- timestamps
- evidence references
- schema/model versions

## Retrieval Interfaces
Provide operations for:
- semantic search
- metadata filtering
- concept lookup
- video lookup
- temporal filtering
- evidence retrieval

## Design Principle
The database implementation must be replaceable.

Do not couple domain logic directly to a specific vector database vendor.

## Consistency
A searchable record must not point to missing source evidence.

## Versioning
Track:
- embedding model version
- representation version
- processing version
- schema version
