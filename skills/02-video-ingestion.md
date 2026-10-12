# Video Ingestion Skill

## Responsibility
Accept a CS lecture video and create the canonical processing job.

## Input
- video file
- optional subject
- optional lecture metadata

## Output
A validated `Video` entity and processing job.

## Workflow

```text
Upload
 -> Validate
 -> Extract metadata
 -> Register video
 -> Create processing job
 -> Dispatch audio + visual pipelines
```

## Requirements
- Validate supported video formats.
- Record duration and basic media metadata.
- Generate a stable `video_id`.
- Never place the entire video inside the vector database.
- Store the original video in file/object storage.
- Preserve the source URI.

## Design Rules
- Ingestion must not perform semantic understanding.
- Ingestion must be restartable.
- Duplicate uploads should be detectable.
- Processing should be asynchronous for long videos.

## Failure Handling
If metadata extraction fails:
- mark the job as failed
- preserve the error
- do not create partial searchable memory.
