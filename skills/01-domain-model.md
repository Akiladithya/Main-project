# Domain Model

## Purpose
Define the canonical concepts used throughout the system.

## Core Entities

### Video
Represents an uploaded lecture.

Suggested fields:
- `video_id`
- `filename`
- `subject`
- `duration`
- `source_uri`
- `processing_version`
- `created_at`

### SpeechSegment
Timestamped transcript evidence.

Fields:
- `segment_id`
- `video_id`
- `start_time`
- `end_time`
- `text`

### FrameEvidence
A visually selected frame.

Fields:
- `frame_id`
- `video_id`
- `timestamp`
- `image_uri`
- `selection_reason`
- `visual_change_score`

### VisualEvidence
Semantic interpretation of a selected frame.

Fields:
- `visual_id`
- `frame_id`
- `video_id`
- `timestamp`
- `caption`
- `entities`
- `concept_candidates`

### VisualEvent
A temporally grouped visual sequence.

Fields:
- `event_id`
- `video_id`
- `start_time`
- `end_time`
- `title`
- `description`
- `frame_ids`

### MultimodalConcept
The primary searchable semantic unit.

Fields:
- `concept_id`
- `video_id`
- `subject`
- `topic`
- `concept`
- `start_time`
- `end_time`
- `speech_text`
- `visual_description`
- `visual_event_id`
- `evidence_ids`
- `embedding`
- `schema_version`

## Relationships

```text
Video
 ├── SpeechSegments
 ├── FrameEvidence
 │    └── VisualEvidence
 │          └── VisualEvent
 └── MultimodalConcept
       ├── SpeechEvidence
       └── VisualEvidence
```

## Invariant
No semantic record should lose its link back to the original video and timestamp.
