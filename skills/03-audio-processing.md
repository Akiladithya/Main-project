# Audio Processing Skill

## Responsibility
Convert lecture speech into timestamped speech evidence.

## Input
- registered video
- audio stream

## Output
Ordered `SpeechSegment` records.

## Workflow

```text
Video
 -> Audio extraction
 -> Speech-to-text
 -> Timestamp preservation
 -> Segment normalization
 -> Speech evidence
```

## Requirements
Use a speech-to-text system such as Whisper.

Every segment must contain:
- text
- start timestamp
- end timestamp
- video ID

## Important Rule
Never reduce the transcript to plain text only. Timestamp information is required for temporal retrieval.

## Normalization
Normalize:
- obvious transcription artifacts
- empty segments
- invalid timestamps

Do not rewrite the lecturer's meaning.

## Quality Checks
Check:
- timestamps are ordered
- end time >= start time
- segments belong to the correct video
- empty speech is handled safely

## Output Example

```json
{
  "video_id": "CN_01",
  "start_time": 250.0,
  "end_time": 275.0,
  "text": "TCP establishes a connection using a three-way handshake."
}
```
