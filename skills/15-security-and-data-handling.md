# Security and Data Handling Skill

## Responsibility
Define safe handling of uploaded lecture media and derived evidence.

## Principles
- Treat uploaded videos as untrusted input.
- Validate media formats before processing.
- Isolate media processing where appropriate.
- Avoid exposing unrelated lecture content during retrieval.
- Apply access control to videos and derived records.
- Do not place raw media in vector metadata.

## Evidence Access
A retrieved concept must only expose evidence the requesting user is authorized to access.

## Data Lifecycle
Track:
- original video
- extracted audio/transcript
- selected frames
- visual evidence
- vector records
- deletion state

Deleting a lecture should have a defined policy for its derived artifacts.

## Logging
Logs should avoid unnecessarily storing sensitive transcript content.

## Model Calls
Do not send data to external model providers unless the deployment and data policy permit it.
