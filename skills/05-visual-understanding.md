# Visual Understanding Skill

## Responsibility
Convert important frames into meaningful visual evidence.

## Model
Use a pretrained vision-language model such as Qwen2.5-VL.

No fine-tuning is required by the current architecture.

## Input
- selected frame
- timestamp
- surrounding context when available

## Output
A `VisualEvidence` record.

## Desired Output
The model should describe semantic content, not merely image presence.

Bad:
> A diagram is displayed.

Good:
> The diagram illustrates the TCP three-way handshake with SYN, SYN-ACK and ACK exchanged between a client and server.

## Prompting Rules
Ask the model to identify:
- diagrams
- code
- tables
- formulas
- labels
- entities
- processes
- state changes
- visible relationships
- likely CS concepts

Do not ask it to invent information that is not visible.

## Constraints
The model must not become responsible for:
- database storage
- retrieval
- final answer verification
- authoritative timestamp assignment

The pipeline supplies timestamps from the source frame.

## Fallback
If visual understanding fails, preserve the frame as unprocessed evidence and continue where possible.
