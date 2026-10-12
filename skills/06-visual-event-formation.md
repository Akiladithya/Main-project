# Visual Event Formation Skill

## Responsibility
Group related visual evidence into meaningful temporal events.

## Problem
One concept may span multiple selected frames.

Example:

```text
04:15 -> SYN
04:20 -> SYN-ACK
04:25 -> ACK
```

These should form one event:

```text
04:15 - 04:25
TCP Three-Way Handshake
```

## Inputs
- ordered visual evidence
- timestamps
- visual semantic descriptions

## Grouping Signals
Use:
- temporal proximity
- semantic similarity
- continuity of visual state
- concept similarity

## Output
A `VisualEvent` with:
- start time
- end time
- title
- description
- supporting frame IDs

## Design Rules
- Do not merge unrelated concepts only because they are close in time.
- Do not split an obviously continuous visual process into unnecessary events.
- Preserve all supporting frame references.

## Quality Principle
Event formation should improve retrieval granularity without destroying evidence-level detail.
