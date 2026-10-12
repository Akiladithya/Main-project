# Grounded Answer Generation Skill

## Responsibility
Generate an answer from retrieved evidence.

## Input

```text
Question
+
Retrieved speech
+
Retrieved visual evidence/events
+
Timestamps
```

## Principle
The LLM reasons over retrieved evidence instead of attempting to understand the entire video from scratch.

## Output
Prefer:

```text
Answer
+
Timestamp
+
Short explanation/summary
```

## Grounding Rules
- Use retrieved evidence as the factual basis.
- Do not invent details absent from evidence.
- Preserve uncertainty where evidence is insufficient.
- Do not fabricate timestamps.
- Do not cite a timestamp that was not associated with retrieved evidence.

## Answer Construction
For educational questions:
1. answer directly
2. explain using evidence
3. provide relevant lecture timestamp
4. provide a short description of what appears there

## Separation
Answer generation must not perform the authoritative verification step itself.
