# Temporal RAG Skill

## Responsibility
Retrieve evidence from multimodal lecture memory while preserving temporal context.

## Query Flow

```text
Student Question
 -> Query Understanding
 -> Query Embedding
 -> Candidate Retrieval
 -> Temporal Expansion
 -> Cross-Modal Re-ranking
 -> Evidence Set
```

## Multi-Stage Retrieval

Use progressive narrowing where useful:

```text
Question
 -> Subject
 -> Topic
 -> Concept
 -> Supporting Evidence
```

Do not require exact metadata classification when semantic retrieval is sufficient.

## Temporal Expansion
If retrieval identifies a point such as `04:20`, expand around the evidence to recover the complete explanation.

```text
04:10 ---- 04:20 ---- 04:30
          match
       <- context ->
```

Expansion width should be configurable and evaluated empirically.

## Cross-Modal Re-ranking
Candidate score should consider:
- semantic relevance
- speech relevance
- visual relevance
- temporal relevance

## Output
Return a ranked evidence set containing:
- evidence text
- visual description
- video ID
- start/end time
- relevance scores
- source references

## Rule
Retrieval should return evidence, not a final answer.
