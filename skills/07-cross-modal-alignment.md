# Cross-Modal Alignment Skill

## Responsibility
Associate timestamped speech evidence with related visual evidence.

## Inputs
- speech segments
- visual events
- visual evidence

## Signals

### Temporal overlap
Evidence occurring in overlapping time windows receives higher alignment priority.

### Semantic similarity
Compare speech meaning with visual descriptions/concept candidates.

### Concept similarity
Determine whether both modalities refer to the same CS concept.

## Conceptual Score

```text
AlignmentScore =
    temporal_score
  + semantic_score
  + concept_score
```

The exact weighting must be configurable and experimentally evaluated.

## Output
A multimodal association connecting:
- speech evidence
- visual evidence/event
- shared concept
- time range
- alignment score

## Design Rules
Alignment must be many-to-many capable.

One speech segment may explain several visuals, and one visual event may correspond to several speech segments.
