# Evaluation Skill

## Purpose
Evaluate whether the proposed architecture actually improves multimodal lecture QA.

## Pipeline Metrics

### Visual efficiency
- raw frames
- candidate frames
- selected frames
- frame reduction ratio
- visual coverage

### Alignment quality
- speech-visual alignment precision
- alignment recall
- temporal overlap quality

### Retrieval quality
- Recall@K
- Precision@K
- MRR where applicable
- timestamp retrieval accuracy

### Answer quality
- answer correctness
- groundedness
- citation/timestamp correctness
- unsupported claim rate

### Verification quality
- supported-claim precision
- unsupported-claim detection
- false rejection rate

## Core Experimental Questions

1. Does visual change detection reduce expensive vision-model calls?
2. Does visual evidence improve QA over transcript-only retrieval?
3. Does visual-event grouping improve retrieval?
4. Does temporal expansion improve answer completeness?
5. Does cross-modal re-ranking improve evidence relevance?
6. Does verification reduce unsupported claims?

## Baselines

At minimum compare against:

```text
Transcript-only RAG
vs
Multimodal Memory RAG
```

Where possible also compare:
- all-frame visual processing
- sampled-only visual processing
- change-detection visual processing

## Rule
Do not claim novelty from using an existing model alone. Evaluate the complete pipeline contribution.
