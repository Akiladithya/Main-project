# Visual Frame Selection Skill

## Responsibility
Reduce thousands of video frames to visually meaningful candidates before expensive visual understanding.

## Principle
Do not send every frame to the vision model.

## Workflow

```text
Video
 -> Initial sampling
 -> Candidate frames
 -> Consecutive-frame comparison
 -> Change detection
 -> Important frames
```

## Sampling
Use an experimentally configurable sampling interval.

Do not hard-code a universal interval as an architectural assumption.

## Change Detection
Possible prototype methods:
- image difference
- histogram difference
- SSIM
- feature-level similarity

The selector should expose a common interface so methods can be compared.

## Selection Output
Each selected frame should preserve:
- frame ID
- video ID
- timestamp
- image location
- selection score/reason

## Design Rules
- Prefer cheap computation before expensive model calls.
- Avoid selecting near-identical frames.
- Do not assume every visual change represents a new concept.
- Preserve temporal ordering.

## Research Metrics
Track:
- frame reduction ratio
- visual coverage
- missed meaningful transitions
- redundant selections
- downstream QA impact
