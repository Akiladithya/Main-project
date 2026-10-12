"""
pipeline/alignment.py
---------------------
Responsibility: associate speech ↔ visual evidence into MultimodalConcepts.
"""
from __future__ import annotations

from typing import List

from lecture_memory.domain.models import (
    Video, SpeechSegment, VisualEvidence, VisualEvent, MultimodalConcept,
)
from lecture_memory.interfaces.aligner import ICrossModalAligner
from lecture_memory.observability.logger import get_logger, stage_timer

logger = get_logger(__name__)


class CrossModalAlignmentPipeline:
    def __init__(self, aligner: ICrossModalAligner) -> None:
        self._aligner = aligner

    def process(
        self,
        video: Video,
        speech_segments: List[SpeechSegment],
        visual_evidence: List[VisualEvidence],
        visual_events: List[VisualEvent],
    ) -> List[MultimodalConcept]:
        with stage_timer(logger, "cross_modal_alignment", video_id=video.video_id):
            concepts = self._aligner.align(
                video_id=video.video_id,
                speech_segments=speech_segments,
                visual_evidence=visual_evidence,
                visual_events=visual_events,
            )
            logger.info(
                f"concepts_aligned | video_id={video.video_id} | count={len(concepts)}"
            )
            return concepts
