from abc import ABC, abstractmethod
from typing import List
from lecture_memory.domain.models import SpeechSegment, VisualEvidence, VisualEvent, MultimodalConcept


class ICrossModalAligner(ABC):
    @abstractmethod
    def align(
        self,
        video_id: str,
        speech_segments: List[SpeechSegment],
        visual_evidence: List[VisualEvidence],
        visual_events: List[VisualEvent],
    ) -> List[MultimodalConcept]:
        """
        Associate speech ↔ visual evidence by temporal overlap + semantic similarity.
        Returns MultimodalConcept records (without embeddings — embedder fills those).
        """
