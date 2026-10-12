from abc import ABC, abstractmethod
from typing import List, Optional

from lecture_memory.domain.models import (
    SpeechSegment,
    VisualEvidence,
    VisualEvent,
)


class IVisualEventFormer(ABC):
    @abstractmethod
    def form_events(
        self,
        video_id: str,
        visual_evidence: List[VisualEvidence],
        speech_segments: Optional[List[SpeechSegment]] = None,
    ) -> List[VisualEvent]:
        """
        Form temporal multimodal events by considering:

        - visual evidence
        - speech segments
        - temporal continuity

        An event represents a coherent episode in the lecture.

        A visual change alone should not necessarily create a new event
        if the lecturer is continuing the same explanation.

        speech_segments is optional so that existing visual-only callers
        remain compatible with the interface.
        """
        pass