from abc import ABC, abstractmethod
from typing import List
from lecture_memory.domain.models import FrameEvidence


class IFrameSelector(ABC):
    @abstractmethod
    def select_frames(
        self,
        video_id: str,
        video_path: str,
        output_dir: str,
        interval_sec: float = 4.0,
    ) -> List[FrameEvidence]:
        """
        Sample and select visually meaningful frames.
        Returns FrameEvidence list in temporal order.
        """
