from abc import ABC, abstractmethod
from typing import List
from lecture_memory.domain.models import SpeechSegment


class ISpeechTranscriber(ABC):
    @abstractmethod
    def transcribe(self, video_id: str, audio_path: str) -> List[SpeechSegment]:
        """Return ordered, timestamped speech segments for the given audio."""
