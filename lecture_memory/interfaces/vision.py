from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from lecture_memory.domain.models import FrameEvidence, VisualEvidence


class IVisionModel(ABC):
    @abstractmethod
    def describe_frame(
        self,
        frame: FrameEvidence,
        context: str | None = None,
    ) -> VisualEvidence:
        """
        Produce a semantic VisualEvidence record from a single frame.
        The pipeline supplies timestamps — the model must not invent them.
        """
