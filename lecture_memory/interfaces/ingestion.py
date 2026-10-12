from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from lecture_memory.domain.models import Video


class IVideoIngester(ABC):
    @abstractmethod
    def ingest(self, file_path: str, subject: str | None = None) -> Video:
        """Validate, extract metadata, register and return a Video entity."""
