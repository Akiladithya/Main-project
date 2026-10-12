from abc import ABC, abstractmethod
from typing import List, Optional
from lecture_memory.domain.models import MultimodalConcept


class IVectorStore(ABC):
    @abstractmethod
    def upsert(self, concepts: List[MultimodalConcept]) -> None:
        """Store or update concepts. Idempotent on concept_id."""

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        video_id: Optional[str] = None,
    ) -> List[MultimodalConcept]:
        """Semantic search with optional metadata filters."""

    @abstractmethod
    def get_by_video(self, video_id: str) -> List[MultimodalConcept]:
        """Return all concepts for a given video."""

    @abstractmethod
    def get_by_time_range(
        self,
        video_id: str,
        start_time: float,
        end_time: float,
    ) -> List[MultimodalConcept]:
        """Temporal filter — used for context expansion."""

    @abstractmethod
    def delete_by_video(self, video_id: str) -> None:
        """Remove all concepts for a video (for data lifecycle management)."""
