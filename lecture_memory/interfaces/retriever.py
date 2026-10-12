from abc import ABC, abstractmethod
from typing import List, Optional
from lecture_memory.domain.models import RetrievedEvidence


class IRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        subject: Optional[str] = None,
        video_id: Optional[str] = None,
        expand_context_sec: float = 30.0,
    ) -> List[RetrievedEvidence]:
        """
        Multi-stage retrieval:
        1. embed query
        2. semantic search
        3. temporal expansion
        4. cross-modal re-ranking
        Returns ranked RetrievedEvidence — not a final answer.
        """
