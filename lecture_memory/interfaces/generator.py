from abc import ABC, abstractmethod
from typing import List
from lecture_memory.domain.models import RetrievedEvidence, GeneratedAnswer


class IAnswerGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        question: str,
        evidence: List[RetrievedEvidence],
    ) -> GeneratedAnswer:
        """
        Reason over retrieved evidence to produce a grounded answer.
        Must not invent timestamps or facts absent from evidence.
        """
