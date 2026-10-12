from abc import ABC, abstractmethod
from typing import List
from lecture_memory.domain.models import GeneratedAnswer, RetrievedEvidence, VerificationResult


class IEvidenceVerifier(ABC):
    @abstractmethod
    def verify(
        self,
        answer: GeneratedAnswer,
        evidence: List[RetrievedEvidence],
    ) -> VerificationResult:
        """
        Extract claims from the answer and check each against retrieved evidence.
        Unsupported claims must not be silently accepted.
        """
