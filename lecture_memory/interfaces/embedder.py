from abc import ABC, abstractmethod
from typing import List


class IEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Return a dense vector for the given text."""

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return dense vectors for a batch of texts."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the underlying embedding model."""
