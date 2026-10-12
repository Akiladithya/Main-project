"""
adapters/sentence_embedder.py
------------------------------
Implements IEmbedder using sentence-transformers (local, no API cost).
Replaceable with any other embedding model.
"""
from __future__ import annotations

from typing import List

from lecture_memory.interfaces.embedder import IEmbedder
from lecture_memory.observability.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceEmbedder(IEmbedder):
    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None  # lazy load

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"embedder_loaded | model={self._model_name}")

    def embed(self, text: str) -> List[float]:
        self._load()
        return self._model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._load()
        return self._model.encode(texts, convert_to_numpy=True).tolist()
