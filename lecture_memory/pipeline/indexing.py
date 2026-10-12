"""
pipeline/indexing.py
--------------------
Responsibility: embed MultimodalConcepts and persist to vector store.
"""
from __future__ import annotations

from typing import List

from lecture_memory.domain.models import MultimodalConcept
from lecture_memory.interfaces.embedder import IEmbedder
from lecture_memory.interfaces.vector_store import IVectorStore
from lecture_memory.observability.logger import get_logger, stage_timer

logger = get_logger(__name__)


class MemoryIndexingPipeline:
    def __init__(self, embedder: IEmbedder, vector_store: IVectorStore) -> None:
        self._embedder = embedder
        self._store = vector_store

    def index(self, concepts: List[MultimodalConcept]) -> None:
        with stage_timer(logger, "memory_indexing", count=len(concepts)):
            texts = [self._searchable_text(c) for c in concepts]
            embeddings = self._embedder.embed_batch(texts)

            for concept, embedding in zip(concepts, embeddings):
                concept.embedding = embedding

            self._store.upsert(concepts)
            logger.info(f"indexed | count={len(concepts)} | model={self._embedder.model_name}")

    def _searchable_text(self, concept: MultimodalConcept) -> str:
        """Build the text representation used for embedding."""
        parts = [concept.concept]
        if concept.speech_text:
            parts.append(concept.speech_text)
        if concept.visual_description:
            parts.append(concept.visual_description)
        return " ".join(parts)
