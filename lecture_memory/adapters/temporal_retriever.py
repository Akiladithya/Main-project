
"""
adapters/temporal_retriever.py

------------------------------

Spatiotemporal retrieval over multimodal lecture concepts.

Retrieval flow:

1. Semantic search finds strong seed concepts.

2. Temporal expansion gathers nearby concepts around the seeds.

3. Expanded concepts are evaluated using semantic similarity
   as the primary relevance signal.

4. Speech overlap, visual overlap, and temporal coherence are
   retained as supporting retrieval metadata.

5. Final results are ordered by semantic relevance.

The temporal expansion preserves nearby lecture context,
while semantic similarity remains the main signal for deciding
which evidence is relevant to the student's question.
"""

from __future__ import annotations

from typing import List, Optional

from lecture_memory.domain.models import (
    MultimodalConcept,
    RetrievedEvidence,
)
from lecture_memory.interfaces.embedder import IEmbedder
from lecture_memory.interfaces.vector_store import IVectorStore
from lecture_memory.observability.logger import get_logger


logger = get_logger(__name__)


class TemporalRetriever:
    def __init__(
        self,
        embedder: IEmbedder,
        vector_store: IVectorStore,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        subject: Optional[str] = None,
        video_id: Optional[str] = None,
        expand_context_sec: float = 30.0,
    ) -> List[RetrievedEvidence]:
        # ---------------------------------------------------------
        # 1. Embed query
        # ---------------------------------------------------------
        query_embedding = self._embedder.embed(query)

        # ---------------------------------------------------------
        # 2. Semantic seed retrieval
        # ---------------------------------------------------------
        candidates = self._store.search(
            query_embedding,
            top_k=top_k,
            subject=subject,
            video_id=video_id,
        )

        if not candidates:
            logger.info(
                f"retrieval_complete | "
                f"query={query} | "
                f"seed_count=0 | "
                f"expansion_seed_count=0 | "
                f"expanded_count=0 | "
                f"returned=0"
            )
            return []

        # ---------------------------------------------------------
        # 3. Use semantic search results as expansion seeds
        # ---------------------------------------------------------
        #
        # The vector store has already returned concepts ordered by
        # semantic relevance. We therefore preserve that ordering
        # instead of introducing another arbitrary seed-ranking rule.
        #
        expansion_seeds = candidates

        # ---------------------------------------------------------
        # 4. Temporal expansion
        # ---------------------------------------------------------
        expanded_concepts: List[MultimodalConcept] = []
        seen_ids = set()

        for seed_concept in expansion_seeds:
            start_time = max(
                0.0,
                seed_concept.start_time - expand_context_sec,
            )

            end_time = (
                seed_concept.end_time
                + expand_context_sec
            )

            nearby = self._store.get_by_time_range(
                seed_concept.video_id,
                start_time,
                end_time,
            )

            for concept in nearby:
                if concept.concept_id in seen_ids:
                    continue

                seen_ids.add(concept.concept_id)
                expanded_concepts.append(concept)

        # ---------------------------------------------------------
        # 5. Evaluate expanded concepts
        # ---------------------------------------------------------
        ranked = self._rerank(
            query=query,
            query_embedding=query_embedding,
            concepts=expanded_concepts,
        )

        # ---------------------------------------------------------
        # 6. Final top-k cutoff
        # ---------------------------------------------------------
        #
        # Semantic relevance is used for the final ordering so the
        # most relevant concepts appear first.
        # ---------------------------------------------------------
        final_results = ranked[:top_k]

        logger.info(
            f"retrieval_complete | "
            f"query={query} | "
            f"seed_count={len(candidates)} | "
            f"expansion_seed_count={len(expansion_seeds)} | "
            f"expanded_count={len(expanded_concepts)} | "
            f"returned={len(final_results)}"
        )

        return final_results

    # =============================================================
    # RERANKING
    # =============================================================

    def _rerank(
        self,
        query: str,
        query_embedding: List[float],
        concepts: List[MultimodalConcept],
    ) -> List[RetrievedEvidence]:
        """
        Evaluate retrieved concepts.

        Semantic similarity is the primary relevance signal.

        Speech overlap, visual overlap, and temporal coherence are
        retained as descriptive signals for downstream components
        and diagnostics, but they do not receive manually selected
        weights that can override semantic relevance.
        """

        results: List[RetrievedEvidence] = []

        query_tokens = self._tokenize(query)

        for concept in concepts:
            # -----------------------------------------------------
            # Semantic similarity
            # -----------------------------------------------------
            semantic_score = self._semantic_similarity(
                query_embedding,
                concept.embedding,
            )

            # -----------------------------------------------------
            # Speech overlap
            # -----------------------------------------------------
            speech_score = self._text_overlap(
                query_tokens,
                self._tokenize(
                    concept.speech_text
                ),
            )

            # -----------------------------------------------------
            # Visual overlap
            # -----------------------------------------------------
            visual_score = self._text_overlap(
                query_tokens,
                self._tokenize(
                    concept.visual_description
                ),
            )

            # -----------------------------------------------------
            # Temporal coherence
            # -----------------------------------------------------
            temporal_score = self._temporal_coherence(
                concept,
                concepts,
            )

            # -----------------------------------------------------
            # Create evidence
            # -----------------------------------------------------
            #
            # The evidence object keeps the individual signals.
            # The semantic score remains the relevance score used
            # for ordering.
            # -----------------------------------------------------
            evidence = RetrievedEvidence(
                concept=concept,
                semantic_score=semantic_score,
                speech_score=speech_score,
                visual_score=visual_score,
                temporal_score=temporal_score,
            )

            results.append(evidence)

        # ---------------------------------------------------------
        # Sort by semantic relevance
        # ---------------------------------------------------------
        results.sort(
            key=lambda evidence: evidence.semantic_score,
            reverse=True,
        )

        return results

    # =============================================================
    # SEMANTIC SIMILARITY
    # =============================================================

    def _semantic_similarity(
        self,
        a: Optional[List[float]],
        b: Optional[List[float]],
    ) -> float:
        if not a or not b:
            return 0.0

        if len(a) != len(b):
            return 0.0

        dot = sum(
            x * y
            for x, y in zip(a, b)
        )

        norm_a = sum(
            x * x
            for x in a
        ) ** 0.5

        norm_b = sum(
            x * x
            for x in b
        ) ** 0.5

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (
            norm_a * norm_b
        )

    # =============================================================
    # TEXT OVERLAP
    # =============================================================

    def _text_overlap(
        self,
        query_tokens: set[str],
        text_tokens: set[str],
    ) -> float:
        if not query_tokens or not text_tokens:
            return 0.0

        intersection = (
            query_tokens
            & text_tokens
        )

        if not intersection:
            return 0.0

        return len(intersection) / len(
            query_tokens
        )

    # =============================================================
    # TEMPORAL COHERENCE
    # =============================================================

    def _temporal_coherence(
        self,
        concept: MultimodalConcept,
        concepts: List[MultimodalConcept],
    ) -> float:
        """
        Measure whether a concept is temporally connected to other
        concepts already present in the expanded evidence set.

        This value is retained as contextual metadata only. It is
        not used to override semantic relevance.
        """

        if not concepts:
            return 0.0

        best = 0.0

        for other in concepts:
            if other.concept_id == concept.concept_id:
                continue

            if other.video_id != concept.video_id:
                continue

            # -----------------------------------------------------
            # Temporal overlap
            # -----------------------------------------------------
            overlap_start = max(
                concept.start_time,
                other.start_time,
            )

            overlap_end = min(
                concept.end_time,
                other.end_time,
            )

            if overlap_end >= overlap_start:
                best = max(
                    best,
                    1.0,
                )

        return best

    # =============================================================
    # TOKENIZATION
    # =============================================================

    def _tokenize(
        self,
        text: Optional[str],
    ) -> set[str]:
        if not text:
            return set()

        words = text.lower().split()

        cleaned = set()

        for word in words:
            token = "".join(
                char
                for char in word
                if char.isalnum()
            )

            if not token:
                continue

            # Ignore very short/common words.
            if len(token) <= 2:
                continue

            cleaned.add(token)

        return cleaned

