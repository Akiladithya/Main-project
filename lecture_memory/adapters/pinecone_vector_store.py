"""
adapters/pinecone_vector_store.py
---------------------------------

Implements IVectorStore using Pinecone cloud vector database.

Pinecone provides managed serverless vector search with low latency,
persistent cloud indexing, and metadata filtering.

This adapter stores and restores the complete multimodal concept,
including:
- speech
- visual description
- visual event
- visual evidence IDs
- frame IDs
- timestamps
- visual change score
- frame selection reasons
- embeddings
"""

from __future__ import annotations

import json
from typing import List, Optional

from pinecone import Pinecone
from pinecone.exceptions import NotFoundException, PineconeException

from lecture_memory.domain.models import MultimodalConcept
from lecture_memory.domain.enums import SelectionReason
from lecture_memory.interfaces.vector_store import IVectorStore
from lecture_memory.config.settings import settings
from lecture_memory.observability.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_DIMENSION = 384
_DEFAULT_METRIC = "cosine"


class PineconeVectorStore(IVectorStore):
    """
    Pinecone cloud implementation of IVectorStore.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or settings.pinecone_api_key
        self._index_name = index_name or settings.pinecone_index_name

        if not self._api_key:
            raise ValueError(
                "Pinecone API key is required. Set PINECONE_API_KEY in your .env file "
                "or pass it explicitly."
            )

        logger.info(f"Connecting to Pinecone index: '{self._index_name}'...")
        self._pc = Pinecone(api_key=self._api_key)

        # Check existing indexes or create if missing
        existing_indexes = [idx.name for idx in self._pc.list_indexes()]
        if self._index_name not in existing_indexes:
            logger.info(
                f"Index '{self._index_name}' not found. Creating serverless index "
                f"(dim={_DEFAULT_DIMENSION}, metric={_DEFAULT_METRIC})..."
            )
            from pinecone import ServerlessSpec
            self._pc.create_index(
                name=self._index_name,
                dimension=_DEFAULT_DIMENSION,
                metric=_DEFAULT_METRIC,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self._index = self._pc.Index(self._index_name)
        self._video_cache: dict[str, List[MultimodalConcept]] = {}
        logger.info(f"pinecone_ready | index={self._index_name}")

    # ========================================================
    # UPSERT
    # ========================================================

    def upsert(
        self,
        concepts: List[MultimodalConcept],
        batch_size: int = 100,
    ) -> None:
        if not concepts:
            return

        total = len(concepts)
        for i in range(0, total, batch_size):
            chunk = concepts[i : i + batch_size]
            vectors = []
            for concept in chunk:
                # Guarantee embedding exists and is non-empty
                emb = concept.embedding if concept.embedding else [0.0] * _DEFAULT_DIMENSION
                vectors.append(
                    {
                        "id": concept.concept_id,
                        "values": emb,
                        "metadata": self._to_meta(concept),
                    }
                )

            self._index.upsert(vectors=vectors)

        for concept in concepts:
            if concept.video_id:
                self._video_cache.pop(concept.video_id, None)

        logger.info(f"pinecone_upserted | concepts={total} | index={self._index_name}")

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        video_id: Optional[str] = None,
    ) -> List[MultimodalConcept]:
        filter_dict = self._build_filter(
            subject=subject,
            topic=topic,
            video_id=video_id,
        )

        try:
            res = self._index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict or None,
                include_metadata=True,
                include_values=True,
            )
        except Exception as e:
            logger.error(f"pinecone_search_error | error={e}")
            return []

        matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
        concepts: List[MultimodalConcept] = []

        for match in matches:
            meta = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
            values = match.get("values", []) if isinstance(match, dict) else getattr(match, "values", [])
            concept = self._from_meta(meta)
            concept.embedding = list(values) if values else []
            concepts.append(concept)

        return concepts

    # ========================================================
    # GET ALL CONCEPTS FOR A VIDEO
    # ========================================================

    def get_by_video(
        self,
        video_id: str,
    ) -> List[MultimodalConcept]:
        if video_id in self._video_cache:
            return list(self._video_cache[video_id])

        filter_dict = {"video_id": {"$eq": video_id}}
        dummy_vector = [0.01] * _DEFAULT_DIMENSION

        try:
            res = self._index.query(
                vector=dummy_vector,
                top_k=1000,
                filter=filter_dict,
                include_metadata=True,
                include_values=True,
            )
        except NotFoundException:
            return []
        except Exception as e:
            logger.warning(f"pinecone_get_by_video_warning | video_id={video_id} | error={e}")
            return []

        matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
        concepts: List[MultimodalConcept] = []

        for match in matches:
            meta = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
            values = match.get("values", []) if isinstance(match, dict) else getattr(match, "values", [])
            concept = self._from_meta(meta)
            concept.embedding = list(values) if values else []
            concepts.append(concept)

        # Sort chronologically by start_time
        concepts.sort(key=lambda c: c.start_time)
        self._video_cache[video_id] = concepts
        return list(concepts)

    # ========================================================
    # GET CONCEPTS IN TIME RANGE
    # ========================================================

    def get_by_time_range(
        self,
        video_id: str,
        start_time: float,
        end_time: float,
    ) -> List[MultimodalConcept]:
        all_concepts = self.get_by_video(video_id)
        return [
            concept
            for concept in all_concepts
            if concept.start_time <= end_time and concept.end_time >= start_time
        ]

    # ========================================================
    # DELETE VIDEO
    # ========================================================

    def delete_by_video(
        self,
        video_id: str,
    ) -> None:
        self._video_cache.pop(video_id, None)
        try:
            self._index.delete(filter={"video_id": {"$eq": video_id}})
            logger.info(f"pinecone_deleted | video_id={video_id}")
        except NotFoundException:
            # Namespace or index is empty
            logger.info(f"pinecone_deleted_noop | namespace empty | video_id={video_id}")
        except Exception as e:
            logger.error(f"pinecone_delete_error | video_id={video_id} | error={e}")

    # ========================================================
    # CONCEPT -> PINECONE METADATA
    # ========================================================

    def _to_meta(
        self,
        concept: MultimodalConcept,
    ) -> dict:
        """
        Convert MultimodalConcept into Pinecone metadata.
        Pinecone supports string, number, bool, and list of strings.
        No None or nested dicts allowed.
        """
        return {
            "concept_id": str(concept.concept_id or ""),
            "video_id": str(concept.video_id or ""),
            "subject": str(concept.subject or ""),
            "topic": str(concept.topic or ""),
            "concept": str(concept.concept or ""),
            "start_time": float(concept.start_time),
            "end_time": float(concept.end_time),
            "speech_text": str(concept.speech_text or "")[:4000],
            "visual_description": str(concept.visual_description or "")[:4000],
            "visual_event_id": str(concept.visual_event_id or ""),
            "evidence_ids": [str(x) for x in concept.evidence_ids] if concept.evidence_ids else [],
            "visual_evidence_ids": [str(x) for x in concept.visual_evidence_ids] if concept.visual_evidence_ids else [],
            "frame_ids": [str(x) for x in concept.frame_ids] if concept.frame_ids else [],
            "visual_change_score": float(concept.visual_change_score or 0.0),
            "selection_reasons": [
                reason.value if hasattr(reason, "value") else str(reason)
                for reason in concept.selection_reasons
            ] if concept.selection_reasons else [],
            "schema_version": str(concept.schema_version or "1.0"),
        }

    # ========================================================
    # PINECONE METADATA -> CONCEPT
    # ========================================================

    def _from_meta(
        self,
        metadata: dict,
    ) -> MultimodalConcept:
        evidence_ids = self._parse_list_field(metadata.get("evidence_ids"))
        visual_evidence_ids = self._parse_list_field(metadata.get("visual_evidence_ids"))
        frame_ids = self._parse_list_field(metadata.get("frame_ids"))
        reason_values = self._parse_list_field(metadata.get("selection_reasons"))

        selection_reasons = []
        for val in reason_values:
            try:
                selection_reasons.append(SelectionReason(val))
            except ValueError:
                pass

        return MultimodalConcept(
            concept_id=metadata.get("concept_id", ""),
            video_id=metadata.get("video_id", ""),
            subject=metadata.get("subject") or None,
            topic=metadata.get("topic") or None,
            concept=metadata.get("concept", ""),
            start_time=float(metadata.get("start_time", 0.0)),
            end_time=float(metadata.get("end_time", 0.0)),
            speech_text=metadata.get("speech_text", ""),
            visual_description=metadata.get("visual_description", ""),
            visual_event_id=metadata.get("visual_event_id") or None,
            evidence_ids=evidence_ids,
            visual_evidence_ids=visual_evidence_ids,
            frame_ids=frame_ids,
            visual_change_score=float(metadata.get("visual_change_score", 0.0)),
            selection_reasons=selection_reasons,
            schema_version=metadata.get("schema_version", "1.0"),
        )

    def _parse_list_field(self, value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x) for x in value]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
            return [value]
        return []

    # ========================================================
    # FILTER BUILDER
    # ========================================================

    def _build_filter(
        self,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        video_id: Optional[str] = None,
    ) -> dict:
        conditions = {}
        if video_id:
            conditions["video_id"] = {"$eq": video_id}
        if subject:
            conditions["subject"] = {"$eq": subject}
        if topic:
            conditions["topic"] = {"$eq": topic}
        return conditions
