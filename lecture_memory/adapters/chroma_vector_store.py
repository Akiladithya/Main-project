"""
adapters/chroma_vector_store.py
--------------------------------

Implements IVectorStore using ChromaDB.

ChromaDB is used as a local persistent vector database.

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

The adapter is intentionally isolated from the rest of the
pipeline so that ChromaDB can later be replaced by another
vector database without changing the application architecture.
"""

from __future__ import annotations

import json
from typing import List, Optional

from lecture_memory.domain.models import MultimodalConcept
from lecture_memory.domain.enums import SelectionReason
from lecture_memory.interfaces.vector_store import IVectorStore
from lecture_memory.config.settings import settings
from lecture_memory.observability.logger import get_logger


logger = get_logger(__name__)


_COLLECTION = "lecture_concepts"


class ChromaVectorStore(IVectorStore):

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        persist_dir: str = settings.vector_store_dir,
    ) -> None:

        import chromadb

        self._client = chromadb.PersistentClient(
            path=persist_dir
        )

        self._col = self._client.get_or_create_collection(
            name=_COLLECTION,
            metadata={
                "hnsw:space": "cosine"
            },
        )

        logger.info(
            f"chroma_ready | "
            f"dir={persist_dir} | "
            f"collection={_COLLECTION}"
        )

    # ========================================================
    # UPSERT
    # ========================================================

    def upsert(
        self,
        concepts: List[MultimodalConcept],
    ) -> None:

        if not concepts:
            return

        self._col.upsert(
            ids=[
                concept.concept_id
                for concept in concepts
            ],

            embeddings=[
                concept.embedding
                for concept in concepts
            ],

            metadatas=[
                self._to_meta(concept)
                for concept in concepts
            ],

            documents=[
                self._build_document(concept)
                for concept in concepts
            ],
        )

        logger.info(
            f"chroma_upserted | "
            f"concepts={len(concepts)}"
        )

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

        where = self._build_filter(
            subject=subject,
            topic=topic,
            video_id=video_id,
        )

        # ----------------------------------------------------
        # Clamp top_k to collection size.
        # ----------------------------------------------------

        count = self._col.count()

        if count == 0:
            return []

        n = min(top_k, count)

        results = self._col.query(
            query_embeddings=[query_embedding],

            n_results=n,

            where=where or None,

            # We explicitly request embeddings here.
            #
            # This is important because the retriever needs
            # embeddings for semantic re-ranking.
            include=[
                "metadatas",
                "embeddings",
            ],
        )

        metadatas = (
            results["metadatas"][0]
            if results.get("metadatas")
            else []
        )

        embeddings = (
            results["embeddings"][0]
            if results.get("embeddings")
            else []
        )

        concepts: List[MultimodalConcept] = []

        for meta, embedding in zip(
            metadatas,
            embeddings,
        ):

            concept = self._from_meta(meta)

            concept.embedding = (
                list(embedding)
                if embedding is not None
                else []
            )

            concepts.append(concept)

        return concepts

    # ========================================================
    # GET ALL CONCEPTS FOR A VIDEO
    # ========================================================

    def get_by_video(
        self,
        video_id: str,
    ) -> List[MultimodalConcept]:

        results = self._col.get(
            where={
                "video_id": video_id
            },

            # IMPORTANT:
            # Include embeddings here as well.
            #
            # Previously get_by_video() returned concepts
            # without embeddings. That meant concepts obtained
            # during temporal expansion later received:
            #
            # semantic_score = 0
            #
            # because their embeddings were empty.
            include=[
                "metadatas",
                "embeddings",
            ],
        )

        metadatas = (
            results.get("metadatas", [])
        )

        embeddings = (
            results.get("embeddings", [])
        )

        concepts: List[MultimodalConcept] = []

        for meta, embedding in zip(
            metadatas,
            embeddings,
        ):

            concept = self._from_meta(meta)

            concept.embedding = (
                list(embedding)
                if embedding is not None
                else []
            )

            concepts.append(concept)

        return concepts

    # ========================================================
    # GET CONCEPTS IN TIME RANGE
    # ========================================================

    def get_by_time_range(
        self,
        video_id: str,
        start_time: float,
        end_time: float,
    ) -> List[MultimodalConcept]:

        all_concepts = self.get_by_video(
            video_id
        )

        return [
            concept
            for concept in all_concepts
            if (
                concept.start_time <= end_time
                and concept.end_time >= start_time
            )
        ]

    # ========================================================
    # DELETE VIDEO
    # ========================================================

    def delete_by_video(
        self,
        video_id: str,
    ) -> None:

        self._col.delete(
            where={
                "video_id": video_id
            }
        )

        logger.info(
            f"deleted | video_id={video_id}"
        )

    # ========================================================
    # BUILD CHROMA DOCUMENT
    # ========================================================

    def _build_document(
        self,
        concept: MultimodalConcept,
    ) -> str:
        """
        Build the searchable text associated with a concept.

        We intentionally combine speech + visual information +
        concept labels.

        This makes the vector representation multimodal at the
        text-embedding level as well.
        """

        parts = [
            concept.concept or "",
            concept.topic or "",
            concept.subject or "",
            concept.speech_text or "",
            concept.visual_description or "",
        ]

        return " ".join(
            part.strip()
            for part in parts
            if part and part.strip()
        )

    # ========================================================
    # CONCEPT -> CHROMA METADATA
    # ========================================================

    def _to_meta(
        self,
        concept: MultimodalConcept,
    ) -> dict:
        """
        Convert a MultimodalConcept into Chroma-compatible
        metadata.

        Chroma metadata values need to be scalar values.

        Therefore lists are serialized as JSON strings.
        """

        return {
            # -----------------------------------------------
            # Identity
            # -----------------------------------------------

            "concept_id": concept.concept_id,

            "video_id": concept.video_id,

            "subject": concept.subject or "",

            "topic": concept.topic or "",

            "concept": concept.concept,

            # -----------------------------------------------
            # Temporal information
            # -----------------------------------------------

            "start_time": concept.start_time,

            "end_time": concept.end_time,

            # -----------------------------------------------
            # Speech
            # -----------------------------------------------

            "speech_text": concept.speech_text,

            # -----------------------------------------------
            # Visual
            # -----------------------------------------------

            "visual_description": (
                concept.visual_description
            ),

            "visual_event_id": (
                concept.visual_event_id or ""
            ),

            # -----------------------------------------------
            # Evidence relationships
            # -----------------------------------------------

            "evidence_ids": json.dumps(
                concept.evidence_ids
            ),

            "visual_evidence_ids": json.dumps(
                concept.visual_evidence_ids
            ),

            "frame_ids": json.dumps(
                concept.frame_ids
            ),

            # -----------------------------------------------
            # Visual importance
            # -----------------------------------------------

            "visual_change_score": float(
                concept.visual_change_score
            ),

            "selection_reasons": json.dumps(
                [
                    reason.value
                    for reason in concept.selection_reasons
                ]
            ),

            # -----------------------------------------------
            # Schema
            # -----------------------------------------------

            "schema_version": (
                concept.schema_version
            ),
        }

    # ========================================================
    # CHROMA METADATA -> CONCEPT
    # ========================================================

    def _from_meta(
        self,
        metadata: dict,
    ) -> MultimodalConcept:
        """
        Restore a MultimodalConcept from Chroma metadata.

        This method is deliberately backward-compatible with
        older records that do not contain the new multimodal
        metadata fields.
        """

        # ----------------------------------------------------
        # Parse JSON list safely.
        # ----------------------------------------------------

        evidence_ids = self._parse_json_list(
            metadata.get("evidence_ids")
        )

        visual_evidence_ids = self._parse_json_list(
            metadata.get("visual_evidence_ids")
        )

        frame_ids = self._parse_json_list(
            metadata.get("frame_ids")
        )

        selection_reason_values = self._parse_json_list(
            metadata.get("selection_reasons")
        )

        # ----------------------------------------------------
        # Convert stored strings back to SelectionReason.
        # ----------------------------------------------------

        selection_reasons = []

        for value in selection_reason_values:

            try:

                selection_reasons.append(
                    SelectionReason(value)
                )

            except ValueError:

                # Ignore unknown values rather than breaking
                # retrieval of an otherwise valid concept.
                logger.warning(
                    f"unknown_selection_reason | "
                    f"value={value}"
                )

        # ----------------------------------------------------
        # Restore concept.
        # ----------------------------------------------------

        return MultimodalConcept(
            concept_id=metadata.get(
                "concept_id",
                "",
            ),

            video_id=metadata.get(
                "video_id",
                "",
            ),

            subject=(
                metadata.get("subject")
                or None
            ),

            topic=(
                metadata.get("topic")
                or None
            ),

            concept=metadata.get(
                "concept",
                "",
            ),

            start_time=float(
                metadata.get(
                    "start_time",
                    0.0,
                )
            ),

            end_time=float(
                metadata.get(
                    "end_time",
                    0.0,
                )
            ),

            speech_text=metadata.get(
                "speech_text",
                "",
            ),

            visual_description=metadata.get(
                "visual_description",
                "",
            ),

            visual_event_id=(
                metadata.get(
                    "visual_event_id"
                )
                or None
            ),

            # -----------------------------------------------
            # New evidence fields
            # -----------------------------------------------

            evidence_ids=evidence_ids,

            visual_evidence_ids=(
                visual_evidence_ids
            ),

            frame_ids=frame_ids,

            # -----------------------------------------------
            # Visual importance
            # -----------------------------------------------

            visual_change_score=float(
                metadata.get(
                    "visual_change_score",
                    0.0,
                )
            ),

            selection_reasons=selection_reasons,

            schema_version=metadata.get(
                "schema_version",
                "1.0",
            ),
        )

    # ========================================================
    # SAFE JSON LIST PARSER
    # ========================================================

    def _parse_json_list(
        self,
        value,
    ) -> List[str]:
        """
        Parse a JSON-encoded list from Chroma metadata.

        Older records may not contain the field, so missing or
        invalid values simply become [].
        """

        if value is None:
            return []

        # Already a list.
        if isinstance(value, list):
            return [
                str(item)
                for item in value
            ]

        try:

            parsed = json.loads(
                str(value)
            )

            if isinstance(parsed, list):

                return [
                    str(item)
                    for item in parsed
                ]

        except Exception:
            pass

        return []

    # ========================================================
    # FILTER BUILDER
    # ========================================================

    def _build_filter(
        self,
        **kwargs,
    ) -> dict:

        conditions = [
            {
                key: {
                    "$eq": value
                }
            }

            for key, value in kwargs.items()

            if value is not None
        ]

        if not conditions:
            return {}

        if len(conditions) == 1:
            return conditions[0]

        return {
            "$and": conditions
        }