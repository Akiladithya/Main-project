
"""
domain/models.py

----------------

Canonical data contracts for the multimodal lecture memory system.

Rules:

- Every entity preserves video_id and timestamps.

- No entity loses its link back to the original video.

- Multimodal concepts preserve both speech and visual information.

- Temporal and visual-change metadata are retained for retrieval.

- No business logic lives here — pure data contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

from .enums import (
    ProcessingStatus,
    SelectionReason,
    VerificationOutcome,
    ModalityType,
)


SCHEMA_VERSION = "1.0"


# ============================================================
# VIDEO
# ============================================================

@dataclass
class Video:
    video_id: str
    filename: str
    source_uri: str
    duration: float
    fps: float
    width: int
    height: int
    has_audio: bool
    subject: Optional[str] = None
    processing_version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: ProcessingStatus = ProcessingStatus.PENDING


# ============================================================
# SPEECH
# ============================================================

@dataclass
class SpeechSegment:
    segment_id: str
    video_id: str
    start_time: float
    end_time: float
    text: str


# ============================================================
# FRAME EVIDENCE
# ============================================================

@dataclass
class FrameEvidence:
    frame_id: str
    video_id: str
    timestamp: float
    image_uri: str
    selection_reason: SelectionReason

    # How strongly this frame differs from the previous frame.
    # Higher value = greater visual change.
    visual_change_score: float = 0.0


# ============================================================
# VISUAL EVIDENCE
# ============================================================

@dataclass
class VisualEvidence:
    visual_id: str
    frame_id: str
    video_id: str
    timestamp: float
    caption: str
    entities: List[str] = field(default_factory=list)
    concept_candidates: List[str] = field(default_factory=list)

    # Preserve information from the original FrameEvidence.
    #
    # This is important because otherwise the information about
    # WHY a frame was selected and HOW much the scene changed
    # would be lost before retrieval.
    selection_reason: Optional[SelectionReason] = None
    visual_change_score: float = 0.0


# ============================================================
# VISUAL EVENT
# ============================================================

@dataclass
class VisualEvent:
    event_id: str
    video_id: str
    start_time: float
    end_time: float
    title: str
    description: str
    frame_ids: List[str] = field(default_factory=list)


# ============================================================
# MULTIMODAL CONCEPT
# ============================================================

@dataclass
class MultimodalConcept:
    """
    A concept-level multimodal memory unit.

    It connects:

        speech
        +
        visual evidence
        +
        visual events
        +
        timestamps

    This is the main unit stored in the vector database.
    """

    concept_id: str
    video_id: str
    subject: Optional[str]
    topic: Optional[str]
    concept: str
    start_time: float
    end_time: float

    # ----------------------------
    # Speech information
    # ----------------------------

    speech_text: str

    # ----------------------------
    # Visual information
    # ----------------------------

    visual_description: str
    visual_event_id: Optional[str]

    # ----------------------------
    # Evidence relationships
    # ----------------------------

    # Existing evidence identifiers.
    evidence_ids: List[str] = field(default_factory=list)

    # IDs of VisualEvidence objects contributing
    # to this multimodal concept.
    visual_evidence_ids: List[str] = field(default_factory=list)

    # Original frame IDs contributing to this concept.
    frame_ids: List[str] = field(default_factory=list)

    # ----------------------------
    # Visual importance
    # ----------------------------

    # Maximum visual-change score among the
    # visual evidence belonging to this concept.
    visual_change_score: float = 0.0

    # Reasons why the contributing frames were selected.
    selection_reasons: List[SelectionReason] = field(
        default_factory=list
    )

    # ----------------------------
    # Retrieval
    # ----------------------------

    embedding: List[float] = field(default_factory=list)

    # ----------------------------
    # Schema
    # ----------------------------

    schema_version: str = SCHEMA_VERSION


# ============================================================
# RETRIEVED EVIDENCE
# ============================================================

@dataclass
class RetrievedEvidence:
    """
    Evidence returned by the retrieval stage.

    Semantic similarity is the primary relevance signal.

    Speech overlap, visual overlap, and temporal coherence are
    retained as supporting signals that describe the relationship
    between the question and the retrieved concept.
    """

    concept: MultimodalConcept
    semantic_score: float
    speech_score: float
    visual_score: float
    temporal_score: float

    @property
    def combined_score(self) -> float:
        """
        Return the primary retrieval relevance score.

        Semantic similarity is the authoritative relevance signal
        used by TemporalRetriever when ordering evidence.

        The other scores remain available separately so that
        downstream components can inspect the supporting evidence
        characteristics without introducing arbitrary weighting.
        """
        return self.semantic_score


# ============================================================
# GENERATED ANSWER
# ============================================================

@dataclass
class GeneratedAnswer:
    question: str
    answer: str
    explanation: str
    timestamp_start: Optional[float]
    timestamp_end: Optional[float]
    video_id: Optional[str]

    # Internal evidence IDs.
    #
    # These should not necessarily be displayed directly
    # to the student in the UI.
    evidence_ids: List[str] = field(default_factory=list)


# ============================================================
# CLAIM VERIFICATION
# ============================================================

@dataclass
class ClaimVerification:
    claim: str
    outcome: VerificationOutcome
    supporting_evidence_ids: List[str] = field(
        default_factory=list
    )
    confidence: float = 0.0


# ============================================================
# VERIFICATION RESULT
# ============================================================

@dataclass
class VerificationResult:
    answer: GeneratedAnswer
    claim_verifications: List[ClaimVerification] = field(
        default_factory=list
    )
    overall_outcome: VerificationOutcome = (
        VerificationOutcome.INSUFFICIENT_EVIDENCE
    )
    verified_answer: Optional[str] = None


# ============================================================
# PROCESSING JOB
# ============================================================

@dataclass
class ProcessingJob:
    job_id: str
    video_id: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    stage: str = "ingestion"
    error: Optional[str] = None
    created_at: datetime = field(
        default_factory=datetime.utcnow
    )
    updated_at: datetime = field(
        default_factory=datetime.utcnow
    )

