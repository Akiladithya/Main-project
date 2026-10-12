"""
adapters/simple_aligner.py
--------------------------

Implements ICrossModalAligner using temporal overlap as the primary signal.

The aligner connects:

    SpeechSegment
          +
    VisualEvidence
          +
    VisualEvent
          ↓
    MultimodalConcept

The concept preserves:
- speech text
- visual descriptions
- visual evidence IDs
- frame IDs
- visual event
- timestamps
- visual change score
- frame selection reasons

Subject/topic extraction currently uses Gemini when a Gemini API
key is configured. If Gemini is unavailable, subject/topic are
left as None rather than inventing information.
"""

from __future__ import annotations

import uuid
from typing import List, Optional, Tuple

from lecture_memory.domain.models import (
    SpeechSegment,
    VisualEvidence,
    VisualEvent,
    MultimodalConcept,
)

from lecture_memory.interfaces.aligner import ICrossModalAligner

from lecture_memory.config.settings import settings

from lecture_memory.observability.logger import get_logger


logger = get_logger(__name__)


# ============================================================
# SUBJECT / TOPIC EXTRACTION
# ============================================================

_EXTRACT_PROMPT = """You are labeling a segment from a Computer Science lecture.

Speech:
{speech}

Visual:
{visual}

Reply with exactly two lines:

Subject: <broad CS subject, e.g. "Computer Networks", "Operating Systems">

Topic: <specific topic within that subject, e.g. "TCP Three-Way Handshake">
"""


class SimpleTemporalAligner(ICrossModalAligner):

    def __init__(self) -> None:
        # Gemini client is created only if subject/topic extraction
        # is actually needed and an API key exists.
        self._client = None

    # ========================================================
    # GEMINI CLIENT
    # ========================================================

    def _gemini_client(self):

        if self._client is None:

            from google import genai

            self._client = genai.Client(
                api_key=settings.gemini_api_key
            )

        return self._client

    # ========================================================
    # SUBJECT / TOPIC EXTRACTION
    # ========================================================

    def _extract_subject_topic(
        self,
        speech: str,
        visual: str,
    ) -> Tuple[Optional[str], Optional[str]]:

        # If Gemini is not configured, do not fail the entire
        # ingestion pipeline.
        if not settings.gemini_api_key:
            return None, None

        try:

            prompt = _EXTRACT_PROMPT.format(
                speech=(speech or "")[:400],
                visual=(visual or "")[:400],
            )

            response = (
                self._gemini_client()
                .models
                .generate_content(
                    model=settings.subject_extraction_model,
                    contents=prompt,
                )
            )

            subject = None
            topic = None

            for line in (response.text or "").splitlines():

                line = line.strip()

                if line.lower().startswith("subject:"):

                    subject = (
                        line.split(":", 1)[1].strip()
                        or None
                    )

                elif line.lower().startswith("topic:"):

                    topic = (
                        line.split(":", 1)[1].strip()
                        or None
                    )

            return subject, topic

        except Exception as exc:

            logger.warning(
                f"subject_extraction_failed | {exc}"
            )

            return None, None

    # ========================================================
    # MAIN ALIGNMENT
    # ========================================================

    def align(
        self,
        video_id: str,
        speech_segments: List[SpeechSegment],
        visual_evidence: List[VisualEvidence],
        visual_events: List[VisualEvent],
    ) -> List[MultimodalConcept]:

        concepts: List[MultimodalConcept] = []

        # ----------------------------------------------------
        # Sort inputs by time.
        #
        # This makes the temporal relationship deterministic.
        # ----------------------------------------------------

        sorted_speech = sorted(
            speech_segments,
            key=lambda segment: segment.start_time,
        )

        sorted_visual = sorted(
            visual_evidence,
            key=lambda evidence: evidence.timestamp,
        )

        sorted_events = sorted(
            visual_events,
            key=lambda event: event.start_time,
        )

        # ====================================================
        # EVENT-BASED MULTIMODAL CONCEPTS
        # ====================================================

        for event in sorted_events:

            # ------------------------------------------------
            # Find speech overlapping this visual event.
            #
            # Two intervals overlap when:
            #
            # speech.start < event.end
            # AND
            # speech.end > event.start
            # ------------------------------------------------

            overlapping_speech = [
                segment
                for segment in sorted_speech
                if (
                    segment.start_time < event.end_time
                    and segment.end_time > event.start_time
                )
            ]

            # ------------------------------------------------
            # Find visual evidence belonging to this event.
            # ------------------------------------------------

            overlapping_visual = [
                evidence
                for evidence in sorted_visual
                if (
                    event.start_time
                    <= evidence.timestamp
                    <= event.end_time
                )
            ]

            # ------------------------------------------------
            # Combine speech.
            # ------------------------------------------------

            speech_text = " ".join(
                segment.text.strip()
                for segment in overlapping_speech
                if segment.text and segment.text.strip()
            )

            # ------------------------------------------------
            # Combine visual descriptions.
            # ------------------------------------------------

            visual_desc = " ".join(
                evidence.caption.strip()
                for evidence in overlapping_visual
                if evidence.caption
                and evidence.caption.strip()
            )

            # ------------------------------------------------
            # Extract subject/topic.
            # ------------------------------------------------

            subject, topic = self._extract_subject_topic(
                speech_text,
                visual_desc,
            )

            # ------------------------------------------------
            # Collect visual evidence IDs.
            # ------------------------------------------------

            visual_evidence_ids = [
                evidence.visual_id
                for evidence in overlapping_visual
            ]

            # ------------------------------------------------
            # Collect frame IDs.
            # ------------------------------------------------

            frame_ids = [
                evidence.frame_id
                for evidence in overlapping_visual
            ]

            # ------------------------------------------------
            # Collect selection reasons.
            #
            # Use a list instead of a set so ordering remains
            # deterministic.
            # ------------------------------------------------

            selection_reasons = []

            for evidence in overlapping_visual:

                reason = evidence.selection_reason

                if (
                    reason is not None
                    and reason not in selection_reasons
                ):
                    selection_reasons.append(reason)

            # ------------------------------------------------
            # Find the strongest visual change in this event.
            #
            # A high value means that at least one important
            # visual transition occurred in this concept.
            # ------------------------------------------------

            visual_change_score = 0.0

            if overlapping_visual:

                visual_change_score = max(
                    evidence.visual_change_score
                    for evidence in overlapping_visual
                )

            # ------------------------------------------------
            # Evidence IDs
            #
            # Keep the existing behaviour:
            #
            # visual IDs + speech segment IDs
            #
            # Also include the event ID because the event itself
            # is an important temporal evidence unit.
            # ------------------------------------------------

            evidence_ids = (
                [evidence.visual_id for evidence in overlapping_visual]
                + [segment.segment_id for segment in overlapping_speech]
                + [event.event_id]
            )

            # ------------------------------------------------
            # Build the multimodal concept.
            # ------------------------------------------------

            concept = MultimodalConcept(
                concept_id=str(uuid.uuid4()),

                video_id=video_id,

                subject=subject,

                topic=topic,

                # Visual event title is the primary concept label.
                concept=event.title,

                start_time=event.start_time,

                end_time=event.end_time,

                speech_text=speech_text,

                visual_description=visual_desc,

                visual_event_id=event.event_id,

                evidence_ids=evidence_ids,

                visual_evidence_ids=visual_evidence_ids,

                frame_ids=frame_ids,

                visual_change_score=visual_change_score,

                selection_reasons=selection_reasons,

                schema_version=settings.schema_version,
            )

            concepts.append(concept)

        # ====================================================
        # SPEECH-ONLY CONCEPTS
        # ====================================================
        #
        # Some lecture speech may not overlap a visual event.
        #
        # Example:
        #
        # Lecturer explains:
        # "A binary search tree maintains an ordering property..."
        #
        # while the slide has not changed.
        #
        # We must NOT lose that speech.
        # ====================================================

        covered_times = [
            (concept.start_time, concept.end_time)
            for concept in concepts
        ]

        for segment in sorted_speech:

            # ------------------------------------------------
            # Check whether the complete speech segment is
            # already represented by an event concept.
            # ------------------------------------------------

            already_covered = any(
                start <= segment.start_time
                and end >= segment.end_time
                for start, end in covered_times
            )

            if already_covered:
                continue

            # ------------------------------------------------
            # Find visual evidence that happens during this
            # speech segment even though it isn't part of an
            # explicit VisualEvent.
            # ------------------------------------------------

            nearby_visual = [
                evidence
                for evidence in sorted_visual
                if (
                    segment.start_time
                    <= evidence.timestamp
                    <= segment.end_time
                )
            ]

            # ------------------------------------------------
            # Combine any visual descriptions available.
            # ------------------------------------------------

            visual_desc = " ".join(
                evidence.caption.strip()
                for evidence in nearby_visual
                if evidence.caption
                and evidence.caption.strip()
            )

            # ------------------------------------------------
            # Subject/topic extraction.
            # ------------------------------------------------

            subject, topic = self._extract_subject_topic(
                segment.text,
                visual_desc,
            )

            # ------------------------------------------------
            # Visual metadata for speech-only concept.
            # ------------------------------------------------

            visual_evidence_ids = [
                evidence.visual_id
                for evidence in nearby_visual
            ]

            frame_ids = [
                evidence.frame_id
                for evidence in nearby_visual
            ]

            selection_reasons = []

            for evidence in nearby_visual:

                reason = evidence.selection_reason

                if (
                    reason is not None
                    and reason not in selection_reasons
                ):
                    selection_reasons.append(reason)

            visual_change_score = 0.0

            if nearby_visual:

                visual_change_score = max(
                    evidence.visual_change_score
                    for evidence in nearby_visual
                )

            evidence_ids = (
                [segment.segment_id]
                + visual_evidence_ids
            )

            # ------------------------------------------------
            # Create speech concept.
            # ------------------------------------------------

            concept = MultimodalConcept(
                concept_id=str(uuid.uuid4()),

                video_id=video_id,

                subject=subject,

                topic=topic,

                # Use the beginning of the speech as the
                # fallback concept label.
                concept=segment.text[:80],

                start_time=segment.start_time,

                end_time=segment.end_time,

                speech_text=segment.text,

                visual_description=visual_desc,

                visual_event_id=None,

                evidence_ids=evidence_ids,

                visual_evidence_ids=visual_evidence_ids,

                frame_ids=frame_ids,

                visual_change_score=visual_change_score,

                selection_reasons=selection_reasons,

                schema_version=settings.schema_version,
            )

            concepts.append(concept)

        # ====================================================
        # FINAL SORTING
        # ====================================================
        #
        # Keep concepts chronologically ordered.
        # This is useful for temporal expansion, debugging,
        # evaluation, and UI evidence display.
        # ====================================================

        concepts.sort(
            key=lambda concept: (
                concept.start_time,
                concept.end_time,
            )
        )

        logger.info(
            f"aligned | "
            f"video_id={video_id} | "
            f"speech_segments={len(speech_segments)} | "
            f"visual_evidence={len(visual_evidence)} | "
            f"visual_events={len(visual_events)} | "
            f"concepts={len(concepts)}"
        )

        return concepts