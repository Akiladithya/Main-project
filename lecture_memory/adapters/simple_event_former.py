from __future__ import annotations

import re
import uuid
from typing import List, Optional

from lecture_memory.domain.models import (
    SpeechSegment,
    VisualEvidence,
    VisualEvent,
)
from lecture_memory.interfaces.event_former import IVisualEventFormer


class SimpleVisualEventFormer(IVisualEventFormer):
    """
    Forms semantic lecture episodes from ordered speech.

    Core principle
    --------------
    Speech is the temporal backbone of event formation.

    A new event is created when the lecturer explicitly signals
    a meaningful discourse transition or when the lecture moves
    into an external outro/sponsor section.

    Visual evidence supports the resulting events but does not
    independently create event boundaries.

    The logic is domain-independent:
    - no CS-specific keywords
    - no fixed timestamps
    - no fixed event count
    - no arbitrary duration threshold
    - no embedding-based boundary tuning
    """

    # ============================================================
    # Explicit discourse transitions
    # ============================================================

    TRANSITION_PATTERNS = (
        r"^(now|next|moving on)\b",
        r"^(let us|let's)\b",
        r"^(below that|above that|after that|before that)\b",
        r"^(this brings us to|that brings us to)\b",
        r"^(we move to|we turn to|we come to)\b",
        r"^(the next|the following)\b",
        r"^(and now)\b",
    )

    # ------------------------------------------------------------
    # Bridge transitions
    #
    # These sentences announce the destination of the transition,
    # but the substantive explanation starts in the next speech
    # unit.
    #
    # Example:
    #
    # "And that brings us to the transport layer."
    # "The Transmission Control Protocol..."
    #
    # The bridge belongs to the preceding event.
    # ------------------------------------------------------------

    BRIDGE_TRANSITION_PATTERNS = (
        r"^(this|that) brings us to\b",
    )

    # ============================================================
    # Explicit structural markers
    # ============================================================

    STRUCTURAL_PATTERNS = (
        r"^(section|chapter|part|stage|step|phase)\s+\w+",
        r"^(the\s+)?(first|second|third|fourth|fifth|sixth|"
        r"seventh|eighth|ninth|tenth)\s+"
        r"(section|part|stage|step|phase)\b",
    )

    # ============================================================
    # Continuation signals
    # ============================================================

    CONTINUATION_PATTERNS = (
        r"^(and|so|because|which|that|this|these|those|"
        r"it|they|also|for example|for instance|"
        r"in other words)\b",
    )

    # ============================================================
    # Lecture conclusion
    # ============================================================

    LECTURE_CONCLUSION_PATTERNS = (
        r"^this has been\b",
        r"^that is how\b",
        r"^in conclusion\b",
        r"^to conclude\b",
    )

    # ============================================================
    # External outro / sponsor
    # ============================================================

    OUTRO_PATTERNS = (
        r"^thanks for watching\b",
        r"^thank you for watching\b",
        r"^if you want to see more\b",
        r"^like and subscribe\b",
        r"^subscribe\b",
        r"^and now a quick word from our sponsor\b",
        r"^a quick word from our sponsor\b",
    )

    def __init__(self, embedder=None) -> None:
        # Kept for compatibility with the existing architecture.
        #
        # Event formation intentionally does not depend on an
        # embedding model. Explicit discourse structure is used
        # instead of arbitrary similarity thresholds.
        self._embedder = embedder

    # ============================================================
    # PUBLIC
    # ============================================================

    def form_events(
        self,
        video_id: str,
        visual_evidence: List[VisualEvidence],
        speech_segments: Optional[List[SpeechSegment]] = None,
    ) -> List[VisualEvent]:

        speech_segments = speech_segments or []

        units = sorted(
            [
                segment
                for segment in speech_segments
                if segment.text and segment.text.strip()
            ],
            key=lambda segment: segment.start_time,
        )

        if not units:
            return self._visual_only_events(
                video_id,
                visual_evidence,
            )

        if len(units) == 1:
            return [
                self._build_event(
                    video_id,
                    units,
                    visual_evidence,
                )
            ]

        boundaries = self._find_boundaries(units)

        print(
            f"EVENT_FORMER | "
            f"speech_units={len(units)} | "
            f"boundaries={boundaries}"
        )

        events = self._build_events(
            video_id,
            units,
            visual_evidence,
            boundaries,
        )

        print(
            f"EVENT_FORMER | "
            f"events_created={len(events)}"
        )

        for index, event in enumerate(events, 1):
            print(
                f"EVENT_FORMER_EVENT | "
                f"{index} | "
                f"{event.start_time:.2f}s-"
                f"{event.end_time:.2f}s | "
                f"{event.title}"
            )

        return events

    # ============================================================
    # BOUNDARY FORMATION
    # ============================================================

    def _find_boundaries(
        self,
        units: List[SpeechSegment],
    ) -> List[int]:

        boundaries: List[int] = []

        for index in range(1, len(units)):

            current_text = self._clean_text(
                units[index].text
            ).lower()

            # ----------------------------------------------------
            # 1. Outro
            #
            # Once the lecture enters the external outro/sponsor
            # section, everything after it belongs to that same
            # final event.
            #
            # This prevents:
            #   "please like and subscribe"
            #   "thanks for watching"
            #   "quick word from our sponsor"
            #
            # from becoming three separate events.
            # ----------------------------------------------------

            if self._is_outro(current_text):
                boundaries.append(index)

                # The remainder of the video is the outro.
                # No more lecture boundaries are required.
                break

            # ----------------------------------------------------
            # 2. Bridge transition
            #
            # A phrase such as:
            #
            # "And that brings us to the transport layer."
            #
            # announces the next topic, but the actual explanation
            # begins in the following speech unit.
            #
            # Therefore the boundary is placed AFTER the bridge
            # sentence.
            # ----------------------------------------------------

            if self._is_bridge_transition(current_text):
                next_index = index + 1

                if next_index < len(units):
                    boundaries.append(next_index)

                continue

            # ----------------------------------------------------
            # 3. Explicit lecture transition
            # ----------------------------------------------------

            if self._is_transition(current_text):
                boundaries.append(index)
                continue

            # ----------------------------------------------------
            # 4. Explicit structural marker
            # ----------------------------------------------------

            if self._is_structural(current_text):
                boundaries.append(index)
                continue

            # ----------------------------------------------------
            # 5. Continuation
            #
            # Do not split ordinary explanatory continuation.
            # ----------------------------------------------------

            if self._is_continuation(current_text):
                continue

            # ----------------------------------------------------
            # 6. Lecture conclusion
            #
            # A conclusion remains part of the current lecture
            # episode. It does not create another event.
            # ----------------------------------------------------

            if self._is_lecture_conclusion(current_text):
                continue

            # ----------------------------------------------------
            # 7. No explicit discourse signal.
            #
            # Do not split simply because a new concept appears.
            # ----------------------------------------------------

            continue

        return self._remove_redundant_boundaries(
            boundaries,
            units,
        )

    # ============================================================
    # DISCOURSE DETECTION
    # ============================================================

    def _is_transition(
        self,
        text: str,
    ) -> bool:

        normalized = self._clean_text(
            text
        ).lower()

        return any(
            re.search(
                pattern,
                normalized,
            )
            for pattern in self.TRANSITION_PATTERNS
        )

    def _is_bridge_transition(
        self,
        text: str,
    ) -> bool:

        normalized = self._clean_text(
            text
        ).lower()

        return any(
            re.search(
                pattern,
                normalized,
            )
            for pattern in self.BRIDGE_TRANSITION_PATTERNS
        )

    def _is_structural(
        self,
        text: str,
    ) -> bool:

        normalized = self._clean_text(
            text
        ).lower()

        return any(
            re.search(
                pattern,
                normalized,
            )
            for pattern in self.STRUCTURAL_PATTERNS
        )

    def _is_continuation(
        self,
        text: str,
    ) -> bool:

        normalized = self._clean_text(
            text
        ).lower()

        return any(
            re.search(
                pattern,
                normalized,
            )
            for pattern in self.CONTINUATION_PATTERNS
        )

    def _is_lecture_conclusion(
        self,
        text: str,
    ) -> bool:

        normalized = self._clean_text(
            text
        ).lower()

        return any(
            re.search(
                pattern,
                normalized
            )
            for pattern in self.LECTURE_CONCLUSION_PATTERNS
        )

    def _is_outro(
        self,
        text: str,
    ) -> bool:

        normalized = self._clean_text(
            text
        ).lower()

        return any(
            re.search(
                pattern,
                normalized
            )
            for pattern in self.OUTRO_PATTERNS
        )

    # ============================================================
    # BOUNDARY CLEANUP
    # ============================================================

    def _remove_redundant_boundaries(
        self,
        boundaries: List[int],
        units: List[SpeechSegment],
    ) -> List[int]:

        if not boundaries:
            return []

        boundaries = sorted(
            set(boundaries)
        )

        result: List[int] = []

        for boundary in boundaries:

            if boundary <= 0:
                continue

            if boundary >= len(units):
                continue

            if not result:
                result.append(boundary)
                continue

            previous = result[-1]

            # Adjacent boundaries describe essentially the same
            # discourse movement. Keep the first meaningful one.
            if boundary == previous + 1:
                continue

            result.append(boundary)

        return result

    # ============================================================
    # EVENT CREATION
    # ============================================================

    def _build_events(
        self,
        video_id: str,
        units: List[SpeechSegment],
        visual_evidence: List[VisualEvidence],
        boundaries: List[int],
    ) -> List[VisualEvent]:

        events: List[VisualEvent] = []

        start = 0

        for boundary in boundaries + [len(units)]:

            group = units[
                start:boundary
            ]

            if group:
                events.append(
                    self._build_event(
                        video_id,
                        group,
                        visual_evidence,
                    )
                )

            start = boundary

        return events

    def _build_event(
        self,
        video_id: str,
        units: List[SpeechSegment],
        visual_evidence: List[VisualEvidence],
    ) -> VisualEvent:

        start_time = units[0].start_time
        end_time = units[-1].end_time

        speech_text = " ".join(
            self._clean_text(
                unit.text
            )
            for unit in units
        )

        frame_ids = [
            visual.frame_id
            for visual in visual_evidence
            if (
                visual.video_id == video_id
                and start_time <= visual.timestamp <= end_time
            )
        ]

        return VisualEvent(
            event_id=str(uuid.uuid4()),
            video_id=video_id,
            start_time=start_time,
            end_time=end_time,
            title=self._build_title(units),
            description=speech_text,
            frame_ids=frame_ids,
        )

    # ============================================================
    # TITLE
    # ============================================================

    def _build_title(
        self,
        units: List[SpeechSegment],
    ) -> str:

        for unit in units:

            text = self._clean_text(
                unit.text
            )

            words = text.split()

            if len(words) < 3:
                continue

            while (
                words
                and words[0].lower()
                in {
                    "and",
                    "so",
                    "now",
                    "next",
                    "finally",
                    "below",
                    "above",
                    "after",
                    "before",
                }
            ):
                words.pop(0)

            if len(words) < 3:
                continue

            return " ".join(
                words[:10]
            ) + (
                "..."
                if len(words) > 10
                else ""
            )

        return "Lecture episode"

    # ============================================================
    # VISUAL-ONLY FALLBACK
    # ============================================================

    def _visual_only_events(
        self,
        video_id: str,
        visual_evidence: List[VisualEvidence],
    ) -> List[VisualEvent]:

        ordered = sorted(
            visual_evidence,
            key=lambda visual: visual.timestamp,
        )

        events: List[VisualEvent] = []

        for visual in ordered:

            events.append(
                VisualEvent(
                    event_id=str(uuid.uuid4()),
                    video_id=video_id,
                    start_time=visual.timestamp,
                    end_time=visual.timestamp,
                    title=self._visual_title(
                        visual.caption
                    ),
                    description=visual.caption,
                    frame_ids=[
                        visual.frame_id
                    ],
                )
            )

        return events

    def _visual_title(
        self,
        caption: str,
    ) -> str:

        words = self._clean_text(
            caption
        ).split()

        if not words:
            return "Visual episode"

        return " ".join(
            words[:10]
        ) + (
            "..."
            if len(words) > 10
            else ""
        )

    # ============================================================
    # TEXT
    # ============================================================

    def _clean_text(
        self,
        text: str,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            text or "",
        ).strip()