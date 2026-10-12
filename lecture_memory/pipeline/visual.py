"""
pipeline/visual.py
------------------
Responsibility:
selected frames -> VisualEvidence -> multimodal VisualEvents.

Cheap computation (frame selection) runs before expensive model calls
(vision).

Event formation uses both visual evidence and speech segments so that
visual changes do not automatically split a continuing lecture
explanation.

speech_segments is optional for backward compatibility with existing
visual-only callers and tests.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from lecture_memory.domain.models import (
    Video,
    SpeechSegment,
    FrameEvidence,
    VisualEvidence,
    VisualEvent,
)

from lecture_memory.interfaces.frame_selector import IFrameSelector
from lecture_memory.interfaces.vision import IVisionModel
from lecture_memory.interfaces.event_former import IVisualEventFormer
from lecture_memory.storage.media_store import MediaStore
from lecture_memory.observability.logger import get_logger, stage_timer


logger = get_logger(__name__)


class VisualProcessingPipeline:

    def __init__(
        self,
        frame_selector: IFrameSelector,
        vision_model: IVisionModel,
        event_former: IVisualEventFormer,
        media_store: MediaStore,
    ) -> None:
        self._selector = frame_selector
        self._vision = vision_model
        self._event_former = event_former
        self._media_store = media_store

    def process(
        self,
        video: Video,
        speech_segments: Optional[List[SpeechSegment]] = None,
    ) -> Tuple[
        List[FrameEvidence],
        List[VisualEvidence],
        List[VisualEvent],
    ]:

        if speech_segments is None:
            speech_segments = []

        with stage_timer(
            logger,
            "visual_processing",
            video_id=video.video_id,
        ):

            frames_dir = self._media_store.frames_dir_for(
                video.video_id
            )

            # ====================================================
            # Stage 1: cheap frame selection
            # ====================================================

            frames = self._selector.select_frames(
                video_id=video.video_id,
                video_path=video.source_uri,
                output_dir=frames_dir,
            )

            logger.info(
                f"frames_selected | "
                f"video_id={video.video_id} | "
                f"count={len(frames)}"
            )

            # ====================================================
            # Stage 2: vision model
            # ====================================================

            visual_evidence: List[VisualEvidence] = []

            prev_context = None

            for frame in frames:

                try:

                    ve = self._vision.describe_frame(
                        frame,
                        context=prev_context,
                    )

                    visual_evidence.append(ve)

                    prev_context = ve.caption

                except Exception as exc:

                    logger.error(
                        f"vision_failed | "
                        f"frame_id={frame.frame_id} | "
                        f"error={exc}"
                    )

            # ====================================================
            # DEBUG: speech captions
            # ====================================================

            print("\n" + "=" * 70)
            print("SPEECH CAPTIONS")
            print("=" * 70)

            for speech in speech_segments:

                print(
                    f"[{speech.start_time:.2f}s - "
                    f"{speech.end_time:.2f}s] "
                    f"{speech.text}"
                )

            # ====================================================
            # DEBUG: visual captions
            # ====================================================

            print("\n" + "=" * 70)
            print("VISUAL CAPTIONS")
            print("=" * 70)

            for visual in visual_evidence:

                print(
                    f"[{visual.timestamp:.2f}s] "
                    f"{visual.caption}"
                )

            print("=" * 70)
            print(
                f"SPEECH_COUNT={len(speech_segments)} | "
                f"VISUAL_COUNT={len(visual_evidence)}"
            )
            print("=" * 70 + "\n")

            # ====================================================
            # Stage 3: multimodal event formation
            # ====================================================

            events = self._event_former.form_events(
                video.video_id,
                visual_evidence,
                speech_segments,
            )

            logger.info(
                f"visual_events | "
                f"video_id={video.video_id} | "
                f"count={len(events)}"
            )

            return (
                frames,
                visual_evidence,
                events,
            )