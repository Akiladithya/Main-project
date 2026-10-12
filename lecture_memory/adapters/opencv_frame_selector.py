"""
adapters/opencv_frame_selector.py

----------------------------------

Implements IFrameSelector using OpenCV.

The selector separates two responsibilities:

1. Candidate sampling
   - Samples the video at the configured interval.
   - This is only a cheap way of observing the video.

2. Important-frame selection
   - Compares consecutive candidates using visual change.
   - Keeps the first frame as visual context.
   - Keeps candidates whose visual change reaches the configured
     visual-change threshold.
   - Consecutive candidates belonging to the same visual transition
     are represented by the strongest local change.
   - max_frames is used only as a hard resource limit.

The selector does NOT assume:
- a fixed number of useful frames,
- a fixed percentage of candidates,
- a fixed minimum time gap,
- a particular lecture domain.

The selector preserves:
- timestamp,
- selection reason,
- visual change score.
"""

from __future__ import annotations

import os
import uuid
from typing import List, Tuple

import cv2
import numpy as np

from lecture_memory.domain.models import FrameEvidence
from lecture_memory.domain.enums import SelectionReason
from lecture_memory.interfaces.frame_selector import IFrameSelector
from lecture_memory.config.settings import settings
from lecture_memory.observability.logger import get_logger


logger = get_logger(__name__)


class OpenCVFrameSelector(IFrameSelector):

    def __init__(
        self,
        interval_sec: float = settings.frame_interval_sec,
        max_frames: int = settings.max_frames_per_video,
        change_threshold: float = settings.visual_change_threshold,
    ) -> None:
        self._interval = interval_sec
        self._max_frames = max_frames
        self._threshold = change_threshold

    # ============================================================
    # PUBLIC API
    # ============================================================

    def select_frames(
        self,
        video_id: str,
        video_path: str,
        output_dir: str,
        interval_sec: float | None = None,
    ) -> List[FrameEvidence]:

        interval = (
            interval_sec
            if interval_sec is not None
            else self._interval
        )

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(
                f"Cannot open video: {video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        )

        duration = (
            total_frames / fps
            if fps > 0
            else 0.0
        )

        timestamps = self._build_timestamps(
            duration=duration,
            interval=interval,
        )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        # ========================================================
        # STAGE 1
        # Cheap candidate extraction
        # ========================================================

        candidates: List[
            Tuple[float, np.ndarray, float]
        ] = []

        previous_frame: np.ndarray | None = None
        previous_hist: np.ndarray | None = None

        try:
            for timestamp in timestamps:

                cap.set(
                    cv2.CAP_PROP_POS_MSEC,
                    timestamp * 1000.0,
                )

                ok, frame = cap.read()

                if not ok or frame is None:
                    continue

                histogram = self._histogram(frame)

                change_score = self._change_score(
                    previous_frame=previous_frame,
                    previous_hist=previous_hist,
                    current_frame=frame,
                    current_hist=histogram,
                )

                candidates.append(
                    (
                        timestamp,
                        frame,
                        change_score,
                    )
                )

                previous_frame = frame
                previous_hist = histogram

        finally:
            cap.release()

        if not candidates:
            logger.warning(
                f"frame_selection_empty | "
                f"video_id={video_id}"
            )
            return []

        # ========================================================
        # STAGE 2
        # Meaningful visual-change selection
        # ========================================================

        selected_candidates = (
            self._select_important_candidates(
                candidates
            )
        )

        # ========================================================
        # STAGE 3
        # Save selected frames
        # ========================================================

        selected: List[FrameEvidence] = []

        for (
            timestamp,
            frame,
            change_score,
            reason,
        ) in selected_candidates:

            frame_id = str(uuid.uuid4())

            image_path = os.path.join(
                output_dir,
                f"{frame_id}.jpg",
            )

            cv2.imwrite(
                image_path,
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    90,
                ],
            )

            selected.append(
                FrameEvidence(
                    frame_id=frame_id,
                    video_id=video_id,
                    timestamp=timestamp,
                    image_uri=image_path,
                    selection_reason=reason,
                    visual_change_score=change_score,
                )
            )

        logger.info(
            f"frame_selection | "
            f"video_id={video_id} | "
            f"candidates={len(candidates)} | "
            f"selected={len(selected)} | "
            f"max_frames={self._max_frames}"
        )

        return selected

    # ============================================================
    # CANDIDATE TIMESTAMPS
    # ============================================================

    def _build_timestamps(
        self,
        duration: float,
        interval: float,
    ) -> List[float]:

        if duration <= 0:
            return []

        if interval <= 0:
            interval = self._interval

        timestamps: List[float] = []

        timestamp = 0.0

        while timestamp < duration:
            timestamps.append(timestamp)
            timestamp += interval

        return timestamps

    # ============================================================
    # IMPORTANT-FRAME SELECTION
    # ============================================================

    def _select_important_candidates(
        self,
        candidates: List[
            Tuple[float, np.ndarray, float]
        ],
    ) -> List[
        Tuple[
            float,
            np.ndarray,
            float,
            SelectionReason,
        ]
    ]:

        if not candidates:
            return []

        candidate_count = len(candidates)

        # --------------------------------------------------------
        # A video always needs its first visual context.
        # --------------------------------------------------------

        selected_indices: set[int] = {0}

        if candidate_count == 1:
            return self._build_selection_result(
                candidates,
                selected_indices,
            )

        # --------------------------------------------------------
        # Find meaningful visual changes.
        #
        # The configured threshold is the only selection
        # sensitivity parameter.
        #
        # No percentage of candidates is assumed.
        # No fixed time gap is assumed.
        # --------------------------------------------------------

        change_indices: List[int] = []

        scores = [
            candidate[2]
            for candidate in candidates
        ]

        for index in range(1, candidate_count):

            score = scores[index]

            if score < self._threshold:
                continue

            # ----------------------------------------------------
            # If several consecutive candidates are part of the
            # same visual transition, retain the strongest one.
            #
            # This uses the actual change scores rather than
            # imposing a time-based gap.
            # ----------------------------------------------------

            previous_score = scores[index - 1]

            next_score = (
                scores[index + 1]
                if index + 1 < candidate_count
                else score
            )

            is_local_peak = (
                score >= previous_score
                and score >= next_score
            )

            if is_local_peak:
                change_indices.append(index)

        # --------------------------------------------------------
        # Add meaningful visual change points.
        # --------------------------------------------------------

        selected_indices.update(change_indices)

        # --------------------------------------------------------
        # If no candidate crossed the configured threshold,
        # preserve the most visually changed candidate.
        #
        # This prevents a visually static video from becoming
        # completely unrepresented while still avoiding arbitrary
        # frame counts.
        # --------------------------------------------------------

        if len(selected_indices) == 1:

            strongest_index = max(
                range(1, candidate_count),
                key=lambda index: candidates[index][2],
            )

            selected_indices.add(
                strongest_index
            )

        # --------------------------------------------------------
        # max_frames is ONLY a resource limit.
        #
        # If the video genuinely contains more significant visual
        # changes than the hardware budget allows, retain the
        # strongest changes.
        # --------------------------------------------------------

        if len(selected_indices) > self._max_frames:

            ranked_indices = sorted(
                selected_indices,
                key=lambda index: candidates[index][2],
                reverse=True,
            )

            # The first frame is mandatory visual context.
            ranked_indices = [
                index
                for index in ranked_indices
                if index != 0
            ]

            allowed_change_count = max(
                self._max_frames - 1,
                0,
            )

            selected_indices = {
                0,
                *ranked_indices[:allowed_change_count],
            }

        return self._build_selection_result(
            candidates,
            selected_indices,
        )

    # ============================================================
    # BUILD RESULT
    # ============================================================

    def _build_selection_result(
        self,
        candidates: List[
            Tuple[float, np.ndarray, float]
        ],
        selected_indices: set[int],
    ) -> List[
        Tuple[
            float,
            np.ndarray,
            float,
            SelectionReason,
        ]
    ]:

        ordered_indices = sorted(
            selected_indices,
            key=lambda index: candidates[index][0],
        )

        result = []

        for index in ordered_indices:

            (
                timestamp,
                frame,
                change_score,
            ) = candidates[index]

            if index == 0:

                reason = (
                    SelectionReason.FORCED_KEYFRAME
                )

            elif change_score >= self._threshold:

                reason = (
                    SelectionReason.VISUAL_CHANGE
                )

            else:

                reason = (
                    SelectionReason.UNIFORM_SAMPLE
                )

            result.append(
                (
                    timestamp,
                    frame,
                    change_score,
                    reason,
                )
            )

        return result

    # ============================================================
    # FRAME REPRESENTATION
    # ============================================================

    def _small_frame(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        return cv2.resize(
            gray,
            (32, 32),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)

    # ============================================================
    # HISTOGRAM
    # ============================================================

    def _histogram(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        hist = cv2.calcHist(
            [gray],
            [0],
            None,
            [64],
            [0, 256],
        )

        return cv2.normalize(
            hist,
            hist,
        ).flatten()

    # ============================================================
    # VISUAL CHANGE SCORE
    # ============================================================

    def _change_score(
        self,
        previous_frame: np.ndarray | None,
        previous_hist: np.ndarray | None,
        current_frame: np.ndarray,
        current_hist: np.ndarray,
    ) -> float:

        # --------------------------------------------------------
        # First candidate establishes the visual baseline.
        # --------------------------------------------------------

        if (
            previous_frame is None
            or previous_hist is None
        ):
            return 0.0

        # --------------------------------------------------------
        # Histogram change
        # --------------------------------------------------------

        histogram_change = float(
            cv2.compareHist(
                previous_hist,
                current_hist,
                cv2.HISTCMP_BHATTACHARYYA,
            )
        )

        # --------------------------------------------------------
        # Structural change
        # --------------------------------------------------------

        previous_small = self._small_frame(
            previous_frame
        )

        current_small = self._small_frame(
            current_frame
        )

        structural_change = float(
            np.mean(
                cv2.absdiff(
                    previous_small,
                    current_small,
                )
            )
        )

        structural_change = min(
            structural_change / 255.0,
            1.0,
        )

        # --------------------------------------------------------
        # Combine independent visual signals.
        #
        # Histogram:
        #   captures global visual distribution.
        #
        # Structural:
        #   captures changes in spatial arrangement.
        #
        # Both are normalized to the same 0-1 scale.
        # --------------------------------------------------------

        score = (
            histogram_change
            + structural_change
        ) / 2.0

        return float(
            min(
                max(score, 0.0),
                1.0,
            )
        )