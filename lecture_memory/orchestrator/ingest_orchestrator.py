
"""
orchestrator/ingest_orchestrator.py

------------------------------------

Coordinates: ingestion → audio → visual → alignment → indexing.

Does not implement any algorithm — delegates to pipelines.

Tracks job state and supports resumability.
"""

from __future__ import annotations

from lecture_memory.domain.models import Video, ProcessingJob
from lecture_memory.domain.enums import ProcessingStatus

from lecture_memory.pipeline.ingestion import VideoIngestionPipeline
from lecture_memory.pipeline.audio import AudioProcessingPipeline
from lecture_memory.pipeline.visual import VisualProcessingPipeline
from lecture_memory.pipeline.alignment import CrossModalAlignmentPipeline
from lecture_memory.pipeline.indexing import MemoryIndexingPipeline

from lecture_memory.storage.media_store import MediaStore
from lecture_memory.orchestrator.job_state import JobStateManager

from lecture_memory.observability.logger import get_logger


logger = get_logger(__name__)


class IngestOrchestrator:

    def __init__(
        self,
        ingestion: VideoIngestionPipeline,
        audio: AudioProcessingPipeline,
        visual: VisualProcessingPipeline,
        alignment: CrossModalAlignmentPipeline,
        indexing: MemoryIndexingPipeline,
        media_store: MediaStore,
        job_manager: JobStateManager | None = None,
        state_manager: JobStateManager | None = None,
    ) -> None:

        self._ingestion = ingestion
        self._audio = audio
        self._visual = visual
        self._alignment = alignment
        self._indexing = indexing
        self._media_store = media_store
        self._jobs = job_manager or state_manager or JobStateManager()

    def run(
        self,
        file_path: str,
        subject: str | None = None,
    ) -> ProcessingJob:

        video: Video | None = None
        job: ProcessingJob | None = None

        try:

            # ============================================================
            # Stage 1: Ingest video
            # ============================================================

            video = self._ingestion.ingest(
                file_path,
                subject=subject,
            )

            stored_uri = self._media_store.save_video(
                video.video_id,
                file_path,
            )

            video.source_uri = stored_uri

            job = self._jobs.create_job(
                video.video_id
            )

            # ============================================================
            # Stage 2: Audio processing
            # ============================================================

            self._jobs.mark_running(
                job.job_id,
                "audio",
            )

            speech_segments = self._audio.process(
                video
            )

            logger.info(
                f"audio_complete | "
                f"video_id={video.video_id} | "
                f"speech_segments={len(speech_segments)}"
            )

            # ============================================================
            # Stage 3: Visual processing
            #
            # Speech segments are passed into the visual pipeline so
            # event formation can consider both speech and visual
            # continuity.
            # ============================================================

            self._jobs.mark_running(
                job.job_id,
                "visual",
            )

            frames, visual_evidence, visual_events = (
                self._visual.process(
                    video,
                    speech_segments,
                )
            )

            logger.info(
                f"visual_complete | "
                f"video_id={video.video_id} | "
                f"frames={len(frames)} | "
                f"visual_evidence={len(visual_evidence)} | "
                f"visual_events={len(visual_events)}"
            )

            # ============================================================
            # Stage 4: Cross-modal alignment
            # ============================================================

            self._jobs.mark_running(
                job.job_id,
                "alignment",
            )

            concepts = self._alignment.process(
                video,
                speech_segments,
                visual_evidence,
                visual_events,
            )

            logger.info(
                f"alignment_complete | "
                f"video_id={video.video_id} | "
                f"concepts={len(concepts)}"
            )

            # ============================================================
            # DEBUG: Verify multimodal information before indexing
            # ============================================================

            speech_count = sum(
                1
                for concept in concepts
                if concept.speech_text
                and concept.speech_text.strip()
            )

            visual_count = sum(
                1
                for concept in concepts
                if concept.visual_description
                and concept.visual_description.strip()
            )

            both_count = sum(
                1
                for concept in concepts
                if (
                    concept.speech_text
                    and concept.speech_text.strip()
                    and concept.visual_description
                    and concept.visual_description.strip()
                )
            )

            logger.info(
                f"concept_debug | "
                f"total={len(concepts)} | "
                f"with_speech={speech_count} | "
                f"with_visual={visual_count} | "
                f"with_both={both_count}"
            )

            # Print a few actual concepts so we can verify
            # whether real lecture content is present.

            for concept in concepts[:3]:

                logger.info(
                    f"concept_sample | "
                    f"id={concept.concept_id} | "
                    f"time={concept.start_time:.1f}-"
                    f"{concept.end_time:.1f} | "
                    f"concept={concept.concept!r} | "
                    f"speech={concept.speech_text[:300]!r} | "
                    f"visual={concept.visual_description[:300]!r}"
                )

            # ============================================================
            # Stage 5: Embed + store
            # ============================================================

            self._jobs.mark_running(
                job.job_id,
                "indexing",
            )

            self._indexing.index(
                concepts
            )

            logger.info(
                f"indexing_complete | "
                f"video_id={video.video_id} | "
                f"concepts_indexed={len(concepts)}"
            )

            # ============================================================
            # Complete job
            # ============================================================

            self._jobs.mark_completed(
                job.job_id
            )

        except Exception as exc:

            logger.error(
                f"ingest_failed | error={exc}"
            )

            if job:
                self._jobs.mark_failed(
                    job.job_id,
                    str(exc),
                )

            raise

        return job
