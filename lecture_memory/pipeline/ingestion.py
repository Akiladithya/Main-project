"""
pipeline/ingestion.py
---------------------
Responsibility: validate video, extract metadata, register as Video entity.
Does NOT perform semantic understanding.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from lecture_memory.domain.models import Video
from lecture_memory.domain.enums import ProcessingStatus
from lecture_memory.interfaces.ingestion import IVideoIngester
from lecture_memory.config.settings import settings
from lecture_memory.observability.logger import get_logger, stage_timer

logger = get_logger(__name__)


class VideoIngestionPipeline(IVideoIngester):
    def ingest(self, file_path: str, subject: str | None = None) -> Video:
        with stage_timer(logger, "ingestion", file=file_path):
            path = Path(file_path)

            if not path.exists():
                raise FileNotFoundError(f"Video not found: {file_path}")

            ext = path.suffix.lower()
            if ext not in settings.supported_formats:
                raise ValueError(f"Unsupported format '{ext}'")

            metadata = self._extract_metadata(file_path)

            if metadata["duration"] > settings.max_video_duration_sec:
                raise ValueError(
                    f"Video exceeds max duration of {settings.max_video_duration_sec}s"
                )

            return Video(
                video_id=str(uuid.uuid4()),
                filename=path.name,
                source_uri=str(path.resolve()),
                duration=metadata["duration"],
                fps=metadata["fps"],
                width=metadata["width"],
                height=metadata["height"],
                has_audio=metadata["has_audio"],
                subject=subject,
                processing_version=settings.processing_version,
                status=ProcessingStatus.PENDING,
            )

    def _extract_metadata(self, file_path: str) -> dict:
        import cv2, shutil, subprocess, json

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError("Could not open video — corrupted or unsupported codec.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        if fps <= 0 or total_frames <= 0:
            raise ValueError("Video has no readable frames.")

        has_audio = False
        if shutil.which("ffprobe"):
            try:
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "a",
                     "-show_entries", "stream=index", "-of", "json", file_path],
                    capture_output=True, text=True, timeout=30,
                )
                has_audio = len(json.loads(result.stdout or "{}").get("streams", [])) > 0
            except Exception:
                pass

        return {
            "duration": total_frames / fps,
            "fps": fps,
            "width": width,
            "height": height,
            "has_audio": has_audio,
        }
