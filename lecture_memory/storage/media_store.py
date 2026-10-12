"""
storage/media_store.py
----------------------
File-based storage for raw videos and extracted frames.
The vector database stores semantic knowledge — raw media lives here.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from lecture_memory.config.settings import settings
from lecture_memory.observability.logger import get_logger

logger = get_logger(__name__)


class MediaStore:
    def __init__(
        self,
        media_dir: str = settings.media_storage_dir,
        frames_dir: str = settings.frames_storage_dir,
    ):
        self.media_dir = Path(media_dir)
        self.frames_dir = Path(frames_dir)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def save_video(self, video_id: str, source_path: str) -> str:
        """Copy uploaded video into managed storage. Returns stored URI."""
        dest = self.media_dir / video_id / Path(source_path).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)
        logger.info(f"video_stored | video_id={video_id} | uri={dest}")
        return str(dest)

    def get_video_path(self, video_id: str, filename: str) -> str:
        return str(self.media_dir / video_id / filename)

    def frames_dir_for(self, video_id: str) -> str:
        """Return (and create) the frames directory for a video."""
        d = self.frames_dir / video_id
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    def delete_video(self, video_id: str) -> None:
        """Remove all stored media for a video (data lifecycle)."""
        for d in [self.media_dir / video_id, self.frames_dir / video_id]:
            if d.exists():
                shutil.rmtree(d)
                logger.info(f"media_deleted | video_id={video_id} | dir={d}")
