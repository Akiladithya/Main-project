"""
adapters/link_downloader.py
---------------------------

Processes and downloads lecture videos from online links / URLs:
- YouTube (youtube.com, youtu.be, shorts)
- Vimeo, Loom, Coursera, Google Drive
- Direct HTTP/HTTPS video links (.mp4, .webm, .mov, etc.)

Uses yt-dlp with optimized resolution constraints (e.g. 720p mp4)
to ensure rapid downloads suitable for lecture analysis.
"""

from __future__ import annotations

import os
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yt_dlp

from lecture_memory.config.settings import settings
from lecture_memory.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DownloadedVideoInfo:
    file_path: str
    title: str
    duration_sec: float
    source_url: str
    thumbnail_url: Optional[str] = None


class VideoLinkDownloader:
    """
    Downloads and prepares video files from online links.
    """

    def __init__(self, download_dir: Optional[str] = None) -> None:
        self.download_dir = Path(download_dir or os.path.join(settings.media_storage_dir, "downloads"))
        self.download_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Validate if a string is a well-formed HTTP/HTTPS URL."""
        if not url or not isinstance(url, str):
            return False
        parsed = urllib.parse.urlparse(url.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)

    def get_info(self, url: str) -> dict:
        """
        Extract video metadata (title, duration, thumbnail) without downloading.
        """
        url = url.strip()
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return {
                    "title": info.get("title", "Untitled Lecture"),
                    "duration": float(info.get("duration") or 0.0),
                    "uploader": info.get("uploader") or info.get("channel") or "",
                    "thumbnail": info.get("thumbnail") or None,
                    "description": info.get("description") or "",
                }
            except Exception as e:
                logger.warning(f"Could not extract online metadata: {e}")
                # Fallback for direct video links
                filename = os.path.basename(urllib.parse.urlparse(url).path) or "downloaded_video.mp4"
                return {
                    "title": os.path.splitext(filename)[0],
                    "duration": 0.0,
                    "uploader": "",
                    "thumbnail": None,
                    "description": "",
                }

    def download(
        self,
        url: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> DownloadedVideoInfo:
        """
        Download video from URL to local disk.
        """
        url = url.strip()
        if not self.is_valid_url(url):
            raise ValueError(f"Invalid video URL: '{url}'")

        # Sanitize output filename pattern
        out_template = str(self.download_dir / "%(id)s_%(title).50s.%(ext)s")

        # Hook to report progress
        def _yt_dlp_hook(d):
            if progress_callback and d.get("status") == "downloading":
                total_bytes = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded_bytes = d.get("downloaded_bytes") or 0
                if total_bytes > 0:
                    percent = downloaded_bytes / total_bytes
                    speed = d.get("_speed_str", "")
                    eta = d.get("_eta_str", "")
                    msg = f"Downloading: {int(percent * 100)}% ({speed}, ETA {eta})"
                    progress_callback(percent, msg)
                else:
                    progress_callback(0.5, "Downloading video stream...")
            elif progress_callback and d.get("status") == "finished":
                progress_callback(1.0, "Processing and merging video...")

        # Optimal format for lecture processing: MP4 up to 720p (saves bandwidth while keeping slide text clear)
        ydl_opts = {
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[ext=mp4]/best",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
            "progress_hooks": [_yt_dlp_hook],
            # Windows filename sanitization
            "windowsfilenames": True,
            "noplaylist": True,
        }

        logger.info(f"Downloading video from URL: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if "requested_downloads" in info and info["requested_downloads"]:
                downloaded_file = info["requested_downloads"][0]["filepath"]
            else:
                downloaded_file = ydl.prepare_filename(info)

            # Ensure file exists
            final_path = Path(downloaded_file)
            if not final_path.exists():
                # Check if it was renamed to .mp4
                mp4_candidate = final_path.with_suffix(".mp4")
                if mp4_candidate.exists():
                    final_path = mp4_candidate
                else:
                    raise FileNotFoundError(f"Expected downloaded video at {final_path}, but file was not found.")

            title = info.get("title", final_path.stem)
            duration = float(info.get("duration") or 0.0)
            thumbnail = info.get("thumbnail")

            logger.info(
                f"Video downloaded successfully: {final_path.name} | "
                f"duration={duration:.1f}s | size={final_path.stat().st_size / (1024*1024):.1f}MB"
            )

            return DownloadedVideoInfo(
                file_path=str(final_path.resolve()),
                title=title,
                duration_sec=duration,
                source_url=url,
                thumbnail_url=thumbnail,
            )
