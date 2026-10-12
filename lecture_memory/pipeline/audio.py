"""
pipeline/audio.py
-----------------
Responsibility: extract audio and produce ordered SpeechSegments.
Delegates transcription to ISpeechTranscriber (replaceable).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import List

from lecture_memory.domain.models import Video, SpeechSegment
from lecture_memory.interfaces.speech import ISpeechTranscriber
from lecture_memory.observability.logger import get_logger, stage_timer

logger = get_logger(__name__)


class AudioProcessingPipeline:
    def __init__(self, transcriber: ISpeechTranscriber) -> None:
        self._transcriber = transcriber

    def process(self, video: Video) -> List[SpeechSegment]:
        with stage_timer(logger, "audio_processing", video_id=video.video_id):
            if not video.has_audio:
                logger.info(f"no_audio | video_id={video.video_id} — skipping transcription")
                return []

            if self._transcriber is None:
                logger.warning(f"no_transcriber | video_id={video.video_id} — whisper not installed")
                return []

            with tempfile.TemporaryDirectory() as tmp:
                wav_path = os.path.join(tmp, "audio.wav")
                self._extract_wav(video.source_uri, wav_path)
                segments = self._transcriber.transcribe(video.video_id, wav_path)

            self._validate(segments, video.video_id)
            logger.info(f"speech_segments | video_id={video.video_id} | count={len(segments)}")
            return segments

    def _extract_wav(self, video_path: str, out_path: str) -> None:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is required for audio extraction.")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", out_path],
            capture_output=True, check=True, timeout=300,
        )

    def _validate(self, segments: List[SpeechSegment], video_id: str) -> None:
        for seg in segments:
            assert seg.video_id == video_id, "Segment belongs to wrong video."
            assert seg.end_time >= seg.start_time, "Invalid timestamp order."
            assert seg.text.strip(), "Empty segment text."
