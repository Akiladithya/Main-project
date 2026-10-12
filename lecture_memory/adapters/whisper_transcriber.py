"""
adapters/whisper_transcriber.py
--------------------------------
Implements ISpeechTranscriber using OpenAI Whisper (local).
"""
from __future__ import annotations

import uuid
from typing import List

from lecture_memory.domain.models import SpeechSegment
from lecture_memory.interfaces.speech import ISpeechTranscriber
from lecture_memory.observability.logger import get_logger

logger = get_logger(__name__)


class WhisperTranscriber(ISpeechTranscriber):
    def __init__(self, model_size: str = "base") -> None:
        self._model_size = model_size
        self._model = None  # lazy load

    def transcribe(self, video_id: str, audio_path: str) -> List[SpeechSegment]:
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self._model_size)
            logger.info(f"whisper_loaded | model={self._model_size}")

        result = self._model.transcribe(audio_path, verbose=False)

        segments = []

        for seg in result.get("segments", []):
            text = (seg.get("text") or "").strip()

            if not text:
                continue

            start_time = float(seg["start"])
            end_time = float(seg["end"])

            speech_segment = SpeechSegment(
                segment_id=str(uuid.uuid4()),
                video_id=video_id,
                start_time=start_time,
                end_time=end_time,
                text=text,
            )

            segments.append(speech_segment)

            # Temporary debugging: inspect actual lecture speech.
            print(
                f"SPEECH_DEBUG | "
                f"{start_time:.1f}s-{end_time:.1f}s | "
                f"{text}"
            )

        logger.info(
            f"transcribed | video_id={video_id} | segments={len(segments)}"
        )

        return segments