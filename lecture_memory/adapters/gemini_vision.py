"""
adapters/gemini_vision.py
--------------------------
Implements IVisionModel using Google Gemini.
Replaceable: swap this file for a Qwen2.5-VL adapter without touching pipelines.
"""
from __future__ import annotations

import uuid
import time
from typing import Optional

import PIL.Image
from google import genai

from lecture_memory.domain.models import FrameEvidence, VisualEvidence
from lecture_memory.interfaces.vision import IVisionModel
from lecture_memory.config.settings import settings
from lecture_memory.observability.logger import get_logger

logger = get_logger(__name__)

_PROMPT = """You are analyzing a single frame from a CS lecture video at timestamp {timestamp:.1f}s.

{context_block}

Identify and describe ONLY what is semantically relevant for a student studying this topic:
- Diagrams, flowcharts, state machines
- Code, pseudocode, algorithms
- Formulas, equations
- Tables, lists
- Labels, annotations, key terms
- Processes, steps, relationships
- Visible CS concepts

Do not invent information not visible in the frame.

Respond in this exact structure:
Caption: <one sentence describing the main visual content>
Entities: <comma-separated list of key entities/terms visible>
Concepts: <comma-separated list of likely CS concepts shown>
"""

_CONTEXT_BLOCK = 'Previous frame context (for continuity only): "{context}"'


class GeminiVisionAdapter(IVisionModel):
    def __init__(self, api_key: str = settings.gemini_api_key, model: str = settings.vision_model):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def describe_frame(
        self, frame: FrameEvidence, context: Optional[str] = None
    ) -> VisualEvidence:
        context_block = _CONTEXT_BLOCK.format(context=context) if context else ""
        prompt = _PROMPT.format(timestamp=frame.timestamp, context_block=context_block)

        for attempt in range(3):
            try:
                image = PIL.Image.open(frame.image_uri)
                response = self._client.models.generate_content(
                    model=self._model, contents=[prompt, image]
                )
                raw = (response.text or "").strip()
                parsed = self._parse(raw)
                return VisualEvidence(
                    visual_id=str(uuid.uuid4()),
                    frame_id=frame.frame_id,
                    video_id=frame.video_id,
                    timestamp=frame.timestamp,
                    caption=parsed["caption"] or "(no caption)",
                    entities=parsed["entities"],
                    concept_candidates=parsed["concepts"],
                )
            except Exception as exc:
                logger.warning(f"vision_retry | attempt={attempt+1} | error={exc}")
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"Vision model failed for frame {frame.frame_id}")

    def _parse(self, text: str) -> dict:
        result = {"caption": "", "entities": [], "concepts": []}
        for line in text.splitlines():
            if line.lower().startswith("caption:"):
                result["caption"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("entities:"):
                result["entities"] = [e.strip() for e in line.split(":", 1)[1].split(",") if e.strip()]
            elif line.lower().startswith("concepts:"):
                result["concepts"] = [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]
        return result
