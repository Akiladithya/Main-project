"""
adapters/gemini_generator.py
-----------------------------
Implements IAnswerGenerator using Gemini text model.
Reasons over retrieved evidence — does not invent timestamps or facts.
"""
from __future__ import annotations

import json
import re
from typing import List

from google import genai

from lecture_memory.domain.models import RetrievedEvidence, GeneratedAnswer
from lecture_memory.interfaces.generator import IAnswerGenerator
from lecture_memory.config.settings import settings
from lecture_memory.observability.logger import get_logger

logger = get_logger(__name__)

_PROMPT = """You are a CS lecture assistant. Answer the student's question using ONLY the evidence below.

QUESTION: {question}

EVIDENCE (ranked by relevance):
{evidence_block}

Rules:
- Base your answer strictly on the evidence provided.
- Do not invent facts or timestamps not present in the evidence.
- If evidence is insufficient, say so explicitly.
- Cite the timestamp from the evidence when relevant.

Respond as valid JSON:
{{
  "answer": "<direct answer>",
  "explanation": "<explanation using evidence>",
  "timestamp_start": <float or null>,
  "timestamp_end": <float or null>,
  "video_id": "<video_id or null>",
  "evidence_ids": ["<concept_id>", ...]
}}"""


class GeminiGeneratorAdapter(IAnswerGenerator):
    def __init__(self, api_key: str = settings.gemini_api_key, model: str = settings.llm_model):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, question: str, evidence: List[RetrievedEvidence]) -> GeneratedAnswer:
        evidence_block = "\n".join(
            f"[{i+1}] concept_id={e.concept.concept_id} | "
            f"t={e.concept.start_time:.1f}s-{e.concept.end_time:.1f}s | "
            f"video={e.concept.video_id}\n"
            f"  speech: {e.concept.speech_text}\n"
            f"  visual: {e.concept.visual_description}"
            for i, e in enumerate(evidence)
        )
        prompt = _PROMPT.format(question=question, evidence_block=evidence_block)

        response = self._client.models.generate_content(model=self._model, contents=prompt)
        raw = (response.text or "").strip()
        data = self._extract_json(raw)

        return GeneratedAnswer(
            question=question,
            answer=data.get("answer", ""),
            explanation=data.get("explanation", ""),
            timestamp_start=data.get("timestamp_start"),
            timestamp_end=data.get("timestamp_end"),
            video_id=data.get("video_id"),
            evidence_ids=data.get("evidence_ids", []),
        )

    def _extract_json(self, text: str) -> dict:
        cleaned = re.sub(r"^```json\s*|^```\s*|```$", "", text, flags=re.MULTILINE).strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]
        return json.loads(cleaned)
