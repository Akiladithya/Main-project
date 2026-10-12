"""
orchestrator/qa_orchestrator.py
--------------------------------
Coordinates: query → retrieve → generate → verify → respond.
"""
from __future__ import annotations

from typing import Optional

from lecture_memory.domain.models import VerificationResult
from lecture_memory.pipeline.qa import QuestionAnsweringPipeline
from lecture_memory.observability.logger import get_logger, stage_timer

logger = get_logger(__name__)


class QAOrchestrator:
    def __init__(self, qa_pipeline: QuestionAnsweringPipeline) -> None:
        self._pipeline = qa_pipeline

    def ask(
        self,
        question: str,
        video_id: Optional[str] = None,
    ) -> VerificationResult:
        with stage_timer(logger, "qa_orchestrator"):
            return self._pipeline.answer(question, video_id=video_id)
