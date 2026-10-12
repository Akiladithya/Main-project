"""
orchestrator/job_state.py
-------------------------
State machine for processing jobs.
PENDING -> RUNNING -> COMPLETED
RUNNING -> FAILED | RETRYING -> RUNNING
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, Optional

from lecture_memory.domain.models import ProcessingJob
from lecture_memory.domain.enums import ProcessingStatus
from lecture_memory.observability.logger import get_logger

logger = get_logger(__name__)


class JobStateManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, ProcessingJob] = {}

    def create_job(self, video_id: str) -> ProcessingJob:
        job = ProcessingJob(
            job_id=str(uuid.uuid4()),
            video_id=video_id,
            status=ProcessingStatus.PENDING,
        )
        self._jobs[job.job_id] = job
        logger.info(f"job_created | job_id={job.job_id} | video_id={video_id}")
        return job

    def transition(
        self,
        job_id: str,
        status: ProcessingStatus,
        stage: Optional[str] = None,
        error: Optional[str] = None,
    ) -> ProcessingJob:
        job = self._jobs[job_id]
        job.status = status
        job.updated_at = datetime.utcnow()
        if stage:
            job.stage = stage
        if error:
            job.error = error
        logger.info(
            f"job_transition | job_id={job_id} | status={status} | stage={job.stage}"
        )
        return job

    def get(self, job_id: str) -> Optional[ProcessingJob]:
        return self._jobs.get(job_id)

    def mark_running(self, job_id: str, stage: str) -> ProcessingJob:
        return self.transition(job_id, ProcessingStatus.RUNNING, stage=stage)

    def mark_completed(self, job_id: str) -> ProcessingJob:
        return self.transition(job_id, ProcessingStatus.COMPLETED)

    def mark_failed(self, job_id: str, error: str) -> ProcessingJob:
        return self.transition(job_id, ProcessingStatus.FAILED, error=error)
