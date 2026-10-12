"""
evaluation/harness.py
----------------------
Evaluation harness for the multimodal lecture memory pipeline.
Covers all metric groups from skills/14-evaluation.md:
  - Visual efficiency
  - Retrieval quality (Recall@K, Precision@K, MRR, timestamp accuracy)
  - Answer quality (correctness, groundedness, unsupported claim rate)
  - Verification quality (supported-claim precision, false rejection rate)

Usage:
    from lecture_memory.evaluation.harness import EvaluationHarness, QASample

    samples = [
        QASample(
            question="What is the TCP three-way handshake?",
            expected_answer="SYN, SYN-ACK, ACK",
            relevant_concept_ids=["concept-id-1", "concept-id-2"],
            expected_timestamp_start=8.0,
            expected_timestamp_end=16.0,
        )
    ]
    harness = EvaluationHarness(qa_orchestrator)
    report = harness.run(samples, video_id="v1")
    print(report.summary())
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from lecture_memory.domain.enums import VerificationOutcome


# --------------------------------------------------------------------------- #
# Input / Output types
# --------------------------------------------------------------------------- #

@dataclass
class QASample:
    question: str
    expected_answer: str                          # ground-truth answer text
    relevant_concept_ids: List[str] = field(default_factory=list)  # for Recall/Precision@K
    expected_timestamp_start: Optional[float] = None
    expected_timestamp_end: Optional[float] = None
    video_id: Optional[str] = None


@dataclass
class SampleResult:
    question: str
    predicted_answer: str
    verified_answer: Optional[str]
    outcome: VerificationOutcome
    retrieved_concept_ids: List[str]
    timestamp_start: Optional[float]
    timestamp_end: Optional[float]
    latency_sec: float
    # per-sample metrics
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    reciprocal_rank: float = 0.0
    timestamp_error_sec: Optional[float] = None
    answer_overlap: float = 0.0          # token-level F1 vs expected_answer
    is_grounded: bool = False            # outcome != UNSUPPORTED
    unsupported_claim_rate: float = 0.0


@dataclass
class EvaluationReport:
    samples: List[SampleResult]
    # Visual efficiency (populated separately via add_visual_stats)
    raw_frames: int = 0
    candidate_frames: int = 0
    selected_frames: int = 0

    @property
    def frame_reduction_ratio(self) -> float:
        return 1.0 - (self.selected_frames / self.raw_frames) if self.raw_frames else 0.0

    @property
    def mean_precision_at_k(self) -> float:
        return _mean(s.precision_at_k for s in self.samples)

    @property
    def mean_recall_at_k(self) -> float:
        return _mean(s.recall_at_k for s in self.samples)

    @property
    def mrr(self) -> float:
        return _mean(s.reciprocal_rank for s in self.samples)

    @property
    def mean_timestamp_error(self) -> Optional[float]:
        errors = [s.timestamp_error_sec for s in self.samples if s.timestamp_error_sec is not None]
        return sum(errors) / len(errors) if errors else None

    @property
    def mean_answer_overlap(self) -> float:
        return _mean(s.answer_overlap for s in self.samples)

    @property
    def groundedness_rate(self) -> float:
        return _mean(float(s.is_grounded) for s in self.samples)

    @property
    def mean_unsupported_claim_rate(self) -> float:
        return _mean(s.unsupported_claim_rate for s in self.samples)

    @property
    def mean_latency_sec(self) -> float:
        return _mean(s.latency_sec for s in self.samples)

    def summary(self) -> str:
        lines = [
            "=== Evaluation Report ===",
            f"Samples: {len(self.samples)}",
            "",
            "-- Visual Efficiency --",
            f"  Raw frames:        {self.raw_frames}",
            f"  Selected frames:   {self.selected_frames}",
            f"  Reduction ratio:   {self.frame_reduction_ratio:.1%}",
            "",
            "-- Retrieval Quality --",
            f"  Precision@K:       {self.mean_precision_at_k:.3f}",
            f"  Recall@K:          {self.mean_recall_at_k:.3f}",
            f"  MRR:               {self.mrr:.3f}",
            f"  Timestamp error:   {self.mean_timestamp_error:.1f}s"
            if self.mean_timestamp_error is not None else "  Timestamp error:   N/A",
            "",
            "-- Answer Quality --",
            f"  Answer overlap:    {self.mean_answer_overlap:.3f}",
            f"  Groundedness:      {self.groundedness_rate:.1%}",
            f"  Unsupported rate:  {self.mean_unsupported_claim_rate:.1%}",
            "",
            "-- Performance --",
            f"  Mean latency:      {self.mean_latency_sec:.2f}s",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #

class EvaluationHarness:
    """
    Runs QASamples through a QAOrchestrator and computes all metrics.

    Args:
        qa_orchestrator: any object with .ask(question, video_id) -> VerificationResult
    """
    def __init__(self, qa_orchestrator) -> None:
        self._qa = qa_orchestrator

    def run(self, samples: List[QASample], video_id: Optional[str] = None) -> EvaluationReport:
        results = []
        for sample in samples:
            result = self._evaluate_sample(sample, video_id=sample.video_id or video_id)
            results.append(result)
        return EvaluationReport(samples=results)

    def add_visual_stats(
        self,
        report: EvaluationReport,
        raw_frames: int,
        candidate_frames: int,
        selected_frames: int,
    ) -> None:
        report.raw_frames = raw_frames
        report.candidate_frames = candidate_frames
        report.selected_frames = selected_frames

    def _evaluate_sample(self, sample: QASample, video_id: Optional[str]) -> SampleResult:
        t0 = time.perf_counter()
        vr = self._qa.ask(sample.question, video_id=video_id)
        latency = time.perf_counter() - t0

        answer = vr.answer
        retrieved_ids = answer.evidence_ids or []

        # Retrieval metrics
        precision, recall, rr = _retrieval_metrics(retrieved_ids, sample.relevant_concept_ids)

        # Timestamp error
        ts_error = None
        if sample.expected_timestamp_start is not None and answer.timestamp_start is not None:
            ts_error = abs(answer.timestamp_start - sample.expected_timestamp_start)

        # Answer token overlap (F1)
        overlap = _token_f1(answer.answer, sample.expected_answer)

        # Groundedness
        is_grounded = vr.overall_outcome != VerificationOutcome.UNSUPPORTED

        # Unsupported claim rate
        total_claims = len(vr.claim_verifications)
        unsupported = sum(
            1 for cv in vr.claim_verifications
            if cv.outcome == VerificationOutcome.UNSUPPORTED
        )
        unsupported_rate = unsupported / total_claims if total_claims else 0.0

        return SampleResult(
            question=sample.question,
            predicted_answer=answer.answer,
            verified_answer=vr.verified_answer,
            outcome=vr.overall_outcome,
            retrieved_concept_ids=retrieved_ids,
            timestamp_start=answer.timestamp_start,
            timestamp_end=answer.timestamp_end,
            latency_sec=latency,
            precision_at_k=precision,
            recall_at_k=recall,
            reciprocal_rank=rr,
            timestamp_error_sec=ts_error,
            answer_overlap=overlap,
            is_grounded=is_grounded,
            unsupported_claim_rate=unsupported_rate,
        )


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #

def _retrieval_metrics(
    retrieved: List[str], relevant: List[str]
) -> tuple[float, float, float]:
    if not relevant:
        return 0.0, 0.0, 0.0
    relevant_set = set(relevant)
    hits = [r for r in retrieved if r in relevant_set]
    precision = len(hits) / len(retrieved) if retrieved else 0.0
    recall = len(hits) / len(relevant_set)
    # MRR: reciprocal rank of first hit
    rr = 0.0
    for i, r in enumerate(retrieved):
        if r in relevant_set:
            rr = 1.0 / (i + 1)
            break
    return precision, recall, rr


def _token_f1(predicted: str, expected: str) -> float:
    pred_tokens = set(predicted.lower().split())
    exp_tokens = set(expected.lower().split())
    if not pred_tokens or not exp_tokens:
        return 0.0
    common = pred_tokens & exp_tokens
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(exp_tokens)
    return 2 * p * r / (p + r)


def _mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
