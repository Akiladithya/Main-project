"""
Tests for EvaluationHarness — no API key required.
"""
import pytest
from lecture_memory.evaluation.harness import (
    EvaluationHarness, QASample, _token_f1, _retrieval_metrics,
)
from lecture_memory.domain.models import GeneratedAnswer, VerificationResult, ClaimVerification
from lecture_memory.domain.enums import VerificationOutcome


class MockQAOrchestrator:
    def ask(self, question, video_id=None):
        answer = GeneratedAnswer(
            question=question,
            answer="TCP uses SYN SYN-ACK ACK for connection setup.",
            explanation="Three-way handshake establishes a reliable connection.",
            timestamp_start=8.0,
            timestamp_end=16.0,
            video_id="v1",
            evidence_ids=["c1", "c2"],
        )
        return VerificationResult(
            answer=answer,
            claim_verifications=[
                ClaimVerification("TCP uses SYN SYN-ACK ACK", VerificationOutcome.SUPPORTED, ["c1"], 0.9),
                ClaimVerification("Three-way handshake establishes a reliable connection", VerificationOutcome.SUPPORTED, ["c2"], 0.8),
            ],
            overall_outcome=VerificationOutcome.SUPPORTED,
            verified_answer="TCP uses SYN SYN-ACK ACK for connection setup.",
        )


def test_token_f1_exact():
    assert _token_f1("SYN ACK", "SYN ACK") == 1.0


def test_token_f1_partial():
    score = _token_f1("SYN ACK handshake", "SYN ACK")
    assert 0.0 < score < 1.0


def test_retrieval_metrics_perfect():
    p, r, rr = _retrieval_metrics(["c1", "c2"], ["c1", "c2"])
    assert p == 1.0 and r == 1.0 and rr == 1.0


def test_retrieval_metrics_miss():
    p, r, rr = _retrieval_metrics(["c3", "c4"], ["c1", "c2"])
    assert p == 0.0 and r == 0.0 and rr == 0.0


def test_harness_runs_and_reports():
    harness = EvaluationHarness(MockQAOrchestrator())
    samples = [
        QASample(
            question="What is the TCP three-way handshake?",
            expected_answer="SYN SYN-ACK ACK",
            relevant_concept_ids=["c1", "c2"],
            expected_timestamp_start=8.0,
        )
    ]
    report = harness.run(samples, video_id="v1")
    assert len(report.samples) == 1
    sr = report.samples[0]
    assert sr.precision_at_k == 1.0
    assert sr.recall_at_k == 1.0
    assert sr.reciprocal_rank == 1.0
    assert sr.timestamp_error_sec == 0.0
    assert sr.answer_overlap > 0.0
    assert sr.is_grounded is True
    assert sr.unsupported_claim_rate == 0.0
    assert report.groundedness_rate == 1.0
    summary = report.summary()
    assert "Precision@K" in summary
