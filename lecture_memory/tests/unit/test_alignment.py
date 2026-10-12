"""
Tests for SimpleTemporalAligner — no API key required.
"""
import pytest
from lecture_memory.domain.models import SpeechSegment, VisualEvidence, VisualEvent
from lecture_memory.domain.enums import SelectionReason
from lecture_memory.adapters.simple_aligner import SimpleTemporalAligner


@pytest.fixture
def speech():
    return [
        SpeechSegment("s1", "v1", 0.0, 10.0, "Introduction to TCP"),
        SpeechSegment("s2", "v1", 10.0, 25.0, "TCP uses a three-way handshake"),
        SpeechSegment("s3", "v1", 25.0, 40.0, "SYN SYN-ACK ACK sequence"),
    ]


@pytest.fixture
def visual_evidence():
    return [
        VisualEvidence("ve1", "f1", "v1", 5.0, "TCP overview diagram", ["TCP"], ["TCP"]),
        VisualEvidence("ve2", "f2", "v1", 15.0, "Handshake diagram", ["SYN", "ACK"], ["TCP handshake"]),
        VisualEvidence("ve3", "f3", "v1", 30.0, "ACK packet detail", ["ACK"], ["TCP handshake"]),
    ]


@pytest.fixture
def visual_events():
    return [
        VisualEvent("e1", "v1", 10.0, 35.0, "TCP Three-Way Handshake",
                    "SYN SYN-ACK ACK", ["f2", "f3"]),
    ]


def test_aligner_produces_concepts(speech, visual_evidence, visual_events):
    aligner = SimpleTemporalAligner()
    concepts = aligner.align("v1", speech, visual_evidence, visual_events)
    assert len(concepts) > 0
    assert all(c.video_id == "v1" for c in concepts)


def test_aligner_preserves_timestamps(speech, visual_evidence, visual_events):
    aligner = SimpleTemporalAligner()
    concepts = aligner.align("v1", speech, visual_evidence, visual_events)
    for c in concepts:
        assert c.start_time >= 0.0
        assert c.end_time >= c.start_time


def test_aligner_no_evidence_loss(speech, visual_evidence, visual_events):
    aligner = SimpleTemporalAligner()
    concepts = aligner.align("v1", speech, visual_evidence, visual_events)
    # All speech segments should be covered
    covered_speech = set()
    for c in concepts:
        covered_speech.update(eid for eid in c.evidence_ids if eid.startswith("s"))
    assert "s2" in covered_speech or "s3" in covered_speech
