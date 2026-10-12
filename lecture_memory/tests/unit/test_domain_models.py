"""
Tests for domain models — no external dependencies required.
"""
import pytest
from lecture_memory.domain.models import (
    Video, SpeechSegment, FrameEvidence, VisualEvidence,
    VisualEvent, MultimodalConcept, RetrievedEvidence,
)
from lecture_memory.domain.enums import (
    ProcessingStatus, SelectionReason, VerificationOutcome,
)


def test_video_defaults():
    v = Video(
        video_id="v1", filename="lecture.mp4", source_uri="/tmp/lecture.mp4",
        duration=600.0, fps=30.0, width=1920, height=1080, has_audio=True,
    )
    assert v.status == ProcessingStatus.PENDING
    assert v.processing_version == "1.0"


def test_speech_segment_fields():
    seg = SpeechSegment(
        segment_id="s1", video_id="v1",
        start_time=10.0, end_time=20.0, text="TCP handshake",
    )
    assert seg.end_time >= seg.start_time
    assert seg.video_id == "v1"


def test_frame_evidence_fields():
    fe = FrameEvidence(
        frame_id="f1", video_id="v1", timestamp=15.0,
        image_uri="/tmp/frame.jpg", selection_reason=SelectionReason.VISUAL_CHANGE,
        visual_change_score=0.75,
    )
    assert fe.visual_change_score == 0.75


def test_multimodal_concept_schema_version():
    c = MultimodalConcept(
        concept_id="c1", video_id="v1", subject="Networks", topic="TCP",
        concept="Three-Way Handshake", start_time=250.0, end_time=270.0,
        speech_text="SYN SYN-ACK ACK", visual_description="Diagram of handshake",
        visual_event_id="e1",
    )
    assert c.schema_version == "1.0"
    assert c.embedding == []


def test_retrieved_evidence_combined_score():
    c = MultimodalConcept(
        concept_id="c1", video_id="v1", subject=None, topic=None,
        concept="TCP", start_time=0.0, end_time=10.0,
        speech_text="", visual_description="", visual_event_id=None,
    )
    r = RetrievedEvidence(
        concept=c, semantic_score=0.8, speech_score=0.6,
        visual_score=0.5, temporal_score=1.0,
    )
    assert 0.0 < r.combined_score <= 1.0
