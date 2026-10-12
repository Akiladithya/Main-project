"""
Integration test for the full ingestion pipeline in offline/mock mode.
No API key required — vision and LLM calls are bypassed via mock adapters.
"""
from __future__ import annotations

import os
import uuid
import pytest
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from lecture_memory.domain.models import FrameEvidence, VisualEvidence, SpeechSegment
from lecture_memory.domain.enums import SelectionReason, VerificationOutcome
from lecture_memory.adapters.simple_aligner import SimpleTemporalAligner
from lecture_memory.adapters.simple_event_former import SimpleVisualEventFormer
from lecture_memory.adapters.simple_verifier import SimpleEvidenceVerifier


# --------------------------------------------------------------------------- #
# Mock adapters — no API calls
# --------------------------------------------------------------------------- #

class MockVisionModel:
    def describe_frame(self, frame: FrameEvidence, context=None) -> VisualEvidence:
        return VisualEvidence(
            visual_id=str(uuid.uuid4()),
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            timestamp=frame.timestamp,
            caption=f"Mock caption at {frame.timestamp:.1f}s",
            entities=["TCP", "diagram"],
            concept_candidates=["TCP handshake"],
        )


class MockEmbedder:
    @property
    def model_name(self): return "mock"
    def embed(self, text): return [0.1] * 384
    def embed_batch(self, texts): return [[0.1] * 384 for _ in texts]


class MockVectorStore:
    def __init__(self): self._store = {}
    def upsert(self, concepts):
        for c in concepts:
            self._store[c.concept_id] = c
    def search(self, q, top_k=10, **kw): return list(self._store.values())[:top_k]
    def get_by_video(self, vid): return [c for c in self._store.values() if c.video_id == vid]
    def get_by_time_range(self, vid, s, e):
        return [c for c in self._store.values() if c.video_id == vid and c.start_time <= e and c.end_time >= s]
    def delete_by_video(self, vid):
        self._store = {k: v for k, v in self._store.items() if v.video_id != vid}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def synthetic_video(tmp_path):
    if not HAS_CV2:
        pytest.skip("opencv not installed")
    path = str(tmp_path / "lecture.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10, (320, 240))
    for i in range(200):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = (i, 80, 120)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def speech_segments():
    return [
        SpeechSegment("s1", "v1", 0.0, 8.0, "Introduction to TCP protocol"),
        SpeechSegment("s2", "v1", 8.0, 16.0, "TCP uses a three-way handshake"),
        SpeechSegment("s3", "v1", 16.0, 20.0, "SYN SYN-ACK ACK sequence completes connection"),
    ]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not HAS_CV2, reason="opencv not installed")
def test_full_ingest_pipeline(synthetic_video, tmp_path, speech_segments):
    from lecture_memory.adapters.opencv_frame_selector import OpenCVFrameSelector
    from lecture_memory.pipeline.visual import VisualProcessingPipeline
    from lecture_memory.pipeline.alignment import CrossModalAlignmentPipeline
    from lecture_memory.pipeline.indexing import MemoryIndexingPipeline
    from lecture_memory.adapters.simple_aligner import SimpleTemporalAligner
    from lecture_memory.adapters.simple_event_former import SimpleVisualEventFormer
    from lecture_memory.storage.media_store import MediaStore
    from lecture_memory.domain.models import Video
    from lecture_memory.domain.enums import ProcessingStatus

    video = Video(
        video_id="v_test", filename="lecture.mp4",
        source_uri=synthetic_video, duration=20.0,
        fps=10.0, width=320, height=240, has_audio=False,
    )

    media_store = MediaStore(
        media_dir=str(tmp_path / "media"),
        frames_dir=str(tmp_path / "frames"),
    )

    visual_pipeline = VisualProcessingPipeline(
        frame_selector=OpenCVFrameSelector(interval_sec=2.0),
        vision_model=MockVisionModel(),
        event_former=SimpleVisualEventFormer(),
        media_store=media_store,
    )
    frames, visual_evidence, events = visual_pipeline.process(video)

    assert len(frames) > 0
    assert len(visual_evidence) == len(frames)
    assert len(events) > 0

    alignment_pipeline = CrossModalAlignmentPipeline(aligner=SimpleTemporalAligner())
    concepts = alignment_pipeline.process(video, speech_segments, visual_evidence, events)
    assert len(concepts) > 0
    assert all(c.video_id == "v_test" for c in concepts)
    assert all(c.end_time >= c.start_time for c in concepts)

    vector_store = MockVectorStore()
    indexing_pipeline = MemoryIndexingPipeline(
        embedder=MockEmbedder(), vector_store=vector_store
    )
    indexing_pipeline.index(concepts)
    assert len(vector_store._store) == len(concepts)
    assert all(len(c.embedding) == 384 for c in concepts)


def test_qa_pipeline_with_mock_store(speech_segments):
    from lecture_memory.pipeline.qa import QuestionAnsweringPipeline
    from lecture_memory.adapters.temporal_retriever import TemporalRetriever
    from lecture_memory.adapters.simple_verifier import SimpleEvidenceVerifier
    from lecture_memory.domain.models import GeneratedAnswer

    class MockGenerator:
        def generate(self, question, evidence):
            return GeneratedAnswer(
                question=question,
                answer="TCP uses a three-way handshake with SYN, SYN-ACK, and ACK.",
                explanation="As shown in the lecture at 8-16s.",
                timestamp_start=8.0, timestamp_end=16.0,
                video_id="v1",
                evidence_ids=[e.concept.concept_id for e in evidence[:2]],
            )

    # Build a store with some concepts
    from lecture_memory.adapters.simple_aligner import SimpleTemporalAligner
    from lecture_memory.adapters.simple_event_former import SimpleVisualEventFormer

    visual_evidence = [
        VisualEvidence("ve1", "f1", "v1", 8.0, "TCP handshake diagram", ["TCP"], ["TCP handshake"]),
        VisualEvidence("ve2", "f2", "v1", 12.0, "SYN ACK packets", ["SYN", "ACK"], ["TCP handshake"]),
    ]
    events = SimpleVisualEventFormer().form_events("v1", visual_evidence)
    concepts = SimpleTemporalAligner().align("v1", speech_segments, visual_evidence, events)

    vector_store = MockVectorStore()
    embedder = MockEmbedder()
    for c in concepts:
        c.embedding = embedder.embed("")
    vector_store.upsert(concepts)

    pipeline = QuestionAnsweringPipeline(
        retriever=TemporalRetriever(embedder=embedder, vector_store=vector_store),
        generator=MockGenerator(),
        verifier=SimpleEvidenceVerifier(),
    )
    result = pipeline.answer("What is the TCP three-way handshake?", video_id="v1")

    assert result.answer.answer != ""
    assert result.overall_outcome in list(VerificationOutcome)
