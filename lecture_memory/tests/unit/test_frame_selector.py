"""
Tests for OpenCVFrameSelector — requires opencv only, no API key.
"""
import os
import tempfile
import pytest
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@pytest.fixture
def synthetic_video(tmp_path):
    """Create a minimal synthetic MP4 for testing."""
    if not HAS_CV2:
        pytest.skip("opencv not installed")
    path = str(tmp_path / "test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10, (320, 240))
    for i in range(100):  # 10 seconds at 10fps
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = (i * 2, 50, 100)
        writer.write(frame)
    writer.release()
    return path


@pytest.mark.skipif(not HAS_CV2, reason="opencv not installed")
def test_frame_selector_returns_frames(synthetic_video, tmp_path):
    from lecture_memory.adapters.opencv_frame_selector import OpenCVFrameSelector
    selector = OpenCVFrameSelector(interval_sec=2.0, max_frames=50)
    frames = selector.select_frames(
        video_id="test_v1",
        video_path=synthetic_video,
        output_dir=str(tmp_path / "frames"),
    )
    assert len(frames) > 0
    assert all(f.video_id == "test_v1" for f in frames)
    assert all(os.path.exists(f.image_uri) for f in frames)


@pytest.mark.skipif(not HAS_CV2, reason="opencv not installed")
def test_frame_selector_temporal_order(synthetic_video, tmp_path):
    from lecture_memory.adapters.opencv_frame_selector import OpenCVFrameSelector
    selector = OpenCVFrameSelector(interval_sec=1.0)
    frames = selector.select_frames("v1", synthetic_video, str(tmp_path / "frames"))
    timestamps = [f.timestamp for f in frames]
    assert timestamps == sorted(timestamps)


@pytest.mark.skipif(not HAS_CV2, reason="opencv not installed")
def test_frame_selector_respects_max(synthetic_video, tmp_path):
    from lecture_memory.adapters.opencv_frame_selector import OpenCVFrameSelector
    selector = OpenCVFrameSelector(interval_sec=0.5, max_frames=5)
    frames = selector.select_frames("v1", synthetic_video, str(tmp_path / "frames"))
    assert len(frames) <= 5
