"""Unit tests for video support utilities."""

from pathlib import Path

import numpy as np
import pytest

from libreyolo.utils.video import (
    VideoSource,
    VideoWriter,
    collect_video_results,
    is_video_file,
)

pytestmark = pytest.mark.unit

cv2 = pytest.importorskip("cv2", reason="opencv-python required for video tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_video(tmp_path):
    """Create a tiny 10-frame 64x64 video for testing."""
    path = str(tmp_path / "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, (64, 64))
    for i in range(10):
        # Each frame has a different shade so they're distinguishable
        frame = np.full((64, 64, 3), fill_value=i * 25, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


# ---------------------------------------------------------------------------
# is_video_file
# ---------------------------------------------------------------------------


class TestIsVideoFile:
    def test_video_extensions(self):
        assert is_video_file("clip.mp4")
        assert is_video_file("clip.avi")
        assert is_video_file("clip.mkv")
        assert is_video_file(Path("clip.mov"))

    def test_image_extensions(self):
        assert not is_video_file("photo.jpg")
        assert not is_video_file("photo.png")

    def test_non_string(self):
        assert not is_video_file(42)
        assert not is_video_file(None)
        assert not is_video_file(np.zeros((3, 3, 3)))

    def test_case_insensitive(self):
        assert is_video_file("CLIP.MP4")
        assert is_video_file("Clip.Avi")


# ---------------------------------------------------------------------------
# VideoSource
# ---------------------------------------------------------------------------


class TestVideoSource:
    def test_metadata(self, sample_video):
        vs = VideoSource(sample_video)
        assert vs.width == 64
        assert vs.height == 64
        assert vs.total_frames == 10
        assert vs.fps == pytest.approx(10.0, abs=1.0)
        vs.release()

    def test_iterate_all_frames(self, sample_video):
        vs = VideoSource(sample_video)
        frames = list(vs)
        assert len(frames) == 10
        # Each element is (frame_bgr, frame_idx)
        for frame, idx in frames:
            assert frame.shape == (64, 64, 3)
            assert frame.dtype == np.uint8

    def test_frame_indices_sequential(self, sample_video):
        vs = VideoSource(sample_video)
        indices = [idx for _, idx in vs]
        assert indices == list(range(10))

    def test_vid_stride(self, sample_video):
        vs = VideoSource(sample_video, vid_stride=3)
        frames = list(vs)
        indices = [idx for _, idx in frames]
        # With stride 3 and 10 frames (0-9): should get frames 0, 3, 6, 9
        assert indices == [0, 3, 6, 9]

    def test_vid_stride_2(self, sample_video):
        vs = VideoSource(sample_video, vid_stride=2)
        frames = list(vs)
        indices = [idx for _, idx in frames]
        assert indices == [0, 2, 4, 6, 8]

    def test_vid_stride_larger_than_total(self, sample_video):
        vs = VideoSource(sample_video, vid_stride=100)
        frames = list(vs)
        # Only frame 0 matches (0 % 100 == 0), all others are skipped
        assert len(frames) == 1
        assert frames[0][1] == 0

    def test_invalid_path(self):
        with pytest.raises(ValueError, match="Cannot open video"):
            VideoSource("/nonexistent/video.mp4")

    def test_re_iteration_raises(self, sample_video):
        vs = VideoSource(sample_video)
        list(vs)  # consume once
        with pytest.raises(RuntimeError, match="consumed or released"):
            list(vs)

    def test_context_manager(self, sample_video):
        with VideoSource(sample_video) as vs:
            frames = list(vs)
        assert len(frames) == 10
        # After exiting context, cap should be released
        assert vs._cap is None

    def test_double_release_safe(self, sample_video):
        vs = VideoSource(sample_video)
        vs.release()
        vs.release()  # should not raise

    def test_repr(self, sample_video):
        vs = VideoSource(sample_video)
        r = repr(vs)
        assert "VideoSource" in r
        assert "64x64" in r
        vs.release()


# ---------------------------------------------------------------------------
# VideoWriter
# ---------------------------------------------------------------------------


class TestVideoWriter:
    def test_write_and_read_back(self, tmp_path):
        out_path = str(tmp_path / "output.mp4")
        writer = VideoWriter(out_path, fps=10.0, width=32, height=32)

        for i in range(5):
            frame = np.full((32, 32, 3), fill_value=i * 50, dtype=np.uint8)
            writer.write_frame(frame)
        writer.release()

        # Read back and verify
        cap = cv2.VideoCapture(out_path)
        assert cap.isOpened()
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        assert count == 5

    def test_creates_parent_dirs(self, tmp_path):
        out_path = str(tmp_path / "sub" / "dir" / "output.mp4")
        writer = VideoWriter(out_path, fps=10.0, width=32, height=32)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        writer.write_frame(frame)
        writer.release()
        assert Path(out_path).exists()

    def test_context_manager(self, tmp_path):
        out_path = str(tmp_path / "ctx_output.mp4")
        with VideoWriter(out_path, fps=10.0, width=32, height=32) as writer:
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            writer.write_frame(frame)
        assert Path(out_path).exists()
        assert writer._writer is None  # released


# ---------------------------------------------------------------------------
# Results.frame_idx
# ---------------------------------------------------------------------------


class TestResultsFrameIdx:
    def test_default_none(self):
        import torch

        from libreyolo.utils.results import Boxes, Results

        boxes = Boxes(torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,)))
        result = Results(boxes=boxes, orig_shape=(480, 640))
        assert result.frame_idx is None

    def test_set_frame_idx(self):
        import torch

        from libreyolo.utils.results import Boxes, Results

        boxes = Boxes(torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,)))
        result = Results(boxes=boxes, orig_shape=(480, 640), frame_idx=42)
        assert result.frame_idx == 42

    def test_cpu_preserves_frame_idx(self):
        import torch

        from libreyolo.utils.results import Boxes, Results

        boxes = Boxes(torch.zeros((1, 4)), torch.zeros((1,)), torch.zeros((1,)))
        result = Results(boxes=boxes, orig_shape=(480, 640), frame_idx=7)
        cpu_result = result.cpu()
        assert cpu_result.frame_idx == 7

    def test_repr_includes_frame_idx(self):
        import torch

        from libreyolo.utils.results import Boxes, Results

        boxes = Boxes(torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,)))
        result = Results(boxes=boxes, orig_shape=(480, 640), frame_idx=3)
        assert "frame_idx=3" in repr(result)

    def test_repr_omits_frame_idx_when_none(self):
        import torch

        from libreyolo.utils.results import Boxes, Results

        boxes = Boxes(torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,)))
        result = Results(boxes=boxes, orig_shape=(480, 640))
        assert "frame_idx" not in repr(result)


class TestCollectVideoResults:
    """Tests for collect_video_results()."""

    def test_collects_all_items(self, sample_video):
        """Generator items are collected into a list."""
        items = ["a", "b", "c"]
        result = collect_video_results(iter(items), sample_video, vid_stride=1)
        assert result == ["a", "b", "c"]

    def test_empty_generator(self, sample_video):
        result = collect_video_results(iter([]), sample_video, vid_stride=1)
        assert result == []

    def test_warns_on_large_video(self, tmp_path):
        """Videos with many frames should emit a warning."""
        # Create a video that reports many frames (>500 threshold)
        path = str(tmp_path / "long.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Write 600 frames
        writer = cv2.VideoWriter(path, fourcc, 30.0, (16, 16))
        for _ in range(600):
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
        writer.release()

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            collect_video_results(iter([1, 2, 3]), path, vid_stride=1)
            memory_warnings = [x for x in w if "stream=True" in str(x.message)]
            assert len(memory_warnings) == 1
