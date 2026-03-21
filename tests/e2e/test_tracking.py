"""
E2E tracking: validate model.track() on a real video with multiple pedestrians.

Downloads a short test video (7.3 MB, 13.6s) from Roboflow's public CDN and
runs ByteTrack through several LibreYOLO detectors, checking that:
  - track IDs are assigned and consistent across frames
  - the same person keeps the same ID across consecutive frames
  - no duplicate IDs within a single frame
  - the tracker works with every supported model family (YOLOX, YOLO9, RF-DETR)

The video auto-downloads and is cached at ~/.cache/libreyolo/tracking/ — no
manual setup needed.

Usage:
    pytest tests/e2e/test_tracking.py -v -m e2e
    pytest tests/e2e/test_tracking.py::TestTrackingYOLOX -v
    pytest tests/e2e/test_tracking.py -k "rfdetr" -v
"""

import urllib.request
from pathlib import Path

import pytest
import torch

from libreyolo import LibreYOLO

from .conftest import (
    cuda_cleanup,
    requires_cuda,
    requires_rfdetr,
)

pytestmark = [pytest.mark.e2e, requires_cuda]

VIDEO_URL = "https://media.roboflow.com/supervision/video-examples/people-walking.mp4"
VIDEO_CACHE = Path.home() / ".cache" / "libreyolo" / "tracking"
VIDEO_PATH = VIDEO_CACHE / "people-walking.mp4"


def download_tracking_video():
    """Download the test video from Roboflow if not already cached."""
    if VIDEO_PATH.exists():
        return
    print(f"\nDownloading tracking test video from {VIDEO_URL} ...")
    VIDEO_CACHE.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(VIDEO_URL, str(VIDEO_PATH))
    print(f"Video cached at {VIDEO_PATH}")


@pytest.fixture(scope="module")
def video_path():
    """Download the test video once per module, skip if network unavailable."""
    try:
        download_tracking_video()
    except Exception as exc:
        pytest.skip(f"Cannot download test video: {exc}")
    return VIDEO_PATH


# ── helpers ──────────────────────────────────────────────────────────────────


def _run_tracker(model, video, n_frames=30):
    """Run model.track() and collect first *n_frames* results."""
    frames = []
    for i, result in enumerate(model.track(video)):
        frames.append(result)
        if i + 1 >= n_frames:
            break
    return frames


def _ids(result):
    """Extract track IDs as a Python set."""
    if result.track_id is None:
        return set()
    return set(result.track_id.tolist())


# ── YOLOX ────────────────────────────────────────────────────────────────────


class TestTrackingYOLOX:
    """Tracking e2e with the smallest YOLOX model."""

    @pytest.fixture(scope="class")
    def model(self):
        m = LibreYOLO("LibreYOLOXn.pt")
        yield m
        del m
        cuda_cleanup()

    def test_track_ids_assigned(self, model, video_path):
        """Every frame should have at least one tracked object with an ID."""
        frames = _run_tracker(model, video_path, n_frames=10)
        assert len(frames) == 10
        for i, f in enumerate(frames):
            assert f.track_id is not None, f"Frame {i}: track_id is None"
            assert isinstance(f.track_id, torch.Tensor)
            assert len(f.track_id) == len(f), f"Frame {i}: track_id length mismatch"
            assert len(f) > 0, f"Frame {i}: no detections"

    def test_id_consistency_across_frames(self, model, video_path):
        """Core IDs should persist across consecutive frames."""
        frames = _run_tracker(model, video_path, n_frames=20)

        stable_count = 0
        for i in range(1, len(frames)):
            prev_ids = _ids(frames[i - 1])
            curr_ids = _ids(frames[i])
            if not prev_ids or not curr_ids:
                continue
            overlap = prev_ids & curr_ids
            survival_rate = len(overlap) / len(prev_ids)
            if survival_rate >= 0.5:
                stable_count += 1

        assert stable_count >= len(frames) // 2, (
            f"Only {stable_count}/{len(frames)-1} frame-pairs had >=50% ID overlap"
        )

    def test_ids_are_positive_integers(self, model, video_path):
        """All track IDs should be positive integers."""
        frames = _run_tracker(model, video_path, n_frames=5)
        for i, f in enumerate(frames):
            if f.track_id is not None and len(f.track_id) > 0:
                assert (f.track_id > 0).all(), (
                    f"Frame {i}: non-positive IDs: {f.track_id.tolist()}"
                )
                assert f.track_id.dtype == torch.int64

    def test_no_duplicate_ids_in_frame(self, model, video_path):
        """Within a single frame, each ID should be unique."""
        frames = _run_tracker(model, video_path, n_frames=10)
        for i, f in enumerate(frames):
            ids = f.track_id.tolist() if f.track_id is not None else []
            assert len(ids) == len(set(ids)), f"Frame {i}: duplicate IDs: {ids}"

    def test_multiple_objects_tracked(self, model, video_path):
        """Video has many pedestrians — should track many distinct objects."""
        frames = _run_tracker(model, video_path, n_frames=10)
        all_ids = set()
        for f in frames:
            all_ids |= _ids(f)
        assert len(all_ids) >= 10, (
            f"Only {len(all_ids)} unique IDs across 10 frames"
        )

    def test_save_creates_annotated_frames(self, model, video_path, tmp_path):
        """save=True should write annotated JPEG frames to output_path."""
        out = tmp_path / "track_output"
        frames = []
        for i, r in enumerate(model.track(video_path, save=True, output_path=str(out))):
            frames.append(r)
            if i >= 4:
                break

        assert out.exists(), "Output directory was not created"
        saved = sorted(out.glob("frame_*.jpg"))
        assert len(saved) == 5, f"Expected 5 saved frames, got {len(saved)}"
        assert saved[0].name == "frame_000000.jpg"
        assert saved[-1].name == "frame_000004.jpg"
        # Verify files are non-empty JPEGs
        for f in saved:
            assert f.stat().st_size > 1000, f"{f.name} is suspiciously small"


# ── YOLO9 ────────────────────────────────────────────────────────────────────


class TestTrackingYOLO9:
    """Tracking e2e with the smallest YOLO9 model."""

    @pytest.fixture(scope="class")
    def model(self):
        m = LibreYOLO("LibreYOLO9t.pt")
        yield m
        del m
        cuda_cleanup()

    def test_track_produces_results(self, model, video_path):
        """YOLO9 tracker should produce tracked results."""
        frames = _run_tracker(model, video_path, n_frames=5)
        assert len(frames) == 5
        for f in frames:
            assert f.track_id is not None
            assert len(f) > 0

    def test_id_consistency(self, model, video_path):
        """IDs should persist across consecutive frames."""
        frames = _run_tracker(model, video_path, n_frames=10)
        stable = sum(
            1
            for i in range(1, len(frames))
            if len(_ids(frames[i - 1]) & _ids(frames[i]))
            >= len(_ids(frames[i - 1])) * 0.5
        )
        assert stable >= len(frames) // 2

    def test_no_duplicate_ids_in_frame(self, model, video_path):
        """Within a single frame, each ID should be unique."""
        frames = _run_tracker(model, video_path, n_frames=10)
        for i, f in enumerate(frames):
            ids = f.track_id.tolist() if f.track_id is not None else []
            assert len(ids) == len(set(ids)), f"Frame {i}: duplicate IDs: {ids}"


# ── RF-DETR ──────────────────────────────────────────────────────────────────


@requires_rfdetr
class TestTrackingRFDETR:
    """Tracking e2e with the smallest RF-DETR model."""

    @pytest.fixture(scope="class")
    def model(self):
        m = LibreYOLO("LibreRFDETRn.pt")
        yield m
        del m
        cuda_cleanup()

    def test_track_produces_results(self, model, video_path):
        """RF-DETR tracker should produce tracked results."""
        frames = _run_tracker(model, video_path, n_frames=5)
        assert len(frames) == 5
        for f in frames:
            assert f.track_id is not None
            assert len(f) > 0

    def test_id_consistency(self, model, video_path):
        """IDs should persist across consecutive frames."""
        frames = _run_tracker(model, video_path, n_frames=10)
        stable = sum(
            1
            for i in range(1, len(frames))
            if len(_ids(frames[i - 1]) & _ids(frames[i]))
            >= len(_ids(frames[i - 1])) * 0.5
        )
        assert stable >= len(frames) // 2

    def test_no_duplicate_ids_in_frame(self, model, video_path):
        """Within a single frame, each ID should be unique."""
        frames = _run_tracker(model, video_path, n_frames=10)
        for i, f in enumerate(frames):
            ids = f.track_id.tolist() if f.track_id is not None else []
            assert len(ids) == len(set(ids)), f"Frame {i}: duplicate IDs: {ids}"
