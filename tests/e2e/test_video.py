"""
E2E video inference: validate model() on a real video with pedestrians.

Downloads a short test video (7.3 MB, 13.6s) from the LibreYOLO HF repo and
runs video inference through LibreYOLO detectors, checking that:
  - stream=True yields a generator of Results
  - stream=False collects Results into a list
  - vid_stride skips frames correctly
  - save=True writes a valid output video
  - show=False does not crash (show=True requires a display)
  - frame_idx is set correctly on each result
  - detections are found (the video contains many pedestrians)

The video auto-downloads and is cached at ~/.cache/libreyolo/tracking/ — no
manual setup needed.

Usage:
    pytest tests/e2e/test_video.py -v -m e2e
    pytest tests/e2e/test_video.py -k "yolo9" -v
"""

import urllib.request
from pathlib import Path

import cv2
import pytest

from libreyolo import LibreYOLO

from .conftest import cuda_cleanup

pytestmark = pytest.mark.e2e

VIDEO_URL = "https://huggingface.co/datasets/LibreYOLO/test-assets/resolve/main/videos/people-walking.mp4"
VIDEO_CACHE = Path.home() / ".cache" / "libreyolo" / "tracking"
VIDEO_PATH = VIDEO_CACHE / "people-walking.mp4"


def download_video():
    """Download the test video from Roboflow if not already cached."""
    if VIDEO_PATH.exists():
        return
    print(f"\nDownloading test video from {VIDEO_URL} ...")
    VIDEO_CACHE.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(VIDEO_URL, str(VIDEO_PATH))
    print(f"Video cached at {VIDEO_PATH}")


@pytest.fixture(scope="module")
def video_path():
    """Download the test video once per module, skip if network unavailable."""
    try:
        download_video()
    except Exception as exc:
        pytest.skip(f"Cannot download test video: {exc}")
    return str(VIDEO_PATH)


@pytest.fixture(scope="module")
def model():
    """Load the smallest YOLO9 model once per module."""
    m = LibreYOLO("LibreYOLO9t.pt")
    yield m
    del m
    cuda_cleanup()


# ---------------------------------------------------------------------------
# stream=True (generator)
# ---------------------------------------------------------------------------


class TestVideoStream:
    """Tests for generator-based video inference."""

    def test_stream_returns_generator(self, model, video_path):
        gen = model(video_path, stream=True, conf=0.25)
        assert hasattr(gen, "__next__"), "stream=True should return a generator"
        # Consume just one frame to verify it works
        result = next(gen)
        assert result is not None
        # Clean up generator
        gen.close()

    def test_stream_yields_all_frames(self, model, video_path):
        results = list(model(video_path, stream=True, conf=0.25))
        # Video has 341 frames
        assert len(results) == 341

    def test_stream_results_have_frame_idx(self, model, video_path):
        indices = []
        for i, result in enumerate(model(video_path, stream=True, conf=0.25)):
            indices.append(result.frame_idx)
            if i >= 9:
                break
        assert indices == list(range(10))

    def test_stream_results_have_path(self, model, video_path):
        result = next(iter(model(video_path, stream=True, conf=0.25)))
        assert result.path == video_path

    def test_stream_detects_people(self, model, video_path):
        """The video has many pedestrians — we should detect objects."""
        total_dets = 0
        for i, result in enumerate(model(video_path, stream=True, conf=0.25)):
            total_dets += len(result)
            if i >= 29:
                break
        # 30 frames of people walking — should find plenty of detections
        assert total_dets > 100, f"Only {total_dets} detections in 30 frames"


# ---------------------------------------------------------------------------
# stream=False (list)
# ---------------------------------------------------------------------------


class TestVideoList:
    """Tests for list-based video inference (stream=False)."""

    def test_list_mode_returns_list(self, model, video_path):
        # Use vid_stride=10 to keep it fast
        results = model(video_path, conf=0.25, vid_stride=10)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_list_mode_results_have_frame_idx(self, model, video_path):
        results = model(video_path, conf=0.25, vid_stride=10)
        for result in results:
            assert result.frame_idx is not None


# ---------------------------------------------------------------------------
# vid_stride
# ---------------------------------------------------------------------------


class TestVidStride:
    """Tests for frame skipping."""

    def test_vid_stride_1(self, model, video_path):
        results = list(model(video_path, stream=True, conf=0.25))
        assert len(results) == 341

    def test_vid_stride_2(self, model, video_path):
        results = list(model(video_path, stream=True, conf=0.25, vid_stride=2))
        assert len(results) == 171  # ceil(341/2)

    def test_vid_stride_5(self, model, video_path):
        results = list(model(video_path, stream=True, conf=0.25, vid_stride=5))
        assert len(results) == 69  # ceil(341/5)

    def test_vid_stride_frame_indices(self, model, video_path):
        results = list(model(video_path, stream=True, conf=0.25, vid_stride=5))
        indices = [r.frame_idx for r in results[:5]]
        assert indices == [0, 5, 10, 15, 20]


# ---------------------------------------------------------------------------
# save=True
# ---------------------------------------------------------------------------


class TestVideoSave:
    """Tests for saving annotated output video."""

    def test_save_creates_video(self, model, video_path, tmp_path):
        output = str(tmp_path / "output.mp4")
        # Use vid_stride=10 to keep it fast
        for _ in model(video_path, stream=True, save=True, output_path=output,
                       conf=0.25, vid_stride=10):
            pass

        assert Path(output).exists()
        assert Path(output).stat().st_size > 1000

    def test_saved_video_is_valid(self, model, video_path, tmp_path):
        output = str(tmp_path / "output.mp4")
        n_input = 0
        for _ in model(video_path, stream=True, save=True, output_path=output,
                       conf=0.25, vid_stride=10):
            n_input += 1

        cap = cv2.VideoCapture(output)
        assert cap.isOpened()
        n_output = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        assert n_output == n_input
        assert width == 1920
        assert height == 1080

    def test_save_auto_path(self, model, video_path):
        """save=True without output_path should create runs/detect/predict*."""
        for _ in model(video_path, stream=True, save=True, conf=0.25, vid_stride=50):
            pass
        # Check that some predict directory was created
        predict_dirs = list(Path("runs/detect").glob("predict*"))
        assert len(predict_dirs) > 0
        mp4s = []
        for d in predict_dirs:
            mp4s.extend(d.glob("*.mp4"))
        assert len(mp4s) > 0


# ---------------------------------------------------------------------------
# conf / classes filters
# ---------------------------------------------------------------------------


class TestVideoFilters:
    """Tests for conf/classes filtering during video inference."""

    def test_high_conf_fewer_detections(self, model, video_path):
        low_conf_dets = 0
        high_conf_dets = 0
        for i, r in enumerate(model(video_path, stream=True, conf=0.1, vid_stride=10)):
            low_conf_dets += len(r)
            if i >= 9:
                break

        for i, r in enumerate(model(video_path, stream=True, conf=0.7, vid_stride=10)):
            high_conf_dets += len(r)
            if i >= 9:
                break

        assert high_conf_dets < low_conf_dets

    def test_classes_filter(self, model, video_path):
        """Filter to class 0 (person) should still find detections."""
        total = 0
        for i, r in enumerate(model(video_path, stream=True, conf=0.25,
                                     classes=[0], vid_stride=10)):
            total += len(r)
            if r.boxes is not None and len(r) > 0:
                # All detections should be class 0
                assert (r.boxes.cls == 0).all()
            if i >= 9:
                break
        assert total > 0


# ---------------------------------------------------------------------------
# orig_shape
# ---------------------------------------------------------------------------


class TestVideoResultShape:
    """Tests for result metadata correctness."""

    def test_orig_shape_matches_video(self, model, video_path):
        result = next(iter(model(video_path, stream=True, conf=0.25)))
        # Video is 1920x1080, orig_shape is (H, W)
        assert result.orig_shape == (1080, 1920)
