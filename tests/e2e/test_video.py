"""
E2E video inference: validate model() on a real video with pedestrians.

Downloads a short test video (7.3 MB, 13.6s) from the LibreYOLO HF repo and
runs video inference through one model per family (YOLOX, YOLO9, RF-DETR),
checking that:
  - stream=True yields a generator of Results
  - stream=False collects Results into a list
  - vid_stride skips frames correctly
  - save=True writes a valid output video
  - frame_idx is set correctly on each result
  - detections are found (the video contains many pedestrians)
  - conf and classes filters work

The video auto-downloads and is cached at ~/.cache/libreyolo/tracking/ — no
manual setup needed.

Usage:
    pytest tests/e2e/test_video.py -v -m e2e
    pytest tests/e2e/test_video.py -k "yolo9" -v
    pytest tests/e2e/test_video.py -k "rfdetr" -v
"""

import urllib.request
from pathlib import Path

import cv2
import pytest

from libreyolo import LibreYOLO

from .conftest import cuda_cleanup, requires_rfdetr

pytestmark = pytest.mark.e2e

VIDEO_URL = "https://huggingface.co/datasets/LibreYOLO/test-assets/resolve/main/videos/people-walking.mp4"
VIDEO_CACHE = Path.home() / ".cache" / "libreyolo" / "tracking"
VIDEO_PATH = VIDEO_CACHE / "people-walking.mp4"


def download_video():
    """Download the test video from HF if not already cached."""
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
def video_total_frames(video_path):
    """Query the actual frame count from the video (avoids hardcoding)."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


# ---------------------------------------------------------------------------
# Model fixtures — one per family, smallest variant
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def yolox_model():
    m = LibreYOLO("LibreYOLOXn.pt")
    yield m
    del m
    cuda_cleanup()


@pytest.fixture(scope="module")
def yolo9_model():
    m = LibreYOLO("LibreYOLO9t.pt")
    yield m
    del m
    cuda_cleanup()


@pytest.fixture(scope="module")
def rfdetr_model():
    m = LibreYOLO("LibreRFDETRn.pt")
    yield m
    del m
    cuda_cleanup()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_n_frames(model, video_path, n=30, **kwargs):
    """Run inference and collect first n frames."""
    results = []
    for i, r in enumerate(model(video_path, stream=True, **kwargs)):
        results.append(r)
        if i + 1 >= n:
            break
    return results


# ---------------------------------------------------------------------------
# YOLOX
# ---------------------------------------------------------------------------


class TestVideoYOLOX:
    """Video inference with YOLOX."""

    def test_stream_returns_generator(self, yolox_model, video_path):
        gen = yolox_model(video_path, stream=True, conf=0.25)
        assert hasattr(gen, "__next__")
        result = next(gen)
        assert result is not None
        assert result.frame_idx == 0
        gen.close()

    def test_detects_people(self, yolox_model, video_path):
        frames = _collect_n_frames(yolox_model, video_path, n=30, conf=0.25)
        total_dets = sum(len(r) for r in frames)
        assert total_dets > 100, f"Only {total_dets} detections in 30 frames"

    def test_vid_stride(self, yolox_model, video_path):
        frames = _collect_n_frames(yolox_model, video_path, n=1000, conf=0.25, vid_stride=5)
        indices = [r.frame_idx for r in frames[:5]]
        assert indices == [0, 5, 10, 15, 20]

    def test_save(self, yolox_model, video_path, tmp_path):
        output = str(tmp_path / "yolox_output.mp4")
        n = 0
        for r in yolox_model(video_path, stream=True, save=True, output_path=output,
                              conf=0.25, vid_stride=10):
            n += 1
        assert Path(output).exists()
        cap = cv2.VideoCapture(output)
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n
        cap.release()

    def test_orig_shape(self, yolox_model, video_path):
        result = next(iter(yolox_model(video_path, stream=True, conf=0.25)))
        assert result.orig_shape == (1080, 1920)


# ---------------------------------------------------------------------------
# YOLO9
# ---------------------------------------------------------------------------


class TestVideoYOLO9:
    """Video inference with YOLO9."""

    def test_stream_returns_generator(self, yolo9_model, video_path):
        gen = yolo9_model(video_path, stream=True, conf=0.25)
        assert hasattr(gen, "__next__")
        result = next(gen)
        assert result is not None
        assert result.frame_idx == 0
        gen.close()

    def test_detects_people(self, yolo9_model, video_path):
        frames = _collect_n_frames(yolo9_model, video_path, n=30, conf=0.25)
        total_dets = sum(len(r) for r in frames)
        assert total_dets > 100, f"Only {total_dets} detections in 30 frames"

    def test_stream_yields_all_frames(self, yolo9_model, video_path, video_total_frames):
        results = list(yolo9_model(video_path, stream=True, conf=0.25))
        assert len(results) == video_total_frames

    def test_list_mode(self, yolo9_model, video_path):
        results = yolo9_model(video_path, conf=0.25, vid_stride=10)
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert r.frame_idx is not None

    def test_vid_stride_2(self, yolo9_model, video_path, video_total_frames):
        results = list(yolo9_model(video_path, stream=True, conf=0.25, vid_stride=2))
        expected = len(range(0, video_total_frames, 2))
        assert len(results) == expected

    def test_vid_stride_5(self, yolo9_model, video_path, video_total_frames):
        results = list(yolo9_model(video_path, stream=True, conf=0.25, vid_stride=5))
        expected = len(range(0, video_total_frames, 5))
        assert len(results) == expected

    def test_vid_stride_frame_indices(self, yolo9_model, video_path):
        results = list(yolo9_model(video_path, stream=True, conf=0.25, vid_stride=5))
        indices = [r.frame_idx for r in results[:5]]
        assert indices == [0, 5, 10, 15, 20]

    def test_save_creates_valid_video(self, yolo9_model, video_path, tmp_path):
        output = str(tmp_path / "yolo9_output.mp4")
        n_input = 0
        for _ in yolo9_model(video_path, stream=True, save=True, output_path=output,
                              conf=0.25, vid_stride=10):
            n_input += 1

        assert Path(output).exists()
        assert Path(output).stat().st_size > 1000
        cap = cv2.VideoCapture(output)
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n_input
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1920
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 1080
        cap.release()

    def test_save_auto_path(self, yolo9_model, video_path, tmp_path, monkeypatch):
        # Run from tmp_path so runs/detect/ is created there, not in the repo
        monkeypatch.chdir(tmp_path)
        for _ in yolo9_model(video_path, stream=True, save=True, conf=0.25, vid_stride=50):
            pass
        predict_dirs = list((tmp_path / "runs" / "detect").glob("predict*"))
        assert len(predict_dirs) > 0
        mp4s = []
        for d in predict_dirs:
            mp4s.extend(d.glob("*.mp4"))
        assert len(mp4s) > 0

    def test_high_conf_fewer_detections(self, yolo9_model, video_path):
        low = sum(len(r) for r in _collect_n_frames(
            yolo9_model, video_path, n=10, conf=0.1, vid_stride=10))
        high = sum(len(r) for r in _collect_n_frames(
            yolo9_model, video_path, n=10, conf=0.7, vid_stride=10))
        assert high < low

    def test_classes_filter(self, yolo9_model, video_path):
        total = 0
        for r in _collect_n_frames(yolo9_model, video_path, n=10,
                                    conf=0.25, classes=[0], vid_stride=10):
            total += len(r)
            if len(r) > 0:
                assert (r.boxes.cls == 0).all()
        assert total > 0

    def test_orig_shape(self, yolo9_model, video_path):
        result = next(iter(yolo9_model(video_path, stream=True, conf=0.25)))
        assert result.orig_shape == (1080, 1920)

    def test_path_on_results(self, yolo9_model, video_path):
        result = next(iter(yolo9_model(video_path, stream=True, conf=0.25)))
        assert result.path == video_path


# ---------------------------------------------------------------------------
# RF-DETR
# ---------------------------------------------------------------------------


@requires_rfdetr
class TestVideoRFDETR:
    """Video inference with RF-DETR."""

    def test_stream_returns_generator(self, rfdetr_model, video_path):
        gen = rfdetr_model(video_path, stream=True, conf=0.25)
        assert hasattr(gen, "__next__")
        result = next(gen)
        assert result is not None
        assert result.frame_idx == 0
        gen.close()

    def test_detects_people(self, rfdetr_model, video_path):
        frames = _collect_n_frames(rfdetr_model, video_path, n=30, conf=0.25)
        total_dets = sum(len(r) for r in frames)
        assert total_dets > 100, f"Only {total_dets} detections in 30 frames"

    def test_vid_stride(self, rfdetr_model, video_path):
        frames = _collect_n_frames(rfdetr_model, video_path, n=1000,
                                    conf=0.25, vid_stride=5)
        indices = [r.frame_idx for r in frames[:5]]
        assert indices == [0, 5, 10, 15, 20]

    def test_save(self, rfdetr_model, video_path, tmp_path):
        output = str(tmp_path / "rfdetr_output.mp4")
        n = 0
        for r in rfdetr_model(video_path, stream=True, save=True, output_path=output,
                               conf=0.25, vid_stride=10):
            n += 1
        assert Path(output).exists()
        cap = cv2.VideoCapture(output)
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n
        cap.release()

    def test_orig_shape(self, rfdetr_model, video_path):
        result = next(iter(rfdetr_model(video_path, stream=True, conf=0.25)))
        assert result.orig_shape == (1080, 1920)
