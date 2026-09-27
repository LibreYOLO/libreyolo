"""
E2E tracking: validate model.track() on a real video with multiple pedestrians.

Downloads official Supervision video assets from Roboflow's public CDN and
runs tracking through several LibreYOLO detectors, checking that:
  - track IDs are assigned and consistent across frames
  - the same person keeps the same ID across consecutive frames
  - no duplicate IDs within a single frame
  - the tracker works with every supported model family (YOLOX, YOLO9, RF-DETR, YOLO-NAS)

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
from PIL import Image

from libreyolo import LibreYOLO
from libreyolo.utils.video import VideoSource

from .conftest import (
    cuda_cleanup,
    requires_cuda,
    requires_rfdetr,
)

pytestmark = [pytest.mark.e2e, requires_cuda]

VIDEO_URL = "https://media.roboflow.com/supervision/video-examples/people-walking.mp4"
BOTSORT_VIDEO_URL = (
    "https://media.roboflow.com/supervision/video-examples/basketball-1.mp4"
)
VIDEO_CACHE = Path.home() / ".cache" / "libreyolo" / "tracking"
VIDEO_PATH = VIDEO_CACHE / "people-walking.mp4"
BOTSORT_VIDEO_PATH = VIDEO_CACHE / "basketball-1.mp4"

OFFICIAL_YOLONAS_S = Path("downloads/yolonas/yolo_nas_s_coco.pth")


def download_tracking_video():
    """Download the test video from Roboflow if not already cached."""
    if VIDEO_PATH.exists():
        return
    print(f"\nDownloading tracking test video from {VIDEO_URL} ...")
    VIDEO_CACHE.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(VIDEO_URL, str(VIDEO_PATH))
    print(f"Video cached at {VIDEO_PATH}")


def download_botsort_video():
    """Download the basketball BoT-SORT smoke-test video when not cached."""
    if BOTSORT_VIDEO_PATH.exists():
        return
    print(f"\nDownloading BoT-SORT test video from {BOTSORT_VIDEO_URL} ...")
    VIDEO_CACHE.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(BOTSORT_VIDEO_URL, str(BOTSORT_VIDEO_PATH))
    print(f"Video cached at {BOTSORT_VIDEO_PATH}")


@pytest.fixture(scope="module")
def video_path():
    """Download the test video once per module, skip if network unavailable."""
    try:
        download_tracking_video()
    except Exception as exc:
        pytest.skip(f"Cannot download test video: {exc}")
    return VIDEO_PATH


@pytest.fixture(scope="module")
def botsort_video_path():
    """Provide a basketball clip with crossings and similar-looking targets."""
    try:
        download_botsort_video()
    except Exception as exc:
        pytest.skip(f"Cannot download BoT-SORT test video: {exc}")
    return BOTSORT_VIDEO_PATH


# ── helpers ──────────────────────────────────────────────────────────────────


def _run_tracker(model, video, n_frames=30, **track_kwargs):
    """Run model.track() and collect first *n_frames* results."""
    frames = []
    for i, result in enumerate(model.track(video, **track_kwargs)):
        frames.append(result)
        if i + 1 >= n_frames:
            break
    return frames


def _load_image_sequence(video, n_frames=3):
    """Load consecutive video frames as an in-memory RGB image sequence."""
    frames = []
    with VideoSource(video) as source:
        for frame_bgr, _ in source:
            frames.append(Image.fromarray(frame_bgr[:, :, ::-1].copy()))
            if len(frames) >= n_frames:
                break
    return frames


def _ids(result):
    """Extract track IDs as a Python set."""
    if result.track_id is None:
        return set()
    return set(result.track_id.tolist())


# ── YOLOX ────────────────────────────────────────────────────────────────────


@pytest.mark.yolox
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
            f"Only {stable_count}/{len(frames) - 1} frame-pairs had >=50% ID overlap"
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
        assert len(all_ids) >= 10, f"Only {len(all_ids)} unique IDs across 10 frames"

    def test_ocsort_tracker_end_to_end(self, model, video_path):
        """The OC-SORT tracker also yields stable, unique per-frame IDs."""
        frames = _run_tracker(model, video_path, n_frames=20, tracker="ocsort")
        assert len(frames) == 20
        stable = 0
        for i, f in enumerate(frames):
            ids = f.track_id.tolist() if f.track_id is not None else []
            assert len(ids) == len(set(ids)), f"Frame {i}: duplicate IDs: {ids}"
            if i > 0:
                prev = _ids(frames[i - 1])
                curr = _ids(frames[i])
                if prev and curr and len(prev & curr) / len(prev) >= 0.5:
                    stable += 1
        assert stable >= len(frames) // 2, "OC-SORT IDs not stable across frames"

    def test_deepocsort_tracker_end_to_end(self, model, video_path):
        """Deep OC-SORT (appearance ReID) yields stable, unique per-frame IDs.

        The OSNet ReID weights are a published autodownload contract
        (LibreYOLO/LibreReID-osnet); a failed download is a real failure,
        not a skip.
        """
        frames = _run_tracker(model, video_path, n_frames=20, tracker="deepocsort")
        assert len(frames) == 20
        stable = 0
        for i, f in enumerate(frames):
            ids = f.track_id.tolist() if f.track_id is not None else []
            assert len(ids) == len(set(ids)), f"Frame {i}: duplicate IDs: {ids}"
            if i > 0:
                prev = _ids(frames[i - 1])
                curr = _ids(frames[i])
                if prev and curr and len(prev & curr) / len(prev) >= 0.5:
                    stable += 1
        assert stable >= len(frames) // 2, "Deep OC-SORT IDs not stable across frames"

    def test_botsort_tracker_end_to_end(self, model, botsort_video_path):
        """BoT-SORT assigns stable, unique IDs on a real basketball clip."""
        frames = _run_tracker(
            model, botsort_video_path, n_frames=120, tracker="botsort"
        )
        assert len(frames) == 120
        stable = 0
        for i, frame in enumerate(frames):
            ids = frame.track_id.tolist() if frame.track_id is not None else []
            assert len(ids) == len(set(ids)), f"Frame {i}: duplicate IDs: {ids}"
            if i > 0:
                previous = _ids(frames[i - 1])
                current = _ids(frame)
                if previous and current and len(previous & current) / len(previous) >= 0.5:
                    stable += 1
        assert stable >= len(frames) // 2, "BoT-SORT IDs not stable across frames"

    def test_save_creates_annotated_video(self, model, video_path, tmp_path):
        """save=True should write an annotated video to output_path."""
        import cv2

        out = tmp_path / "track_output.mp4"
        frames = []
        for i, r in enumerate(model.track(video_path, save=True, output_path=str(out))):
            frames.append(r)
            if i >= 4:
                break

        assert out.exists(), "Output video was not created"
        assert out.stat().st_size > 1000, "Output video is suspiciously small"
        cap = cv2.VideoCapture(str(out))
        assert cap.isOpened(), "Output video is not a valid video file"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert frame_count == 5, f"Expected 5 frames, got {frame_count}"

    def test_track_raises_on_missing_video(self, model):
        """track() should raise FileNotFoundError for non-existent video."""
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            next(model.track("/tmp/nonexistent_video_12345.mp4"))


# ── YOLO9 ────────────────────────────────────────────────────────────────────


@pytest.mark.flagship_nightly
@pytest.mark.yolo9
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

    def test_track_image_sequence(self, model, video_path):
        frames = _load_image_sequence(video_path)

        results = list(model.track(frames))

        assert len(results) == len(frames)
        assert all(result.track_id is not None for result in results)
        assert all(result.path is None for result in results)

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

    def test_botsort_produces_results(self, model, video_path):
        """The BoT-SORT path works with flagship YOLO9 detectors."""
        frames = _run_tracker(model, video_path, n_frames=5, tracker="botsort")
        assert len(frames) == 5
        assert all(frame.track_id is not None and len(frame) > 0 for frame in frames)


# ── YOLO-NAS ─────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not OFFICIAL_YOLONAS_S.exists(),
    reason="Official YOLO-NAS-S checkpoint not present in downloads/yolonas/",
)
@pytest.mark.yolonas
class TestTrackingYOLONAS:
    """Tracking e2e with the smallest YOLO-NAS model."""

    @pytest.fixture(scope="class")
    def model(self):
        m = LibreYOLO(str(OFFICIAL_YOLONAS_S))
        yield m
        del m
        cuda_cleanup()

    def test_track_produces_results(self, model, video_path):
        """YOLO-NAS tracker should produce tracked results."""
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
@pytest.mark.flagship_nightly
@pytest.mark.rfdetr
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

    def test_track_image_sequence(self, model, video_path):
        frames = _load_image_sequence(video_path)

        results = list(model.track(frames))

        assert len(results) == len(frames)
        assert all(result.track_id is not None for result in results)
        assert all(result.path is None for result in results)

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

    def test_botsort_produces_results(self, model, video_path):
        """The BoT-SORT path works with flagship RF-DETR detectors."""
        frames = _run_tracker(model, video_path, n_frames=5, tracker="botsort")
        assert len(frames) == 5
        assert all(frame.track_id is not None and len(frame) > 0 for frame in frames)


# ── RF-DETR Segmentation + Tracking ─────────────────────────────────────────


@requires_rfdetr
@pytest.mark.flagship_nightly
@pytest.mark.rfdetr
class TestTrackingRFDETRSeg:
    """Tracking e2e with a segmentation model: masks must survive tracking."""

    @pytest.fixture(scope="class")
    def model(self):
        m = LibreYOLO("LibreRFDETRn-seg.pt")
        yield m
        del m
        cuda_cleanup()

    def test_tracked_results_have_masks(self, model, video_path):
        """Each tracked frame should carry masks alongside track IDs."""
        frames = _run_tracker(model, video_path, n_frames=10)
        frames_with_masks = [f for f in frames if len(f) > 0 and f.masks is not None]
        assert len(frames_with_masks) >= len(frames) // 2, (
            f"Only {len(frames_with_masks)}/{len(frames)} frames had masks"
        )
        for f in frames_with_masks:
            assert len(f.masks) == len(f), (
                f"Mask count ({len(f.masks)}) != detection count ({len(f)})"
            )
            assert f.track_id is not None
            assert len(f.track_id) == len(f)

    def test_masks_are_image_resolution(self, model, video_path):
        """Tracked masks should be at the original frame resolution."""
        frames = _run_tracker(model, video_path, n_frames=5)
        for f in frames:
            if len(f) > 0 and f.masks is not None:
                h, w = f.orig_shape
                assert f.masks.data.shape[1:] == (h, w), (
                    f"Mask shape {f.masks.data.shape[1:]} != frame {(h, w)}"
                )
                return  # one verified frame is enough
        pytest.skip("No frames with detections+masks to verify")
