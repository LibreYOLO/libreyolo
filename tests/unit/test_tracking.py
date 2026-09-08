"""Unit tests for the ByteTrack tracking module."""

import io
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.models.base.inference import InferenceRunner
from libreyolo.models.base.model import BaseModel
from libreyolo.tracking.config import TrackConfig
from libreyolo.tracking.kalman_filter import KalmanFilterXYAH
from libreyolo.tracking.matching import (
    bbox_iou_batch,
    fuse_score,
    iou_distance,
    linear_assignment,
)
from libreyolo.tracking.strack import STrack, TrackState
from libreyolo.tracking.tracker import ByteTracker
from libreyolo.utils.image_loader import ImageLoader
from libreyolo.utils.results import Boxes, Masks, Results

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _make_results(boxes_list, confs, classes, orig_shape=(480, 640)):
    """Build a Results object from raw lists for testing."""
    boxes = torch.tensor(boxes_list, dtype=torch.float32)
    conf = torch.tensor(confs, dtype=torch.float32)
    cls = torch.tensor(classes, dtype=torch.float32)
    return Results(
        boxes=Boxes(boxes, conf, cls),
        orig_shape=orig_shape,
    )


def test_obb_tracking_rejected_before_axis_aligned_tracker():
    model = type("OBBModel", (), {"task": "obb"})()

    with pytest.raises(NotImplementedError, match="oriented boxes"):
        next(BaseModel.track(model, "missing.mp4"))


# --------------------------------------------------------------------------
# TrackConfig
# --------------------------------------------------------------------------


class TestTrackConfig:
    def test_defaults(self):
        cfg = TrackConfig()
        assert cfg.track_high_thresh == 0.25
        assert cfg.track_low_thresh == 0.1
        assert cfg.new_track_thresh == 0.25
        assert cfg.match_thresh == 0.8
        assert cfg.track_buffer == 30
        assert cfg.frame_rate == 30
        assert cfg.fuse_score is True
        assert cfg.minimum_consecutive_frames == 1

    def test_custom_values(self):
        cfg = TrackConfig(track_high_thresh=0.5, track_buffer=60)
        assert cfg.track_high_thresh == 0.5
        assert cfg.track_buffer == 60

    def test_from_kwargs(self):
        cfg = TrackConfig.from_kwargs(track_high_thresh=0.3, unknown_key=42)
        assert cfg.track_high_thresh == 0.3

    def test_from_kwargs_warns_on_unknown(self):
        with pytest.warns(UserWarning, match="Unknown tracking config"):
            TrackConfig.from_kwargs(bogus=True)

    def test_rejects_zero_frame_rate(self):
        with pytest.raises(ValueError, match="frame_rate must be > 0"):
            TrackConfig(frame_rate=0)

    def test_rejects_negative_threshold(self):
        with pytest.raises(ValueError, match="track_high_thresh must be in"):
            TrackConfig(track_high_thresh=-0.5)

    def test_rejects_threshold_over_one(self):
        with pytest.raises(ValueError, match="track_low_thresh must be in"):
            TrackConfig(track_low_thresh=1.5)

    def test_rejects_high_below_low(self):
        with pytest.raises(
            ValueError, match="track_high_thresh .* must be >= track_low_thresh"
        ):
            TrackConfig(track_high_thresh=0.1, track_low_thresh=0.5)

    def test_rejects_negative_track_buffer(self):
        with pytest.raises(ValueError, match="track_buffer must be >= 0"):
            TrackConfig(track_buffer=-1)

    def test_rejects_zero_minimum_consecutive_frames(self):
        with pytest.raises(ValueError, match="minimum_consecutive_frames must be >= 1"):
            TrackConfig(minimum_consecutive_frames=0)


# --------------------------------------------------------------------------
# KalmanFilter
# --------------------------------------------------------------------------


class TestKalmanFilter:
    def test_initiate_shapes(self):
        kf = KalmanFilterXYAH()
        measurement = np.array([100.0, 200.0, 0.5, 80.0])
        mean, cov = kf.initiate(measurement)
        assert mean.shape == (8,)
        assert cov.shape == (8, 8)
        np.testing.assert_array_almost_equal(mean[:4], measurement)
        np.testing.assert_array_almost_equal(mean[4:], 0.0)

    def test_predict_advances_position(self):
        kf = KalmanFilterXYAH()
        mean = np.array([100.0, 200.0, 0.5, 80.0, 5.0, 10.0, 0.0, 2.0])
        cov = np.eye(8) * 1.0
        pred_mean, pred_cov = kf.predict(mean, cov)
        # Position should advance by velocity (dt=1).
        assert pred_mean[0] == pytest.approx(105.0)
        assert pred_mean[1] == pytest.approx(210.0)
        assert pred_mean[3] == pytest.approx(82.0)

    def test_update_corrects_state(self):
        kf = KalmanFilterXYAH()
        measurement = np.array([100.0, 200.0, 0.5, 80.0])
        mean, cov = kf.initiate(measurement)
        mean, cov = kf.predict(mean, cov)

        # Measurement slightly different from prediction.
        new_meas = np.array([102.0, 198.0, 0.5, 81.0])
        updated_mean, updated_cov = kf.update(mean, cov, new_meas)

        # Updated state should be between prediction and measurement.
        assert 100.0 < updated_mean[0] < 103.0
        assert 197.0 < updated_mean[1] < 201.0

    def test_multi_predict_matches_single(self):
        kf = KalmanFilterXYAH()
        m1 = np.array([100.0, 200.0, 0.5, 80.0])
        m2 = np.array([300.0, 400.0, 0.8, 120.0])
        mean1, cov1 = kf.initiate(m1)
        mean2, cov2 = kf.initiate(m2)

        # Single predictions.
        sp1, sc1 = kf.predict(mean1.copy(), cov1.copy())
        sp2, sc2 = kf.predict(mean2.copy(), cov2.copy())

        # Batch prediction.
        means = np.stack([mean1, mean2])
        covs = np.stack([cov1, cov2])
        bp, bc = kf.multi_predict(means, covs)

        np.testing.assert_array_almost_equal(bp[0], sp1)
        np.testing.assert_array_almost_equal(bp[1], sp2)
        np.testing.assert_array_almost_equal(bc[0], sc1)
        np.testing.assert_array_almost_equal(bc[1], sc2)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


class TestMatching:
    def test_iou_identical_boxes(self):
        a = np.array([[10, 20, 50, 60]], dtype=np.float64)
        iou = bbox_iou_batch(a, a)
        assert iou[0, 0] == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[20, 20, 30, 30]], dtype=np.float64)
        iou = bbox_iou_batch(a, b)
        assert iou[0, 0] == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[5, 5, 15, 15]], dtype=np.float64)
        iou = bbox_iou_batch(a, b)
        # Intersection = 5*5=25, union = 100+100-25=175.
        assert iou[0, 0] == pytest.approx(25.0 / 175.0)

    def test_iou_distance_is_complement(self):
        tracks = [STrack(np.array([0, 0, 10, 10]), 0.9, 0, 0)]
        tracks[0].mean = np.array([5, 5, 1.0, 10, 0, 0, 0, 0], dtype=np.float64)
        dets = np.array([[0, 0, 10, 10]], dtype=np.float64)
        cost = iou_distance(tracks, dets)
        assert cost[0, 0] == pytest.approx(1.0 - 1.0)  # identical = IoU 1

    def test_fuse_score_formula(self):
        cost = np.array([[0.3]], dtype=np.float64)  # IoU sim = 0.7
        scores = np.array([0.9], dtype=np.float64)
        fused = fuse_score(cost, scores)
        expected = 1.0 - (0.7 * 0.9)
        assert fused[0, 0] == pytest.approx(expected)

    def test_linear_assignment_perfect_match(self):
        cost = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float64)
        matches, ua, ub = linear_assignment(cost, 0.5)
        assert len(matches) == 2
        assert len(ua) == 0
        assert len(ub) == 0

    def test_linear_assignment_threshold(self):
        cost = np.array([[0.8]], dtype=np.float64)
        matches, ua, ub = linear_assignment(cost, 0.5)
        assert len(matches) == 0
        assert len(ua) == 1
        assert len(ub) == 1

    def test_linear_assignment_empty(self):
        cost = np.empty((0, 0), dtype=np.float64)
        matches, ua, ub = linear_assignment(cost, 0.5)
        assert matches.shape == (0, 2)
        assert len(ua) == 0
        assert len(ub) == 0


# --------------------------------------------------------------------------
# STrack
# --------------------------------------------------------------------------


class TestSTrack:
    def test_activate_sets_tracked(self):
        kf = KalmanFilterXYAH()
        st = STrack(np.array([10, 20, 50, 60]), 0.9, 0, 0)
        st.activate(kf, frame_id=1, track_id=1)
        assert st.state == TrackState.Tracked
        assert st.is_activated is True
        assert st.track_id == 1

    def test_xyxy_to_xyah_and_back(self):
        xyxy = np.array([10.0, 20.0, 50.0, 80.0])
        xyah = STrack.xyxy_to_xyah(xyxy)
        # cx=30, cy=50, a=40/60, h=60
        assert xyah[0] == pytest.approx(30.0)
        assert xyah[1] == pytest.approx(50.0)
        assert xyah[2] == pytest.approx(40.0 / 60.0)
        assert xyah[3] == pytest.approx(60.0)

    def test_mark_lost_and_removed(self):
        st = STrack(np.array([0, 0, 10, 10]), 0.9, 0, 0)
        st.state = TrackState.Tracked
        st.mark_lost()
        assert st.state == TrackState.Lost
        st.mark_removed()
        assert st.state == TrackState.Removed


# --------------------------------------------------------------------------
# ByteTracker
# --------------------------------------------------------------------------


class TestByteTracker:
    def test_empty_results(self):
        tracker = ByteTracker()
        result = _make_results([], [], [])
        tracked = tracker.update(result)
        assert len(tracked) == 0
        assert tracked.track_id is not None
        assert len(tracked.track_id) == 0

    def test_single_detection_gets_id(self):
        tracker = ByteTracker()
        r1 = _make_results([[100, 100, 200, 200]], [0.9], [0])
        t1 = tracker.update(r1)
        assert len(t1) == 1
        assert t1.track_id is not None
        id1 = t1.track_id[0].item()

        # Same object, slightly moved.
        r2 = _make_results([[105, 105, 205, 205]], [0.9], [0])
        t2 = tracker.update(r2)
        assert len(t2) == 1
        assert t2.track_id[0].item() == id1

    def test_two_detections_different_ids(self):
        tracker = ByteTracker()
        r = _make_results(
            [[100, 100, 200, 200], [400, 400, 500, 500]],
            [0.9, 0.8],
            [0, 0],
        )
        t = tracker.update(r)
        assert len(t) == 2
        assert t.track_id[0].item() != t.track_id[1].item()

    def test_lost_track_recovery(self):
        tracker = ByteTracker(track_buffer=10)

        # Frame 1: object appears.
        r1 = _make_results([[100, 100, 200, 200]], [0.9], [0])
        t1 = tracker.update(r1)
        id1 = t1.track_id[0].item()

        # Frames 2-4: object disappears.
        for _ in range(3):
            tracker.update(_make_results([], [], []))

        # Frame 5: object reappears at similar position.
        r5 = _make_results([[105, 105, 205, 205]], [0.9], [0])
        t5 = tracker.update(r5)
        assert len(t5) == 1
        assert t5.track_id[0].item() == id1

    def test_low_confidence_recovery(self):
        tracker = ByteTracker(track_high_thresh=0.5, track_low_thresh=0.1)

        # Frame 1: high confidence.
        r1 = _make_results([[100, 100, 200, 200]], [0.9], [0])
        t1 = tracker.update(r1)
        id1 = t1.track_id[0].item()

        # Frame 2: same object, but low confidence (occluded).
        r2 = _make_results([[103, 103, 203, 203]], [0.3], [0])
        t2 = tracker.update(r2)
        assert len(t2) == 1
        assert t2.track_id[0].item() == id1

    def test_minimum_consecutive_frames(self):
        tracker = ByteTracker(minimum_consecutive_frames=3)

        # Frame 1: new detection — should NOT be output yet.
        r1 = _make_results([[100, 100, 200, 200]], [0.9], [0])
        t1 = tracker.update(r1)
        assert len(t1) == 0

        # Frame 2: matched — still not enough frames.
        r2 = _make_results([[102, 102, 202, 202]], [0.9], [0])
        t2 = tracker.update(r2)
        assert len(t2) == 0

        # Frame 3: third consecutive match — now confirmed.
        r3 = _make_results([[104, 104, 204, 204]], [0.9], [0])
        t3 = tracker.update(r3)
        assert len(t3) == 1

    def test_reset_clears_state(self):
        tracker = ByteTracker()
        r = _make_results([[100, 100, 200, 200]], [0.9], [0])
        tracker.update(r)

        tracker.reset()

        t2 = tracker.update(r)
        id2 = t2.track_id[0].item()
        # After reset, IDs start from 1 again.
        assert id2 == 1

    def test_track_id_tensor_on_results(self):
        tracker = ByteTracker()
        r = _make_results([[100, 100, 200, 200]], [0.9], [0])
        t = tracker.update(r)
        assert isinstance(t.track_id, torch.Tensor)
        assert t.track_id.dtype == torch.int64
        assert t.track_id.shape == (1,)

    def test_per_instance_id_counter(self):
        t1 = ByteTracker()
        t2 = ByteTracker()

        r = _make_results([[100, 100, 200, 200]], [0.9], [0])
        res1 = t1.update(r)
        res2 = t2.update(r)

        # Both should start from 1 independently.
        assert res1.track_id[0].item() == 1
        assert res2.track_id[0].item() == 1

    def test_results_backward_compatible(self):
        """Results without track_id should still work."""
        r = Results(
            boxes=Boxes(torch.rand(2, 4), torch.rand(2), torch.rand(2)),
            orig_shape=(480, 640),
        )
        assert r.track_id is None
        assert "track_ids" not in repr(r)

    def test_results_cpu_with_track_id(self):
        r = Results(
            boxes=Boxes(torch.rand(2, 4), torch.rand(2), torch.rand(2)),
            orig_shape=(480, 640),
            track_id=torch.tensor([1, 2]),
        )
        cpu_r = r.cpu()
        assert cpu_r.track_id is not None
        assert cpu_r.track_id.device.type == "cpu"
        assert torch.equal(cpu_r.track_id, torch.tensor([1, 2]))

    def test_update_accepts_numpy_results(self):
        tracker = ByteTracker()
        r = _make_results([[100, 100, 200, 200]], [0.9], [0]).numpy()

        tracked = tracker.update(r)

        assert isinstance(tracked.track_id, np.ndarray)
        assert tracked.track_id.dtype == np.int64
        assert tracked.boxes.id is tracked.track_id
        assert tracked.track_id.tolist() == [1]


def _make_results_with_masks(boxes_list, confs, classes, orig_shape=(480, 640)):
    """Build a Results object with fake instance masks for testing."""
    n = len(boxes_list)
    h, w = orig_shape
    boxes = torch.tensor(boxes_list, dtype=torch.float32)
    conf = torch.tensor(confs, dtype=torch.float32)
    cls = torch.tensor(classes, dtype=torch.float32)
    # One binary mask per detection, each a simple filled rectangle
    mask_data = torch.zeros((n, h, w), dtype=torch.uint8)
    for i, (x1, y1, x2, y2) in enumerate(boxes_list):
        mask_data[i, int(y1) : int(y2), int(x1) : int(x2)] = 1
    return Results(
        boxes=Boxes(boxes, conf, cls),
        orig_shape=orig_shape,
        masks=Masks(mask_data, orig_shape),
    )


class TestByteTrackerMasks:
    """Verify that segmentation masks survive through ByteTracker.update()."""

    def test_masks_preserved_through_tracking(self):
        """Tracked results should carry masks sliced to matched detections."""
        tracker = ByteTracker()
        r = _make_results_with_masks(
            [[100, 100, 200, 200], [400, 400, 480, 480]],
            [0.9, 0.8],
            [0, 1],
        )
        tracked = tracker.update(r)

        assert tracked.masks is not None, "Masks should survive tracking"
        assert len(tracked.masks) == len(tracked), "One mask per tracked detection"
        # Masks should be actual filled regions, not empty
        for i in range(len(tracked.masks)):
            assert tracked.masks.data[i].sum() > 0

    def test_masks_sliced_to_correct_detections(self):
        """When tracker drops a detection, its mask should also be dropped."""
        tracker = ByteTracker(track_high_thresh=0.5, track_low_thresh=0.1)

        # Frame 1: two objects
        r1 = _make_results_with_masks(
            [[100, 100, 200, 200], [400, 400, 480, 480]],
            [0.9, 0.8],
            [0, 1],
        )
        t1 = tracker.update(r1)
        assert t1.masks is not None
        assert len(t1.masks) == len(t1)

        # Frame 2: only first object remains (second disappears)
        r2 = _make_results_with_masks(
            [[105, 105, 205, 205]],
            [0.9],
            [0],
        )
        t2 = tracker.update(r2)
        assert t2.masks is not None
        assert len(t2.masks) == len(t2)
        # The surviving mask should cover the first object's region
        assert t2.masks.data[0, 150, 150] == 1  # center of first box

    def test_no_masks_when_input_has_none(self):
        """Detection-only results should not gain masks through tracking."""
        tracker = ByteTracker()
        r = _make_results([[100, 100, 200, 200]], [0.9], [0])
        tracked = tracker.update(r)
        assert tracked.masks is None

    def test_empty_frame_with_seg_model(self):
        """Empty tracked output from a seg model should have no masks."""
        tracker = ByteTracker(minimum_consecutive_frames=5)
        r = _make_results_with_masks(
            [[100, 100, 200, 200]],
            [0.9],
            [0],
        )
        tracked = tracker.update(r)
        # With min_consecutive_frames=5, first frame yields nothing
        assert len(tracked) == 0


class TestDrawBoxesWithTrackIds:
    """Tests for draw_boxes() with the track_ids parameter."""

    def _draw(self, **kwargs):
        from libreyolo.utils.drawing import draw_boxes

        img = Image.new("RGB", (200, 200), (255, 255, 255))
        boxes = [[10, 10, 90, 90], [110, 110, 190, 190]]
        scores = [0.9, 0.8]
        classes = [0, 1]
        return draw_boxes(img, boxes, scores, classes, **kwargs)

    def test_without_track_ids(self):
        result = self._draw()
        assert isinstance(result, Image.Image)
        assert result.size == (200, 200)

    def test_with_track_ids(self):
        result = self._draw(track_ids=[1, 2])
        assert isinstance(result, Image.Image)
        # Tracked image should differ from non-tracked (different label text)
        arr_tracked = np.array(result)
        arr_plain = np.array(self._draw())
        assert not np.array_equal(arr_tracked, arr_plain)

    def test_track_ids_color_by_id(self):
        """Two boxes with same class but different track IDs get different colors."""
        from libreyolo.utils.drawing import draw_boxes

        img = Image.new("RGB", (300, 100), (255, 255, 255))
        # Same class (0) for both, but different track IDs
        r1 = draw_boxes(img, [[10, 10, 90, 90]], [0.9], [0], track_ids=[1])
        r2 = draw_boxes(img, [[10, 10, 90, 90]], [0.9], [0], track_ids=[2])
        # Different track IDs → different box colors → different images
        assert not np.array_equal(np.array(r1), np.array(r2))

    def test_does_not_modify_original(self):
        img = Image.new("RGB", (200, 200), (128, 128, 128))
        original_arr = np.array(img).copy()
        from libreyolo.utils.drawing import draw_boxes

        draw_boxes(img, [[10, 10, 90, 90]], [0.9], [0], track_ids=[1])
        assert np.array_equal(np.array(img), original_arr)


# --------------------------------------------------------------------------
# track() over image sequences, not just video files
# --------------------------------------------------------------------------


class _StubTrackModel:
    """Minimal stand-in for a BaseModel driving BaseModel.track().

    Always reports one fixed-position detection, which is enough for
    ByteTrack to assign and keep a single, stable track_id across frames.
    """

    task = "detect"
    names = {0: "thing"}
    device = torch.device("cpu")

    def __init__(self, box=(1.0, 1.0, 5.0, 5.0), score=0.9):
        self._box = list(box)
        self._score = score

    def _get_input_size(self):
        return 32

    def _get_model_name(self):
        return "stub"

    def _preprocess(self, image, color_format="auto", input_size=None):
        pil = ImageLoader.load(image, color_format=color_format)
        return torch.zeros(1, 3, 32, 32), pil, pil.size, 1.0

    def _forward(self, tensor):
        return tensor

    def _postprocess(
        self,
        output,
        conf,
        iou,
        original_size,
        max_det=300,
        ratio=1.0,
        classes=None,
        **kwargs,
    ):
        return {
            "boxes": [self._box],
            "scores": [self._score],
            "classes": [0],
            "num_detections": 1,
        }

    @property
    def _runner(self):
        if getattr(self, "_runner_instance", None) is None:
            self._runner_instance = InferenceRunner(self)
        return self._runner_instance


def _make_frames(n, size=(20, 16), color=(50, 50, 50)):
    return [Image.new("RGB", size, color) for _ in range(n)]


class TestTrackImageSequences:
    def test_tracks_a_list_of_pil_images(self):
        model = _StubTrackModel()

        results = list(BaseModel.track(model, _make_frames(4)))

        assert len(results) == 4
        assert [r.frame_idx for r in results] == [0, 1, 2, 3]
        assert all(r.path is None for r in results)
        # The fixed-position detection matches itself frame over frame, so
        # ByteTrack keeps assigning it the same identity.
        assert [int(r.track_id[0]) for r in results] == [1, 1, 1, 1]

    def test_tracks_a_directory_of_images_in_sorted_order(self, tmp_path):
        for i in range(3):
            Image.new("RGB", (20, 16), (i * 40, 0, 0)).save(tmp_path / f"{i:03d}.png")
        model = _StubTrackModel()

        results = list(BaseModel.track(model, tmp_path))

        assert len(results) == 3
        assert [Path(r.path).name for r in results] == [
            "000.png",
            "001.png",
            "002.png",
        ]

    def test_tracks_a_lazy_generator_without_materializing_it(self):
        model = _StubTrackModel()
        pulled = []

        def frames():
            for i in range(1000):
                pulled.append(i)
                yield Image.new("RGB", (20, 16))

        gen = BaseModel.track(model, frames())
        first_three = [next(gen) for _ in range(3)]

        assert len(first_three) == 3
        # Only as many source frames were pulled as were actually consumed
        # from the tracking generator -- the iterator must stay lazy.
        assert len(pulled) == 3

    def test_tracks_a_single_image_as_a_one_frame_sequence(self):
        model = _StubTrackModel()

        results = list(BaseModel.track(model, Image.new("RGB", (20, 16))))

        assert len(results) == 1

    def test_tracks_a_single_bytesio_image(self):
        buffer = io.BytesIO()
        Image.new("RGB", (20, 16)).save(buffer, format="PNG")
        buffer.seek(0)
        model = _StubTrackModel()

        results = list(BaseModel.track(model, buffer))

        assert len(results) == 1

    def test_bgr_numpy_frames_reach_the_model_as_rgb(self):
        model = _StubTrackModel()
        seen_pixels = []
        original_preprocess = model._preprocess

        def capture_preprocess(image, color_format="auto", input_size=None):
            seen_pixels.append(np.asarray(image)[0, 0].tolist())
            return original_preprocess(
                image, color_format=color_format, input_size=input_size
            )

        model._preprocess = capture_preprocess
        frame_bgr = np.zeros((16, 20, 3), dtype=np.uint8)
        frame_bgr[:] = [0, 0, 255]

        list(BaseModel.track(model, [frame_bgr], color_format="bgr"))

        assert seen_pixels == [[255, 0, 0]]

    def test_empty_directory_yields_no_results(self, tmp_path):
        model = _StubTrackModel()

        assert list(BaseModel.track(model, tmp_path)) == []

    def test_live_stream_sources_are_not_yet_supported(self):
        model = _StubTrackModel()

        with pytest.raises(NotImplementedError, match="stream"):
            next(BaseModel.track(model, 0))

    def test_save_writes_output_video_for_an_image_list(self, tmp_path):
        pytest.importorskip("cv2", reason="opencv-python required for video tests")
        model = _StubTrackModel()
        output_path = tmp_path / "tracked.mp4"

        results = list(
            BaseModel.track(
                model, _make_frames(3), save=True, output_path=str(output_path)
            )
        )

        assert len(results) == 3
        assert output_path.exists()

    def test_save_rejects_images_with_changed_dimensions(self, tmp_path):
        pytest.importorskip("cv2", reason="opencv-python required for video tests")
        model = _StubTrackModel()
        output_path = tmp_path / "tracked.mp4"
        frames = [
            Image.new("RGB", (20, 16)),
            Image.new("RGB", (30, 24)),
        ]

        with pytest.raises(ValueError, match="frame size changed"):
            list(
                BaseModel.track(
                    model, frames, save=True, output_path=str(output_path)
                )
            )

    @pytest.mark.parametrize("fps", [0, -1, np.nan, np.inf])
    def test_rejects_invalid_image_sequence_fps(self, fps):
        model = _StubTrackModel()

        with pytest.raises(ValueError, match="fps must be a finite value > 0"):
            list(BaseModel.track(model, _make_frames(1), fps=fps))

    def test_fps_seeds_bytetrack_frame_rate_for_image_sequences(self, monkeypatch):
        captured = {}
        original = TrackConfig.from_kwargs.__func__

        def spy(cls, **kwargs):
            captured.update(kwargs)
            return original(cls, **kwargs)

        monkeypatch.setattr(TrackConfig, "from_kwargs", classmethod(spy))
        model = _StubTrackModel()

        list(BaseModel.track(model, _make_frames(2), fps=12.0))

        assert captured["frame_rate"] == 12

    def test_fps_seeds_frame_rate_at_the_retained_frame_cadence(self, monkeypatch):
        # tracker.update() only ever sees retained frames, so frame_rate
        # must be scaled by vid_stride -- not the raw fps -- or a lost
        # track's real-world expiry stretches out by vid_stride times.
        captured = {}
        original = TrackConfig.from_kwargs.__func__

        def spy(cls, **kwargs):
            captured.update(kwargs)
            return original(cls, **kwargs)

        monkeypatch.setattr(TrackConfig, "from_kwargs", classmethod(spy))
        model = _StubTrackModel()

        list(BaseModel.track(model, _make_frames(6), fps=30.0, vid_stride=3))

        assert captured["frame_rate"] == 10

    def test_fps_is_not_pre_rounded_before_reaching_track_config(self, monkeypatch):
        # tracker.py truncates (int()) frame_rate itself; pre-rounding here
        # too would double-round and can shift max_time_lost by a frame at
        # the boundary (e.g. an exact 6.6 cadence: round()->7, int()->6).
        captured = {}
        original = TrackConfig.from_kwargs.__func__

        def spy(cls, **kwargs):
            captured.update(kwargs)
            return original(cls, **kwargs)

        monkeypatch.setattr(TrackConfig, "from_kwargs", classmethod(spy))
        model = _StubTrackModel()

        list(BaseModel.track(model, _make_frames(6), fps=33.0, vid_stride=5))

        assert captured["frame_rate"] == pytest.approx(6.6)

    def test_explicit_frame_rate_kwarg_overrides_fps(self, monkeypatch):
        captured = {}
        original = TrackConfig.from_kwargs.__func__

        def spy(cls, **kwargs):
            captured.update(kwargs)
            return original(cls, **kwargs)

        monkeypatch.setattr(TrackConfig, "from_kwargs", classmethod(spy))
        model = _StubTrackModel()

        list(BaseModel.track(model, _make_frames(2), fps=12.0, frame_rate=5))

        assert captured["frame_rate"] == 5

    def test_fps_does_not_seed_frame_rate_for_video_sources(self, tmp_path, monkeypatch):
        cv2 = pytest.importorskip("cv2", reason="opencv-python required for video tests")
        path = str(tmp_path / "clip.mp4")
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (20, 16)
        )
        for _ in range(2):
            writer.write(np.zeros((16, 20, 3), dtype=np.uint8))
        writer.release()

        captured = {}
        original = TrackConfig.from_kwargs.__func__

        def spy(cls, **kwargs):
            captured.update(kwargs)
            return original(cls, **kwargs)

        monkeypatch.setattr(TrackConfig, "from_kwargs", classmethod(spy))
        model = _StubTrackModel()

        list(BaseModel.track(model, path, fps=12.0))

        assert "frame_rate" not in captured

    def test_fps_does_not_leak_into_ocsort_config(self):
        # OCSortConfig has no frame_rate field; passing it through would
        # otherwise trigger a spurious "unknown config key" warning.
        model = _StubTrackModel()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            list(BaseModel.track(model, _make_frames(2), fps=12.0, tracker="ocsort"))

    def test_mismatched_typed_tracker_config_frame_rate_warns(self):
        # tracker_config is never silently overwritten (same contract as
        # track_conf), but its frame_rate silently governs lost-track
        # timing for this sequence, so a mismatch must not stay quiet.
        model = _StubTrackModel()
        config = TrackConfig()  # frame_rate=30, left at the class default

        with pytest.warns(UserWarning, match="frame_rate"):
            list(
                BaseModel.track(
                    model,
                    _make_frames(6),
                    fps=30.0,
                    vid_stride=3,  # retained rate is 10, not config's 30
                    tracker_config=config,
                )
            )

        # tracker_config itself is never mutated.
        assert config.frame_rate == 30

    def test_typed_tracker_config_frame_rate_already_matching_is_quiet(self):
        model = _StubTrackModel()
        config = TrackConfig(frame_rate=10)  # caller already did the math

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            list(
                BaseModel.track(
                    model,
                    _make_frames(6),
                    fps=30.0,
                    vid_stride=3,
                    tracker_config=config,
                )
            )
