"""Unit tests for the ByteTrack tracking module."""

import pytest
import numpy as np
import torch

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
from libreyolo.utils.results import Boxes, Results

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
        t1 = tracker.update(r)
        id1 = t1.track_id[0].item()

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
