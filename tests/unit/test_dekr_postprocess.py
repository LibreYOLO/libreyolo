"""LibreDEKR decoding: centres, offsets, pose NMS, batch safety, derived boxes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.postprocess.dekr import (
    DEKR_KEYPOINT_THRESHOLD,
    decode_poses,
    derive_boxes_from_keypoints,
    postprocess_dekr,
)

pytestmark = [pytest.mark.unit, pytest.mark.dekr]

K = 12  # > nms_num_threshold (8) so poses can survive the close-joint gate
H = W = 32
STRIDE = 4


def planted(centers, *, k=K, h=H, w=W, batch=1):
    """Build a synthetic ``(heatmap_logits, offsets)`` pair with known centres.

    Joint ``j`` is given offset ``-(j + 1)`` in both axes, so the decoded pose
    fans out diagonally from its centre. A pose must have non-zero extent to
    survive pose NMS at all: the suppression radius is a fraction of the pose's
    own diagonal, so joints collapsed onto a single point yield radius zero and
    are always dropped (see ``test_collapsed_pose_is_suppressed``).
    """
    heatmap = torch.full((batch, k + 1, h, w), -8.0)
    offsets = torch.zeros(batch, 2 * k, h, w)
    for joint in range(k):
        offsets[:, 2 * joint, :, :] = -(joint + 1)
        offsets[:, 2 * joint + 1, :, :] = -(joint + 1)
    for item, item_centers in enumerate(centers):
        heatmap[item, :k, :, :] = 2.0
        for row, col in item_centers:
            heatmap[item, -1, row, col] = 6.0
    return heatmap, offsets


def expected_xy(row, col, k=K):
    """Decoded coordinates for :func:`planted`: ``(location - offset) * stride``."""
    joints = np.arange(1, k + 1)
    return (col + joints) * STRIDE, (row + joints) * STRIDE


def test_single_centre_decodes_to_expected_coordinates():
    heatmap, offsets = planted([[(8, 10)]])
    (poses, scores), = decode_poses(heatmap, offsets)
    assert poses.shape == (1, K, 3)
    xs, ys = expected_xy(8, 10)
    np.testing.assert_allclose(poses[0, :, 0], xs)
    np.testing.assert_allclose(poses[0, :, 1], ys)
    assert scores.shape == (1,)


def test_offset_sign_convention_is_location_minus_offset():
    heatmap, offsets = planted([[(8, 10)]])
    # Flipping the offset sign must mirror the decoded joints about the centre.
    (positive, _), = decode_poses(heatmap, -offsets)
    xs, _ = expected_xy(8, 10)
    mirrored = (10 * STRIDE) - (xs - 10 * STRIDE)
    np.testing.assert_allclose(positive[0, :, 0], mirrored)


def test_collapsed_pose_is_suppressed_like_upstream():
    # Zero offsets put every joint on the centre: zero diagonal, zero NMS
    # radius, so upstream's `distance < radius` count never clears the
    # close-joint threshold and the pose is dropped.
    heatmap, _ = planted([[(8, 10)]])
    zeroed = torch.zeros(1, 2 * K, H, W)
    (poses, _), = decode_poses(heatmap, zeroed)
    assert len(poses) == 0


def test_zero_candidates_returns_empty_arrays():
    heatmap = torch.full((1, K + 1, H, W), -20.0)
    offsets = torch.zeros(1, 2 * K, H, W)
    (poses, scores), = decode_poses(heatmap, offsets)
    assert poses.shape == (0, K, 3)
    assert scores.shape == (0,)


def test_candidates_at_all_four_borders_survive():
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    heatmap, offsets = planted([corners])
    (poses, _), = decode_poses(heatmap, offsets)
    # Distinct corners are far apart, so pose NMS keeps all four.
    assert len(poses) == 4


def test_more_peaks_than_max_num_people_is_deterministic_top_k():
    centers = [(2 * r, 2 * c) for r in range(6) for c in range(8)]  # 48 peaks
    heatmap, offsets = planted([centers])
    (first, _), = decode_poses(heatmap, offsets, max_num_people=10)
    (second, _), = decode_poses(heatmap, offsets, max_num_people=10)
    assert len(first) <= 10
    np.testing.assert_array_equal(first, second)


def test_batch_two_decodes_each_item_independently():
    heatmap_a, offsets_a = planted([[(4, 4)]])
    heatmap_b, offsets_b = planted([[(6, 6), (20, 20), (28, 8)]])
    decoded = decode_poses(
        torch.cat([heatmap_a, heatmap_b]), torch.cat([offsets_a, offsets_b])
    )
    assert len(decoded) == 2
    assert len(decoded[0][0]) == 1
    assert len(decoded[1][0]) == 3
    # Item 0 must not leak into item 1 or vice versa.
    np.testing.assert_allclose(decoded[0][0][0, :, 0], expected_xy(4, 4)[0])


def test_batch_decode_matches_per_item_decode():
    heatmap_a, offsets_a = planted([[(4, 4)]])
    heatmap_b, offsets_b = planted([[(6, 6), (20, 20)]])
    batched = decode_poses(
        torch.cat([heatmap_a, heatmap_b]), torch.cat([offsets_a, offsets_b])
    )
    for index, (heatmap, offsets) in enumerate(
        ((heatmap_a, offsets_a), (heatmap_b, offsets_b))
    ):
        (poses, scores), = decode_poses(heatmap, offsets)
        np.testing.assert_array_equal(batched[index][0], poses)
        np.testing.assert_array_equal(batched[index][1], scores)


def test_non_seventeen_keypoint_head_decodes():
    heatmap, offsets = planted([[(8, 8)]], k=9)
    (poses, _), = decode_poses(heatmap, offsets, nms_num_threshold=4)
    assert poses.shape == (1, 9, 3)


def test_min_confidence_filters_low_scoring_poses():
    heatmap, offsets = planted([[(8, 8)]])
    assert len(decode_poses(heatmap, offsets, min_confidence=0.0)[0][0]) == 1
    assert len(decode_poses(heatmap, offsets, min_confidence=1.01)[0][0]) == 0


def test_half_precision_output_is_promoted_before_decode():
    heatmap, offsets = planted([[(8, 10)]])
    (poses, _), = decode_poses(heatmap.half(), offsets.half())
    assert poses.dtype == np.float32
    np.testing.assert_allclose(poses[0, :, 0], expected_xy(8, 10)[0])


def test_apply_sigmoid_false_treats_input_as_probabilities():
    heatmap, offsets = planted([[(8, 8)]])
    assert len(decode_poses(heatmap.sigmoid(), offsets, apply_sigmoid=False)[0][0]) == 1


def test_shape_contract_errors_are_actionable():
    heatmap = torch.zeros(1, K + 1, H, W)
    offsets = torch.zeros(1, 2 * K, H, W)
    with pytest.raises(ValueError, match="4D"):
        decode_poses(heatmap[0], offsets[0])
    with pytest.raises(ValueError, match="batch mismatch"):
        decode_poses(heatmap, offsets.repeat(2, 1, 1, 1))
    with pytest.raises(ValueError, match="centre channel"):
        decode_poses(torch.zeros(1, K, H, W), offsets)
    with pytest.raises(ValueError, match="spatial mismatch"):
        decode_poses(heatmap, torch.zeros(1, 2 * K, H, W + 4))


# --- derived boxes (LibreYOLO adapter, not an upstream head) -------------


def test_derived_box_is_tight_around_confident_joints():
    pose = np.zeros((1, 4, 3), dtype=np.float32)
    pose[0, :, :2] = [[10, 20], [30, 60], [15, 25], [200, 200]]
    pose[0, :, 2] = [0.9, 0.9, 0.9, 0.0]  # last joint is below threshold
    box = derive_boxes_from_keypoints(pose, (640, 480))
    np.testing.assert_allclose(box[0], [10, 20, 30, 60])


def test_derived_box_falls_back_when_too_few_confident_joints():
    pose = np.zeros((1, 3, 3), dtype=np.float32)
    pose[0, :, :2] = [[10, 20], [30, 60], [50, 40]]
    pose[0, :, 2] = [0.9, 0.0, 0.0]  # only one confident joint
    box = derive_boxes_from_keypoints(pose, (640, 480))
    np.testing.assert_allclose(box[0], [10, 20, 50, 60])


def test_derived_box_is_clipped_to_the_original_canvas():
    pose = np.zeros((1, 2, 3), dtype=np.float32)
    pose[0, :, :2] = [[-50, -20], [900, 700]]
    pose[0, :, 2] = [0.9, 0.9]
    box = derive_boxes_from_keypoints(pose, (640, 480))
    np.testing.assert_allclose(box[0], [0, 0, 640, 480])


def test_derived_box_ignores_non_finite_joints():
    pose = np.zeros((1, 3, 3), dtype=np.float32)
    pose[0, :, :2] = [[10, 20], [np.nan, np.inf], [30, 60]]
    pose[0, :, 2] = [0.9, 0.9, 0.9]
    box = derive_boxes_from_keypoints(pose, (640, 480))
    np.testing.assert_allclose(box[0], [10, 20, 30, 60])


def test_empty_pose_set_yields_empty_boxes():
    boxes = derive_boxes_from_keypoints(np.zeros((0, K, 3), np.float32), (640, 480))
    assert boxes.shape == (0, 4)


# --- LibreYOLO result contract -------------------------------------------


def test_postprocess_returns_the_pose_contract_on_the_original_canvas():
    heatmap, offsets = planted([[(8, 10)]])
    result = postprocess_dekr(
        (heatmap, offsets), conf_thres=0.0, original_size=(320, 240), ratio=0.5
    )
    assert set(result) >= {"num_detections", "boxes", "scores", "classes", "keypoints"}
    assert result["keypoints"].shape == (1, K, 3)
    assert result["boxes"].shape == (1, 4)
    assert result["num_detections"] == 1
    assert result["classes"].tolist() == [0]
    # ratio 0.5 halves the network-canvas coordinate back onto the source image.
    np.testing.assert_allclose(result["keypoints"][0, :, 0], expected_xy(8, 10)[0] / 0.5)


def test_postprocess_clips_restored_keypoints_to_the_canvas():
    heatmap, offsets = planted([[(30, 30)]])
    result = postprocess_dekr(
        (heatmap, offsets), conf_thres=0.0, original_size=(50, 40), ratio=1.0
    )
    assert result["keypoints"][:, :, 0].max() <= 50
    assert result["keypoints"][:, :, 1].max() <= 40


def test_postprocess_handles_no_people():
    heatmap = torch.full((1, K + 1, H, W), -20.0)
    offsets = torch.zeros(1, 2 * K, H, W)
    result = postprocess_dekr(
        (heatmap, offsets), conf_thres=0.0, original_size=(320, 240), ratio=1.0
    )
    assert result["num_detections"] == 0
    assert result["keypoints"].shape == (0, K, 3)
    assert result["boxes"].shape == (0, 4)


def test_postprocess_batch_index_selects_the_right_item():
    heatmap_a, offsets_a = planted([[(4, 4)]])
    heatmap_b, offsets_b = planted([[(20, 24)]])
    raw = (torch.cat([heatmap_a, heatmap_b]), torch.cat([offsets_a, offsets_b]))
    second = postprocess_dekr(
        raw, conf_thres=0.0, original_size=(320, 240), ratio=1.0, batch_index=1
    )
    np.testing.assert_allclose(second["keypoints"][0, :, 0], expected_xy(20, 24)[0])


def test_keypoint_threshold_default_matches_upstream():
    assert DEKR_KEYPOINT_THRESHOLD == 0.05
