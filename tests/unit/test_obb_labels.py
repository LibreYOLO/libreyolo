"""Tests for OBB label parsing."""

import math

import numpy as np
import pytest

from libreyolo.data.obb import (
    corners_to_xywhr,
    parse_yolo_obb_label_line,
    xywhr_iou,
    xywhr_to_proxy_xyxy,
)

pytestmark = pytest.mark.unit


def test_parse_yolo_obb_label_line():
    cls_id, corners = parse_yolo_obb_label_line(
        "1 0.10 0.20 0.50 0.20 0.50 0.40 0.10 0.40",
        num_classes=3,
    )

    assert cls_id == 1
    assert corners.shape == (4, 2)
    assert corners.dtype == np.float32
    np.testing.assert_allclose(
        corners,
        np.array(
            [[0.10, 0.20], [0.50, 0.20], [0.50, 0.40], [0.10, 0.40]],
            dtype=np.float32,
        ),
    )


def test_parse_yolo_obb_label_line_accepts_split_parts():
    cls_id, corners = parse_yolo_obb_label_line(
        ["0", "0", "0", "1", "0", "1", "1", "0", "1"], num_classes=1
    )

    assert cls_id == 0
    assert corners.shape == (4, 2)


def test_parse_yolo_obb_label_line_can_clip_crop_boundary_rows():
    cls_id, corners = parse_yolo_obb_label_line(
        "0 -0.01 0.2 1.01 0.2 1.01 0.4 -0.01 0.4",
        num_classes=1,
        clip=True,
    )

    assert cls_id == 0
    np.testing.assert_allclose(
        corners,
        np.array([[0.0, 0.2], [1.0, 0.2], [1.0, 0.4], [0.0, 0.4]], dtype=np.float32),
    )


def test_corners_to_xywhr_and_proxy_box():
    _, corners = parse_yolo_obb_label_line(
        "0 0.10 0.20 0.50 0.20 0.50 0.40 0.10 0.40",
        num_classes=1,
    )

    xywhr = corners_to_xywhr(corners)
    proxy = xywhr_to_proxy_xyxy(xywhr)

    np.testing.assert_allclose(xywhr[:4], [0.30, 0.30, 0.40, 0.20], atol=1e-6)
    assert xywhr[4] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(proxy, [0.10, 0.20, 0.50, 0.40], atol=1e-6)


def test_xywhr_iou_handles_rotated_identity_and_disjoint_boxes():
    box = [32.0, 32.0, 20.0, 10.0, 0.5]

    assert xywhr_iou(box, box) == pytest.approx(1.0, abs=1e-6)
    assert xywhr_iou(box, [100.0, 100.0, 20.0, 10.0, 0.5]) == pytest.approx(0.0)


def test_xywhr_iou_is_pi_periodic():
    box = [32.0, 32.0, 20.0, 10.0, 0.5]
    same_box_pi_period = [32.0, 32.0, 20.0, 10.0, 0.5 + math.pi]

    assert xywhr_iou(box, same_box_pi_period) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("0 0.5 0.5 0.2 0.2", "Expected 9 fields"),
        ("1.5 0 0 1 0 1 1 0 1", "integer"),
        ("2 0 0 1 0 1 1 0 1", "out of range"),
        ("0 0 0 1 0 1 1 0 1.1", r"\[0, 1\]"),
        ("0 0 0 1 0 nan 1 0 1", "finite"),
        ("0 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5", "non-degenerate"),
        ("0 0.1 0.1 0.2 0.2 0.3 0.3 0.4 0.4", "non-degenerate"),
    ],
)
def test_parse_yolo_obb_label_line_rejects_invalid_rows(line, message):
    with pytest.raises(ValueError, match=message):
        parse_yolo_obb_label_line(line, num_classes=2)
