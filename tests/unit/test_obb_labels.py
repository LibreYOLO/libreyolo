"""Tests for OBB label parsing."""

import numpy as np
import pytest

from libreyolo.data.obb import parse_yolo_obb_label_line

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
