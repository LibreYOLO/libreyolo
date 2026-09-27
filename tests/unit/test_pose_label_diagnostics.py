"""Unit tests for pose label rejection diagnostics (issue #873).

A skipped label line must be traceable: the warning names the file, the line
number and the actual reason, and a numeric parse error is not reported as a
field-count error.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest

from libreyolo.data import YOLOPoseDataset, parse_yolo_pose_label_line
from libreyolo.data.pose_dataset import MAX_REPORTED_BAD_LINES

pytestmark = pytest.mark.unit

GOOD_XY = "0 0.5 0.5 0.2 0.2 0.4 0.4 0.6 0.6"
GOOD_XYV = "0 0.5 0.5 0.2 0.2 0.4 0.4 2 0.6 0.6 2"


def _write_dataset(tmp_path, label_lines):
    """Write one image per label file and return the image paths."""
    img_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    img_files = []
    for name, lines in label_lines.items():
        cv2.imwrite(str(img_dir / f"{name}.jpg"), np.zeros((32, 32, 3), np.uint8))
        (label_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
        img_files.append(img_dir / f"{name}.jpg")
    return sorted(img_files)


class TestValidLayouts:
    def test_xy_layout_keeps_every_line(self, tmp_path, caplog):
        img_files = _write_dataset(tmp_path, {"a": [GOOD_XY, GOOD_XY]})
        with caplog.at_level(logging.WARNING):
            ds = YOLOPoseDataset(img_files, num_keypoints=2, keypoint_dim=2)
        assert ds.labels[0][0].shape[0] == 2
        assert ds.labels[0][2].shape == (2, 2, 3)
        assert "skipped" not in caplog.text

    def test_xyv_layout_keeps_every_line(self, tmp_path, caplog):
        img_files = _write_dataset(tmp_path, {"a": [GOOD_XYV, GOOD_XYV]})
        with caplog.at_level(logging.WARNING):
            ds = YOLOPoseDataset(img_files, num_keypoints=2, keypoint_dim=3)
        assert ds.labels[0][0].shape[0] == 2
        assert "skipped" not in caplog.text


class TestSkipDiagnostics:
    def test_warning_names_file_line_and_reason(self, tmp_path, caplog):
        img_files = _write_dataset(
            tmp_path,
            {"a": [GOOD_XY, "0 0.5 0.5 0.2 0.2 0.4 0.4 0.6 0.6 0.7 0.7"]},
        )
        with caplog.at_level(logging.WARNING):
            YOLOPoseDataset(img_files, num_keypoints=2, keypoint_dim=2)
        assert "a.txt:2" in caplog.text
        assert "got 11" in caplog.text

    def test_numeric_error_is_not_reported_as_field_count(self, tmp_path, caplog):
        img_files = _write_dataset(
            tmp_path, {"a": ["0 0.5 0.5 0.2 0.2 0.4 0.4 bad 0.6"]}
        )
        with caplog.at_level(logging.WARNING):
            YOLOPoseDataset(img_files, num_keypoints=2, keypoint_dim=2)
        assert "Non-numeric coordinate 'bad'" in caplog.text
        assert "Expected 9 fields" not in caplog.text

    def test_class_id_out_of_range_names_the_line(self, tmp_path, caplog):
        img_files = _write_dataset(
            tmp_path, {"a": [GOOD_XY, "3 0.5 0.5 0.2 0.2 0.4 0.4 0.6 0.6"]}
        )
        with caplog.at_level(logging.WARNING):
            YOLOPoseDataset(
                img_files, num_keypoints=2, keypoint_dim=2, num_classes=2
            )
        assert "a.txt:2: class id 3" in caplog.text

    def test_quoted_lines_are_capped_and_the_rest_counted(self, tmp_path, caplog):
        bad = "0 0.5 0.5 0.2 0.2 0.4 0.4"
        n_bad = MAX_REPORTED_BAD_LINES + 3
        img_files = _write_dataset(tmp_path, {"a": [bad] * n_bad})
        with caplog.at_level(logging.WARNING):
            YOLOPoseDataset(img_files, num_keypoints=2, keypoint_dim=2)
        assert f"skipped {n_bad} unparsable" in caplog.text
        assert caplog.text.count("a.txt:") == MAX_REPORTED_BAD_LINES
        assert "... and 3 more" in caplog.text


class TestParserMessages:
    def test_wrong_keypoint_dim_is_hinted(self):
        parts = GOOD_XYV.split()
        with pytest.raises(ValueError, match=r"kpt_shape \[2, 3\]"):
            parse_yolo_pose_label_line(parts, num_keypoints=2, keypoint_dim=2)

    def test_extra_keypoints_are_counted(self):
        parts = "0 0.5 0.5 0.2 0.2 0.4 0.4 0.6 0.6 0.7 0.7 0.8 0.8".split()
        with pytest.raises(ValueError, match="carries 4 keypoint"):
            parse_yolo_pose_label_line(parts, num_keypoints=2, keypoint_dim=2)

    def test_non_numeric_class_id(self):
        parts = "person 0.5 0.5 0.2 0.2 0.4 0.4 0.6 0.6".split()
        with pytest.raises(ValueError, match="Class id 'person' is not a number"):
            parse_yolo_pose_label_line(parts, num_keypoints=2, keypoint_dim=2)


class TestNonFiniteClassIds:
    """`float()` accepts these, `int()` cannot convert them (Greptile P1)."""

    @pytest.mark.parametrize("token", ["inf", "-inf", "1e400", "nan"])
    def test_non_finite_class_id_is_skipped_not_raised(self, token):
        parts = f"{token} 0.5 0.5 0.2 0.2 0.4 0.4 0.6 0.6".split()
        with pytest.raises(ValueError, match="is not a number"):
            parse_yolo_pose_label_line(parts, num_keypoints=2, keypoint_dim=2)

    def test_dataset_survives_a_non_finite_class_id(self, tmp_path, caplog):
        img_files = _write_dataset(
            tmp_path, {"a": [GOOD_XY, "inf 0.5 0.5 0.2 0.2 0.4 0.4 0.6 0.6"]}
        )
        with caplog.at_level(logging.WARNING):
            ds = YOLOPoseDataset(img_files, num_keypoints=2, keypoint_dim=2)
        assert ds.labels[0][0].shape[0] == 1
        assert "a.txt:2: Class id 'inf' is not a number" in caplog.text
