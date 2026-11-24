"""Snapshot construction, label parsing, and the format guard."""

from collections import Counter

import pytest

from libreyolo.data.yolo_coco_api import parse_yolo_label_line
from libreyolo.doctor.config import NotADetectionDatasetError
from libreyolo.doctor.runner import diagnose
from libreyolo.doctor.snapshot import (
    build_snapshot,
    detect_non_detection,
    parse_label_file,
)

pytestmark = pytest.mark.unit


class TestParseLabelFile:
    def test_valid_lines(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05\n")
        boxes, issues, digest, polygons = parse_label_file(f, Counter())
        assert boxes.shape == (2, 5)
        assert not issues
        assert digest is not None
        assert polygons == 0

    def test_blank_lines_skipped(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("\n0 0.5 0.5 0.2 0.2\n\n")
        boxes, issues, _, _ = parse_label_file(f, Counter())
        assert boxes.shape == (1, 5)
        assert not issues

    def test_too_few_fields(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("0 0.5 0.5 0.2\n")
        boxes, issues, _, _ = parse_label_file(f, Counter())
        assert boxes.shape[0] == 0
        assert len(issues) == 1 and "4" in issues[0].reason

    def test_polygon_lines_become_boxes_like_training(self, tmp_path):
        # YOLODataset._load_label consumes >5-field rows as polygons and
        # derives the box from their extent; doctor must accept them too.
        f = tmp_path / "a.txt"
        f.write_text("1 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")
        boxes, issues, _, polygons = parse_label_file(f, Counter())
        assert not issues
        assert polygons == 1
        assert boxes.shape == (1, 5)
        cls, cx, cy, w, h = boxes[0]
        assert cls == 1
        assert (cx, cy) == pytest.approx((0.3, 0.3))
        assert (w, h) == pytest.approx((0.4, 0.4))

    # Rows that parse are in bounds and non-degenerate, so only the line
    # format decides whether training keeps them; class and geometry checks
    # stay out of it.
    @pytest.mark.parametrize(
        "row,kept",
        [
            pytest.param("0 0.5 0.5 0.2 0.2", True, id="box"),
            pytest.param("0 0.2 0.2 0.8 0.2 0.5 0.8", True, id="3-vertex-polygon"),
            pytest.param("0 0.5 0.5 0.2", False, id="4-fields"),
            pytest.param("0 0.5 oops 0.2 0.2", False, id="non-numeric"),
            pytest.param("1.5 0.5 0.5 0.2 0.2", False, id="float-class"),
            pytest.param("0 nan 0.5 0.2 0.2", False, id="nan"),
            pytest.param("0 0.5 0.5 0.2 0.2 0.87", False, id="box-plus-confidence"),
            pytest.param(
                "0 0.10 0.10 0.90 0.10 0.90 0.90 0.10", False, id="7-coordinates"
            ),
        ],
    )
    def test_parity_with_training_parser(self, tmp_path, row, kept):
        img = 64
        f = tmp_path / "a.txt"
        f.write_text(row + "\n")
        boxes, issues, _, _ = parse_label_file(f, Counter())
        trained = parse_yolo_label_line(row, img, img, num_classes=1)

        assert (trained is not None) == kept
        if not kept:
            assert len(issues) == 1 and boxes.shape[0] == 0
            return
        assert not issues and boxes.shape == (1, 5)
        cls, cx, cy, w, h = boxes[0]
        assert cls == trained[0]
        doctor_xyxy = (
            (cx - w / 2) * img,
            (cy - h / 2) * img,
            (cx + w / 2) * img,
            (cy + h / 2) * img,
        )
        assert doctor_xyxy == pytest.approx(trained[1:5], abs=1e-3)

    def test_malformed_polygon_is_an_issue_but_still_tallied(self, tmp_path):
        # No box and no polygon count, but the field tally keeps the row so
        # the format guard still classifies pose rows by their shape.
        f = tmp_path / "a.txt"
        f.write_text("0 0.5 0.5 0.2 0.2 0.87\n")
        counts = Counter()
        boxes, issues, _, polygons = parse_label_file(f, counts)
        assert boxes.shape[0] == 0
        assert polygons == 0
        assert len(issues) == 1 and "6 fields" in issues[0].reason
        assert counts == Counter({6: 1})

    def test_non_numeric_and_non_finite(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x 0.5 0.5 0.2 0.2\n0 nan 0.5 0.2 0.2\n1.5 0.5 0.5 0.2 0.2\n")
        boxes, issues, _, _ = parse_label_file(f, Counter())
        assert boxes.shape[0] == 0
        assert len(issues) == 3

    def test_empty_file_has_no_digest(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("  \n")
        boxes, issues, digest, _ = parse_label_file(f, Counter())
        assert boxes.shape[0] == 0 and not issues and digest is None


class TestBuildSnapshot:
    def test_splits_and_names(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        ds.sample("val", "b.jpg")
        snap = build_snapshot(str(ds.yaml_path))
        assert {s.name for s in snap.splits} == {"train", "val"}
        assert snap.nc == 2
        assert snap.names == {0: "cat", 1: "dog"}

    def test_missing_label_is_background(self, make_dataset):
        ds = make_dataset()
        ds.image("train", "a.jpg")
        ds.sample("val", "b.jpg")
        snap = build_snapshot(str(ds.yaml_path))
        record = snap.split("train").records[0]
        assert not record.label_exists
        assert record.is_background

    def test_utf8_yaml_with_non_ascii_names(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        ds.sample("val", "b.jpg")
        import yaml as yaml_mod

        config = yaml_mod.safe_load(ds.yaml_path.read_text(encoding="utf-8"))
        config["names"] = ["人", "犬"]
        ds.yaml_path.write_text(
            yaml_mod.safe_dump(config, allow_unicode=True), encoding="utf-8"
        )
        snap = build_snapshot(str(ds.yaml_path))
        assert snap.names == {0: "人", 1: "犬"}


class TestFormatGuard:
    def test_clean_detection_passes(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        ds.sample("val", "b.jpg")
        snap = build_snapshot(str(ds.yaml_path))
        assert detect_non_detection(snap) is None

    def test_kpt_shape_means_pose(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        ds.set_yaml(kpt_shape=[17, 3])
        snap = build_snapshot(str(ds.yaml_path))
        assert "pose" in detect_non_detection(snap)

    def test_nine_field_lines(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg", boxes="0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n" * 3)
        snap = build_snapshot(str(ds.yaml_path))
        suspected = detect_non_detection(snap)
        assert suspected is not None and "obb or segment" in suspected

    def test_polygon_lines(self, make_dataset):
        ds = make_dataset()
        ds.sample(
            "train",
            "a.jpg",
            boxes="0 0.1 0.1 0.5 0.1 0.5 0.5 0.3 0.6 0.1 0.5\n"
            "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5 0.2 0.2 0.3 0.3 0.4 0.1\n",
        )
        snap = build_snapshot(str(ds.yaml_path))
        suspected = detect_non_detection(snap)
        assert suspected is not None and "segment" in suspected

    def test_pose_rows_without_kpt_shape(self, make_dataset):
        # 56 fields = 55 coordinates, an invalid polygon: the rows are syntax
        # issues for the parser but must still trip the format guard.
        pose_row = "0 0.5 0.5 0.2 0.2 " + " ".join(["0.5 0.5 2"] * 17)
        ds = make_dataset()
        ds.sample("train", "a.jpg", boxes=f"{pose_row}\n" * 3)
        ds.sample("val", "b.jpg", boxes=f"{pose_row}\n" * 3)
        snap = build_snapshot(str(ds.yaml_path))
        assert snap.line_field_counts == {56: 6}
        assert "pose" in detect_non_detection(snap)
        with pytest.raises(NotADetectionDatasetError):
            diagnose(str(ds.yaml_path), fast=True, progress=False)

    def test_diagnose_raises_for_pose(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        ds.set_yaml(kpt_shape=[17, 3])
        with pytest.raises(NotADetectionDatasetError):
            diagnose(str(ds.yaml_path), fast=True, progress=False)

    def test_mostly_valid_detect_not_guarded(self, make_dataset):
        # A few broken lines should be syntax errors, not a format mismatch.
        ds = make_dataset()
        ds.sample(
            "train",
            "a.jpg",
            boxes="0 0.5 0.5 0.2 0.2\n" * 8 + "0 0.5 0.5 0.2 0.2 0.9\n",
        )
        snap = build_snapshot(str(ds.yaml_path))
        assert detect_non_detection(snap) is None
