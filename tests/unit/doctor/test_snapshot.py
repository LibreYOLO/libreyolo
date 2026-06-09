"""Snapshot construction, label parsing, and the format guard."""

from collections import Counter

import pytest

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
