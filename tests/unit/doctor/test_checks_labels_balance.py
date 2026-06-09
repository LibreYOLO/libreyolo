"""Label-content (labels.*) and class-balance (balance.*) checks."""

import pytest

from libreyolo.doctor.runner import diagnose

from .conftest import finding_ids, findings_for

pytestmark = pytest.mark.unit


def run_fast(ds, **kwargs):
    return diagnose(str(ds.yaml_path), fast=True, progress=False, **kwargs)


def clean(ds):
    ds.sample("train", "a.jpg", boxes="0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    ds.sample("train", "b.jpg", boxes="0 0.4 0.4 0.2 0.2\n1 0.7 0.7 0.1 0.1\n")
    ds.sample("val", "c.jpg", boxes="0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")


class TestLabelChecks:
    def test_clean_dataset_has_no_label_errors(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        report = run_fast(ds)
        assert not report.errors

    def test_polygon_minority_is_info_not_error(self, make_dataset):
        # A few segmentation-style rows train fine (polygon -> box extent),
        # so doctor must not fail the dataset; it informs instead.
        ds = make_dataset()
        clean(ds)
        ds.label(
            "train",
            "a.txt",
            "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n",
        )
        report = run_fast(ds)
        assert "labels.syntax" not in finding_ids(report)
        (f,) = findings_for(report, "labels.polygon_line")
        assert f.count == 1 and f.severity.value == "info"
        assert not report.errors

    def test_syntax(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.label("train", "a.txt", "0 0.5 0.5 0.2 0.2\n0 0.5 oops 0.2 0.2\n")
        report = run_fast(ds)
        (f,) = findings_for(report, "labels.syntax")
        assert f.count == 1 and f.severity.value == "error"

    def test_class_out_of_range(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.label("train", "a.txt", "5 0.5 0.5 0.2 0.2\n-1 0.5 0.5 0.2 0.2\n")
        report = run_fast(ds)
        (f,) = findings_for(report, "labels.class_out_of_range")
        assert f.count == 2

    def test_pixel_coords_flagged(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.label("train", "a.txt", "0 320 240 100 80\n")
        report = run_fast(ds)
        errors = [
            f
            for f in findings_for(report, "labels.coords_out_of_range")
            if f.severity.value == "error"
        ]
        assert errors and "pixel" in errors[0].message

    def test_box_spill_is_warning(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.label("train", "a.txt", "0 0.95 0.5 0.2 0.2\n")
        report = run_fast(ds)
        warnings = [
            f
            for f in findings_for(report, "labels.coords_out_of_range")
            if f.severity.value == "warning"
        ]
        assert warnings and warnings[0].count == 1

    def test_degenerate_box(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.label("train", "a.txt", "0 0.5 0.5 0.0 0.2\n")
        report = run_fast(ds)
        assert "labels.degenerate_box" in finding_ids(report)

    def test_tiny_object(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        # 0.001 * 640 = 0.64 px on the short side.
        ds.label("train", "a.txt", "0 0.5 0.5 0.001 0.2\n")
        report = run_fast(ds)
        (f,) = findings_for(report, "labels.tiny_object")
        assert f.count == 1

    def test_huge_box(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.label("train", "a.txt", "0 0.5 0.5 0.99 0.99\n")
        report = run_fast(ds)
        assert "labels.huge_box" in finding_ids(report)

    def test_extreme_aspect(self, make_dataset):
        # In fast mode (no image dims) the threshold is doubled to absorb
        # image-aspect uncertainty, so use a 180:1 sliver.
        ds = make_dataset()
        clean(ds)
        ds.label("train", "a.txt", "0 0.5 0.5 0.9 0.005\n")
        report = run_fast(ds)
        assert "labels.extreme_aspect" in finding_ids(report)

    def test_extreme_aspect_exact_with_image_dims(self, make_dataset):
        # With the decode pass, pixel dims make a 90:1 sliver detectable.
        ds = make_dataset()
        clean(ds)
        ds.label("train", "a.txt", "0 0.5 0.5 0.9 0.01\n")
        report = diagnose(str(ds.yaml_path), progress=False)
        assert "labels.extreme_aspect" in finding_ids(report)

    def test_duplicate_box(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.label("train", "a.txt", "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n")
        report = run_fast(ds)
        (f,) = findings_for(report, "labels.duplicate_box")
        assert f.count == 1

    def test_identical_files(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        for i in range(6):
            ds.sample("train", f"copy{i}.jpg", boxes="1 0.4 0.4 0.3 0.3\n")
        report = run_fast(ds)
        (f,) = findings_for(report, "labels.identical_files")
        assert f.count == 6


class TestBalanceChecks:
    def test_zero_and_few_instances(self, make_dataset):
        ds = make_dataset(nc=3, names=["cat", "dog", "bird"])
        ds.sample("train", "a.jpg", boxes="0 0.5 0.5 0.2 0.2\n" * 3)
        ds.sample("train", "b.jpg", boxes="1 0.5 0.5 0.2 0.2\n")
        ds.sample("val", "c.jpg", boxes="0 0.5 0.5 0.2 0.2\n")
        report = run_fast(ds)
        (zero,) = findings_for(report, "balance.class_zero_instances")
        assert "bird" in zero.message
        (few,) = findings_for(report, "balance.class_few_instances")
        assert "dog" in few.message

    def test_imbalance_warns_past_ratio(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg", boxes="0 0.5 0.5 0.2 0.2\n" * 150)
        ds.sample("train", "b.jpg", boxes="1 0.5 0.5 0.2 0.2\n")
        ds.sample("val", "c.jpg", boxes="0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
        report = run_fast(ds)
        (f,) = findings_for(report, "balance.imbalance")
        assert f.severity.value == "warning"

    def test_split_coverage(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg", boxes="0 0.5 0.5 0.2 0.2\n")
        ds.sample("val", "b.jpg", boxes="1 0.5 0.5 0.2 0.2\n")
        report = run_fast(ds)
        messages = [f.message for f in findings_for(report, "balance.split_coverage")]
        assert any("in val but never in train" in m for m in messages)
        assert any("in train but never in val" in m for m in messages)

    def test_background_ratio_warns(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        for i in range(3):
            ds.image("train", f"bg{i}.jpg")
        ds.sample("val", "c.jpg")
        report = run_fast(ds)
        train_findings = [
            f
            for f in findings_for(report, "balance.background_ratio")
            if f.split == "train"
        ]
        assert train_findings and train_findings[0].severity.value == "warning"

    def test_background_ratio_info_when_low(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.image("train", "bg.jpg")  # one background image: low but non-zero
        report = run_fast(ds)
        train_findings = [
            f
            for f in findings_for(report, "balance.background_ratio")
            if f.split == "train"
        ]
        assert train_findings and train_findings[0].severity.value == "info"

    def test_background_ratio_silent_at_zero(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        report = run_fast(ds)
        assert "balance.background_ratio" not in finding_ids(report)

    def test_imbalance_silent_when_balanced(self, make_dataset):
        ds = make_dataset()
        clean(ds)  # cat:dog is 1:1
        report = run_fast(ds)
        assert "balance.imbalance" not in finding_ids(report)

    def test_split_skew(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg", boxes="0 0.5 0.5 0.2 0.2\n" * 9)
        ds.sample("train", "b.jpg", boxes="1 0.5 0.5 0.2 0.2\n")
        ds.sample("val", "c.jpg", boxes="0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
        report = run_fast(ds)
        (f,) = findings_for(report, "balance.split_skew")
        assert f.severity.value == "info" and "cat" in f.message

    def test_crowded_image(self, make_dataset):
        ds = make_dataset()
        for i in range(100):
            ds.sample("train", f"t{i}.jpg", boxes="0 0.5 0.5 0.2 0.2\n")
        ds.sample("train", "crowd.jpg", boxes="0 0.5 0.5 0.01 0.01\n" * 60)
        ds.sample("val", "v.jpg", boxes="0 0.5 0.5 0.2 0.2\n")
        report = run_fast(ds)
        (f,) = findings_for(report, "labels.crowded_image")
        assert f.count == 1 and "crowd" in str(f.paths[0])
