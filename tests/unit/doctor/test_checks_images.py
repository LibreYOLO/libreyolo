"""Image-content (images.*) and leakage (splits.*) checks."""

import shutil

import pytest

from libreyolo.doctor.runner import diagnose

from .conftest import finding_ids, findings_for

pytestmark = pytest.mark.unit


def run_full(ds, **kwargs):
    return diagnose(str(ds.yaml_path), progress=False, **kwargs)


def clean(ds):
    ds.sample("train", "a.jpg", seed=1)
    ds.sample("train", "b.jpg", seed=2)
    ds.sample("val", "c.jpg", seed=3)


class TestImageChecks:
    def test_clean_dataset(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        report = run_full(ds)
        assert not report.errors
        for cid in ("images.corrupt", "splits.leakage_exact"):
            assert cid not in finding_ids(report)

    def test_corrupt_image(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.image("train", "broken.jpg", raw_bytes=b"this is not a jpeg")
        ds.label("train", "broken.txt", "0 0.5 0.5 0.2 0.2\n")
        report = run_full(ds)
        (f,) = findings_for(report, "images.corrupt")
        assert f.count == 1 and f.severity.value == "error"

    def test_zero_byte_image(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.image("train", "empty.jpg", raw_bytes=b"")
        report = run_full(ds)
        (f,) = findings_for(report, "images.corrupt")
        assert "zero-byte" in (f.details.get("error") or f.message) or f.count == 1

    def test_exif_orientation(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.sample("train", "rotated.jpg", seed=4, exif_orientation=6)
        report = run_full(ds)
        (f,) = findings_for(report, "images.exif_orientation")
        assert f.count == 1

    def test_odd_mode(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.sample("train", "gray.png", seed=5, mode="L")
        report = run_full(ds)
        (f,) = findings_for(report, "images.odd_mode")
        assert f.count == 1 and "L" in f.details["modes"]

    def test_tiny_image(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.sample("train", "tiny.jpg", size=(16, 16))
        report = run_full(ds)
        (f,) = findings_for(report, "images.tiny_or_extreme")
        assert f.count == 1

    def test_uniform_image(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.sample("train", "flat.png", color=(0, 0, 0))
        report = run_full(ds)
        (f,) = findings_for(report, "images.uniform")
        assert f.count == 1

    def test_exact_duplicates_within_split(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        src = ds.root / "images" / "train" / "a.jpg"
        dup = ds.root / "images" / "train" / "a_copy.jpg"
        shutil.copyfile(src, dup)
        ds.label("train", "a_copy.txt", "0 0.5 0.5 0.2 0.2\n")
        report = run_full(ds)
        (f,) = findings_for(report, "images.exact_duplicates")
        assert f.split == "train" and f.count == 1

    def test_leakage_exact_across_splits(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        src = ds.root / "images" / "train" / "a.jpg"
        leak = ds.root / "images" / "val" / "leaked.jpg"
        shutil.copyfile(src, leak)
        ds.label("val", "leaked.txt", "0 0.5 0.5 0.2 0.2\n")
        report = run_full(ds)
        (f,) = findings_for(report, "splits.leakage_exact")
        assert f.severity.value == "error" and f.count == 1

    def test_near_duplicates_within_split(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        # Same noise pattern saved as png and jpg: different bytes, near-equal
        # pixels, so dHash distance is small but sha1 differs.
        ds.sample("train", "near1.png", seed=42)
        ds.sample("train", "near2.jpg", seed=42)
        report = run_full(ds)
        findings = findings_for(report, "images.near_duplicates")
        assert findings and findings[0].split == "train"

    def test_fast_mode_skips_image_checks(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.image("train", "broken.jpg", raw_bytes=b"junk")
        report = diagnose(str(ds.yaml_path), fast=True, progress=False)
        assert "images.corrupt" not in finding_ids(report)
        assert "images.corrupt" in report.skipped_checks
