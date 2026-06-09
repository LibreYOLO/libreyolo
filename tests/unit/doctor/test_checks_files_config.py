"""YAML (config.*) and file-layout (files.*) checks."""

import pytest

from libreyolo.doctor.runner import diagnose

from .conftest import finding_ids, findings_for

pytestmark = pytest.mark.unit


def run_fast(ds, **kwargs):
    return diagnose(str(ds.yaml_path), fast=True, progress=False, **kwargs)


class TestConfigChecks:
    def test_nc_names_mismatch(self, make_dataset):
        ds = make_dataset(nc=5, names=["cat", "dog"])
        ds.sample("train", "a.jpg")
        ds.sample("val", "b.jpg")
        report = run_fast(ds)
        assert "config.nc_names_mismatch" in finding_ids(report)

    def test_missing_val_split_warns(self, make_dataset, tmp_path):
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        import yaml as yaml_mod

        config = yaml_mod.safe_load(ds.yaml_path.read_text())
        del config["val"]
        ds.yaml_path.write_text(yaml_mod.safe_dump(config))
        report = run_fast(ds)
        findings = findings_for(report, "config.missing_split")
        assert findings and findings[0].severity.value == "warning"

    def test_path_not_found(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        ds.set_yaml(val="images/nonexistent")
        report = run_fast(ds)
        findings = findings_for(report, "config.path_not_found")
        assert findings and findings[0].split == "val"

    def test_duplicate_names(self, make_dataset):
        ds = make_dataset(nc=2, names=["cat", "cat"])
        ds.sample("train", "a.jpg")
        ds.sample("val", "b.jpg")
        report = run_fast(ds)
        assert "config.duplicate_names" in finding_ids(report)

    def test_nc_as_float_still_checked(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        ds.sample("val", "b.jpg")
        ds.set_yaml(nc=5.0)  # YAML floats must not bypass the mismatch check
        report = run_fast(ds)
        assert "config.nc_names_mismatch" in finding_ids(report)

    def test_list_split_with_empty_dir(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        ds.sample("val", "b.jpg")
        (ds.root / "images" / "train_extra").mkdir()
        ds.set_yaml(train=["images/train", "images/train_extra"])
        report = run_fast(ds)
        findings = [
            f
            for f in findings_for(report, "config.path_not_found")
            if f.split == "train"
        ]
        assert findings and "no images" in findings[0].message


class TestFileChecks:
    def test_missing_label_counted(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        ds.image("train", "nolabel.jpg")
        ds.sample("val", "b.jpg")
        report = run_fast(ds)
        train_findings = [
            f for f in findings_for(report, "files.missing_label") if f.split == "train"
        ]
        assert train_findings and train_findings[0].count == 1

    def test_orphan_label(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        ds.sample("val", "b.jpg")
        ds.label("train", "ghost.txt", "0 0.5 0.5 0.2 0.2\n")
        report = run_fast(ds)
        (f,) = findings_for(report, "files.orphan_label")
        assert f.count == 1 and "ghost" in str(f.paths[0])

    def test_classes_txt_not_orphan(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        ds.sample("val", "b.jpg")
        ds.label("train", "classes.txt", "cat\ndog\n")
        report = run_fast(ds)
        assert "files.orphan_label" not in finding_ids(report)

    def test_unsupported_ext(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        ds.sample("val", "b.jpg")
        (ds.root / "images" / "train" / "video.mp4").write_bytes(b"\x00")
        report = run_fast(ds)
        (f,) = findings_for(report, "files.unsupported_ext")
        assert f.count == 1

    def test_missing_image_in_txt_split(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        ds.sample("val", "b.jpg")
        (ds.root / "train_list.txt").write_text(
            "images/train/a.jpg\nimages/train/deleted.jpg\n"
        )
        ds.set_yaml(train="train_list.txt")
        report = run_fast(ds)
        (f,) = findings_for(report, "files.missing_image")
        assert f.severity.value == "error" and f.count == 1
        assert "deleted" in str(f.paths[0])

    def test_missing_image_not_double_reported(self, make_dataset):
        # A missing listed image is files.missing_image's finding alone:
        # not a decode failure, not background, not a missing label.
        from libreyolo.doctor.runner import diagnose

        ds = make_dataset()
        ds.sample("train", "a.jpg")
        ds.sample("val", "b.jpg")
        (ds.root / "train_list.txt").write_text(
            "images/train/a.jpg\nimages/train/deleted.jpg\n"
        )
        ds.set_yaml(train="train_list.txt")
        report = diagnose(str(ds.yaml_path), progress=False)
        assert "files.missing_image" in finding_ids(report)
        assert "images.corrupt" not in finding_ids(report)
        train_missing_label = [
            f for f in findings_for(report, "files.missing_label") if f.split == "train"
        ]
        assert not train_missing_label

    def test_missing_names(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        ds.sample("val", "b.jpg")
        ds.set_yaml(names=[])
        report = run_fast(ds)
        assert "config.missing_names" in finding_ids(report)

    def test_txt_split_with_no_images(self, make_dataset):
        # An existing list file that resolves to zero images must not pass.
        ds = make_dataset()
        ds.sample("val", "b.jpg")
        (ds.root / "train_list.txt").write_text("# nothing here\n")
        ds.set_yaml(train="train_list.txt")
        report = run_fast(ds)
        findings = [
            f
            for f in findings_for(report, "config.path_not_found")
            if f.split == "train"
        ]
        assert findings and "no images" in findings[0].message
        assert report.exit_code() == 1

    def test_no_orphans_when_splits_share_label_dir(self, make_dataset):
        # txt-list splits can point into the same images/labels directories;
        # a label claimed by another split is not an orphan.
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        ds.sample("train", "b.jpg")
        (ds.root / "val_list.txt").write_text("images/train/a.jpg\n")
        ds.set_yaml(val="val_list.txt")
        report = run_fast(ds)
        assert "files.orphan_label" not in finding_ids(report)

    def test_case_collision(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg")
        ds.image("train", "a.png")  # same stem -> same label file
        ds.sample("val", "b.jpg")
        report = run_fast(ds)
        (f,) = findings_for(report, "files.case_collision")
        assert f.count == 1
