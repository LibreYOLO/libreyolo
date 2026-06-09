"""diagnose() orchestration, selection, report rendering, JSON contract."""

import json

import pytest

from libreyolo.doctor import (
    DatasetNotFoundError,
    DoctorError,
    UnknownCheckError,
    diagnose,
)
from libreyolo.doctor.report import Finding, Report, Severity

pytestmark = pytest.mark.unit


def clean(ds):
    ds.sample("train", "a.jpg", seed=1, boxes="0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    ds.sample("train", "b.jpg", seed=2, boxes="0 0.4 0.4 0.2 0.2\n1 0.7 0.7 0.1 0.1\n")
    ds.sample("val", "c.jpg", seed=3, boxes="0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")


class TestDiagnose:
    def test_healthy_dataset_exit_zero(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        report = diagnose(str(ds.yaml_path), progress=False)
        assert report.exit_code() == 0
        assert not report.errors

    def test_errors_drive_exit_code(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.label("train", "a.txt", "9 0.5 0.5 0.2 0.2\n")
        report = diagnose(str(ds.yaml_path), fast=True, progress=False)
        assert report.exit_code() == 1

    def test_strict_promotes_warnings(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.label("train", "a.txt", "0 0.5 0.5 0.99 0.99\n")  # huge box: warning
        report = diagnose(str(ds.yaml_path), fast=True, progress=False)
        assert report.exit_code() == 0
        assert report.exit_code(strict=True) == 1

    def test_skip_family(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.label("train", "a.txt", "0 0.5 0.5 0.99 0.99\n")
        report = diagnose(str(ds.yaml_path), fast=True, progress=False, skip=["labels"])
        assert not any(f.check_id.startswith("labels.") for f in report.findings)
        assert "labels.huge_box" in report.skipped_checks

    def test_only_selector(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        report = diagnose(
            str(ds.yaml_path), fast=True, progress=False, only=["balance"]
        )
        assert all(f.check_id.startswith("balance.") for f in report.findings)

    def test_unknown_selector_raises(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        with pytest.raises(UnknownCheckError):
            diagnose(str(ds.yaml_path), progress=False, skip=["imagez"])

    def test_missing_yaml_raises_doctor_error(self, tmp_path):
        with pytest.raises(DatasetNotFoundError):
            diagnose(str(tmp_path / "nope.yaml"), progress=False)
        assert issubclass(DatasetNotFoundError, DoctorError)

    def test_empty_selection_raises(self, make_dataset):
        # --fast removes all image checks; --only images then leaves nothing,
        # which must not masquerade as a clean run.
        ds = make_dataset()
        clean(ds)
        with pytest.raises(UnknownCheckError):
            diagnose(str(ds.yaml_path), fast=True, progress=False, only=["images"])

    def test_crashed_check_becomes_error_finding(self, make_dataset, monkeypatch):
        from libreyolo.doctor import checks as checks_pkg
        from libreyolo.doctor.report import Finding, Severity

        def exploding(snap, cfg):
            yield Finding("balance.imbalance", Severity.INFO, "partial")
            raise RuntimeError("boom")

        checks_pkg._load()
        monkeypatch.setitem(checks_pkg._REGISTRY, "balance.imbalance", exploding)
        ds = make_dataset()
        clean(ds)
        report = diagnose(str(ds.yaml_path), fast=True, progress=False)
        crash = [f for f in report.findings if f.check_id == "balance.imbalance"]
        # The partial finding is discarded; one ERROR reports the crash.
        assert len(crash) == 1
        assert crash[0].severity.value == "error" and "boom" in crash[0].message
        assert report.exit_code() == 1

    def test_imgsz_changes_tiny_threshold(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        # 0.03 * 64 px image -> ~2 px at imgsz=64, ~19 px at imgsz=640.
        ds.label("train", "a.txt", "0 0.5 0.5 0.03 0.5\n")
        small = diagnose(str(ds.yaml_path), imgsz=64, fast=True, progress=False)
        large = diagnose(str(ds.yaml_path), imgsz=640, fast=True, progress=False)
        small_ids = {f.check_id for f in small.findings}
        large_ids = {f.check_id for f in large.findings}
        assert "labels.tiny_object" in small_ids
        assert "labels.tiny_object" not in large_ids


class TestReport:
    def test_to_dict_is_json_serializable(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.label("train", "a.txt", "9 320 240 0 -1\n")
        report = diagnose(str(ds.yaml_path), progress=False)
        payload = json.dumps(report.to_dict())
        decoded = json.loads(payload)
        assert decoded["summary"]["errors"] >= 1
        assert decoded["stats"]["splits"]["train"]["images"] == 2

    def test_render_human_is_ascii(self, make_dataset):
        ds = make_dataset()
        clean(ds)
        ds.label("train", "a.txt", "9 0.5 0.5 0.2 0.2\n")
        report = diagnose(str(ds.yaml_path), fast=True, progress=False)
        text = report.render_human()
        text.encode("ascii")  # must not raise on cp1252-ish consoles
        assert "ERRORS" in text

    def test_findings_sorted_errors_first(self):
        report = Report(
            findings=[
                Finding("z.info", Severity.INFO, "i"),
                Finding("a.warn", Severity.WARNING, "w"),
                Finding("m.err", Severity.ERROR, "e"),
            ],
            stats={},
        )
        ordered = [f["severity"] for f in report.to_dict()["findings"]]
        assert ordered == ["error", "warning", "info"]
