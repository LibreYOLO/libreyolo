"""Unit tests for the live training status writer and the monitor server."""

from __future__ import annotations

import json
import logging

import pytest

from libreyolo.training.artifacts import TrainingStatusCallback
from libreyolo.training.callbacks import (
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from libreyolo.ui import train_monitor

pytestmark = pytest.mark.unit


def _start(save_dir, *, start_epoch=1):
    return TrainStartEvent(
        start_epoch=start_epoch,
        total_epochs=4,
        model_family="yolo9",
        model_size="s",
        task="detect",
        save_dir=str(save_dir),
        config={"epochs": 4},
    )


def _epoch(save_dir, *, epoch=1, metric=0.5, best_epoch=1):
    return TrainEpochEvent(
        epoch=epoch,
        total_epochs=4,
        model_family="yolo9",
        model_size="s",
        task="detect",
        save_dir=str(save_dir),
        train_loss=2.0 - epoch * 0.1,
        train_loss_items={"box": 0.2, "cls": 0.3},
        lr={"group0": 0.01},
        val_metrics={"metrics/mAP50-95": metric},
        validated=True,
        is_best=True,
        current_metric=metric,
        current_metric_name="metrics/mAP50-95",
        best_metric=metric,
        best_metric_name="metrics/mAP50-95",
        best_epoch=best_epoch,
        epoch_seconds=1.0,
    )


def _end(save_dir):
    return TrainEndEvent(
        total_epochs=4,
        completed_epochs=4,
        model_family="yolo9",
        model_size="s",
        task="detect",
        save_dir=str(save_dir),
        final_loss=1.6,
        best_metric=0.62,
        best_epoch=3,
        total_seconds=4.0,
        results={
            "best_checkpoint": str(save_dir / "weights" / "best.pt"),
            "last_checkpoint": str(save_dir / "weights" / "last.pt"),
        },
    )


def _load(save_dir):
    return json.loads((save_dir / "status.json").read_text())


def test_status_running_then_completed(tmp_path):
    cb = TrainingStatusCallback()
    cb.on_train_start(_start(tmp_path))
    running = _load(tmp_path)
    assert running["state"] == "running"
    assert running["total_epochs"] == 4
    assert running["pid"] > 0

    cb.on_train_epoch_end(_epoch(tmp_path, epoch=1, metric=0.5))
    mid = _load(tmp_path)
    assert mid["state"] == "running"
    # Epoch numbers are one-based end to end: the first completed epoch is 1.
    assert mid["current_epoch"] == 1
    assert mid["completed_epochs"] == 1
    assert mid["progress"] == pytest.approx(0.25)
    assert mid["metrics"]["mAP50-95"] == pytest.approx(0.5)
    # ETA = mean epoch time (1.0s) * remaining (3) = 3.0
    assert mid["eta_seconds"] == pytest.approx(3.0, abs=0.01)

    cb.on_train_end(_end(tmp_path))
    done = _load(tmp_path)
    assert done["state"] == "completed"
    assert done["best_metric"] == pytest.approx(0.62)
    assert done["checkpoints"]["best"].endswith("best.pt")


def test_status_failed_records_error(tmp_path):
    cb = TrainingStatusCallback()
    cb.on_train_start(_start(tmp_path))
    cb.on_train_epoch_end(_epoch(tmp_path, epoch=1))
    exc = RuntimeError("CUDA out of memory")
    cb.on_train_exception(
        TrainExceptionEvent(
            epoch=1,
            total_epochs=4,
            model_family="yolo9",
            model_size="s",
            task="detect",
            save_dir=str(tmp_path),
            exception=exc,
            exception_type="RuntimeError",
            exception_message="CUDA out of memory",
            elapsed_seconds=2.0,
        )
    )
    failed = _load(tmp_path)
    assert failed["state"] == "failed"
    assert failed["error"]["type"] == "RuntimeError"
    assert "out of memory" in failed["error"]["message"]


def test_status_atomic_write_is_valid_json_every_time(tmp_path):
    """status.json must always parse; never a half-written file."""
    cb = TrainingStatusCallback(write_log=False)
    cb.on_train_start(_start(tmp_path))
    for e in range(1, 5):
        cb.on_train_epoch_end(_epoch(tmp_path, epoch=e))
        json.loads((tmp_path / "status.json").read_text())  # raises if corrupt


def test_log_teeing_captures_libreyolo_output(tmp_path):
    cb = TrainingStatusCallback()
    cb.on_train_start(_start(tmp_path))
    logging.getLogger("libreyolo").info("epoch 1 done, mAP=0.5")
    cb.on_train_end(_end(tmp_path))
    log = (tmp_path / "train.log").read_text()
    assert "epoch 1 done" in log
    # handler must be detached after the run (no leak onto the global logger)
    assert cb._log_handler is None


def test_monitor_reads_status_and_results_csv_fallback(tmp_path):
    """With no metrics.jsonl, the monitor falls back to the family results.csv."""
    cb = TrainingStatusCallback(write_log=False)
    cb.on_train_start(_start(tmp_path))  # writes status.json, not metrics.jsonl

    # results.csv is written by TrainingArtifactsCallback; emulate a couple rows.
    (tmp_path / "results.csv").write_text(
        "epoch,train/loss,metrics/mAP50-95\n0,2.0,0.5\n1,1.9,0.55\n"
    )
    assert not (tmp_path / "metrics.jsonl").exists()

    assert train_monitor._read_status(tmp_path)["state"] == "running"
    metrics = train_monitor._read_metrics(tmp_path)
    assert "metrics/mAP50-95" in metrics["columns"]
    assert metrics["rows"][1]["metrics/mAP50-95"] == pytest.approx(0.55)


def test_metrics_jsonl_written_and_read(tmp_path):
    """Every family gets a universal, chart-ready metrics.jsonl history."""
    cb = TrainingStatusCallback(write_log=False)
    cb.on_train_start(_start(tmp_path))
    cb.on_train_epoch_end(_epoch(tmp_path, epoch=1, metric=0.5))
    cb.on_train_epoch_end(_epoch(tmp_path, epoch=2, metric=0.55))

    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2

    parsed = train_monitor._read_metrics(tmp_path)
    assert "epoch" in parsed["columns"]
    assert "train/loss" in parsed["columns"]
    assert "metrics/mAP50-95" in parsed["columns"]
    assert parsed["rows"][1]["metrics/mAP50-95"] == pytest.approx(0.55)


def test_metrics_jsonl_reset_on_fresh_run(tmp_path):
    cb = TrainingStatusCallback(write_log=False)
    cb.on_train_start(_start(tmp_path))
    cb.on_train_epoch_end(_epoch(tmp_path, epoch=1))
    # A new fresh run (start_epoch=1) must clear stale history.
    cb.on_train_start(_start(tmp_path, start_epoch=1))
    cb.on_train_epoch_end(_epoch(tmp_path, epoch=1))
    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1


def test_metrics_jsonl_skips_torn_line(tmp_path):
    (tmp_path / "metrics.jsonl").write_text(
        '{"epoch": 0, "train/loss": 2.0}\n{"epoch": 1, "train/loss": 1.9}\n{"epoch": 2, "trai'
    )
    parsed = train_monitor._read_metrics(tmp_path)
    assert len(parsed["rows"]) == 2  # torn final line dropped


def test_monitor_missing_status_is_graceful(tmp_path):
    assert train_monitor._read_status(tmp_path)["state"] == "missing"
    assert train_monitor._read_metrics(tmp_path) == {"columns": [], "rows": []}
    assert train_monitor._read_log_tail(tmp_path) == ""


def test_find_latest_run_prefers_status(tmp_path):
    old = tmp_path / "runs" / "exp"
    new = tmp_path / "runs" / "exp2"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "status.json").write_text("{}")
    (new / "status.json").write_text("{}")
    # make `new` newer
    import os
    import time

    now = time.time()
    os.utime(old / "status.json", (now - 100, now - 100))
    os.utime(new / "status.json", (now, now))
    assert train_monitor.find_latest_run(tmp_path / "runs") == new


def test_image_path_sandbox_blocks_escape(tmp_path):
    (tmp_path / "val.png").write_bytes(b"x")
    assert train_monitor._resolve_in_run(tmp_path, "val.png") is not None
    assert train_monitor._resolve_in_run(tmp_path, "../secret.txt") is None
    assert train_monitor._resolve_in_run(tmp_path, "../../etc/passwd") is None


def _make_run(root, rel, state="running"):
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(json.dumps({"state": state, "save_dir": str(d)}))
    return d


def test_discover_runs_finds_all_and_newest_first(tmp_path):
    import os
    import time

    root = tmp_path / "runs"
    a = _make_run(root, "train/exp")
    b = _make_run(root, "train/exp2")
    c = _make_run(root, "detect/predict")  # different subtree
    now = time.time()
    os.utime(a / "status.json", (now - 300, now - 300))
    os.utime(b / "status.json", (now, now))
    os.utime(c / "status.json", (now - 100, now - 100))

    runs = train_monitor.discover_runs(root)
    assert set(runs) == {a, b, c}
    assert runs[0] == b  # newest first


def test_discover_runs_includes_root_when_root_is_a_run(tmp_path):
    _make_run(tmp_path, ".")  # tmp_path itself is a run
    runs = train_monitor.discover_runs(tmp_path)
    assert tmp_path in runs


def test_run_id_and_resolution_roundtrip(tmp_path):
    root = tmp_path / "runs"
    run = _make_run(root, "train/exp2")
    run_id = train_monitor.run_id_for(root, run)
    assert run_id == "train/exp2"
    assert train_monitor._run_dir(root, run_id) == run.resolve()


def test_run_dir_rejects_escape_and_unknown(tmp_path):
    root = tmp_path / "runs"
    _make_run(root, "train/exp")
    assert train_monitor._run_dir(root, "../../etc") is None
    assert train_monitor._run_dir(root, "train/does_not_exist") is None
    assert train_monitor._run_dir(root, "") is None
    assert train_monitor._run_dir(root, None) is None


def test_run_summary_reports_state(tmp_path):
    root = tmp_path / "runs"
    run = _make_run(root, "train/exp", state="completed")
    (run / "status.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "model_family": "yolo9",
                "model_size": "t",
                "task": "detect",
                "progress": 1.0,
                "total_epochs": 8,
                "best_metric": 0.5,
                "best_metric_name": "mAP50-95",
            }
        )
    )
    summ = train_monitor._run_summary(root, run)
    assert summ["id"] == "train/exp"
    assert summ["state"] == "completed"
    assert summ["model"] == "yolo9t"
    assert summ["best_metric"] == 0.5


def test_server_serves_multiple_runs_by_id(tmp_path):
    """One server, two runs, each addressable by its own ?run= id."""
    import threading
    import urllib.request

    root = tmp_path / "runs"
    _make_run(root, "train/a", state="running")
    _make_run(root, "train/b", state="completed")

    httpd, url = train_monitor.serve(root, port=0, open_browser=False)
    httpd.server_port = httpd.server_address[1]  # port=0 -> OS-assigned
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        runs = json.load(urllib.request.urlopen(base + "/api/runs", timeout=5))["runs"]
        ids = {r["id"] for r in runs}
        assert ids == {"train/a", "train/b"}

        sa = json.load(urllib.request.urlopen(base + "/api/status?run=train/a", timeout=5))
        sb = json.load(urllib.request.urlopen(base + "/api/status?run=train/b", timeout=5))
        assert sa["state"] == "running"
        assert sb["state"] == "completed"

        # unknown / escaping run ids are refused, not served
        code = urllib.request.urlopen(base + "/", timeout=5).status
        assert code == 200
        try:
            urllib.request.urlopen(base + "/api/status?run=../../etc", timeout=5)
            raise AssertionError("escape not blocked")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
