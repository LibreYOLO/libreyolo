"""Unit tests for the optical flow validator."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from PIL import Image

from libreyolo.data.flow_dataset import write_flo
from libreyolo.validation import FlowValidator, ValidationConfig
from libreyolo.validation.flow_validator import flow_pair_metrics

pytestmark = pytest.mark.unit

H, W = 10, 14


def _make_dataset(tmp_path, n_pairs=3):
    """One sequence of ``n_pairs + 1`` frames with a (3, -1) px flow field."""
    seq = tmp_path / "images" / "val" / "seq"
    for i in range(n_pairs + 1):
        seq.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (W, H), color=(20 * i, 20 * i, 20 * i)).save(seq / f"f{i}.png")
    for i in range(n_pairs):
        flow = np.zeros((H, W, 2), dtype=np.float32)
        flow[..., 0] = 3.0
        flow[..., 1] = -1.0
        write_flo(tmp_path / "flows" / "val" / "seq" / f"f{i}.flo", flow)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": str(tmp_path), "val": "images/val"}))
    return yaml_path


class _StubFlowModel:
    """Model double: predict() returns the true field plus an offset."""

    size = "s"
    names = {0: "flow"}
    task = "flow"

    def __init__(self, offset=(0.0, 0.0)):
        self.model = nn.Identity()
        self.offset = offset
        self.calls = []

    def _get_model_name(self):
        return "stub"

    def predict(self, source, **kwargs):
        self.calls.append(list(source))
        flow = torch.zeros(H, W, 2)
        flow[..., 0] = 3.0 + self.offset[0]
        flow[..., 1] = -1.0 + self.offset[1]
        return [SimpleNamespace(flow=SimpleNamespace(data=flow))]


def _validate(tmp_path, model, **extra):
    config = ValidationConfig(
        data=str(_make_dataset(tmp_path)),
        batch_size=2,
        imgsz=32,
        device="cpu",
        verbose=False,
        save_dir=str(tmp_path / "run"),
        **extra,
    )
    return FlowValidator(model=model, config=config)()


def test_perfect_prediction(tmp_path):
    model = _StubFlowModel()
    metrics = _validate(tmp_path, model)
    assert metrics["metrics/epe"] == pytest.approx(0.0)
    assert metrics["metrics/px1"] == pytest.approx(1.0)
    assert metrics["metrics/px3"] == pytest.approx(1.0)
    assert metrics["metrics/fl_all"] == pytest.approx(0.0)
    assert metrics["fitness"] == pytest.approx(1.0)
    assert metrics["speed/images_seen"] == 3
    # Every pair went through predict([frame, next_frame]).
    assert len(model.calls) == 3
    assert model.calls[0][0].endswith("f0.png") and model.calls[0][1].endswith("f1.png")


def test_constant_two_pixel_offset(tmp_path):
    metrics = _validate(tmp_path, _StubFlowModel(offset=(2.0, 0.0)))
    assert metrics["metrics/epe"] == pytest.approx(2.0)
    assert metrics["metrics/px1"] == pytest.approx(0.0)
    assert metrics["metrics/px3"] == pytest.approx(1.0)
    assert metrics["metrics/fl_all"] == pytest.approx(0.0)
    assert metrics["fitness"] == pytest.approx(0.0)


def test_large_offset_is_an_outlier(tmp_path):
    metrics = _validate(tmp_path, _StubFlowModel(offset=(0.0, 5.0)))
    assert metrics["metrics/epe"] == pytest.approx(5.0)
    assert metrics["metrics/px3"] == pytest.approx(0.0)
    assert metrics["metrics/fl_all"] == pytest.approx(1.0)


def test_pair_metrics_respect_valid_mask():
    pred = torch.zeros(4, 4, 2)
    target = torch.zeros(4, 4, 2)
    target[0, :, 0] = 10.0  # wrong row, masked out
    valid = torch.ones(4, 4, dtype=torch.bool)
    valid[0] = False
    metrics = flow_pair_metrics(pred, target, valid)
    assert metrics["metrics/epe"] == 0.0 and metrics["metrics/px1"] == 1.0
    assert flow_pair_metrics(pred, target, torch.zeros(4, 4, dtype=torch.bool)) == {}


def test_fl_all_needs_both_thresholds():
    # Error of 4 px on a 100 px vector is under 5 percent: not an outlier.
    pred = torch.zeros(2, 2, 2)
    target = torch.zeros(2, 2, 2)
    target[..., 0] = 100.0
    pred[..., 0] = 96.0
    valid = torch.ones(2, 2, dtype=torch.bool)
    metrics = flow_pair_metrics(pred, target, valid)
    assert metrics["metrics/epe"] == pytest.approx(4.0)
    assert metrics["metrics/fl_all"] == 0.0


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="does not match"):
        flow_pair_metrics(torch.zeros(2, 2, 2), torch.zeros(3, 2, 2), torch.ones(3, 2, dtype=torch.bool))


def test_missing_flow_attribute_is_an_error(tmp_path):
    class _NoFlow(_StubFlowModel):
        def predict(self, source, **kwargs):
            return [SimpleNamespace(flow=None)]

    with pytest.raises(ValueError, match="Results.flow"):
        _validate(tmp_path, _NoFlow())


def test_requires_data_yaml(tmp_path):
    # data_dir satisfies the generic config, but flow needs the dataset YAML.
    config = ValidationConfig(data=None, data_dir=str(tmp_path), imgsz=32, device="cpu", verbose=False)
    with pytest.raises(ValueError, match="data="):
        FlowValidator(model=_StubFlowModel(), config=config)()
