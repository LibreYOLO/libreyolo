"""Unit tests for the optical flow validator."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from PIL import Image

from libreyolo.data.flow_dataset import write_flo
from libreyolo.models.base.inference import InferenceRunner
from libreyolo.utils.image_loader import ImageLoader
from libreyolo.validation import FlowValidator, ValidationConfig
from libreyolo.validation.flow_validator import flow_pair_metrics, group_sequence_frames

pytestmark = pytest.mark.unit

H, W = 10, 14


def _make_dataset(tmp_path, n_pairs=3, sequences=("seq",)):
    """Sequences of ``n_pairs + 1`` frames each with a (3, -1) px flow field.

    Frame ``i`` of every sequence is a solid shade ``20 * i``, so a family
    that records the mean brightness of the frames it receives reveals
    which frames made up each window.
    """
    for name in sequences:
        seq = tmp_path / "images" / "val" / name
        for i in range(n_pairs + 1):
            seq.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (W, H), color=(20 * i, 20 * i, 20 * i)).save(seq / f"f{i}.png")
        for i in range(n_pairs):
            flow = np.zeros((H, W, 2), dtype=np.float32)
            flow[..., 0] = 3.0
            flow[..., 1] = -1.0
            write_flo(tmp_path / "flows" / "val" / name / f"f{i}.flo", flow)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": str(tmp_path), "val": "images/val"}))
    return yaml_path


class _StubFlowModel:
    """Model double: predict(frames, stream=True) yields the true field plus an offset.

    Like the real runner it yields one result per consecutive pair of the
    frame sequence it is given, each carrying the path of its first frame.
    """

    size = "s"
    names = {0: "flow"}
    task = "flow"

    def __init__(self, offset=(0.0, 0.0)):
        self.model = nn.Identity()
        self.offset = offset
        self.calls = []
        self.kwargs = []

    def _get_model_name(self):
        return "stub"

    def _result(self, path):
        flow = torch.zeros(H, W, 2)
        flow[..., 0] = 3.0 + self.offset[0]
        flow[..., 1] = -1.0 + self.offset[1]
        return SimpleNamespace(flow=SimpleNamespace(data=flow), path=path)

    def predict(self, source, **kwargs):
        frames = list(source)
        self.calls.append(frames)
        self.kwargs.append(dict(kwargs))
        return (self._result(path) for path in frames[:-1])


class _RunnerFlowModel:
    """A flow family double that predicts through the real InferenceRunner.

    ``_forward_flow`` records the mean shade of every frame in each window it
    receives, so a test can see whether the runner handed it the real
    neighbouring frames of the sequence or edge-padded copies.
    """

    task = "flow"
    family = "stubflow"
    TTA_ENABLED = False
    size = "s"
    names = {0: "flow"}
    device = torch.device("cpu")

    def __init__(self, context=(0, 0)):
        self.FLOW_CONTEXT = context
        self.model = nn.Identity()
        self.window_means = []
        self.input_sizes = []

    def _get_model_name(self):
        return "stub"

    def _get_input_size(self):
        return 32

    def predict(self, source, **kwargs):
        return InferenceRunner(self)(source, **kwargs)

    def _preprocess(self, image, color_format="auto", input_size=None):
        self.input_sizes.append(input_size)
        pil = ImageLoader.load(image, color_format=color_format)
        tensor = torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1).float().unsqueeze(0)
        return tensor, pil, pil.size, 1.0

    def _forward_flow(self, window):
        self.window_means.append([round(float(frame.mean())) for frame in window])
        before = self.FLOW_CONTEXT[0]
        return {"delta": float(window[before + 1].mean() - window[before].mean())}

    def _postprocess(self, output, conf, iou, original_size, max_det=300, ratio=1.0, classes=None, **kwargs):
        w, h = original_size
        return {"flow": torch.full((h, w, 2), output["delta"])}


def _validate(tmp_path, model, imgsz=32, sequences=("seq",), **extra):
    config = ValidationConfig(
        data=str(_make_dataset(tmp_path, sequences=sequences)),
        batch_size=2,
        imgsz=imgsz,
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
    # The whole sequence went through predict(frames, stream=True) once.
    assert len(model.calls) == 1
    assert [Path(p).name for p in model.calls[0]] == ["f0.png", "f1.png", "f2.png", "f3.png"]
    assert model.kwargs[0]["stream"] is True


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
        def _result(self, path):
            return SimpleNamespace(flow=None, path=path)

    with pytest.raises(ValueError, match="Results.flow"):
        _validate(tmp_path, _NoFlow())


def test_too_few_results_is_an_error(tmp_path):
    class _Short(_StubFlowModel):
        def predict(self, source, **kwargs):
            frames = list(source)
            return (self._result(path) for path in frames[:1])

    with pytest.raises(ValueError, match="fewer results"):
        _validate(tmp_path, _Short())


def test_result_order_mismatch_is_an_error(tmp_path):
    class _Reversed(_StubFlowModel):
        def predict(self, source, **kwargs):
            frames = list(source)
            return (self._result(path) for path in reversed(frames[:-1]))

    with pytest.raises(ValueError, match="frame order"):
        _validate(tmp_path, _Reversed())


def test_multi_frame_family_gets_real_previous_frame(tmp_path):
    # FLOW_CONTEXT (1, 0): the family sees frames t-1, t, t+1. Shades are
    # 0, 20, 40, 60, so interior windows must start with the real previous
    # frame, not an edge-padded copy of frame t.
    model = _RunnerFlowModel(context=(1, 0))
    _validate(tmp_path, model, imgsz=None)
    assert model.window_means == [[0, 0, 20], [0, 20, 40], [20, 40, 60]]


def test_sequences_do_not_share_context(tmp_path):
    # Two sequences: the second one starts edge padded with its own first
    # frame, never with the last frame of the first sequence.
    model = _RunnerFlowModel(context=(1, 0))
    _validate(tmp_path, model, imgsz=None, sequences=("a", "b"))
    assert model.window_means == [
        [0, 0, 20], [0, 20, 40], [20, 40, 60],
        [0, 0, 20], [0, 20, 40], [20, 40, 60],
    ]


def test_group_sequence_frames_orders_frames_per_directory(tmp_path):
    pairs = [
        (tmp_path / "a" / "f0.png", tmp_path / "a" / "f1.png"),
        (tmp_path / "a" / "f1.png", tmp_path / "a" / "f2.png"),
        (tmp_path / "b" / "f0.png", tmp_path / "b" / "f1.png"),
    ]
    frames = group_sequence_frames(pairs)
    assert list(frames) == [str(tmp_path / "a"), str(tmp_path / "b")]
    assert [Path(p).name for p in frames[str(tmp_path / "a")]] == ["f0.png", "f1.png", "f2.png"]
    assert [Path(p).name for p in frames[str(tmp_path / "b")]] == ["f0.png", "f1.png"]


def test_explicit_imgsz_reaches_predict(tmp_path):
    model = _StubFlowModel()
    _validate(tmp_path, model, imgsz=512)
    assert model.kwargs[0]["imgsz"] == 512


def test_default_imgsz_is_not_forwarded(tmp_path):
    # val() leaves imgsz None unless the user sets it; predict then applies
    # the family default, so nothing is accepted and ignored.
    model = _StubFlowModel()
    _validate(tmp_path, model, imgsz=None)
    assert "imgsz" not in model.kwargs[0]


def test_runner_family_receives_explicit_imgsz(tmp_path):
    model = _RunnerFlowModel()
    _validate(tmp_path, model, imgsz=64)
    assert model.input_sizes and all(size == 64 for size in model.input_sizes)


def test_requires_data_yaml(tmp_path):
    # data_dir satisfies the generic config, but flow needs the dataset YAML.
    config = ValidationConfig(data=None, data_dir=str(tmp_path), imgsz=32, device="cpu", verbose=False)
    with pytest.raises(ValueError, match="data="):
        FlowValidator(model=_StubFlowModel(), config=config)()
