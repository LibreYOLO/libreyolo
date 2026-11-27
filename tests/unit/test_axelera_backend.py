"""AxeleraBackend with a fake Axelera SDK: routing, output mapping, parity, errors."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from libreyolo.backends import axelera as axrt
from libreyolo.export import axelera as ax


@pytest.fixture(autouse=True)
def _libreyolo_logs_reach_caplog(monkeypatch):
    """libreyolo's logging setup turns propagation off on the "libreyolo" logger
    (utils/logging.py); turn it back on so caplog sees the records whatever ran
    before this test."""
    monkeypatch.setattr(logging.getLogger("libreyolo"), "propagate", True)


pytestmark = pytest.mark.unit

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


class _Field:
    def __init__(self, alias=None):
        self.alias = alias


class _FakeConfig:
    model_fields = {  # noqa: RUF012 - mirrors the SDK's pydantic class attribute
        name: _Field()
        for name in (
            "quantization_scheme", "resources_used", "aipu_cores_used",
            "output_axm_format", "save_error_artifact", "model_name", "output_dir",
        )
    }

    def __init__(self, **settings):
        self.__dict__.update(settings)

    def to_toml(self):
        return "".join(f"{k} = {v!r}\n" for k, v in sorted(self.__dict__.items()))


class _CompiledModel:
    """Plays the device: runs the exported FP32 ONNX, returns each pyramid
    level's outputs in reverse order as the real compiler does."""

    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        names = [o.name for o in self.session.get_outputs()]
        per_level = len(names) // 3
        if len(names) % 3:  # not three pyramid levels: return them all, reversed
            self.order = list(reversed(range(len(names))))
        else:
            self.order = [
                i for level in range(3)
                for i in reversed(range(level * per_level, (level + 1) * per_level))
            ] or list(range(len(names)))  # classifiers: one output
        self.calls = 0

    def run(self, array):
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 3:
            array = array[None]
        self.calls += 1
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: array})
        return [outputs[i] for i in self.order]

    def __call__(self, tensor):  # quantized-model interface (torch tensor in)
        return self.run(tensor.numpy())


@pytest.fixture(scope="module")
def sdk():
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield _install_fake_sdk(monkeypatch)


def _install_fake_sdk(monkeypatch):
    compiled: dict[str, _CompiledModel] = {}
    state = {"load_error": None}

    compiler = types.ModuleType("axelera.compiler")
    compiler.CompilerConfig = _FakeConfig

    def quantize(*, model, config, calibration_dataset, transform_fn=None):
        list(calibration_dataset)
        return _CompiledModel(model)

    def compile(*, model, config, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        axm = Path(output_dir).parent / f"{config.model_name}.axm"
        axm.write_bytes(b"axm")
        compiled[axm.name] = model

    compiler.quantize, compiler.compile = quantize, compile

    runtime = types.ModuleType("axelera.runtime")
    for name in ("DeviceInUse", "IncompatibleDevice"):
        setattr(runtime, name, type(name, (RuntimeError,), {}))
    runtime.ConnectionError = type("ConnectionError", (OSError,), {})

    class _Loaded:
        """Stands in for the runtime's loaded model: call one image, or batch a list."""

        def __init__(self, model):
            self.model = model
            self.batches = []

        def __call__(self, array):
            return self.model.run(array)

        def batch(self, arrays):
            self.batches.append(len(arrays))
            return [self.model.run(array) for array in arrays]

    def load(path, **kwargs):
        if state["load_error"] is not None:
            raise state["load_error"]
        return _Loaded(compiled[Path(path).name])

    runtime.op = types.SimpleNamespace(load=load)
    package = types.ModuleType("axelera")
    package.compiler, package.runtime = compiler, runtime
    for name, module in (
        ("axelera", package), ("axelera.compiler", compiler),
        ("axelera.runtime", runtime), ("axelera.runtime.op", runtime.op),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(axrt, "_module_available", lambda module: True)
    monkeypatch.setattr(axrt, "_installed_version", lambda package: axrt.AXELERA_SDK_VERSION)
    monkeypatch.setattr(ax, "check_compile_platform", lambda: None)
    return types.SimpleNamespace(compiled=compiled, state=state, runtime=runtime)


def _dataset(tmp_path, n=3):
    from PIL import Image

    images = tmp_path / "data" / "images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        Image.fromarray(rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)).save(images / f"{i}.jpg")
    path = tmp_path / "data" / "data.yaml"
    path.write_text(f"path: {tmp_path / 'data'}\ntrain: images\nval: images\nnc: 3\nnames: [a, b, c]\n")
    return path


def _well_conditioned_yolo9():
    """A YOLO9-t whose predictions on ``_image(shape=(96, 96, 3))`` can be
    compared after NMS. At default init the head's input is ~1e-6, so every
    anchor gets the same score and NMS order is decided by ties. BatchNorm
    statistics are set in eval mode, layer by layer in forward order, on that
    image plus three random ones; then each level's last convs are scaled by
    their measured input std: class logits about -3 +- 1.5 (seven candidates above conf=0.5, some overlapping), box bins peaked
    near 1.5 strides, all inside YOLO9_RAW_OUTPUT_WINDOWS and the class
    activation bound. The parity test checks the margins before comparing."""
    import torch

    from libreyolo import LibreYOLO9

    torch.manual_seed(0)
    model = LibreYOLO9(None, size="t", nb_classes=3, device="cpu")
    net = model.model.eval()
    blob = model._preprocess(_image(shape=(96, 96, 3)), input_size=96)[0]
    calib = torch.cat([blob, torch.rand(3, 3, 96, 96, generator=torch.Generator().manual_seed(3))])

    def first_inputs(modules, x):
        seen: dict = {}
        hooks = [m.register_forward_hook(lambda m, i, o: seen.setdefault(m, i[0])) for m in modules]
        with torch.no_grad():
            net(x)
        for hook in hooks:
            hook.remove()
        return seen

    norms = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for norm in list(first_inputs(norms, calib)):  # dict order = forward order
        x = first_inputs([norm], calib)[norm]
        with torch.no_grad():
            norm.running_mean.copy_(x.mean(dim=(0, 2, 3)))
            # The floor keeps near-dead channels from amplifying float rounding.
            norm.running_var.copy_(x.var(dim=(0, 2, 3)) + 1e-1)

    head = net.head
    lasts = [t[-1] for t in head.class_convs] + [t[-1] for t in head.anchor_convs]
    input_std = {m: float(x.std()) for m, x in first_inputs(lasts, blob).items()}
    generator = torch.Generator().manual_seed(24)
    bins = torch.arange(16, dtype=torch.float32)
    with torch.no_grad():
        for tower in head.class_convs:
            conv = tower[-1]
            scale = 1.5 / (input_std[conv] * conv.in_channels**0.5)
            conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator) * scale)
            conv.bias.fill_(-3.0)
        for tower in head.anchor_convs:
            conv = tower[-1]
            scale = 0.5 / (input_std[conv] * conv.in_channels**0.5)
            conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator) * scale)
            conv.bias.copy_(-0.7 * (bins - 1.5).abs().repeat(4))
    return model


def _export(tmp_path, task="detect"):
    """``task`` is a YOLO-NAS task, ``"yolo9"`` for a YOLO9-t detector,
    ``"yolox"`` for a YOLOX-n detector or ``"yolo4"`` for a YOLOv4-tiny detector."""
    import torch

    from libreyolo import LibreYOLO4, LibreYOLO9, LibreYOLONAS, LibreYOLOX

    torch.manual_seed(0)
    if task == "yolo9":
        model = _well_conditioned_yolo9()
    elif task == "yolo4":
        model = LibreYOLO4(None, size="t", nb_classes=3, device="cpu")
        generator = torch.Generator().manual_seed(1)
        with torch.no_grad():  # spread scores so boxes pass conf without ties
            for spec in model.model.net.detection_specs():
                conv = model.model.net.layers[spec.layer_index - 1].conv
                conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator) * 0.05)
                conv.bias.copy_(torch.randn(conv.bias.shape, generator=generator) + 1.0)
    elif task == "yolox":
        model = LibreYOLOX(None, size="n", nb_classes=3, device="cpu")
        generator = torch.Generator().manual_seed(1)
        with torch.no_grad():  # spread scores so boxes pass conf without ties
            for conv in (*model.model.head.obj_preds, *model.model.head.cls_preds):
                conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator) * 0.05)
                conv.bias.copy_(torch.linspace(1.0, -1.0, conv.bias.numel()))
    else:
        size, nc = {"detect": ("s", 3), "pose": ("n", 1)}[task]
        model = LibreYOLONAS(None, size=size, nb_classes=nc, task=task, device="cpu")
        with torch.no_grad():  # sharper class logits so a few detections pass conf
            for head in (model.model.heads.head1, model.model.heads.head2, model.model.heads.head3):
                head.cls_pred.bias.add_(4.0)
    from unittest import mock

    # The pose input bound changes FP32 keypoints by design (tested in
    # test_yolonas_raw_heads); without it these stay strict decode tests.
    with mock.patch("libreyolo.export.exporter._bound_yolonas_pose_inputs", lambda *a, **k: {}):
        out = model.export(
            format="axelera", data=str(_dataset(tmp_path)), imgsz=96,
            output_path=str(tmp_path / f"{task}_axelera_model"),
        )
    return model, Path(out)


@pytest.fixture(scope="module")
def exports(sdk, tmp_path_factory):
    cache = {}

    def get(task="detect"):
        if task not in cache:
            cache[task] = _export(tmp_path_factory.mktemp(task), task)
        return cache[task]

    return get


def _copy(artifact, tmp_path):
    import shutil

    return Path(shutil.copytree(artifact, tmp_path / artifact.name))


def _image(seed=1, shape=(72, 96, 3)):
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


@pytest.mark.parametrize("task", ["detect", "pose"])
def test_backend_matches_the_exported_graph(exports, tmp_path, task):
    """Decoded tensors equal the PyTorch decoded graph; predictions equal the
    ONNX backend's, which shares the backend postprocessing."""
    import torch

    from libreyolo import LibreYOLO
    from libreyolo.export.exporter import _YOLONASExportWrapper

    model, artifact = exports(task)
    backend = LibreYOLO(str(artifact))
    assert type(backend).__name__ == "AxeleraBackend"
    assert backend.task == task and backend.model_family == "yolonas"

    image = _image()
    blob = model._preprocess(image, input_size=96)[0]
    blob = blob.numpy() if hasattr(blob, "numpy") else np.asarray(blob)
    with torch.no_grad():
        reference = _YOLONASExportWrapper(model.model.eval())(torch.from_numpy(blob))
    decoded = backend._run_inference(blob)
    assert len(decoded) == len(reference)
    for actual, expected in zip(decoded, reference):
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-6, atol=1e-4)

    onnx_backend = LibreYOLO(
        model.export(format="onnx", imgsz=96, dynamic=False, output_path=str(tmp_path / "ref.onnx"))
    )
    expected = onnx_backend(image, conf=0.3)
    actual = backend(image, conf=0.3)
    assert len(expected) > 0
    assert len(actual) == len(expected)
    np.testing.assert_allclose(actual.boxes.xyxy, expected.boxes.xyxy, atol=1e-3)
    np.testing.assert_allclose(actual.boxes.conf, expected.boxes.conf, atol=1e-5)
    np.testing.assert_array_equal(actual.boxes.cls, expected.boxes.cls)
    if task == "pose":
        np.testing.assert_allclose(actual.keypoints.data, expected.keypoints.data, atol=1e-3)


def test_yolo9_backend_matches_the_exported_graph(exports, tmp_path):
    """YOLO9 raw heads decode on the host to the decoded graph's tensor, and
    predictions equal the ONNX backend's."""
    import torch

    from libreyolo import LibreYOLO

    model, artifact = exports("yolo9")
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["output_layout"] == "yolo9_raw_heads_v1" and meta["reg_max"] == 16
    backend = LibreYOLO(str(artifact))
    assert type(backend).__name__ == "AxeleraBackend"
    assert backend.model_family == "yolo9" and backend.task == "detect"

    image = _image(shape=(96, 96, 3))  # square: no letterbox padding to clamp into
    blob = model._preprocess(image, input_size=96)[0]
    blob = blob.numpy() if hasattr(blob, "numpy") else np.asarray(blob)
    with torch.no_grad():
        reference = model.model.eval()(torch.from_numpy(blob))["predictions"].numpy()
    (decoded,) = backend._run_inference(blob)
    assert decoded.shape == reference.shape  # (1, 4 + nc, N)
    # Export fuses conv+BN: float rounding through the whole net. Boxes are bin
    # expectations times the stride (up to 32 px), so 0.1 px is 0.003 of a bin;
    # measured up to 0.03 px across CI platforms.
    np.testing.assert_allclose(decoded[:, :4], reference[:, :4], atol=1e-1)
    # Scores: this synthetic net amplifies FP32 summation order; PyTorch FP32 alone
    # is 6e-3 from a float64 run, and ONNX Runtime vs PyTorch measured up to 4e-3
    # (it moves with thread count and CPU). Scores sit >= 2e-2 from conf.
    np.testing.assert_allclose(decoded[:, 4:], reference[:, 4:], atol=1e-2)

    # Predictions can only be compared if no score sits near conf, no same-class
    # pair near the NMS IoU and no overlapping pair on a score tie, with margins
    # well above what the tolerances above can move (a 0.1 px shift moves the
    # IoU of these 20+ px boxes by under 0.01).
    conf, iou = 0.5, 0.5
    boxes, scores = np.clip(reference[0, :4].T, 0, 96), reference[0, 4:].T
    assert np.abs(scores - conf).min() > 2e-2
    above = np.argwhere(scores > conf)
    assert 2 <= len(above) <= 40
    for cls in np.unique(above[:, 1]):
        rows = boxes[above[above[:, 1] == cls, 0]]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                assert abs(inter / union - iou) > 2e-2
                if inter > 0:  # overlapping boxes: NMS order must not hang on a tie
                    rows_idx = above[above[:, 1] == cls, 0]
                    assert abs(scores[rows_idx[i], cls] - scores[rows_idx[j], cls]) > 5e-3

    onnx_backend = LibreYOLO(
        model.export(format="onnx", imgsz=96, dynamic=False, output_path=str(tmp_path / "ref.onnx"))
    )
    expected = onnx_backend(image, conf=conf, iou=iou)
    actual = backend(image, conf=conf, iou=iou)
    assert len(expected) > 0 and len(actual) == len(expected)
    np.testing.assert_allclose(actual.boxes.xyxy, expected.boxes.xyxy, atol=1e-3)
    np.testing.assert_allclose(actual.boxes.conf, expected.boxes.conf, atol=1e-5)
    np.testing.assert_array_equal(actual.boxes.cls, expected.boxes.cls)


def test_classifier_backend_matches_the_onnx_backend(sdk, tmp_path):
    import torch

    from libreyolo import LibreResNet, LibreYOLO

    torch.manual_seed(0)
    model = LibreResNet(size="18", nb_classes=5, device="cpu")
    artifact = model.export(
        format="axelera", data=str(_dataset(tmp_path)), imgsz=64,
        output_path=str(tmp_path / "cls_axelera_model"),
    )
    backend = LibreYOLO(artifact)
    assert type(backend).__name__ == "AxeleraBackend" and backend.task == "classify"
    reference = LibreYOLO(model.export(format="onnx", imgsz=64, dynamic=False, output_path=str(tmp_path / "cls.onnx")))
    images = [_image(1), _image(2), _image(3)]
    for single, expected in zip(backend.predict(images, batch=3), [reference(image) for image in images]):
        np.testing.assert_allclose(single.probs.data, expected.probs.data, atol=1e-5)
        assert int(single.probs.top1) == int(expected.probs.top1)


def test_yolox_backend_matches_the_exported_graph(exports, tmp_path):
    """YOLOX raw heads decode on the host to the decoded graph's tensor, and
    predictions equal the ONNX backend's."""
    import torch

    from libreyolo import LibreYOLO

    model, artifact = exports("yolox")
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["output_layout"] == "yolox_raw_heads_v1" and "reg_max" not in meta
    assert meta["grid_cell_offset"] == 0.0
    backend = LibreYOLO(str(artifact))
    assert type(backend).__name__ == "AxeleraBackend"
    assert backend.model_family == "yolox" and backend.task == "detect"

    image = _image()
    blob = model._preprocess(image, input_size=96)[0]
    blob = blob.numpy() if hasattr(blob, "numpy") else np.asarray(blob)
    net = model.model.eval()
    net.head.export = True
    try:
        with torch.no_grad():
            reference = net(torch.from_numpy(blob)).numpy()
    finally:
        net.head.export = False
    (decoded,) = backend._run_inference(blob)
    assert decoded.shape == reference.shape  # (1, N, 5 + nc)
    np.testing.assert_allclose(decoded[..., :4], reference[..., :4], rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(decoded[..., 4:], reference[..., 4:], atol=1e-5)

    onnx_backend = LibreYOLO(
        model.export(format="onnx", imgsz=96, dynamic=False, output_path=str(tmp_path / "ref.onnx"))
    )
    expected = onnx_backend(image, conf=0.3)
    actual = backend(image, conf=0.3)
    assert len(expected) > 0 and len(actual) == len(expected)

    def rows(result):  # a random head scores many boxes alike: compare as sets
        table = np.column_stack(
            [np.asarray(result.boxes.xyxy), np.asarray(result.boxes.conf), np.asarray(result.boxes.cls)]
        )
        return table[np.lexsort(np.round(table, 2).T[::-1])]

    np.testing.assert_allclose(rows(actual), rows(expected), atol=1e-3)


def test_yolo4_backend_matches_the_exported_graph(exports, tmp_path):
    """YOLOv4 raw heads decode on the host to the decoded graph's tensor, and
    predictions equal the ONNX backend's."""
    import torch

    from libreyolo import LibreYOLO
    from libreyolo.models.darknet.export import DarknetExportWrapper

    model, artifact = exports("yolo4")
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["output_layout"] == "darknet_raw_heads_v1" and len(meta["anchors"]) == 2
    backend = LibreYOLO(str(artifact))
    assert type(backend).__name__ == "AxeleraBackend"
    assert backend.model_family == "yolo4" and backend.task == "detect"

    image = _image()
    blob = model._preprocess(image, input_size=96)[0]
    blob = blob.numpy() if hasattr(blob, "numpy") else np.asarray(blob)
    with torch.no_grad():
        reference = DarknetExportWrapper(model.model.eval())(torch.from_numpy(blob)).numpy()
    (decoded,) = backend._run_inference(blob)
    assert decoded.shape == reference.shape  # (1, 4 + nc, N)
    np.testing.assert_allclose(decoded[:, :4], reference[:, :4], rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(decoded[:, 4:], reference[:, 4:], atol=1e-5)

    onnx_backend = LibreYOLO(
        model.export(format="onnx", imgsz=96, dynamic=False, output_path=str(tmp_path / "ref.onnx"))
    )
    expected = onnx_backend(image, conf=0.3)
    actual = backend(image, conf=0.3)
    assert len(expected) > 0 and len(actual) == len(expected)
    np.testing.assert_allclose(actual.boxes.xyxy, expected.boxes.xyxy, atol=1e-3)
    np.testing.assert_allclose(actual.boxes.conf, expected.boxes.conf, atol=1e-5)
    np.testing.assert_array_equal(actual.boxes.cls, expected.boxes.cls)


def test_axm_path_and_batched_predict(exports):
    from libreyolo import LibreYOLO

    _, artifact = exports()
    backend = LibreYOLO(str(artifact / "detect.axm"))
    assert type(backend).__name__ == "AxeleraBackend"
    images = [_image(1), _image(2), _image(3)]
    batched = backend.predict(images, conf=0.3, batch=3)
    assert backend._model.batches == [3]  # one batched submission to the runtime
    single = [backend(image, conf=0.3) for image in images]
    for a, b in zip(batched, single):
        np.testing.assert_allclose(a.boxes.xyxy, b.boxes.xyxy)
        np.testing.assert_array_equal(a.boxes.cls, b.boxes.cls)


def test_outputs_are_mapped_by_name_not_position(exports):
    from libreyolo.backends.axelera import AxeleraBackend

    _, artifact = exports()
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["output_names"][:2] == ["p3_cls", "p3_reg"]
    backend = AxeleraBackend(artifact)
    raw = backend._invoke(np.zeros((1, 3, 96, 96), np.float32))
    assert [a.shape[1] for a in raw[:2]] == [68, 3]  # layout order: reg, cls


def test_wrong_input_size_names_the_exported_size(exports):
    from libreyolo.backends.axelera import AxeleraBackend

    _, artifact = exports()
    with pytest.raises(ValueError, match="fixed 96x96 input"):
        AxeleraBackend(artifact)._run_inference(np.zeros((1, 3, 64, 64), np.float32))


def test_output_count_mismatch_asks_for_recompile(exports):
    from libreyolo.backends.axelera import AxeleraBackend

    _, artifact = exports()
    backend = AxeleraBackend(artifact)
    backend._model = lambda array: [np.zeros((1, 68, 12, 12), np.float32)]  # no .batch
    with pytest.raises(ValueError, match="recompiling"):
        backend._run_inference(np.zeros((1, 3, 96, 96), np.float32))


def test_foreign_and_incomplete_directories_are_rejected(exports, tmp_path):
    from libreyolo.backends.axelera import AxeleraBackend

    _, artifact = exports()
    foreign = tmp_path / "foreign_model"
    foreign.mkdir()
    (foreign / "m.axm").write_bytes(b"axm")
    (foreign / "metadata.yaml").write_text("task: detect\nnames: {0: a}\n")
    with pytest.raises(ValueError, match="only loads Axelera models it exported"):
        AxeleraBackend(foreign)

    lone = tmp_path / "lone"
    lone.mkdir()
    (lone / "m.axm").write_bytes(b"axm")
    with pytest.raises(ValueError, match="no metadata.yaml"):
        AxeleraBackend(lone / "m.axm")

    artifact = _copy(artifact, tmp_path)
    (artifact / "second.axm").write_bytes(b"axm")
    with pytest.raises(ValueError, match="more than one .axm"):
        AxeleraBackend(artifact)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No .axm"):
        AxeleraBackend(empty)


def test_newer_output_layout_asks_for_upgrade(exports, tmp_path):
    from libreyolo.backends.axelera import AxeleraBackend

    _, artifact = exports()
    artifact = _copy(artifact, tmp_path)
    meta_path = artifact / "metadata.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    meta["output_layout"] = "yolonas_raw_heads_v9"
    meta_path.write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError, match="Upgrade LibreYOLO"):
        AxeleraBackend(artifact)


def test_pose_windows_are_recorded_and_validated(exports, tmp_path):
    from libreyolo.backends.axelera import AxeleraBackend
    from libreyolo.postprocess.yolonas_raw import YOLONAS_RAW_OUTPUT_WINDOWS_BY_TASK

    _, artifact = exports("pose")
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    windows = YOLONAS_RAW_OUTPUT_WINDOWS_BY_TASK["pose"]
    assert meta["output_windows"] == {part: list(bounds) for part, bounds in windows.items()}
    assert AxeleraBackend(artifact)._decode_params["output_windows"] == windows

    artifact = _copy(artifact, tmp_path)
    meta["output_windows"] = {"rot": [-1.0, 1.0]}
    (artifact / "metadata.yaml").write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError, match="Re-export the model"):
        AxeleraBackend(artifact)


def test_missing_runtime_names_the_install(exports, monkeypatch):
    from libreyolo.backends.axelera import AxeleraBackend

    _, artifact = exports()
    monkeypatch.delenv(axrt.AUTOINSTALL_ENV, raising=False)
    monkeypatch.setattr(axrt, "_module_available", lambda module: False)
    with pytest.raises(ImportError, match="axelera-rt==1.9.0"):
        AxeleraBackend(artifact)


def test_busy_or_missing_device_points_at_axdevice(sdk, exports):
    from libreyolo.backends.axelera import AxeleraBackend

    _, artifact = exports()
    sdk.state["load_error"] = sdk.runtime.DeviceInUse("metis-0:3:0 is in use")
    try:
        with pytest.raises(RuntimeError, match="axdevice"):
            AxeleraBackend(artifact)
    finally:
        sdk.state["load_error"] = None


def test_device_argument(exports, caplog):
    from libreyolo.backends.axelera import AxeleraBackend

    _, artifact = exports()
    for device in ("auto", "axelera", "metis", "cpu"):
        assert AxeleraBackend(artifact, device=device).device == "axelera"
    with caplog.at_level(logging.WARNING, logger=axrt.__name__):
        AxeleraBackend(artifact, device="cuda:0")
    assert "ignored" in caplog.text
    with pytest.raises(ValueError, match="Unsupported device"):
        AxeleraBackend(artifact, device="tpu")


# ---------------------------------------------------------------------------
# Whole-graph dense exports (layout graph_outputs_v1)
# ---------------------------------------------------------------------------


@pytest.fixture
def dense_allowed(monkeypatch):
    """Let a dense family through the support gate whatever its committed tier."""
    import libreyolo.export.exporter as exporter
    from libreyolo.export.support import INT8_EXPORTS, SupportEntry

    original = exporter.get_support

    def gate(family, task, fmt):
        if fmt == "axelera":
            return SupportEntry("available", "unit test")
        return original(family, task, fmt)

    monkeypatch.setattr(exporter, "get_support", gate)
    pairs = {("fcn", "semantic"), ("realesrgan", "restore"), ("ssd", "detect")}
    monkeypatch.setitem(INT8_EXPORTS, "axelera", INT8_EXPORTS["axelera"] | pairs)


def _dense_model(family="fcn"):
    import torch

    from libreyolo import LibreFCN

    torch.manual_seed(0)
    return LibreFCN(None, size="r50", nb_classes=21, device="cpu"), 72


def test_dense_backend_matches_the_onnx_backend(sdk, dense_allowed, tmp_path, family="fcn"):
    """graph_outputs_v1 hands the ONNX graph's tensors to the shared parsers."""
    import yaml

    from libreyolo import LibreYOLO

    model, imgsz = _dense_model(family)
    artifact = Path(model.export(
        format="axelera", data=str(_dataset(tmp_path)), imgsz=imgsz,
        output_path=str(tmp_path / f"{family}_axelera_model"),
    ))
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["output_layout"] == "graph_outputs_v1"
    assert meta["graph_outputs"] == meta["output_names"] and len(meta["graph_outputs"]) == 1
    backend = LibreYOLO(str(artifact))
    assert type(backend).__name__ == "AxeleraBackend" and backend.task == model.task
    reference = LibreYOLO(model.export(
        format="onnx", imgsz=imgsz, dynamic=False, output_path=str(tmp_path / f"{family}.onnx"),
    ))
    image = _image(4)
    actual, expected = backend(image)[0], reference(image)[0]
    assert np.mean(np.asarray(actual.semantic_mask.data) == np.asarray(expected.semantic_mask.data)) > 0.999


def test_graph_outputs_metadata_must_list_the_outputs(sdk, dense_allowed, tmp_path):
    import yaml

    from libreyolo import LibreYOLO

    model, imgsz = _dense_model()
    artifact = Path(model.export(
        format="axelera", data=str(_dataset(tmp_path)), imgsz=imgsz,
        output_path=str(tmp_path / "fcn2_axelera_model"),
    ))
    meta_path = artifact / "metadata.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    meta.pop("graph_outputs")
    meta_path.write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError, match="graph_outputs"):
        LibreYOLO(str(artifact))


def test_quicksrnet_backend_matches_the_onnx_backend(sdk, tmp_path):
    """Whole-graph restore: the device returns the restored image. The export
    equalizes QuickSRNet's channels and corrects its biases for INT8 weights,
    so outputs move by a few grey levels; smaller images pad into the canvas."""
    import torch
    import yaml

    from libreyolo import LibreQuickSRNet, LibreYOLO

    torch.manual_seed(0)
    model, imgsz = LibreQuickSRNet(None, size="m2", device="cpu"), 32
    artifact = Path(model.export(
        format="axelera", data=str(_dataset(tmp_path)), imgsz=imgsz,
        output_path=str(tmp_path / "sr_axelera_model"),
    ))
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["output_layout"] == "graph_outputs_v1" and meta["graph_outputs"] == ["restored"]
    backend = LibreYOLO(str(artifact))
    reference = LibreYOLO(model.export(
        format="onnx", imgsz=imgsz, dynamic=False, output_path=str(tmp_path / "sr.onnx"),
    ))
    image = _image(4, shape=(imgsz, imgsz, 3))
    actual, expected = backend(image)[0], reference(image)[0]
    diff = np.abs(
        np.asarray(actual.restored.data, dtype=np.int16) - np.asarray(expected.restored.data, dtype=np.int16)
    )
    assert diff.max() <= 8
    smaller = backend(_image(5, shape=(imgsz - 8, imgsz - 4, 3)))[0]
    assert np.asarray(smaller.restored.data).shape[:2] == (2 * (imgsz - 8), 2 * (imgsz - 4))
    with pytest.raises(ValueError, match="fixed-resolution"):
        backend(_image(5, shape=(imgsz + 8, imgsz, 3)))


def test_midas_backend_matches_the_onnx_backend(sdk, tmp_path):
    """Whole-graph depth: the device returns the ONNX graph's inverse depth;
    calibration stretches images to the canvas like the backend."""
    pytest.importorskip("timm")  # MiDaS-s backbone (libreyolo[midas])
    import torch
    import yaml

    from libreyolo import LibreMiDaS, LibreYOLO
    from libreyolo.backends.base import BaseBackend
    from libreyolo.export.exporter import AxeleraExporter

    torch.manual_seed(0)
    model, imgsz = LibreMiDaS(None, size="s", device="cpu"), 64
    rgb = _image(6, shape=(60, 100, 3))
    chw, _ = AxeleraExporter(model)._calibration_preprocess()(rgb, (imgsz, imgsz))
    expected_chw = BaseBackend._preprocess_depth(rgb, (imgsz, imgsz), "rgb", cubic=True)[0][0]
    torch.testing.assert_close(torch.from_numpy(chw), expected_chw)
    artifact = Path(model.export(
        format="axelera", data=str(_dataset(tmp_path)), imgsz=imgsz,
        output_path=str(tmp_path / "midas_axelera_model"),
    ))
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["output_layout"] == "graph_outputs_v1" and meta["graph_outputs"] == ["depth"]
    backend = LibreYOLO(str(artifact))
    reference = LibreYOLO(model.export(
        format="onnx", imgsz=imgsz, dynamic=False, output_path=str(tmp_path / "midas.onnx"),
    ))
    image = _image(4)
    np.testing.assert_allclose(
        backend(image)[0].depth_map.data, reference(image)[0].depth_map.data, rtol=1e-4, atol=1e-4
    )


def test_realesrgan_trunk_runs_on_device_and_tail_on_host(sdk, dense_allowed, tmp_path):
    """restore_shuffle_skip_v1: the host pixel shuffle + input skip reproduce the full graph."""
    import torch
    import yaml

    from libreyolo import LibreRealESRGAN, LibreYOLO
    from libreyolo.backends.axelera import shuffle_and_skip

    torch.manual_seed(0)
    model = LibreRealESRGAN(None, size="x4t", device="cpu")
    artifact = Path(model.export(
        format="axelera", data=str(_dataset(tmp_path)), imgsz=32,
        output_path=str(tmp_path / "esr_axelera_model"),
    ))
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["output_layout"] == "restore_shuffle_skip_v1" and meta["upscale"] == 4
    backend = LibreYOLO(str(artifact))
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        expected = model.model(x).numpy()
    actual = backend._run_inference(x.numpy())[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    body = np.random.default_rng(0).standard_normal((1, 48, 4, 5)).astype(np.float32)
    image = np.random.default_rng(1).standard_normal((1, 3, 4, 5)).astype(np.float32)
    reference = torch.nn.functional.pixel_shuffle(torch.from_numpy(body), 4) + torch.nn.functional.interpolate(
        torch.from_numpy(image), scale_factor=4, mode="nearest")
    np.testing.assert_allclose(shuffle_and_skip(body, image, 4), reference.numpy(), atol=1e-6)


@pytest.fixture(scope="module")
def ssd_export(sdk, tmp_path_factory):
    """One two-part SSD300 export (random weights) shared by the chain tests."""
    import torch

    from libreyolo import LibreSSD
    from libreyolo.export import exporter
    from libreyolo.export.support import INT8_EXPORTS, SupportEntry

    tmp_path = tmp_path_factory.mktemp("ssd")
    with pytest.MonkeyPatch.context() as monkeypatch:
        original = exporter.get_support
        monkeypatch.setattr(
            exporter, "get_support",
            lambda f, t, fmt: SupportEntry("available", "unit test") if fmt == "axelera" else original(f, t, fmt),
        )
        monkeypatch.setitem(INT8_EXPORTS, "axelera", INT8_EXPORTS["axelera"] | {("ssd", "detect")})
        torch.manual_seed(0)
        model = LibreSSD(None, device="cpu")
        artifact = Path(model.export(
            format="axelera", data=str(_dataset(tmp_path)), output_path=str(tmp_path / "ssd_axelera_model"),
        ))
    return model, artifact


def test_ssd_export_leaves_no_intermediate_graphs(ssd_export):
    _, artifact = ssd_export
    assert not list(artifact.parent.glob("*.export_intermediate*"))


def test_ssd_export_cleans_up_when_part2_fails(sdk, tmp_path, monkeypatch):
    """Part 1 is on disk when part 2's export fails; both intermediates go."""
    import torch

    from libreyolo import LibreSSD
    from libreyolo.export import exporter, onnx
    from libreyolo.export.support import INT8_EXPORTS, SupportEntry

    original = exporter.get_support
    monkeypatch.setattr(
        exporter, "get_support",
        lambda f, t, fmt: SupportEntry("available", "unit test") if fmt == "axelera" else original(f, t, fmt),
    )
    monkeypatch.setitem(INT8_EXPORTS, "axelera", INT8_EXPORTS["axelera"] | {("ssd", "detect")})
    real_export = onnx._export_onnx_graph

    def failing_part2(*args, output_path, **kwargs):
        if output_path.endswith(".part2.onnx"):
            Path(output_path).write_bytes(b"partial")
            raise RuntimeError("part 2 failed")
        return real_export(*args, output_path=output_path, **kwargs)

    monkeypatch.setattr(onnx, "_export_onnx_graph", failing_part2)
    torch.manual_seed(0)
    model = LibreSSD(None, device="cpu")
    with pytest.raises(RuntimeError, match="part 2 failed"):
        model.export(format="axelera", data=str(_dataset(tmp_path)), output_path=str(tmp_path / "ssd_axelera_model"))
    assert not list(tmp_path.glob("*.export_intermediate*"))


def test_ssd_exports_two_parts_with_the_l2norm_on_the_host(ssd_export):
    import yaml

    _, artifact = ssd_export
    assert sorted(p.name for p in artifact.glob("*.axm")) == ["ssd.axm", "ssd_part2.axm"]
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["output_layout"] == "ssd_raw_heads_v1" and meta["axm"] == "ssd.axm"
    first, host, second = meta["parts"]
    assert first["axm"] == "ssd.axm" and first["input"] == "images"
    assert sorted(first["outputs"]) == sorted(["conv4_3", *meta["output_names"][2:]])
    assert host["host"] == "l2norm" and host["input"] == "conv4_3" and host["output"] == "conv4_3_norm"
    assert len(host["scale"]) == 512
    assert second == {"axm": "ssd_part2.axm", "input": "conv4_3_norm", "outputs": second["outputs"]}
    assert sorted(second["outputs"]) == sorted(meta["output_names"][:2])
    assert len(meta["class_index_map"]) == 80


def test_ssd_chain_matches_the_in_graph_decode(ssd_export):
    import torch

    from libreyolo import LibreYOLO
    from libreyolo.models.ssd.nn import SSDExportWrapper

    model, artifact = ssd_export
    backend = LibreYOLO(str(artifact))
    assert backend._chain is not None and len([m for m in backend._chain_models if m is not None]) == 2
    image = np.random.default_rng(3).random((1, 3, 300, 300), dtype=np.float32)
    (decoded,) = backend._run_inference(image)
    with torch.no_grad():
        expected = SSDExportWrapper(model.model.eval()).eval()(torch.from_numpy(image)).numpy()
    assert decoded.shape == expected.shape == (1, 84, 8732)
    np.testing.assert_allclose(decoded[:, :4], expected[:, :4], atol=1e-2)
    np.testing.assert_allclose(decoded[:, 4:], expected[:, 4:], atol=1e-5)


def test_ssd_chain_batches_every_part(ssd_export):
    from libreyolo import LibreYOLO

    _, artifact = ssd_export
    backend = LibreYOLO(str(artifact))
    rng = np.random.default_rng(4)
    images = [rng.random((1, 3, 300, 300), dtype=np.float32) for _ in range(3)]
    assert backend._supports_batched_inference()
    batched = backend._invoke_batch(images)
    single = [backend._invoke(image) for image in images]
    assert [m.batches for m in backend._chain_models if m is not None] == [[3], [3]]
    for got, want in zip(batched, single):
        for a, b in zip(got, want):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    ("edit", "error"),
    [
        (lambda parts: parts[1].update(host="softplus"), "host operation 'softplus'"),
        (lambda parts: parts[2].update(input="nowhere"), "reads 'nowhere'"),
        (lambda parts: parts[2].update(outputs=parts[2]["outputs"][:1]), "do not produce"),
        (lambda parts: parts.insert(0, {"host": "l2norm", "input": "images", "output": "x", "scale": [1.0]}),
         "must be a compiled model"),
    ],
)
def test_ssd_chain_metadata_is_validated(ssd_export, tmp_path, edit, error):
    import shutil

    import yaml

    from libreyolo import LibreYOLO

    _, artifact = ssd_export
    copy_dir = tmp_path / "ssd_axelera_model"
    shutil.copytree(artifact, copy_dir)
    meta = yaml.safe_load((copy_dir / "metadata.yaml").read_text())
    edit(meta["parts"])
    (copy_dir / "metadata.yaml").write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError, match=error):
        LibreYOLO(str(copy_dir))


def test_ssd_chain_needs_every_part_file(ssd_export, tmp_path):
    import shutil

    from libreyolo import LibreYOLO

    _, artifact = ssd_export
    copy_dir = tmp_path / "ssd_axelera_model"
    shutil.copytree(artifact, copy_dir)
    (copy_dir / "ssd_part2.axm").unlink()
    with pytest.raises(FileNotFoundError, match="ssd_part2.axm"):
        LibreYOLO(str(copy_dir))


def test_ssd_part2_input_feeds_one_op(tmp_path):
    """A graph input that fans out straight into two convolutions is cut into two AIPU functions."""
    import onnx
    import torch

    from libreyolo import LibreSSD
    from libreyolo.export.exporter import _SSDPartBWrapper

    model = LibreSSD(None, device="cpu").model.eval()
    part = _SSDPartBWrapper(model, ["p3_reg", "p3_cls"]).eval()
    x = torch.rand(1, 512, 38, 38)
    with torch.no_grad():
        reg, cls = part(x)
        assert torch.equal(reg, model.head.regression_head.module_list[0](x))
        assert torch.equal(cls, model.head.classification_head.module_list[0](x))
    path = tmp_path / "part2.onnx"
    torch.onnx.export(part, x, str(path), input_names=["conv4_3_norm"], output_names=part.output_names, dynamo=False)
    graph = onnx.load(str(path)).graph
    consumers = [node for node in graph.node if "conv4_3_norm" in node.input]
    assert len(consumers) == 1 and consumers[0].op_type == "Conv"


def test_host_l2norm_matches_torch_normalize():
    import torch

    from libreyolo.backends.axelera import HOST_OPS

    rng = np.random.default_rng(5)
    x = rng.standard_normal((1, 8, 5, 6)).astype(np.float32)
    x[0, :, 0, 0] = 0.0  # an all-zero pixel stays zero
    scale = rng.random(8).astype(np.float32) * 20
    expected = scale.reshape(1, -1, 1, 1) * torch.nn.functional.normalize(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(HOST_OPS["l2norm"](x, scale=scale.tolist()), expected, rtol=1e-6, atol=1e-6)


def test_single_axm_artifacts_take_the_one_model_path(exports, tmp_path):
    """Artifacts without ``parts`` load and run exactly one .axm, as before chained exports existed."""
    import shutil

    from libreyolo import LibreYOLO
    from libreyolo.backends.axelera import resolve_axelera_artifact

    _, artifact = exports("detect")
    backend = LibreYOLO(str(artifact))
    assert backend._chain is None and backend._chain_models == []
    h, w = backend._fixed_input_hw
    before = backend._model.model.calls
    outputs = backend._invoke(np.zeros((1, 3, h, w), dtype=np.float32))
    assert backend._model.model.calls == before + 1 and len(outputs) == len(backend._output_names)
    copy_dir = tmp_path / "two_axm"
    shutil.copytree(artifact, copy_dir)
    (copy_dir / "extra.axm").write_bytes(b"axm")
    with pytest.raises(ValueError, match="more than one .axm"):
        resolve_axelera_artifact(copy_dir)


# --- Multi-part exports, general form (parts v2) ------------------------------


def test_ssd_metadata_keeps_the_legacy_two_part_form(ssd_export, tmp_path):
    """The generic exporter writes SSD300's ``parts`` byte for byte as the two-part exporter did."""
    from libreyolo.export.axelera import _write_yaml

    _, artifact = ssd_export
    text = (artifact / "metadata.yaml").read_text(encoding="utf-8")
    meta = yaml.safe_load(text)
    first, host, second = meta["parts"]
    assert list(first) == ["axm", "input", "outputs"] and list(second) == ["axm", "input", "outputs"]
    assert list(host) == ["host", "input", "output", "scale"]
    assert "host_arrays" not in meta
    legacy = dict(meta)
    legacy["parts"] = [
        {"axm": "ssd.axm", "input": "images", "outputs": list(first["outputs"])},
        {"host": "l2norm", "input": "conv4_3", "output": "conv4_3_norm", "scale": [float(v) for v in host["scale"]]},
        {"axm": "ssd_part2.axm", "input": "conv4_3_norm", "outputs": list(second["outputs"])},
    ]
    _write_yaml(tmp_path / "legacy.yaml", legacy)
    assert (tmp_path / "legacy.yaml").read_text(encoding="utf-8") == text


def test_legacy_ssd_metadata_loads_as_chain_steps(ssd_export):
    from libreyolo import LibreYOLO

    _, artifact = ssd_export
    before = (artifact / "metadata.yaml").read_bytes()
    backend = LibreYOLO(str(artifact))
    assert [step.kind for step in backend._chain] == ["axm", "host", "axm"]
    host = backend._chain[1]
    assert (host.source, host.inputs, host.outputs) == ("l2norm", ["conv4_3"], ["conv4_3_norm"])
    assert list(host.params) == ["scale"] and len(host.params["scale"]) == 512
    assert backend._chain[2].inputs == ["conv4_3_norm"] and not backend._chain[2].per_item
    assert (artifact / "metadata.yaml").read_bytes() == before
    backend._run_inference(np.zeros((1, 3, 300, 300), dtype=np.float32))
    timing = backend.chain_timing
    assert timing["images"] == 1 and timing["aipu_ms"] > 0 and timing["host_ms"] > 0


def _onnx(module, inputs, names_in, names_out, path):
    import torch

    torch.onnx.export(
        module.eval(), tuple(inputs), str(path), input_names=names_in, output_names=names_out,
        opset_version=17, dynamo=False,
    )
    return str(path)


class _Named:
    """A torch module plus the ``output_names`` the exporter reads."""

    def __init__(self, module, names):
        self.module, self.output_names = module.eval(), list(names)

    def __call__(self, tensor):
        import torch

        out = self.module(tensor)
        return (out,) if isinstance(out, torch.Tensor) else tuple(out)


def _toy_mix(a, b, *, k):
    """A two-in, two-out host operation for the tests."""
    k = np.asarray(k, dtype=np.float32).reshape(1, -1, 1, 1)
    return (a + b).astype(np.float32), (a * k).astype(np.float32)


def _toy_crops(feat, *, count):
    """``count`` 4x4 crops of ``feat`` stacked as items (a stand-in for RoIAlign)."""
    return np.concatenate([feat[:, :, i : i + 4, i : i + 4] for i in range(int(count))], axis=0).astype(np.float32)


@pytest.fixture
def toy_parts(sdk, tmp_path, monkeypatch):
    """A four-step export: compiled part, two-in/two-out host op (npz array), host ONNX graph, compiled part."""
    import torch

    from libreyolo.export.axelera import export_axelera_parts

    monkeypatch.setitem(axrt.HOST_OPS, "toy_mix", _toy_mix)
    torch.manual_seed(0)

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.Conv2d(3, 4, 1)

        def forward(self, x):
            return self.a(x), torch.relu(self.b(x))

    class Middle(torch.nn.Module):
        def forward(self, s, p):
            return s - 0.5 * p

    head, tail = Head(), torch.nn.Conv2d(4, 2, 3, padding=1)
    x = torch.rand(1, 3, 8, 8)
    f = torch.rand(1, 4, 8, 8)
    head_onnx = _onnx(head, [x], ["images"], ["a", "b"], tmp_path / "head.onnx")
    middle_onnx = _onnx(Middle(), [f, f], ["s", "p"], ["q"], tmp_path / "middle.onnx")
    tail_onnx = _onnx(tail, [f], ["q"], ["out"], tmp_path / "tail.onnx")
    k = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float32)
    recorded: dict[str, list] = {}
    original = sys.modules["axelera.compiler"].quantize

    def quantize(*, model, config, calibration_dataset, transform_fn=None):
        samples = list(calibration_dataset)
        recorded[config.model_name] = samples
        return original(model=model, config=config, calibration_dataset=iter(samples))

    monkeypatch.setattr(sys.modules["axelera.compiler"], "quantize", quantize)
    rng = np.random.default_rng(7)
    calibration = [rng.random((1, 3, 8, 8), dtype=np.float32) for _ in range(3)]
    held_out = [rng.random((1, 3, 8, 8), dtype=np.float32) for _ in range(2)]
    artifact = Path(export_axelera_parts(
        parts=[
            {"compile": head_onnx, "reference": _Named(head, ["a", "b"]), "input": "images"},
            {"host": "toy_mix", "inputs": ["a", "b"], "outputs": ["s", "p"], "k": "@npz:k"},
            {"host_onnx": middle_onnx, "inputs": ["s", "p"], "outputs": ["q"]},
            {"compile": tail_onnx, "reference": _Named(tail, ["out"]), "input": "q"},
        ],
        output_path=str(tmp_path / "toy_axelera_model"), stem="toy",
        metadata={"output_layout": "graph_outputs_v1", "graph_outputs": ["out"]},
        calibration=calibration, calibration_info={"source": "unit"},
        host_arrays={"k": k}, verify=True, verify_samples=held_out,
    ))

    def reference(image):
        with torch.no_grad():
            a, b = (t.numpy() for t in head(torch.from_numpy(image)))
            s, p = _toy_mix(a, b, k=k)
            return a, b, s, p, tail(torch.from_numpy(s - 0.5 * p)).numpy()

    return types.SimpleNamespace(artifact=artifact, recorded=recorded, calibration=calibration, reference=reference, k=k)


def _chain_backend(artifact: Path, final_names: list[str]):
    """An AxeleraBackend holding just a chain, built the way __init__ builds it."""
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    axm = artifact / meta["axm"]
    backend = object.__new__(axrt.AxeleraBackend)
    arrays = axrt._load_host_arrays(artifact, meta.get("host_arrays"), axm)
    backend._chain = axrt._validate_chain(meta["parts"], axm, final_names, arrays)
    backend._chain_sessions = {
        i: axrt._host_onnx_session(artifact / s.source, s, axm) for i, s in enumerate(backend._chain) if s.kind == "onnx"
    }
    backend._chain_models = [
        axrt.AxeleraBackend._load(artifact / s.source) if s.kind == "axm" else None for s in backend._chain
    ]
    backend._output_names, backend._layout_order = list(final_names), list(range(len(final_names)))
    backend.chain_timing = {"images": 0, "aipu_ms": 0.0, "host_ms": 0.0}
    return backend


def test_parts_metadata_lists_every_step_kind(toy_parts):
    meta = yaml.safe_load((toy_parts.artifact / "metadata.yaml").read_text())
    assert [next(iter(part)) for part in meta["parts"]] == ["axm", "host", "onnx", "axm"]
    first, host, graph, last = meta["parts"]
    assert first["axm"] == "toy.axm" and sorted(first["outputs"]) == ["a", "b"]
    assert host == {"host": "toy_mix", "inputs": ["a", "b"], "outputs": ["s", "p"], "k": "@npz:k"}
    assert graph == {"onnx": "toy_host1.onnx", "inputs": ["s", "p"], "outputs": ["q"]}
    assert last == {"axm": "toy_part2.axm", "input": "q", "outputs": ["out"]}
    assert meta["host_arrays"] == "toy_host.npz"
    names = sorted(p.name for p in toy_parts.artifact.iterdir())
    for name in ("toy.axm", "toy_part2.axm", "toy_host1.onnx", "toy_host.npz", "compiler_config_final.part2.toml"):
        assert name in names


def test_each_compiled_part_is_calibrated_on_its_fp32_input(toy_parts):
    head_samples, tail_samples = toy_parts.recorded["toy"], toy_parts.recorded["toy_part2"]
    assert len(head_samples) == len(tail_samples) == 3
    for image, seen_head, seen_tail in zip(toy_parts.calibration, head_samples, tail_samples):
        np.testing.assert_array_equal(np.asarray(seen_head), image)
        _, _, s, p, _ = toy_parts.reference(image)
        np.testing.assert_allclose(np.asarray(seen_tail), s - 0.5 * p, rtol=1e-5, atol=1e-6)


def test_parts_chain_runs_host_numpy_onnx_and_device_steps(toy_parts):
    backend = _chain_backend(toy_parts.artifact, ["out"])
    assert [s.kind for s in backend._chain] == ["axm", "host", "onnx", "axm"]
    np.testing.assert_array_equal(backend._chain[1].params["k"], toy_parts.k)
    image = np.random.default_rng(9).random((1, 3, 8, 8), dtype=np.float32)
    (out,) = backend._run_chain([image], batch=False)[0]
    np.testing.assert_allclose(out, toy_parts.reference(image)[-1], rtol=1e-4, atol=1e-5)
    images = [np.random.default_rng(i).random((1, 3, 8, 8), dtype=np.float32) for i in range(3)]
    batched = backend._run_chain(images, batch=True)
    assert [m.batches[-1] for m in backend._chain_models if m is not None] == [3, 3]
    for got, image in zip(batched, images):
        np.testing.assert_allclose(got[0], toy_parts.reference(image)[-1], rtol=1e-4, atol=1e-5)
    timing = backend.chain_timing
    assert timing["images"] == 4 and timing["aipu_ms"] > 0 and timing["host_ms"] > 0


def test_parts_verify_reports_every_compiled_part(toy_parts):
    import json

    report = json.loads((toy_parts.artifact / "toy.parity.json").read_text())
    assert [part["axm"] for part in report["parts"]] == ["toy.axm", "toy_part2.axm"]
    for part in report["parts"]:
        assert part["images"] == 2
        assert min(o["min_cosine"] for o in part["quantized_vs_fp32"]) > 0.999


@pytest.mark.parametrize(
    ("edit", "error", "kind"),
    [
        (lambda parts: parts[3].update(input=None, inputs=["q", "s"]), "compiled part with 2 inputs", ValueError),
        (lambda parts: parts[1].update(k="@npz:missing"), "host array 'missing'", ValueError),
        (lambda parts: parts[2].update(onnx="gone.onnx"), "gone.onnx", FileNotFoundError),
        (lambda parts: parts[2].update(inputs=["s"]), "toy_host1.onnx takes", ValueError),
        (lambda parts: parts[1].pop("outputs"), "lists no output", ValueError),
    ],
)
def test_parts_metadata_is_validated(toy_parts, edit, error, kind):
    meta = yaml.safe_load((toy_parts.artifact / "metadata.yaml").read_text())
    edit(meta["parts"])
    for part in meta["parts"]:
        if part.get("input", "") is None:
            del part["input"]
    axm = toy_parts.artifact / "toy.axm"
    with pytest.raises(kind, match=error):
        arrays = axrt._load_host_arrays(toy_parts.artifact, meta.get("host_arrays"), axm)
        steps = axrt._validate_chain(meta["parts"], axm, ["out"], arrays)
        for step in steps:
            if step.kind == "onnx":
                axrt._host_onnx_session(toy_parts.artifact / step.source, step, axm)


def test_host_onnx_steps_need_onnxruntime(toy_parts, monkeypatch):
    meta = yaml.safe_load((toy_parts.artifact / "metadata.yaml").read_text())
    axm = toy_parts.artifact / "toy.axm"
    arrays = axrt._load_host_arrays(toy_parts.artifact, meta.get("host_arrays"), axm)
    step = axrt._validate_chain(meta["parts"], axm, ["out"], arrays)[2]
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match="pip install onnxruntime"):
        axrt._host_onnx_session(toy_parts.artifact / step.source, step, axm)


def test_per_item_parts_run_every_item(sdk, tmp_path, monkeypatch):
    import torch

    from libreyolo.export.axelera import export_axelera_parts

    monkeypatch.setitem(axrt.HOST_OPS, "toy_crops", _toy_crops)
    torch.manual_seed(1)
    trunk, item_head = torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.Conv2d(4, 2, 3, padding=1)
    trunk_onnx = _onnx(trunk, [torch.rand(1, 3, 8, 8)], ["images"], ["feat"], tmp_path / "trunk.onnx")
    item_onnx = _onnx(item_head, [torch.rand(1, 4, 4, 4)], ["crops"], ["y"], tmp_path / "item.onnx")
    rng = np.random.default_rng(2)
    artifact = Path(export_axelera_parts(
        parts=[
            {"compile": trunk_onnx, "reference": _Named(trunk, ["feat"]), "input": "images"},
            {"host": "toy_crops", "input": "feat", "output": "crops", "count": 3},
            {"compile": item_onnx, "reference": _Named(item_head, ["y"]), "input": "crops", "per_item": True},
        ],
        output_path=str(tmp_path / "items_axelera_model"), stem="items",
        metadata={"output_layout": "graph_outputs_v1", "graph_outputs": ["y"]},
        calibration=[rng.random((1, 3, 8, 8), dtype=np.float32) for _ in range(2)],
        calibration_info={},
    ))
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["parts"][2] == {
        "axm": "items_part2.axm", "input": "crops", "outputs": ["y"], "per_item": True, "item_shapes": {"y": [2, 4, 4]},
    }
    backend = _chain_backend(artifact, ["y"])
    images = [rng.random((1, 3, 8, 8), dtype=np.float32) for _ in range(2)]
    results = backend._run_chain(images, batch=False)
    assert backend._chain_models[2].batches[-1] == 6  # both images' items in one submission
    for (y,), image in zip(results, images):
        with torch.no_grad():
            crops = _toy_crops(trunk(torch.from_numpy(image)).numpy(), count=3)
            expected = item_head(torch.from_numpy(crops)).numpy()
        assert y.shape == (3, 2, 4, 4)
        np.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-5)
    env = {"crops": np.zeros((0, 4, 4, 4), dtype=np.float32)}
    backend._run_per_item(backend._chain[2], backend._chain_models[2], [env], 2)
    assert env["y"].shape == (0, 2, 4, 4)


def _toy_boxes(feat, *, boxes):
    """Crops of ``feat`` at nested ``[y, x, h, w]`` integer boxes, stacked on the channel axis."""
    return np.concatenate([feat[:, :, y : y + h, x : x + w] for y, x, h, w in boxes], axis=1).astype(np.float32)


def test_host_params_may_nest_and_keep_their_integers(sdk, tmp_path, monkeypatch):
    import torch

    from libreyolo.export.axelera import export_axelera_parts

    monkeypatch.setitem(axrt.HOST_OPS, "toy_boxes", _toy_boxes)
    torch.manual_seed(2)
    trunk, tail = torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.Conv2d(8, 2, 1)
    trunk_onnx = _onnx(trunk, [torch.rand(1, 3, 8, 8)], ["images"], ["feat"], tmp_path / "trunk.onnx")
    tail_onnx = _onnx(tail, [torch.rand(1, 8, 4, 4)], ["crops"], ["y"], tmp_path / "tail.onnx")
    boxes = [[0, 0, 4, 4], [2, 3, 4, 4]]
    rng = np.random.default_rng(3)
    artifact = Path(export_axelera_parts(
        parts=[
            {"compile": trunk_onnx, "reference": _Named(trunk, ["feat"]), "input": "images"},
            {"host": "toy_boxes", "input": "feat", "output": "crops", "boxes": boxes},
            {"compile": tail_onnx, "reference": _Named(tail, ["y"]), "input": "crops"},
        ],
        output_path=str(tmp_path / "boxes_axelera_model"), stem="boxes",
        metadata={"output_layout": "graph_outputs_v1", "graph_outputs": ["y"]},
        calibration=[rng.random((1, 3, 8, 8), dtype=np.float32) for _ in range(2)], calibration_info={},
    ))
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["parts"][1]["boxes"] == boxes and all(isinstance(v, int) for box in meta["parts"][1]["boxes"] for v in box)
    backend = _chain_backend(artifact, ["y"])
    image = rng.random((1, 3, 8, 8), dtype=np.float32)
    (y,) = backend._run_chain([image], batch=False)[0]
    with torch.no_grad():
        expected = tail(torch.from_numpy(_toy_boxes(trunk(torch.from_numpy(image)).numpy(), boxes=boxes))).numpy()
    np.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-5)


def test_fp32_chain_stops_at_the_requested_part(monkeypatch):
    from libreyolo.export.axelera import _run_fp32_chain

    calls = []
    head = _Named(_identity_module(), ["a"])
    host_ops = {"count": lambda a: calls.append(1) or a}
    parts = [
        {"compile": "unused.onnx", "reference": head, "input": "images"},
        {"host": "count", "input": "a", "output": "b"},
        {"compile": "unused.onnx", "reference": _Named(_identity_module(), ["c"]), "input": "b"},
    ]
    seen = []
    image = np.ones((1, 3, 2, 2), dtype=np.float32)
    _run_fp32_chain(parts, image, host_ops, {}, {}, lambda i, x: seen.append(i), stop=0)
    assert seen == [0] and calls == []  # nothing after part 0 ran
    _run_fp32_chain(parts, image, host_ops, {}, {}, lambda i, x: seen.append(i), stop=2)
    assert seen == [0, 0, 2] and calls == [1]


def _identity_module():
    import torch

    return torch.nn.Identity()


def test_host_map_keeps_order_and_runs_single_items_inline(monkeypatch):
    import threading

    from libreyolo.backends import axelera as ax

    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    assert ax._host_map(lambda x: x * x, list(range(9))) == [x * x for x in range(9)]
    caller = threading.get_ident()
    assert ax._host_map(lambda _: threading.get_ident(), [0]) == [caller]
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert set(ax._host_map(lambda _: threading.get_ident(), [0, 1, 2])) == {caller}


def test_batched_decode_on_host_threads_equals_per_image_decode(exports, monkeypatch):
    """A batch decodes its images on a thread pool; every output is bitwise the per-image result."""
    from libreyolo import LibreYOLO

    model, artifact = exports("yolo9")
    backend = LibreYOLO(str(artifact))
    rng = np.random.default_rng(3)
    images = [rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8) for _ in range(4)]
    blobs = [np.asarray(model._preprocess(image, input_size=96)[0]) for image in images]
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    single = [backend._run_inference(blob) for blob in blobs]
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    batched = backend._run_inference(np.concatenate(blobs, axis=0))
    for index, outputs in enumerate(single):
        for got, want in zip(batched, outputs):
            np.testing.assert_array_equal(got[index : index + 1], want)


def test_batched_predict_on_host_threads_equals_single_image_predict(exports, monkeypatch):
    """predict(batch=4) pre-processes and decodes on host threads; Results equal one-at-a-time predict."""
    from libreyolo import LibreYOLO
    from libreyolo.backends.base import BaseBackend

    model, artifact = exports("yolo9")
    backend = LibreYOLO(str(artifact))
    assert type(backend)._map_images is not BaseBackend._map_images
    rng = np.random.default_rng(5)
    images = [rng.integers(0, 255, size=(90 + 7 * i, 96, 3), dtype=np.uint8) for i in range(5)]
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    single = [backend(image, conf=0.01) for image in images]
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    batched = backend(images, batch=4, conf=0.01)
    assert len(batched) == len(single)
    for got, want in zip(batched, single):
        for field in ("xyxy", "conf", "cls"):
            np.testing.assert_array_equal(np.asarray(getattr(got.boxes, field)), np.asarray(getattr(want.boxes, field)))


def test_host_map_items_see_the_callers_context_variables(monkeypatch):
    from contextvars import ContextVar

    from libreyolo.backends import axelera as ax

    flag = ContextVar("flag", default="default")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    token = flag.set("caller")
    try:
        assert ax._host_map(lambda _: flag.get(), list(range(6))) == ["caller"] * 6
    finally:
        flag.reset(token)


def test_result_building_runs_on_host_threads_only_for_dense_tasks(exports, monkeypatch):
    import threading

    from libreyolo import LibreYOLO

    _, artifact = exports("yolo9")
    backend = LibreYOLO(str(artifact))
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    caller = threading.get_ident()
    ident = lambda _: threading.get_ident()  # noqa: E731
    assert set(backend._map_images(ident, [0, 1, 2], stage="results")) == {caller}
    monkeypatch.setattr(backend, "task", "semantic")
    assert len(backend._map_images(lambda _: threading.current_thread().name, [0, 1, 2], stage="results")) == 3
    assert all(
        name.startswith("libreyolo-axelera-host")
        for name in backend._map_images(lambda _: threading.current_thread().name, [0, 1, 2], stage="results")
    )


def test_lists_directories_and_videos_run_batch_8_by_default_with_per_image_results(exports, tmp_path):
    """On the Metis, predict() batches lists, directories and video files 8 at a time unless ``batch=1``;
    every Result is bitwise the image-by-image one."""
    cv2 = pytest.importorskip("cv2")
    from PIL import Image

    from libreyolo import LibreYOLO
    from libreyolo.backends.base import BaseBackend

    assert BaseBackend.default_predict_batch == 1
    _, artifact = exports("yolo9")
    backend = LibreYOLO(str(artifact))
    assert backend.default_predict_batch == 8
    rng = np.random.default_rng(11)
    images = [rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8) for _ in range(10)]
    folder = tmp_path / "images"
    folder.mkdir()
    for index, image in enumerate(images):
        Image.fromarray(image).save(folder / f"{index:02d}.png")
    video = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 96))
    for image in images:
        writer.write(image)
    writer.release()

    def same(got, want):
        assert len(got) == len(want) == len(images)
        for a, b in zip(got, want):
            assert a.path == b.path and getattr(a, "frame_idx", None) == getattr(b, "frame_idx", None)
            for field in ("xyxy", "conf", "cls"):
                np.testing.assert_array_equal(np.asarray(getattr(a.boxes, field)), np.asarray(getattr(b.boxes, field)))

    for source in (images, str(folder), video):
        backend._model.batches.clear()
        default = list(backend(source, conf=0.01))
        assert backend._model.batches == [8, 2], source if isinstance(source, str) else "list"
        backend._model.batches.clear()
        single = list(backend(source, conf=0.01, batch=1))
        assert backend._model.batches == []
        same(default, single)
    with pytest.raises(ValueError, match="batch must be >= 1"):
        backend(images, batch=0)
