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


def _export(tmp_path, task="detect"):
    """``task`` is a YOLO-NAS task, or ``"yolo9"`` for a YOLO9-t detector."""
    import torch

    from libreyolo import LibreYOLO9, LibreYOLONAS

    torch.manual_seed(0)
    if task == "yolo9":
        model = LibreYOLO9(None, size="t", nb_classes=3, device="cpu")
        with torch.no_grad():  # class bias starts at -10; lift it so boxes pass conf
            for tower in model.model.head.class_convs:
                tower[-1].bias.fill_(0.0)
    else:
        size, nc = {"detect": ("s", 3), "pose": ("n", 1)}[task]
        model = LibreYOLONAS(None, size=size, nb_classes=nc, task=task, device="cpu")
        with torch.no_grad():  # sharper class logits so a few detections pass conf
            for head in (model.model.heads.head1, model.model.heads.head2, model.model.heads.head3):
                head.cls_pred.bias.add_(4.0)
    from unittest import mock

    from libreyolo.export.support import REFUSED_SIZES

    # The pose input bound changes FP32 keypoints by design (tested in
    # test_yolonas_raw_heads); without it these stay strict decode tests. The
    # small test pose model is size n, refused on Metis by measurement.
    with mock.patch("libreyolo.export.exporter._bound_yolonas_pose_inputs", lambda *a, **k: {}), \
            mock.patch.dict(REFUSED_SIZES, {("yolonas", "pose", "axelera"): {}}):
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

    image = _image()
    blob = model._preprocess(image, input_size=96)[0]
    blob = blob.numpy() if hasattr(blob, "numpy") else np.asarray(blob)
    with torch.no_grad():
        reference = model.model.eval()(torch.from_numpy(blob))["predictions"].numpy()
    (decoded,) = backend._run_inference(blob)
    assert decoded.shape == reference.shape  # (1, 4 + nc, N)
    np.testing.assert_allclose(decoded[:, :4], reference[:, :4], rtol=1e-6, atol=1e-3)
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
