"""Axelera export: SDK helpers and exporter contract, with a fake SDK (no device)."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

from libreyolo.backends import axelera as axrt
from libreyolo.export import axelera as ax


@pytest.fixture(autouse=True)
def _libreyolo_logs_reach_caplog(monkeypatch):
    """libreyolo's logging setup turns propagation off on the "libreyolo" logger
    (utils/logging.py); turn it back on so caplog sees the records whatever ran
    before this test."""
    monkeypatch.setattr(logging.getLogger("libreyolo"), "propagate", True)


pytestmark = pytest.mark.unit


class _Field:
    def __init__(self, alias=None):
        self.alias = alias


class _FakeCompilerConfig:
    """Stand-in for axelera.compiler.CompilerConfig: same field/alias surface."""

    model_fields: ClassVar[dict[str, _Field]] = {
        "quantization_scheme": _Field("ptq_scheme"),
        "resources_used": _Field("resources"),
        "aipu_cores_used": _Field("aipu_cores"),
        "output_axm_format": _Field(),
        "save_error_artifact": _Field(),
        "model_name": _Field(),
        "output_dir": _Field(),
        "model_metadata": _Field(),
        "tiling_depth": _Field(),
    }

    def __init__(self, **settings):
        unknown = set(settings) - set(self.model_fields)
        if unknown:
            raise ValueError(f"extra fields not permitted: {sorted(unknown)}")
        if settings.get("tiling_depth", 1) < 1:
            raise ValueError("tiling_depth must be >= 1")
        self.__dict__.update(settings)

    def to_toml(self):
        return "".join(f"{k} = {v!r}\n" for k, v in sorted(self.__dict__.items()))


@pytest.fixture
def fake_sdk(monkeypatch, tmp_path):
    calls = []
    compiler = types.ModuleType("axelera.compiler")
    compiler.CompilerConfig = _FakeCompilerConfig
    state = {"fail_compile": False}

    def quantize(*, model, config, calibration_dataset, transform_fn=None):
        samples = list(calibration_dataset)
        calls.append(("quantize", model, len(samples)))
        return types.SimpleNamespace(samples=samples)

    def compile(*, model, config, output_dir):
        calls.append(("compile", Path(output_dir)))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "manifest.json").write_text("{}")
        if state["fail_compile"]:
            (Path(output_dir) / "error_artifact.zip").write_bytes(b"zip")
            raise RuntimeError("tiling failed for node Conv_12")
        (Path(output_dir).parent / f"{config.model_name}.axm").write_bytes(b"axm")

    compiler.quantize = quantize
    compiler.compile = compile
    package = types.ModuleType("axelera")
    package.compiler = compiler
    monkeypatch.setitem(sys.modules, "axelera", package)
    monkeypatch.setitem(sys.modules, "axelera.compiler", compiler)
    return types.SimpleNamespace(calls=calls, state=state, module=compiler)


def test_install_command_is_exact():
    assert ax.axelera_install_command("axelera-devkit") == (
        "pip install --extra-index-url "
        "https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple "
        "axelera-devkit==1.9.0"
    )


@pytest.mark.parametrize(
    ("require", "package"),
    ((ax.require_axelera_devkit, "axelera-devkit"), (ax.require_axelera_rt, "axelera-rt")),
)
def test_missing_sdk_raises_with_exact_command(monkeypatch, require, package):
    monkeypatch.delenv(ax.AUTOINSTALL_ENV, raising=False)
    monkeypatch.setattr(axrt, "_module_available", lambda module: False)
    monkeypatch.setattr(axrt.subprocess, "check_call", pytest.fail)
    with pytest.raises(ImportError) as info:
        require()
    message = str(info.value)
    assert ax.axelera_install_command(package) in message
    assert "LIBREYOLO_AUTOINSTALL=1" in message
    assert "EULA" in message


@pytest.mark.parametrize("value", ["1", "true", "TRUE"])
def test_autoinstall_runs_pip_with_axelera_index(monkeypatch, caplog, value):
    monkeypatch.setenv(ax.AUTOINSTALL_ENV, value)
    installed = {"done": False}
    monkeypatch.setattr(axrt, "_module_available", lambda module: installed["done"])
    monkeypatch.setattr(axrt, "_installed_version", lambda package: ax.AXELERA_SDK_VERSION)
    commands = []

    def check_call(command):
        commands.append(command)
        installed["done"] = True

    monkeypatch.setattr(axrt.subprocess, "check_call", check_call)
    with caplog.at_level(logging.WARNING, logger=axrt.__name__):
        assert ax.require_axelera_devkit() == ax.AXELERA_SDK_VERSION
    assert commands == [[
        sys.executable, "-m", "pip", "install", "--extra-index-url",
        ax.AXELERA_INDEX_URL, "axelera-devkit==1.9.0",
    ]]
    assert "proprietary" in caplog.text and "axelera-devkit" in caplog.text


@pytest.mark.parametrize("value", ["", "0", "yes", "false"])
def test_autoinstall_needs_explicit_opt_in(monkeypatch, value):
    monkeypatch.setenv(ax.AUTOINSTALL_ENV, value)
    monkeypatch.setattr(axrt, "_module_available", lambda module: False)
    monkeypatch.setattr(axrt.subprocess, "check_call", pytest.fail)
    with pytest.raises(ImportError):
        ax.require_axelera_rt()


def test_other_sdk_version_warns_and_continues(monkeypatch, caplog):
    monkeypatch.setattr(axrt, "_module_available", lambda module: True)
    monkeypatch.setattr(axrt, "_installed_version", lambda package: "1.8.0")
    with caplog.at_level(logging.WARNING, logger=axrt.__name__):
        assert ax.require_axelera_rt() == "1.8.0"
    assert "1.8.0" in caplog.text and ax.AXELERA_SDK_VERSION in caplog.text


@pytest.mark.parametrize("platform", ["darwin", "win32", "cygwin"])
def test_compile_platform_must_be_linux(monkeypatch, platform):
    monkeypatch.setattr(ax.sys, "platform", platform)
    with pytest.raises(RuntimeError, match=platform):
        ax.check_compile_platform()


def test_compile_platform_linux_passes(monkeypatch):
    monkeypatch.setattr(ax.sys, "platform", "linux")
    ax.check_compile_platform()


def test_overrides_accept_field_names_and_aliases():
    assert ax.normalize_compiler_overrides(
        {"ptq_scheme": "per_tensor_min_max", "tiling_depth": 6}, _FakeCompilerConfig
    ) == {"quantization_scheme": "per_tensor_min_max", "tiling_depth": 6}
    assert ax.normalize_compiler_overrides(None, _FakeCompilerConfig) == {}


def test_overrides_reject_unknown_fields_and_list_valid_ones():
    with pytest.raises(ValueError, match="Unknown compiler_config field.*nope") as info:
        ax.normalize_compiler_overrides({"nope": 1}, _FakeCompilerConfig)
    assert "quantization_scheme" in str(info.value)


@pytest.mark.parametrize(
    "key", ["output_axm_format", "aipu_cores", "aipu_cores_used", "model_name", "output_dir", "model_metadata"]
)
def test_overrides_reject_fields_libreyolo_owns(key):
    with pytest.raises(ValueError, match="cannot set"):
        ax.normalize_compiler_overrides({key: 1}, _FakeCompilerConfig)


def test_overrides_reject_duplicates_and_non_mappings():
    with pytest.raises(ValueError, match="more than once"):
        ax.normalize_compiler_overrides(
            {"ptq_scheme": "a", "quantization_scheme": "b"}, _FakeCompilerConfig
        )
    with pytest.raises(TypeError, match="must be a dict"):
        ax.normalize_compiler_overrides(["tiling_depth"], _FakeCompilerConfig)


def test_build_compiler_config_forces_artifact_settings(fake_sdk, tmp_path):
    config, overrides = ax.build_compiler_config(
        "LibreYOLONASs",
        build_dir=tmp_path / "build",
        overrides={"resources": 0.5, "tiling_depth": 6},
    )
    assert overrides == {"resources_used": 0.5, "tiling_depth": 6}
    assert config.model_name == "LibreYOLONASs"
    assert config.aipu_cores_used == 1
    assert config.output_axm_format is True
    assert config.save_error_artifact is True
    assert config.resources_used == 0.5
    assert config.quantization_scheme == ax.DEFAULT_QUANTIZATION_SCHEME
    assert config.output_dir == tmp_path / "build"


def test_build_compiler_config_reports_invalid_values(fake_sdk, tmp_path):
    with pytest.raises(ValueError, match="Invalid compiler_config"):
        ax.build_compiler_config("m", build_dir=tmp_path, overrides={"tiling_depth": 0})


def test_compile_to_axm_finds_the_axm_next_to_the_build_dir(fake_sdk, tmp_path):
    config, _ = ax.build_compiler_config("LibreYOLONASs", build_dir=tmp_path / "x")
    axm, quantized = ax.compile_to_axm("model.onnx", [1, 2, 3], config, tmp_path / "work")
    assert axm == tmp_path / "work" / "LibreYOLONASs.axm"
    assert len(quantized.samples) == 3
    assert fake_sdk.calls == [("quantize", "model.onnx", 3), ("compile", tmp_path / "work" / "build")]


def test_compile_failure_names_error_artifact_and_docs(fake_sdk, tmp_path):
    fake_sdk.state["fail_compile"] = True
    config, _ = ax.build_compiler_config("m", build_dir=tmp_path / "x")
    with pytest.raises(RuntimeError) as info:
        ax.compile_to_axm("model.onnx", [1, 2], config, tmp_path / "work")
    message = str(info.value)
    assert "tiling failed for node Conv_12" in message
    assert "error_artifact.zip" in message
    assert "docs/axelera.md#troubleshooting" in message


def test_publish_directory_replaces_previous_export(tmp_path):
    destination = tmp_path / "weights" / "m_axelera_model"
    destination.mkdir(parents=True)
    (destination / "old.axm").write_bytes(b"old")
    staging = tmp_path / "weights" / ".staging"
    staging.mkdir()
    (staging / "m.axm").write_bytes(b"new")

    assert ax.publish_directory(staging, destination) == destination
    assert sorted(p.name for p in destination.iterdir()) == ["m.axm"]
    assert sorted(p.name for p in destination.parent.iterdir()) == ["m_axelera_model"]


def test_publish_directory_keeps_previous_export_on_failure(tmp_path, monkeypatch):
    destination = tmp_path / "m_axelera_model"
    destination.mkdir()
    (destination / "old.axm").write_bytes(b"old")
    missing_staging = tmp_path / "never_built"

    with pytest.raises(OSError):
        ax.publish_directory(missing_staging, destination)
    assert (destination / "old.axm").read_bytes() == b"old"


# ---------------------------------------------------------------------------
# AxeleraExporter end to end with a fake SDK (no compiler, no device)
# ---------------------------------------------------------------------------


@pytest.fixture
def exporter_sdk(monkeypatch, fake_sdk):
    """Fake SDK plus the checks the real environment would pass."""
    pytest.importorskip("onnx")
    monkeypatch.setattr(axrt, "_module_available", lambda module: True)
    monkeypatch.setattr(axrt, "_installed_version", lambda package: ax.AXELERA_SDK_VERSION)
    monkeypatch.setattr(ax, "check_compile_platform", lambda: None)
    seen = {}
    original_quantize = fake_sdk.module.quantize

    def quantize(*, model, config, calibration_dataset, transform_fn=None):
        import onnx

        graph = onnx.load(model).graph
        seen["onnx_outputs"] = [o.name for o in graph.output]
        seen["onnx_ops"] = [node.op_type for node in graph.node]
        seen["onnx_input"] = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
        samples = list(calibration_dataset)
        seen["samples"] = samples
        seen["config"] = config
        original_quantize(model=model, config=config, calibration_dataset=iter(samples))
        return _ReorderingQuantizedModel(model, seen)

    fake_sdk.module.quantize = quantize
    fake_sdk.seen = seen
    return fake_sdk


class _ReorderingQuantizedModel:
    """Runs the exported ONNX on CPU and, like the real compiler, returns each
    level's outputs in a different order than the graph (reversed per level)."""

    def __init__(self, onnx_path, seen):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        names = [o.name for o in self.session.get_outputs()]
        per_level = len(names) // 3
        self.order = [
            i for level in range(3)
            for i in reversed(range(level * per_level, (level + 1) * per_level))
        ] or list(range(len(names)))  # classifiers: one output
        seen["device_order"] = [names[i] for i in self.order]

    def __call__(self, tensor):
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: tensor.numpy()})
        return [outputs[i] for i in self.order]


def _dataset(tmp_path, n=4):
    import numpy as np
    from PIL import Image

    images = tmp_path / "data" / "images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        Image.fromarray(rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)).save(images / f"{i}.jpg")
    yaml_path = tmp_path / "data" / "data.yaml"
    yaml_path.write_text(
        f"path: {tmp_path / 'data'}\ntrain: images\nval: images\nnc: 3\nnames: [a, b, c]\n"
    )
    return yaml_path


def _model(task="detect"):
    import torch

    from libreyolo import LibreYOLONAS

    torch.manual_seed(0)
    size, nc = {"detect": ("s", 3), "pose": ("n", 1), "obb": ("s", 3)}[task]
    return LibreYOLONAS(None, size=size, nb_classes=nc, task=task, device="cpu")


def _classifier(nb_classes=5):
    import torch

    from libreyolo import LibreResNet

    torch.manual_seed(0)
    return LibreResNet(size="18", nb_classes=nb_classes, device="cpu")


def test_classify_export_writes_the_logits_layout(exporter_sdk, tmp_path):
    import yaml

    destination = tmp_path / "r18_axelera_model"
    # 72 is not a multiple of 32: only the YOLO-NAS host decode needs that.
    _classifier().export(format="axelera", data=str(_dataset(tmp_path)), imgsz=72, output_path=str(destination))
    assert exporter_sdk.seen["onnx_outputs"] == ["logits"]
    assert exporter_sdk.seen["onnx_input"] == [1, 3, 72, 72]
    meta = yaml.safe_load((destination / "metadata.yaml").read_text())
    assert meta["task"] == "classify" and meta["model_family"] == "resnet"
    assert meta["output_layout"] == "classify_logits_v1" and meta["output_names"] == ["logits"]
    assert "strides" not in meta and "output_windows" not in meta
    # The eval transform travels with the artifact (crop, interpolation).
    assert meta["crop_pct"] == 0.95 and meta["interpolation"] == "bicubic"
    assert meta["axelera"]["calibration"]["preprocess"] == "classify"


def test_mobilenetv4_export_equalizes_streams_and_uses_histogram_ranges(exporter_sdk, tmp_path, caplog):
    import logging

    import torch
    import yaml

    from libreyolo import LibreMobileNetV4

    torch.manual_seed(0)
    model = LibreMobileNetV4(size="m", nb_classes=5, device="cpu")  # s and l are refused
    live = {k: v.clone() for k, v in model.model.state_dict().items()}
    destination = tmp_path / "mnv4_axelera_model"
    with caplog.at_level(logging.INFO, logger="libreyolo.export.exporter"):
        model.export(format="axelera", data=str(_dataset(tmp_path)), imgsz=64, output_path=str(destination))
    assert "residual streams equalized" in caplog.text
    meta = yaml.safe_load((destination / "metadata.yaml").read_text())
    assert meta["output_layout"] == "classify_logits_v1"
    assert meta["axelera"]["quantization_scheme"] == "hybrid_per_tensor_per_channel"
    # The transform runs on a copy: the caller's model keeps its weights.
    after = model.model.state_dict()
    assert all(torch.equal(live[k], after[k]) for k in live)


def test_default_quantization_scheme_per_model():
    assert ax.default_quantization_scheme("mobilenetv4", "classify") == "hybrid_per_tensor_per_channel"
    assert ax.default_quantization_scheme("resnet", "classify") == "per_tensor_min_max"
    assert ax.default_quantization_scheme("yolonas", "pose") == "hybrid_per_tensor_per_channel"
    assert ax.default_quantization_scheme("yolonas", "detect") == "per_tensor_min_max"


def test_classify_calibration_defaults_to_coco128(exporter_sdk):
    from libreyolo.export.exporter import AxeleraExporter

    assert AxeleraExporter(_classifier())._resolve_calibration_data(True, None) == "coco128.yaml"


@pytest.mark.parametrize("task", ["detect", "pose"])
def test_export_writes_the_artifact_directory(exporter_sdk, tmp_path, task, caplog):
    import cv2
    import numpy as np
    import yaml

    from libreyolo.backends.metadata import parse_export_metadata
    from libreyolo.postprocess.yolonas_raw import (
        YOLONAS_RAW_LAYOUT_BY_TASK,
        raw_head_output_names,
    )

    data = _dataset(tmp_path)
    model = _model(task)
    destination = tmp_path / "out" / "custom_axelera_model"
    with caplog.at_level(logging.INFO):
        path = model.export(format="axelera", data=str(data), imgsz=64, output_path=str(destination))

    assert Path(path) == destination
    assert sorted(p.name for p in destination.iterdir()) == [
        "compiler_config_final.toml", "custom.axm", "metadata.yaml",
    ]
    assert sorted(p.name for p in destination.parent.iterdir()) == ["custom_axelera_model"]
    assert "INT8" in caplog.text
    assert "100-400" in caplog.text or "recommended" in caplog.text

    layout = YOLONAS_RAW_LAYOUT_BY_TASK[task]
    seen = exporter_sdk.seen
    assert seen["onnx_outputs"] == raw_head_output_names(layout)
    assert seen["onnx_input"] == [1, 3, 64, 64]
    assert seen["config"].aipu_cores_used == 1 and seen["config"].output_axm_format is True
    assert seen["config"].model_name == "custom"

    # Calibration: one image at a time, exactly the model's own preprocessing.
    assert len(seen["samples"]) == 4
    preprocess = model._get_preprocess_numpy()
    first = min((tmp_path / "data" / "images").glob("*.jpg"))
    rgb = cv2.cvtColor(cv2.imread(str(first)), cv2.COLOR_BGR2RGB)
    expected, _ = preprocess(rgb, (64, 64))
    assert seen["samples"][0].shape == (1, 3, 64, 64)
    assert seen["samples"][0].dtype == np.float32
    np.testing.assert_array_equal(seen["samples"][0][0], expected)

    meta = yaml.safe_load((destination / "metadata.yaml").read_text())
    parsed = parse_export_metadata(meta, artifact="metadata.yaml", strict=True)
    assert parsed.task == task and parsed.model_family == "yolonas"
    assert meta["format"] == "axelera"
    assert meta["precision"] == "int8" and meta["dynamic"] is False and meta["batch"] == 1
    assert meta["axm"] == "custom.axm"
    assert meta["output_layout"] == layout
    # Metadata lists outputs in the order the compiled model returns them.
    assert meta["output_names"] == seen["device_order"]
    assert meta["output_names"] != raw_head_output_names(layout)
    assert sorted(meta["output_names"]) == sorted(raw_head_output_names(layout))
    assert meta["strides"] == [8, 16, 32] and meta["reg_max"] == 16
    assert meta["axelera"]["devkit_version"] == ax.AXELERA_SDK_VERSION
    assert meta["axelera"]["aipu_cores_used"] == 1
    assert meta["axelera"]["calibration"]["num_images"] == 4
    assert meta["axelera"]["calibration"]["preprocess"] == task
    assert "Deci" in meta["weights_license"]
    if task == "pose":
        assert meta["num_keypoints"] == 17
        # Logits and keypoint offsets get bounded int8 ranges (ReLU6 windows).
        assert meta["output_windows"] == {
            "cls": [-8.0, 8.0], "reg": [-12.0, 12.0], "pose": [-10.0, 18.0],
        }
        assert sorted(meta["pose_head_bounds"]) == ["head1.cls_convs", "head2.cls_convs", "head3.cls_convs"]
        assert meta["axelera"]["quantization_scheme"] == "hybrid_per_tensor_per_channel"
    else:
        assert "pose_head_bounds" not in meta
        assert meta["axelera"]["quantization_scheme"] == "per_tensor_min_max"
        assert "output_windows" not in meta


def test_yolo9_export_uses_the_yolo9_raw_head_layout(exporter_sdk, tmp_path):
    import torch
    import yaml

    from libreyolo import LibreYOLO9
    from libreyolo.postprocess.yolonas_raw import (
        YOLO9_RAW_OUTPUT_WINDOWS,
        raw_head_output_names,
    )

    torch.manual_seed(0)
    model = LibreYOLO9(None, size="t", nb_classes=3, device="cpu")
    destination = tmp_path / "y9_axelera_model"
    model.export(format="axelera", data=str(_dataset(tmp_path)), imgsz=64, output_path=str(destination))

    layout = "yolo9_raw_heads_v1"
    assert exporter_sdk.seen["onnx_outputs"] == raw_head_output_names(layout)
    meta = yaml.safe_load((destination / "metadata.yaml").read_text())
    assert meta["model_family"] == "yolo9" and meta["task"] == "detect"
    assert meta["output_layout"] == layout
    assert meta["output_names"] == exporter_sdk.seen["device_order"]
    assert meta["strides"] == [8, 16, 32] and meta["reg_max"] == 16
    assert meta["grid_cell_offset"] == 0.5
    # Class logits are bounded like the TFLite INT8 export's; bins too.
    assert meta["output_windows"] == {
        part: list(bounds) for part, bounds in YOLO9_RAW_OUTPUT_WINDOWS.items()
    }
    assert meta["output_windows"]["cls"] == [-8.0, 8.0]
    assert "weights_license" not in meta


@pytest.mark.parametrize("size", ["n", "t"])
def test_yolox_export_uses_the_yolox_raw_head_layout(exporter_sdk, tmp_path, size):

    import torch
    import yaml

    from libreyolo import LibreYOLOX
    from libreyolo.postprocess.yolonas_raw import raw_head_output_names

    torch.manual_seed(0)
    model = LibreYOLOX(None, size=size, nb_classes=3, device="cpu")
    destination = tmp_path / "yx_axelera_model"
    model.export(format="axelera", data=str(_dataset(tmp_path)), imgsz=64, output_path=str(destination))

    layout = "yolox_raw_heads_v1"
    assert exporter_sdk.seen["onnx_outputs"] == raw_head_output_names(layout)
    meta = yaml.safe_load((destination / "metadata.yaml").read_text())
    assert meta["model_family"] == "yolox" and meta["task"] == "detect"
    assert meta["output_layout"] == layout
    assert meta["output_names"] == exporter_sdk.seen["device_order"]
    assert meta["strides"] == [8, 16, 32] and "reg_max" not in meta
    assert meta["grid_cell_offset"] == 0.0
    assert "weights_license" not in meta and "output_windows" not in meta
    # Depthwise graph (n): per-channel ranges for depthwise inputs.
    expected = "hybrid_per_tensor_per_channel" if size == "n" else "per_tensor_min_max"
    assert meta["axelera"]["quantization_scheme"] == expected


def test_yolox_n_exports_with_its_measured_drop(exporter_sdk, tmp_path, caplog):
    """No accuracy gate: a size below the 1.0-point bar exports and says so once."""
    from libreyolo import LibreYOLOX

    model = LibreYOLOX(None, size="n", nb_classes=3, device="cpu")
    with caplog.at_level(logging.WARNING):
        out = model.export(format="axelera", data=str(_dataset(tmp_path)), imgsz=64,
                           output_path=str(tmp_path / "yxn_axelera_model"))
    assert Path(out).exists()
    warnings = [r.getMessage() for r in caplog.records if "supported but not validated" in r.getMessage()]
    assert len(warnings) == 1 and "1.81 points" in warnings[0] and "COCO val2017" in warnings[0]


def test_yolo4_export_uses_the_anchor_raw_head_layout(exporter_sdk, tmp_path):
    import torch
    import yaml

    from libreyolo import LibreYOLO4
    from libreyolo.postprocess.yolonas_raw import raw_head_output_names

    torch.manual_seed(0)
    model = LibreYOLO4(None, size="t", nb_classes=3, device="cpu")
    destination = tmp_path / "y4_axelera_model"
    model.export(format="axelera", data=str(_dataset(tmp_path)), imgsz=64, output_path=str(destination))

    layout = "darknet_raw_heads_v1"
    assert exporter_sdk.seen["onnx_outputs"] == raw_head_output_names(layout, (32, 16))
    meta = yaml.safe_load((destination / "metadata.yaml").read_text())
    assert meta["model_family"] == "yolo4" and meta["output_layout"] == layout
    # Calibration used Darknet's own letterbox (it takes the (h, w) pair the loader passes).
    assert len(exporter_sdk.seen["samples"]) == 4
    assert meta["output_names"] == exporter_sdk.seen["device_order"]
    assert meta["strides"] == [32, 16] and "reg_max" not in meta
    specs = model.model.net.detection_specs()
    assert meta["anchors"] == [[list(pair) for pair in spec.anchors] for spec in specs]
    assert meta["scale_x_y"] == [spec.scale_x_y for spec in specs]
    assert meta["axelera"]["quantization_scheme"] == "per_tensor_min_max"
    assert "output_windows" not in meta


def test_yolo3_tiny_export_folds_norms_and_maxpool_padding(exporter_sdk, tmp_path):
    import torch
    import yaml

    from libreyolo import LibreYOLO3
    from libreyolo.postprocess.yolonas_raw import raw_head_output_names

    torch.manual_seed(0)
    model = LibreYOLO3(None, size="t", nb_classes=3, device="cpu")
    destination = tmp_path / "y3_axelera_model"
    model.export(
        format="axelera",
        data=str(_dataset(tmp_path)),
        imgsz=64,
        output_path=str(destination),
    )

    layout = "darknet_raw_heads_v1"
    assert exporter_sdk.seen["onnx_outputs"] == raw_head_output_names(layout, (32, 16))
    ops = exporter_sdk.seen["onnx_ops"]
    # yolov3-tiny's stride-1 maxpool reaches the compiler as one padded
    # MaxPool, and the Darknet norms are folded into the convolutions.
    assert "Pad" not in ops and "Div" not in ops and "Sub" not in ops
    meta = yaml.safe_load((destination / "metadata.yaml").read_text())
    assert meta["model_family"] == "yolo3" and meta["output_layout"] == layout
    assert meta["output_names"] == exporter_sdk.seen["device_order"]


def test_yolo7_export_uses_the_anchor_raw_head_layout(exporter_sdk, tmp_path):
    import torch
    import yaml

    from libreyolo import LibreYOLO7
    from libreyolo.postprocess.yolonas_raw import raw_head_output_names

    torch.manual_seed(0)
    model = LibreYOLO7(None, size="b", nb_classes=3, device="cpu")
    destination = tmp_path / "y7_axelera_model"
    model.export(format="axelera", data=str(_dataset(tmp_path)), imgsz=64, output_path=str(destination))

    layout = "yolo7_raw_heads_v1"
    assert exporter_sdk.seen["onnx_outputs"] == raw_head_output_names(layout)
    meta = yaml.safe_load((destination / "metadata.yaml").read_text())
    assert meta["model_family"] == "yolo7" and meta["output_layout"] == layout
    assert meta["output_names"] == exporter_sdk.seen["device_order"]
    assert meta["strides"] == [8, 16, 32] and "scale_x_y" not in meta
    net = model.model
    assert meta["anchors"] == [[list(pair) for pair in zip(a[0::2], a[1::2])] for a in net.anchors]
    assert len(exporter_sdk.seen["samples"]) == 4


def test_ppyoloe_export_uses_the_ppyoloe_raw_head_layout(exporter_sdk, tmp_path):

    import torch
    import yaml

    from libreyolo import LibrePPYOLOE
    from libreyolo.postprocess.yolonas_raw import raw_head_output_names

    torch.manual_seed(0)
    model = LibrePPYOLOE(None, size="s", nb_classes=3, device="cpu")
    destination = tmp_path / "pp_axelera_model"
    model.export(format="axelera", data=str(_dataset(tmp_path)), imgsz=64, output_path=str(destination))

    layout = "ppyoloe_raw_heads_v1"
    assert exporter_sdk.seen["onnx_outputs"] == raw_head_output_names(layout, (32, 16, 8))
    meta = yaml.safe_load((destination / "metadata.yaml").read_text())
    assert meta["model_family"] == "ppyoloe" and meta["output_layout"] == layout
    assert meta["output_names"] == exporter_sdk.seen["device_order"]
    assert meta["strides"] == [32, 16, 8] and meta["reg_max"] == 16
    assert meta["grid_cell_offset"] == 0.5 and "output_windows" not in meta
    assert meta["axelera"]["quantization_scheme"] == "per_tensor_min_max"


def test_depthwise_detection():
    import torch

    from libreyolo.export.exporter import _has_depthwise_convs

    assert _has_depthwise_convs(torch.nn.Conv2d(8, 8, 3, groups=8))
    assert not _has_depthwise_convs(torch.nn.Conv2d(8, 16, 3, groups=2))
    assert not _has_depthwise_convs(torch.nn.Conv2d(1, 4, 3, groups=1))


def test_default_output_path_and_reexport_replaces(exporter_sdk, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path)
    model = _model()
    path = Path(model.export(format="axelera", data=str(data), imgsz=64))
    assert path == Path("weights") / "LibreYOLONASs_axelera_model"
    (path / "stale.txt").write_text("from an older export")
    model.export(format="axelera", data=str(data), imgsz=64)
    assert not (path / "stale.txt").exists()
    assert (path / "LibreYOLONASs.axm").is_file()


def test_failed_compile_keeps_previous_export(exporter_sdk, tmp_path):
    data = _dataset(tmp_path)
    model = _model()
    destination = tmp_path / "m_axelera_model"
    model.export(format="axelera", data=str(data), imgsz=64, output_path=str(destination))
    exporter_sdk.state["fail_compile"] = True
    with pytest.raises(RuntimeError, match="Axelera compilation failed"):
        model.export(format="axelera", data=str(data), imgsz=64, output_path=str(destination))
    assert (destination / "m.axm").read_bytes() == b"axm"
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".")) == []


def test_batch_sizes_only_the_calibration_loader(exporter_sdk, tmp_path):
    data = _dataset(tmp_path, n=5)
    _model().export(format="axelera", data=str(data), imgsz=64, batch=4, output_path=str(tmp_path / "b_axelera_model"), simplify=False)
    samples = exporter_sdk.seen["samples"]
    assert len(samples) == 5  # the padded duplicate in the last batch is dropped
    assert all(s.shape == (1, 3, 64, 64) for s in samples)
    assert exporter_sdk.seen["onnx_input"] == [1, 3, 64, 64]


def test_fraction_reaches_the_loader(exporter_sdk, tmp_path):
    data = _dataset(tmp_path, n=4)
    _model().export(format="axelera", data=str(data), imgsz=64, fraction=0.5, output_path=str(tmp_path / "f_axelera_model"), simplify=False)
    assert len(exporter_sdk.seen["samples"]) == 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"half": True}, "INT8 only"),
        ({"quantize": 16}, "INT8 only"),
        ({"quantize": 32}, "INT8 only"),
        ({"dynamic": True}, "fixed input shapes"),
        ({"opset": 13}, "opset 14-17"),
        ({"opset": 18}, "opset 14-17"),
        ({"imgsz": 80}, "multiple of 32"),
        ({"compiler_config": {"not_a_field": 1}}, "Unknown compiler_config"),
        ({"compiler_config": {"aipu_cores": 4}}, "cannot set"),
    ],
)
def test_rejected_arguments(exporter_sdk, tmp_path, kwargs, match):
    data = _dataset(tmp_path)
    kwargs = {"imgsz": 64, **kwargs}
    with pytest.raises(ValueError, match=match):
        _model().export(format="axelera", data=str(data), output_path=str(tmp_path / "x_axelera_model"), **kwargs)
    assert not (tmp_path / "x_axelera_model").exists()


def test_quantize_8_and_int8_are_accepted(exporter_sdk, tmp_path):
    data = _dataset(tmp_path)
    _model().export(
        format="axelera", data=str(data), imgsz=64, quantize="8", int8=True,
        simplify=False, output_path=str(tmp_path / "q_axelera_model"),
    )


def test_embedded_nms_is_not_supported(exporter_sdk, tmp_path):
    with pytest.raises(NotImplementedError, match="embedded NMS"):
        _model().export(format="axelera", data=str(_dataset(tmp_path)), imgsz=64, nms=True)


def test_obb_export_needs_data_and_writes_the_obb_layout(exporter_sdk, tmp_path):
    import yaml

    with pytest.raises(ValueError, match="pass data="):
        _model("obb").export(format="axelera", imgsz=64, output_path=str(tmp_path / "o_axelera_model"))
    destination = tmp_path / "obb_axelera_model"
    _model("obb").export(format="axelera", data=str(_dataset(tmp_path)), imgsz=64, output_path=str(destination))
    meta = yaml.safe_load((destination / "metadata.yaml").read_text())
    assert meta["task"] == "obb" and meta["output_layout"] == "yolonas_obb_raw_heads_v1"
    assert meta["axelera"]["calibration"]["preprocess"] == "obb"


def test_obb_calibration_has_no_default(exporter_sdk):
    from libreyolo.export.exporter import AxeleraExporter

    with pytest.raises(ValueError, match="pass data="):
        AxeleraExporter(_model("obb"))._resolve_calibration_data(True, None)


def test_too_few_calibration_images(exporter_sdk, tmp_path):
    data = _dataset(tmp_path, n=1)
    with pytest.raises(ValueError, match="at least 2 images"):
        _model().export(format="axelera", data=str(data), imgsz=64, output_path=str(tmp_path / "t_axelera_model"))


def test_non_linux_export_fails_before_tracing(exporter_sdk, tmp_path, monkeypatch):
    def refuse():
        raise RuntimeError("Axelera export compiles with Axelera's toolchain, which runs on Linux")

    monkeypatch.setattr(ax, "check_compile_platform", refuse)
    with pytest.raises(RuntimeError, match="runs on Linux"):
        _model().export(format="axelera", data=str(_dataset(tmp_path)), imgsz=64)
    assert exporter_sdk.calls == []


def test_missing_devkit_fails_before_tracing(fake_sdk, tmp_path, monkeypatch):
    monkeypatch.delenv(ax.AUTOINSTALL_ENV, raising=False)
    monkeypatch.setattr(axrt, "_module_available", lambda module: False)
    with pytest.raises(ImportError, match="axelera-devkit==1.9.0"):
        _model().export(format="axelera", data=str(_dataset(tmp_path)), imgsz=64)
    assert fake_sdk.calls == []


def test_other_families_are_blocked(exporter_sdk):
    from libreyolo import LibrePICODET

    model = LibrePICODET(None, size="s", device="cpu")
    with pytest.raises(NotImplementedError, match="blocked"):
        model.export(format="axelera")


def test_verify_writes_a_parity_report(exporter_sdk, tmp_path, monkeypatch, caplog):
    import json

    import libreyolo.backends.axelera as backend_module

    data = _dataset(tmp_path)
    destination = tmp_path / "v_axelera_model"
    # No Metis device in the PR gate: the runtime import is reported, not fatal.
    monkeypatch.setattr(
        backend_module, "require_axelera_rt", lambda: (_ for _ in ()).throw(ImportError("no axelera-rt"))
    )
    with caplog.at_level(logging.INFO):
        _model().export(format="axelera", data=str(data), imgsz=64, verify=True, output_path=str(destination))
    report = json.loads((destination / "v.parity.json").read_text())
    # data.yaml's val split is its train split, so verify falls back to the bundled samples.
    assert report["source"].startswith("bundled sample images")
    assert report["images"] >= 1
    names = [row["name"] for row in report["quantized_vs_fp32"]]
    assert sorted(names) == sorted(exporter_sdk.seen["onnx_outputs"])
    # The fake quantized model is the FP32 graph itself, reordered.
    assert all(row["min_cosine"] > 0.9999 for row in report["quantized_vs_fp32"])
    assert report["device_vs_quantized"].startswith("not run")
    assert "Axelera verify on" in caplog.text


def test_parity_report_lists_constant_reference_outputs_instead_of_scoring_them():
    import numpy as np

    from libreyolo.export.axelera import parity_report

    rng = np.random.default_rng(0)
    varied = rng.normal(size=(1, 4, 8, 8)).astype(np.float32)
    flat = np.zeros((1, 2, 8, 8), np.float32)  # e.g. a fully clamped class map
    noisy = flat.copy()
    noisy[0, 0, 0, 0] = 0.0235  # one quantization step of noise
    collected = [(None, [varied, flat], [varied, noisy])]
    report = parity_report(collected, ["p3_reg", "p3_cls"])
    assert [row["name"] for row in report["quantized_vs_fp32"]] == ["p3_reg"]
    assert report["quantized_vs_fp32"][0]["min_cosine"] > 0.9999
    assert report["constant_reference"] == ["p3_cls"]


def test_export_directory_gets_normal_permissions(exporter_sdk, tmp_path):
    import os
    import stat

    mask = os.umask(0)
    os.umask(mask)
    destination = tmp_path / "perm_axelera_model"
    _model().export(format="axelera", data=str(_dataset(tmp_path)), imgsz=64, output_path=str(destination))
    assert stat.S_IMODE(destination.stat().st_mode) == 0o777 & ~mask


def test_busy_device_during_verify_keeps_the_export(exporter_sdk, tmp_path, monkeypatch):
    """The runtime connects to the device on the first call, not in load():
    a busy device must leave a 'not run' note, not discard the compile."""
    import json

    import libreyolo.export.axelera as ax_module

    def busy_runner(*_args, **_kwargs):
        def run(_sample):
            raise RuntimeError("AXR_ERROR_CONNECTION_ERROR: device busy")

        return run

    monkeypatch.setattr(ax_module, "_device_runner", busy_runner)
    destination = tmp_path / "busy_axelera_model"
    _model().export(
        format="axelera", data=str(_dataset(tmp_path)), imgsz=64, verify=True,
        output_path=str(destination),
    )
    report = json.loads((destination / "busy.parity.json").read_text())
    assert report["device_vs_quantized"].startswith("not run: AXR_ERROR_CONNECTION_ERROR")
    assert (destination / "busy.axm").is_file()


def test_verify_reads_the_quantized_model_before_compile_lowers_it(exporter_sdk, tmp_path, monkeypatch):
    """The real compiler lowers the quantized model in place; verify and the
    output-order probe must have used it before that happens."""
    import json

    import libreyolo.backends.axelera as backend_module

    monkeypatch.setattr(
        backend_module, "require_axelera_rt", lambda: (_ for _ in ()).throw(ImportError("no axelera-rt"))
    )
    original_compile = exporter_sdk.module.compile

    def lowering_compile(*, model, config, output_dir):
        original_compile(model=model, config=config, output_dir=output_dir)
        model.order = model.order[::-1]  # after "lowering" the CPU model is unusable for parity
        model.session = None

    exporter_sdk.module.compile = lowering_compile
    destination = tmp_path / "w_axelera_model"
    _model().export(format="axelera", data=str(_dataset(tmp_path)), imgsz=64, verify=True,
                    simplify=False, output_path=str(destination))
    report = json.loads((destination / "w.parity.json").read_text())
    assert all(row["min_cosine"] > 0.9999 for row in report["quantized_vs_fp32"])
    meta = __import__("yaml").safe_load((destination / "metadata.yaml").read_text())
    assert meta["output_names"] == exporter_sdk.seen["device_order"]


def test_support_matrix_rows_for_axelera():
    from libreyolo.export.support import get_support, precision_support

    from libreyolo.export.support import SIZE_ACCURACY_NOTES

    for task in ("detect", "pose", "obb"):
        assert get_support("yolonas", task, "axelera").tier == "validated"
        assert precision_support("yolonas", task, "axelera") == (True, True)
    assert "DOTA" in get_support("yolonas", "obb", "axelera").reason
    assert "validated sizes: m, l" in get_support("yolonas", "pose", "axelera").constraint
    assert sorted(SIZE_ACCURACY_NOTES[("yolonas", "pose", "axelera")]) == ["n", "s"]
    assert get_support("yolo9", "detect", "axelera").tier in {"available", "validated"}
    assert precision_support("yolo9", "detect", "axelera") == (True, True)
    for family in ("yolo9_e2e", "yolo9_p2"):
        assert get_support(family, "detect", "axelera").tier == "blocked"
    rfdetr = get_support("rfdetr", "detect", "axelera")
    assert rfdetr.tier == "blocked" and "attention" in rfdetr.reason
    yolox = get_support("yolox", "detect", "axelera")
    assert yolox.tier == "validated" and "validated sizes: t, s, m, l" in yolox.constraint
    assert precision_support("yolox", "detect", "axelera") == (True, True)
    yolo3 = get_support("yolo3", "detect", "axelera")
    assert yolo3.tier == "validated" and "validated sizes: t, b, spp" in yolo3.constraint
    assert precision_support("yolo3", "detect", "axelera") == (True, True)
    yolo4 = get_support("yolo4", "detect", "axelera")
    assert yolo4.tier == "validated" and "validated sizes: t" in yolo4.constraint
    assert sorted(SIZE_ACCURACY_NOTES[("yolo4", "detect", "axelera")]) == ["b"]
    assert precision_support("yolo4", "detect", "axelera") == (True, True)
    ppyoloe = get_support("ppyoloe", "detect", "axelera")
    assert ppyoloe.tier == "validated" and "validated sizes: l" in ppyoloe.constraint
    assert sorted(SIZE_ACCURACY_NOTES[("ppyoloe", "detect", "axelera")]) == ["m", "s", "x"]
    yolo7 = get_support("yolo7", "detect", "axelera")
    assert yolo7.tier == "validated" and "validated sizes: b" in yolo7.constraint
    other = get_support("rtmdet", "detect", "axelera")
    assert other.tier == "blocked" and "not been compiled" in other.reason
    mnv4 = get_support("mobilenetv4", "classify", "axelera")
    assert mnv4.tier == "validated" and "validated sizes: m;" in mnv4.constraint
    assert sorted(SIZE_ACCURACY_NOTES[("mobilenetv4", "classify", "axelera")]) == ["l", "s"]
    assert sorted(SIZE_ACCURACY_NOTES[("yolox", "detect", "axelera")]) == ["n"]
    for notes in SIZE_ACCURACY_NOTES.values():
        assert all(note.startswith("measured ") and " vs FP32 " in note for note in notes.values())
    fcn = get_support("fcn", "semantic", "axelera")
    assert fcn.tier == "validated" and "validated sizes: r50" in fcn.constraint


def test_pose_sizes_below_the_bar_export_with_a_warning(exporter_sdk, tmp_path, caplog):
    """Pose n misses the 1.0-point bar on Metis: it exports, and the warning names the numbers."""
    with caplog.at_level(logging.WARNING):
        out = _model("pose").export(format="axelera", data=str(_dataset(tmp_path)), imgsz=64,
                                    output_path=str(tmp_path / "pn_axelera_model"))
    assert Path(out).exists()
    assert any("yolonas-n pose export to axelera is supported but not validated" in r.getMessage()
               and "1.30 points" in r.getMessage() for r in caplog.records)


def test_size_recipes_reach_the_export(exporter_sdk, tmp_path, monkeypatch):
    """Each per-size recipe changes the export it is keyed on (checked on small
    stand-ins whose size attribute is set to the recipe's key)."""
    import torch
    import yaml

    from libreyolo import LibreMobileNetV4, LibreYOLO4
    from libreyolo.export.axelera import AXELERA_SIZE_RECIPES, size_recipe

    assert size_recipe("yolo4", "detect", "t") == {}
    assert set(AXELERA_SIZE_RECIPES) == {
        ("yolo4", "detect", "b"), ("mobilenetv4", "classify", "s"),
        ("mobilenetv4", "classify", "l"), ("yolonas", "pose", "s"),
    }
    data = str(_dataset(tmp_path))
    # YOLOv4-b: histogram ranges and a [-6, 6] window on the box output.
    torch.manual_seed(0)
    model = LibreYOLO4(None, size="t", nb_classes=3, device="cpu")
    monkeypatch.setattr(model, "size", "b")
    out = model.export(format="axelera", data=data, imgsz=64,
                       output_path=str(tmp_path / "y4b_axelera_model"))
    meta = yaml.safe_load((Path(out) / "metadata.yaml").read_text())
    assert meta["output_windows"] == {"reg": [-6.0, 6.0]}
    assert meta["axelera"]["quantization_scheme"] == "hybrid_per_tensor_per_channel"
    # MobileNetV4-s: no residual-stream transform, histogram ranges.
    import libreyolo.models.mobilenetv4.quantize as q

    calls = []
    original = q.prepare_for_int8
    monkeypatch.setattr(q, "prepare_for_int8", lambda *a, **k: calls.append(1) or original(*a, **k))
    small = LibreMobileNetV4(size="s", nb_classes=5, device="cpu")
    out = small.export(format="axelera", data=data, imgsz=64,
                       output_path=str(tmp_path / "mn4s_axelera_model"))
    meta = yaml.safe_load((Path(out) / "metadata.yaml").read_text())
    assert calls == [] and meta["axelera"]["quantization_scheme"] == "hybrid_per_tensor_per_channel"


def _pad_maxpool_graph(pad_value):
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    pads = numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int64), "pads")
    value = numpy_helper.from_array(np.array(pad_value, np.float32), "value")
    nodes = [
        helper.make_node("LeakyRelu", ["x"], ["act"], alpha=0.1),
        helper.make_node("Pad", ["act", "pads", "value"], ["padded"], mode="constant"),
        helper.make_node("MaxPool", ["padded"], ["y"], kernel_shape=[2, 2], strides=[1, 1]),
    ]
    graph = helper.make_graph(
        nodes,
        "pad_maxpool",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 5, 5])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 5, 5])],
        initializer=[pads, value],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)


def test_merge_pad_into_maxpool_is_exact(tmp_path):
    np = pytest.importorskip("numpy")
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    source = tmp_path / "model.onnx"
    onnx.save(_pad_maxpool_graph(np.finfo(np.float32).min), str(source))

    merged = ax.merge_pad_into_maxpool(str(source), tmp_path)

    assert merged != str(source)
    graph = onnx.load(merged).graph
    assert [node.op_type for node in graph.node] == ["LeakyRelu", "MaxPool"]
    pool = graph.node[1]
    assert list(next(a for a in pool.attribute if a.name == "pads").ints) == [0, 0, 1, 1]
    x = -np.abs(np.random.default_rng(0).normal(size=(1, 2, 5, 5)).astype(np.float32))
    before = ort.InferenceSession(str(source)).run(None, {"x": x})[0]
    after = ort.InferenceSession(merged).run(None, {"x": x})[0]
    np.testing.assert_array_equal(before, after)


def test_merge_pad_into_maxpool_leaves_other_graphs_alone(tmp_path):
    onnx = pytest.importorskip("onnx")
    source = tmp_path / "model.onnx"
    onnx.save(_pad_maxpool_graph(0.0), str(source))  # zero padding is not -inf padding
    before = source.read_bytes()

    assert ax.merge_pad_into_maxpool(str(source), tmp_path) == str(source)
    assert source.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.onnx"]


def test_efficientnetv2_b0_exports_with_histogram_ranges(exporter_sdk, tmp_path, caplog):
    import yaml

    from libreyolo import LibreEfficientNetV2

    assert ax.default_quantization_scheme("efficientnetv2", "classify") == "hybrid_per_tensor_per_channel"
    model = LibreEfficientNetV2(size="b0", nb_classes=5, device="cpu")
    with caplog.at_level(logging.WARNING):
        out = model.export(format="axelera", data=str(_dataset(tmp_path)), imgsz=64,
                           output_path=str(tmp_path / "effb0_axelera_model"))
    meta = yaml.safe_load((Path(out) / "metadata.yaml").read_text())
    assert meta["output_layout"] == "classify_logits_v1"
    assert meta["axelera"]["quantization_scheme"] == "hybrid_per_tensor_per_channel"
    assert any("efficientnetv2-b0 classify export to axelera is supported but not validated" in r.getMessage()
               for r in caplog.records)


@pytest.mark.parametrize("size", ["b1", "b2", "b3"])
def test_efficientnetv2_sizes_that_do_not_compile_are_blocked(tmp_path, size):
    """Structural blocks only: these sizes do not compile (or were never compiled)."""
    from libreyolo import LibreEfficientNetV2

    model = LibreEfficientNetV2(size=size, nb_classes=5, device="cpu")
    with pytest.raises(NotImplementedError, match=rf"efficientnetv2-{size} classify export to axelera is blocked"):
        model.export(format="axelera", data=str(_dataset(tmp_path)), imgsz=64)

