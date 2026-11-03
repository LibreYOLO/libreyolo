"""Unit tests for MNN runtime loading and inference contracts."""

from __future__ import annotations

import builtins
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from libreyolo.backends.mnn import MNNBackend, _SUPPORTED_FAMILIES
from libreyolo.models import LibreYOLO

pytestmark = pytest.mark.unit


class _FakeVar:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.shape = list(self.value.shape)

    def read(self):
        return self.value


class _FakeModule:
    def __init__(self, outputs):
        self.outputs = outputs
        self.inputs = []

    def forward(self, inputs):
        self.inputs.append(inputs[0].value.copy())
        return [_FakeVar(output) for output in self.outputs]


def _install_fake_mnn(monkeypatch, outputs):
    module = _FakeModule(outputs)
    calls = {}

    def create_runtime_manager(config):
        calls["config"] = config
        return "runtime-manager"

    def load_module_from_file(
        path,
        input_names,
        output_names,
        *,
        runtime_manager,
        dynamic,
        shape_mutable,
    ):
        calls["load"] = {
            "path": path,
            "input_names": input_names,
            "output_names": output_names,
            "runtime_manager": runtime_manager,
            "dynamic": dynamic,
            "shape_mutable": shape_mutable,
        }
        return module

    fake = types.SimpleNamespace(
        nn=types.SimpleNamespace(
            create_runtime_manager=create_runtime_manager,
            load_module_from_file=load_module_from_file,
        ),
        expr=types.SimpleNamespace(
            NCHW="nchw",
            float="float32",
            const=lambda value, shape, layout, dtype: _FakeVar(
                np.asarray(value).reshape(shape)
            ),
            convert=lambda value, layout: value,
        ),
    )
    monkeypatch.setitem(sys.modules, "MNN", fake)
    return module, calls


def _write_artifact(
    tmp_path: Path,
    *,
    family: str = "yolo9",
    batch: int = 2,
    outputs: tuple[str, ...] = ("boxes", "logits"),
) -> Path:
    path = tmp_path / "model.mnn"
    path.write_bytes(b"mnn")
    metadata = {
        "schema_version": "1.0",
        "format": "mnn",
        "dynamic": False,
        "precision": "fp32",
        "model_family": family,
        "size": "t" if family == "yolo9" else "n",
        "model_size": "t" if family == "yolo9" else "n",
        "task": "detect",
        "supported_tasks": ["detect"],
        "default_task": "detect",
        "nc": 2,
        "names": {"0": "a", "1": "b"},
        "imgsz": 8,
        "imgsz_h": 8,
        "imgsz_w": 8,
        "mnn_backend": "cpu",
        "mnn_version": "3.6.1",
        "mnn_input_names": ["images"],
        "mnn_output_names": list(outputs),
        "mnn_input_shape": [batch, 3, 8, 8],
        "mnn_batch": batch,
    }
    Path(f"{path}.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_backend_loads_cpu_module_and_preserves_output_order(tmp_path, monkeypatch):
    first = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    second = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    fake_module, calls = _install_fake_mnn(monkeypatch, (first, second))
    path = _write_artifact(tmp_path)

    backend = MNNBackend(str(path))
    outputs = backend._run_inference(np.ones((1, 3, 8, 8), dtype=np.float32))

    assert backend.device == "cpu"
    assert backend.model_family == "yolo9"
    assert backend.task == "detect"
    assert backend.nb_classes == 2
    assert backend.names == {0: "a", 1: "b"}
    assert backend.input_names == ["images"]
    assert backend.output_names == ["boxes", "logits"]
    # The backend clamps threads to min(cpu_count, 4); CI runners can have
    # fewer than 4 cores, so mirror the clamp instead of hardcoding 4.
    expected_threads = min(max(os.cpu_count() or 1, 1), 4)
    assert calls["config"] == (
        {"backend": 0, "precision": 1, "numThread": expected_threads},
    )
    assert calls["load"]["dynamic"] is False
    assert calls["load"]["shape_mutable"] is False
    assert fake_module.inputs[0].shape == (2, 3, 8, 8)
    np.testing.assert_array_equal(fake_module.inputs[0][0], 1.0)
    np.testing.assert_array_equal(fake_module.inputs[0][1], 0.0)
    np.testing.assert_array_equal(outputs[0], first[:1])
    np.testing.assert_array_equal(outputs[1], second[:1])


def test_backend_routes_rfdetr_outputs_through_shared_parser(tmp_path, monkeypatch):
    boxes = np.array([[[0.5, 0.5, 0.5, 0.5]]], dtype=np.float32)
    logits = np.array([[[8.0, -8.0]]], dtype=np.float32)
    _install_fake_mnn(monkeypatch, (boxes, logits))
    backend = MNNBackend(str(_write_artifact(tmp_path, family="rfdetr", batch=1)))

    parsed_boxes, scores, classes, masks = backend._parse_outputs(
        backend._run_inference(np.zeros((1, 3, 8, 8), dtype=np.float32)),
        8,
        (100, 80),
        conf=0.25,
        max_det=10,
    )

    np.testing.assert_allclose(parsed_boxes, [[25.0, 20.0, 75.0, 60.0]])
    assert scores[0] > 0.99
    np.testing.assert_array_equal(classes, [0])
    assert masks is None


@pytest.mark.parametrize("family", sorted(_SUPPORTED_FAMILIES))
def test_backend_accepts_every_implemented_detection_family(tmp_path, monkeypatch, family):
    _install_fake_mnn(monkeypatch, (np.zeros((1, 1, 6), dtype=np.float32),))

    backend = MNNBackend(str(_write_artifact(tmp_path, family=family, batch=1)))

    assert backend.model_family == family
    assert backend.task == "detect"


def test_backend_requires_sidecar(tmp_path, monkeypatch):
    _install_fake_mnn(monkeypatch, ())
    path = tmp_path / "missing-sidecar.mnn"
    path.write_bytes(b"mnn")

    with pytest.raises(FileNotFoundError, match="metadata sidecar"):
        MNNBackend(str(path))


def test_backend_reports_optional_dependency_install_hint(tmp_path, monkeypatch):
    path = _write_artifact(tmp_path, batch=1)
    real_import = builtins.__import__

    def missing_mnn(name, *args, **kwargs):
        if name == "MNN":
            raise ImportError("missing MNN")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "MNN", raising=False)
    monkeypatch.setattr(builtins, "__import__", missing_mnn)
    with pytest.raises(ImportError, match=r"pip install libreyolo\[mnn\]"):
        MNNBackend(str(path))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("dynamic", True, "dynamic=false"),
        ("precision", "fp16", "precision='fp32'"),
        ("model_family", "yolox", "no runtime contract"),
        ("task", "segment", "Supported tasks: detect"),
        ("mnn_backend", "opencl", "mnn_backend='cpu'"),
        ("names", {"1": "a", "2": "b"}, "keys must cover the range"),
        ("mnn_input_shape", [1, 3, -1, 8], "fixed positive NCHW"),
        ("mnn_batch", 1, "does not match"),
    ],
)
def test_backend_rejects_invalid_contract(tmp_path, monkeypatch, key, value, message):
    _install_fake_mnn(monkeypatch, ())
    path = _write_artifact(tmp_path)
    sidecar = Path(f"{path}.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata[key] = value
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises((ValueError, NotImplementedError), match=message):
        MNNBackend(str(path))


def test_backend_rejects_mismatched_nb_classes_override(tmp_path, monkeypatch):
    _install_fake_mnn(monkeypatch, ())
    path = _write_artifact(tmp_path)

    with pytest.raises(ValueError, match="nb_classes override does not match"):
        MNNBackend(str(path), nb_classes=3)


def test_libreyolo_routes_mnn_suffix_to_backend(tmp_path, monkeypatch):
    path = _write_artifact(tmp_path, batch=1)
    sentinel = object()
    calls = {}

    def fake_backend(model_path, nb_classes=None, device="auto", task=None):
        calls.update(
            model_path=model_path,
            nb_classes=nb_classes,
            device=device,
            task=task,
        )
        return sentinel

    monkeypatch.setattr("libreyolo.backends.mnn.MNNBackend", fake_backend)
    result = LibreYOLO(str(path), nb_classes=7, device="cuda", task="detect")

    assert result is sentinel
    assert calls == {
        "model_path": str(path),
        "nb_classes": 7,
        "device": "cuda",
        "task": "detect",
    }


def test_import_mnn_explains_the_executable_stack_refusal(monkeypatch):
    # glibc 2.41+ refuses MNN's Linux wheel, which is installed; the message
    # must not tell the user to install it again (#989).
    import builtins

    from libreyolo.backends.mnn import import_mnn

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "MNN":
            raise ImportError(
                "_mnncengine.so: cannot enable executable stack as shared "
                "object requires: Invalid argument"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(ImportError, match="GLIBC_TUNABLES=glibc.rtld.execstack=2"):
        import_mnn("MNN export")


def test_import_mnn_points_a_missing_package_at_the_extra(monkeypatch):
    from libreyolo.backends.mnn import import_mnn

    monkeypatch.setitem(sys.modules, "MNN", None)
    with pytest.raises(ImportError, match=r"pip install libreyolo\[mnn\]"):
        import_mnn("MNN inference")
