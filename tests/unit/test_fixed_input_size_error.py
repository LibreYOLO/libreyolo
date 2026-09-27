"""Fixed-shape exports run at another imgsz fail with a LibreYOLO error.

The runtimes' own errors (onnxruntime INVALID_ARGUMENT, a TensorRT buffer
copy, OpenVINO and CoreML shape errors) do not say which size the artifact was
exported at. The backend names both sizes before calling the runtime.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_HAS_ORT = (
    importlib.util.find_spec("onnx") is not None
    and importlib.util.find_spec("onnxruntime") is not None
)
# test_export_coreml.py stubs coremltools into sys.modules when it is missing,
# and find_spec() on that stub raises; look for the installed package instead.
_HAS_COREML = importlib.machinery.PathFinder.find_spec("coremltools") is not None


def _backend(cls, fixed_hw, **attrs):
    backend = cls.__new__(cls)
    backend._fixed_input_hw = fixed_hw
    for key, value in attrs.items():
        setattr(backend, key, value)
    return backend


def _tensorrt():
    from libreyolo.backends.tensorrt import TensorRTBackend

    return TensorRTBackend, {"_infer": MagicMock()}


def _openvino():
    from libreyolo.backends.openvino import OpenVINOBackend

    return OpenVINOBackend, {"compiled_model": MagicMock()}


def _coreml():
    from libreyolo.backends.coreml import CoreMLBackend

    return CoreMLBackend, {"model": MagicMock(), "output_names": []}


def _onnx():
    from libreyolo.backends.onnx import OnnxBackend

    return OnnxBackend, {"session": MagicMock(), "input_name": "images"}


@pytest.mark.parametrize(
    "make", [_onnx, _tensorrt, _openvino, _coreml],
    ids=["onnx", "tensorrt", "openvino", "coreml"],
)
def test_mismatched_input_names_both_sizes_before_the_runtime_runs(make):
    cls, attrs = make()
    backend = _backend(cls, (64, 64), **attrs)
    runtime = next(iter(attrs.values()))

    with pytest.raises(ValueError, match=r"fixed 64x64 input.*96x128.*imgsz=64"):
        backend._run_inference(np.zeros((1, 3, 96, 128), np.float32))
    runtime.assert_not_called()
    getattr(runtime, "run", MagicMock()).assert_not_called()
    getattr(runtime, "predict", MagicMock()).assert_not_called()


def test_rectangular_export_suggests_its_tuple_imgsz():
    from libreyolo.backends.onnx import OnnxBackend

    backend = _backend(OnnxBackend, (32, 64), session=MagicMock(), input_name="x")
    with pytest.raises(ValueError, match=r"fixed 32x64 input.*imgsz=\(32, 64\)"):
        backend._run_inference(np.zeros((1, 3, 64, 64), np.float32))


@pytest.mark.onnx
@pytest.mark.skipif(not _HAS_ORT, reason="onnx/onnxruntime not installed")
def test_fixed_onnx_export_predicted_at_another_imgsz(tmp_path):
    from libreyolo import LibreYOLO, LibreYOLO9

    model = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    path = model.export(
        "onnx",
        output_path=str(tmp_path / "y9.onnx"),
        imgsz=64,
        dynamic=False,
        simplify=False,
    )
    backend = LibreYOLO(path, device="cpu")
    image = np.zeros((100, 120, 3), np.uint8)

    assert backend._fixed_input_hw == (64, 64)
    backend.predict(image, imgsz=64)
    with pytest.raises(ValueError, match=r"fixed 64x64 input.*96x96"):
        backend.predict(image, imgsz=96)


@pytest.mark.coreml
@pytest.mark.skipif(sys.platform != "darwin", reason="CoreML inference needs macOS")
@pytest.mark.skipif(not _HAS_COREML, reason="coremltools not installed")
def test_coreml_export_records_its_fixed_input(tmp_path):
    from libreyolo import LibreYOLO, LibreYOLO9

    model = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    path = model.export(
        "coreml", output_path=str(tmp_path / "y9.mlpackage"), imgsz=64
    )
    backend = LibreYOLO(path, device="cpu")

    assert backend._fixed_input_hw == (64, 64)
    with pytest.raises(ValueError, match=r"fixed 64x64 input.*96x96"):
        backend.predict(np.zeros((100, 120, 3), np.uint8), imgsz=96)
