"""Export smoke tests for the native DEIMv2 family."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit

DEIMV2_SIZE_CASES = [
    ("atto", 320, 100),
    ("femto", 416, 150),
    ("pico", 640, 200),
    ("n", 640, 300),
    ("s", 640, 300),
    ("m", 640, 300),
    ("l", 640, 300),
    ("x", 640, 300),
]


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@pytest.mark.parametrize(("size", "input_size", "queries"), DEIMV2_SIZE_CASES)
def test_deimv2_export_wrapper_returns_tuple_and_deploy_is_idempotent(
    size, input_size, queries
):
    """The deploy wrapper may be constructed more than once for repeat exports."""
    from libreyolo import LibreDEIMv2
    from libreyolo.models.deimv2.nn import DEIMv2ExportWrapper

    wrapper = LibreDEIMv2(None, size=size, device="cpu")

    for _ in range(2):
        exp = DEIMv2ExportWrapper(wrapper.model)
        exp.eval()
        with torch.no_grad():
            out = exp(torch.randn(1, 3, input_size, input_size))
        assert isinstance(out, tuple) and len(out) == 2
        assert out[0].shape == (1, queries, 80)
        assert out[1].shape == (1, queries, 4)


@pytest.mark.skipif(
    not (_has_module("onnx") and _has_module("onnxruntime")),
    reason="onnx/onnxruntime not installed",
)
@pytest.mark.parametrize(("size", "input_size", "queries"), DEIMV2_SIZE_CASES)
def test_deimv2_onnx_export_roundtrip_all_sizes(tmp_path, size, input_size, queries):
    """Export every DEIMv2 size to ONNX and run it through ONNX Runtime."""
    import onnx
    import onnxruntime as ort

    from libreyolo import LibreDEIMv2

    model = LibreDEIMv2(None, size=size, device="cpu")
    out_path = tmp_path / f"LibreDEIMv2_{size}.onnx"
    model.export("onnx", output_path=str(out_path), simplify=False, dynamic=False)

    proto = onnx.load(str(out_path))
    output_names = [o.name for o in proto.graph.output]
    assert output_names == ["pred_logits", "pred_boxes"]
    assert max(opset.version for opset in proto.opset_import) >= 17

    metadata = {p.key: p.value for p in proto.metadata_props}
    assert metadata.get("model_family") == "deimv2"
    assert metadata.get("model_size") == size
    assert metadata.get("nb_classes") == "80"
    assert metadata.get("imgsz") == str(input_size)

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    x = (
        np.random.default_rng(0)
        .standard_normal((1, 3, input_size, input_size))
        .astype(np.float32)
    )
    pred_logits, pred_boxes = sess.run(None, {"images": x})
    assert pred_logits.shape == (1, queries, 80)
    assert pred_boxes.shape == (1, queries, 4)

    repeat_path = tmp_path / f"LibreDEIMv2_{size}.repeat.onnx"
    model.export("onnx", output_path=str(repeat_path), simplify=False, dynamic=False)
    assert repeat_path.exists()


@pytest.mark.parametrize(("size", "input_size", "queries"), DEIMV2_SIZE_CASES)
def test_deimv2_torchscript_export_roundtrip_all_sizes(
    tmp_path, size, input_size, queries
):
    """TorchScript export should preserve metadata and tuple output order."""
    from libreyolo import LibreDEIMv2, LibreYOLO

    model = LibreDEIMv2(None, size=size, device="cpu")
    out_path = tmp_path / f"LibreDEIMv2_{size}.torchscript"
    model.export("torchscript", output_path=str(out_path))

    ts = torch.jit.load(str(out_path), map_location="cpu")
    ts.eval()
    with torch.no_grad():
        out = ts(torch.randn(1, 3, input_size, input_size))
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == (1, queries, 80)
    assert out[1].shape == (1, queries, 4)

    backend = LibreYOLO(str(out_path), device="cpu")
    assert backend.model_family == "deimv2"
    assert backend.model_size == size
    assert backend.imgsz == input_size


@pytest.mark.skipif(not _has_module("openvino"), reason="openvino not installed")
@pytest.mark.parametrize(("size", "input_size", "queries"), DEIMV2_SIZE_CASES)
def test_deimv2_openvino_export_backend_outputs_all_sizes(
    tmp_path, size, input_size, queries
):
    """OpenVINO export should load through LibreYOLO and keep raw DETR outputs."""
    from libreyolo import LibreDEIMv2, LibreYOLO

    model = LibreDEIMv2(None, size=size, device="cpu")
    out_dir = tmp_path / f"LibreDEIMv2_{size}_openvino"
    exported = model.export(
        "openvino",
        output_path=str(out_dir),
        simplify=False,
        dynamic=False,
    )

    exported_dir = Path(exported)
    assert (exported_dir / "model.xml").exists()
    assert (exported_dir / "model.bin").exists()
    assert (exported_dir / "metadata.yaml").exists()

    backend = LibreYOLO(exported, device="cpu")
    assert backend.model_family == "deimv2"
    assert backend.model_size == size
    assert backend.imgsz == input_size

    x = (
        np.random.default_rng(1)
        .standard_normal((1, 3, input_size, input_size))
        .astype(np.float32)
    )
    outputs = backend._run_inference(x)
    assert [out.shape for out in outputs] == [(1, queries, 80), (1, queries, 4)]


@pytest.mark.parametrize(("size", "input_size", "_queries"), DEIMV2_SIZE_CASES)
def test_deimv2_tensorrt_metadata_is_deimv2(size, input_size, _queries):
    """TensorRT sidecar metadata must preserve the DEIMv2 family."""
    from libreyolo import LibreDEIMv2
    from libreyolo.export.exporter import TensorRTExporter

    model = LibreDEIMv2(None, size=size, device="cpu")
    metadata = TensorRTExporter(model)._build_metadata(
        precision="fp16",
        dynamic=False,
        onnx_path=f"LibreDEIMv2_{size}.onnx",
    )

    assert metadata["model_family"] == "deimv2"
    assert metadata["model_size"] == size
    assert metadata["imgsz"] == input_size
    assert metadata["exported_from"] == f"LibreDEIMv2_{size}.onnx"


def test_deimv2_ncnn_export_is_blocked(tmp_path):
    """NCNN cannot run DETR-style decoders, so fail before invoking PNNX."""
    from libreyolo import LibreDEIMv2

    model = LibreDEIMv2(None, size="atto", device="cpu")
    with pytest.raises(
        NotImplementedError,
        match="NCNN export is not supported for DEIMv2",
    ):
        model.export("ncnn", output_path=str(tmp_path / "deimv2_ncnn"))
