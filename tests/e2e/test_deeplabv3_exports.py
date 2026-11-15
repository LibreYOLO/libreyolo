"""Trained-checkpoint deployment parity for all DeepLabv3 variants."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.supported_backend,
    pytest.mark.deeplabv3,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.slow,
]

CHECKPOINTS = {
    "r50": "a8910db2cb2827ec19fce65a051f4d651bee73f5a46ba8d1c431c0d7042dca7c",
    "r101": "4575b7d5b1b70e9c67225ae76c00f552b29c2e54b07d55cfee8da218a9f41429",
    "mv3": "fb83a67bca845817d816d139af6fb6a4b9d809c0a813ebcfcb1e2a5fbd222682",
}


def _format_param(format_name: str):
    marks = [getattr(pytest.mark, format_name)]
    if format_name == "tensorrt":
        marks.append(pytest.mark.trt)
    return pytest.param(format_name, marks=marks, id=format_name)


FORMATS = tuple(
    _format_param(format_name)
    for format_name in ("onnx", "torchscript", "openvino", "tensorrt")
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.parametrize("size", tuple(CHECKPOINTS))
@pytest.mark.parametrize("format_name", FORMATS)
def test_official_checkpoint_export_runtime_parity(tmp_path, size, format_name):
    if format_name == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    elif format_name == "openvino":
        pytest.importorskip("openvino")
    elif format_name == "tensorrt":
        pytest.importorskip("tensorrt")
        if not torch.cuda.is_available():
            pytest.skip("TensorRT parity requires CUDA")

    huggingface_hub = pytest.importorskip("huggingface_hub")
    from libreyolo import LibreYOLO

    filename = f"LibreDeepLabv3{size}-sem.pt"
    checkpoint = Path(
        huggingface_hub.hf_hub_download(
            repo_id=f"LibreYOLO/LibreDeepLabv3{size}-sem",
            filename=filename,
            token=False,
        )
    )
    assert _sha256(checkpoint) == CHECKPOINTS[size]

    device = "cuda" if format_name == "tensorrt" else "cpu"
    native = LibreYOLO(str(checkpoint), device=device)
    image = Path("tests/data/parkour.jpg")
    input_tensor, _, _, _ = native._preprocess(image, input_size=520)
    with torch.inference_mode():
        expected_raw = native.model(input_tensor.to(device)).detach().cpu().numpy()

    suffixes = {
        "onnx": ".onnx",
        "torchscript": ".torchscript",
        "openvino": "_openvino",
        "tensorrt": ".engine",
    }
    artifact = native.export(
        format=format_name,
        output_path=str(tmp_path / f"deeplabv3-{size}{suffixes[format_name]}"),
        imgsz=520,
        dynamic=False,
        simplify=False,
        half=False,
        int8=False,
    )
    runtime = LibreYOLO(artifact, device=device)
    actual_raw = np.asarray(runtime._run_inference(input_tensor.numpy())[0])
    assert actual_raw.shape == expected_raw.shape == (1, 21, 520, 520)

    raw_argmax_agreement = float(
        np.mean(actual_raw.argmax(axis=1) == expected_raw.argmax(axis=1))
    )
    if format_name == "torchscript":
        np.testing.assert_array_equal(actual_raw, expected_raw)
    elif format_name == "onnx":
        np.testing.assert_allclose(actual_raw, expected_raw, rtol=1e-4, atol=5e-5)
    elif format_name == "openvino":
        # OpenVINO's default CPU precision hint may select reduced-precision
        # kernels even for an FP32 IR. Validate the semantic decision surface.
        assert raw_argmax_agreement > 0.998
    else:
        assert raw_argmax_agreement > 0.9997

    expected = native.predict(image, imgsz=520).semantic_mask.data.cpu()
    actual = runtime.predict(image).semantic_mask.data.cpu()
    public_agreement = float((expected == actual).float().mean())
    if format_name in {"onnx", "torchscript"}:
        assert public_agreement == 1.0
    else:
        assert public_agreement > 0.9997
    assert runtime.FAMILY == "deeplabv3"
    assert runtime.task == "semantic"
    assert runtime.size == size
