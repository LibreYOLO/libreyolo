"""ONNX + TorchScript export tests for YOLOv9 E2E.

Random-weight smoke tests — no checkpoint required. Skipped if onnx /
onnxruntime are not installed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None
    or importlib.util.find_spec("onnxruntime") is None,
    reason="onnx/onnxruntime not installed",
)
def test_yolo9_e2e_onnx_export_t_roundtrip(tmp_path):
    """Export tiny model to ONNX, run via onnxruntime, verify output shape."""
    import onnx
    import onnxruntime as ort

    from libreyolo import LibreYOLO9E2E

    m = LibreYOLO9E2E(None, size="t", device="cpu")
    out_path = tmp_path / "LibreYOLO9E2Et.onnx"
    m.export("onnx", output_path=str(out_path), simplify=False, dynamic=False, opset=17)

    proto = onnx.load(str(out_path))
    metadata = {p.key: p.value for p in proto.metadata_props}
    assert metadata.get("model_family") == "yolo9_e2e"
    assert metadata.get("model_size") == "t"
    assert metadata.get("nb_classes") == "80"

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, 640, 640).astype(np.float32)
    outs = sess.run(None, {"images": x})
    # Inference output is the standard yolo9 (B, 4+nc, num_anchors) tensor;
    # 80*80 + 40*40 + 20*20 = 8400 anchors at 640.
    pred = outs[0]
    assert pred.shape == (1, 84, 8400)


def test_yolo9_e2e_torchscript_export_roundtrip(tmp_path):
    """TorchScript export traces cleanly + the saved module runs."""
    from libreyolo import LibreYOLO9E2E

    m = LibreYOLO9E2E(None, size="t", device="cpu")
    out_path = tmp_path / "LibreYOLO9E2Et.torchscript"
    m.export("torchscript", output_path=str(out_path))

    ts = torch.jit.load(str(out_path), map_location="cpu")
    ts.eval()
    with torch.no_grad():
        out = ts(torch.zeros(1, 3, 640, 640))
    # YOLOv9-style export output: predictions tensor (B, 4+nc, num_anchors)
    pred = out[0] if isinstance(out, (tuple, list)) else out
    assert pred.shape[1] == 84
    assert pred.shape[2] == 8400
