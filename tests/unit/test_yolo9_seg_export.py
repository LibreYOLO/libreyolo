"""Random-weight export smoke tests for YOLO9 segmentation."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None
    or importlib.util.find_spec("onnxruntime") is None,
    reason="onnx/onnxruntime not installed",
)
def test_yolo9_seg_onnx_export_roundtrip(tmp_path):
    import onnx
    import onnxruntime as ort

    from libreyolo import LibreYOLO, LibreYOLO9

    model = LibreYOLO9(None, size="t", nb_classes=2, task="segment", device="cpu")
    out_path = tmp_path / "LibreYOLO9t-seg.onnx"
    model.export(
        "onnx",
        output_path=str(out_path),
        simplify=False,
        dynamic=False,
        imgsz=64,
        opset=17,
    )

    proto = onnx.load(str(out_path))
    output_names = [o.name for o in proto.graph.output]
    assert output_names == ["predictions", "proto", "mask_coeffs"]

    metadata = {p.key: p.value for p in proto.metadata_props}
    assert metadata.get("model_family") == "yolo9"
    assert metadata.get("task") == "segment"
    assert metadata.get("segmentation") == "true"
    assert metadata.get("nb_classes") == "2"

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"images": np.random.randn(1, 3, 64, 64).astype(np.float32)})
    assert [tuple(out.shape) for out in outs] == [
        (1, 6, 84),
        (1, 32, 16, 16),
        (1, 32, 84),
    ]

    loaded = LibreYOLO(str(out_path), device="cpu")
    assert loaded.task == "segment"
    assert loaded.imgsz == 64
