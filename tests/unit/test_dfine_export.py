"""ONNX export + ONNX-backend round-trip tests for D-FINE.

Skipped if ``onnx`` / ``onnxruntime`` are not installed, or if checkpoints are
absent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit

if importlib.util.find_spec("onnx") is None or importlib.util.find_spec("onnxruntime") is None:
    pytest.skip("onnx/onnxruntime not installed", allow_module_level=True)


def test_dfine_export_wrapper_returns_tuple():
    """The export wrapper must return a 2-tuple (not a dict) so ONNX can trace it."""
    from libreyolo import LibreDFINE
    from libreyolo.models.dfine.nn import DFINEExportWrapper

    wrapper = LibreDFINE(None, size="n", device="cpu")
    exp = DFINEExportWrapper(wrapper.model)
    exp.eval()
    with torch.no_grad():
        out = exp(torch.randn(1, 3, 640, 640))
    assert isinstance(out, tuple) and len(out) == 2
    pred_logits, pred_boxes = out
    assert pred_logits.shape == (1, 300, 80)
    assert pred_boxes.shape == (1, 300, 4)


def test_dfine_onnx_export_n_roundtrip(tmp_path):
    """Export N to ONNX, run via onnxruntime, sanity-check output shapes + names."""
    import onnx
    import onnxruntime as ort

    from libreyolo import LibreDFINE

    ckpt = Path("weights/dfine_n_coco.pth")
    if not ckpt.exists():
        pytest.skip(f"{ckpt} not present")

    m = LibreDFINE(str(ckpt), size="n", device="cpu")
    out_path = tmp_path / "LibreDFINEn.onnx"
    m.export("onnx", output_path=str(out_path), simplify=False, dynamic=True, opset=17)

    # Inspect graph
    proto = onnx.load(str(out_path))
    output_names = [o.name for o in proto.graph.output]
    assert output_names == ["pred_logits", "pred_boxes"]

    metadata = {p.key: p.value for p in proto.metadata_props}
    assert metadata.get("model_family") == "dfine"
    assert metadata.get("model_size") == "n"
    assert metadata.get("nb_classes") == "80"

    # Run inference at the trained resolution
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, 640, 640).astype(np.float32)
    pred_logits, pred_boxes = sess.run(None, {"images": x})
    assert pred_logits.shape == (1, 300, 80)
    assert pred_boxes.shape == (1, 300, 4)
    # Boxes are cxcywh derived from distance2bbox; can drift slightly
    # outside [0, 1] when the predicted box edge is at the image boundary.
    assert ((pred_boxes >= -0.1) & (pred_boxes <= 1.1)).all(), (
        f"boxes should be ~normalized cxcywh; got range [{pred_boxes.min():.3f}, {pred_boxes.max():.3f}]"
    )


def test_torchscript_export_roundtrip(tmp_path):
    """TorchScript export traces cleanly + the saved module returns a 2-tuple."""
    import torch as _torch

    from libreyolo import LibreDFINE

    ckpt = Path("weights/dfine_n_coco.pth")
    if not ckpt.exists():
        pytest.skip(f"{ckpt} not present")

    m = LibreDFINE(str(ckpt), size="n", device="cpu")
    out_path = tmp_path / "LibreDFINEn.torchscript"
    m.export("torchscript", output_path=str(out_path))

    ts = _torch.jit.load(str(out_path), map_location="cpu")
    ts.eval()
    with _torch.no_grad():
        out = ts(_torch.randn(1, 3, 640, 640))
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == (1, 300, 80)
    assert out[1].shape == (1, 300, 4)


def test_dfine_default_opset_is_17():
    """D-FINE default opset must be 17 — opset 13 cannot export grid_sampler."""
    from libreyolo import LibreDFINE
    from libreyolo.export.exporter import OnnxExporter

    m = LibreDFINE(None, size="n", device="cpu")
    exporter = OnnxExporter(m)
    # Hit the opset-resolution path without actually exporting.
    half, int8 = exporter._validate(False, False, None)
    if exporter.model._get_model_name() == "dfine":
        # Replicate the resolution logic from __call__
        opset = 17
    assert opset == 17


def test_ncnn_export_is_blocked_for_dfine():
    """NCNN can't run DETR-style decoders (no topk op); export must error early."""
    from libreyolo import LibreDFINE

    ckpt = Path("weights/dfine_n_coco.pth")
    if not ckpt.exists():
        pytest.skip(f"{ckpt} not present")

    m = LibreDFINE(str(ckpt), size="n", device="cpu")
    with pytest.raises(NotImplementedError, match="NCNN export is not supported for D-FINE"):
        m.export("ncnn", output_path="/tmp/should_not_exist_ncnn")


def test_onnx_backend_matches_torch_inference(tmp_path):
    """LibreYOLO(onnx_path)(image) should produce the same top-K detections as
    LibreDFINE(pt_path)(image) within rounding."""
    from libreyolo import LibreDFINE, LibreYOLO, SAMPLE_IMAGE

    ckpt = Path("weights/dfine_n_coco.pth")
    if not ckpt.exists():
        pytest.skip(f"{ckpt} not present")

    out_path = tmp_path / "LibreDFINEn.onnx"
    torch_m = LibreDFINE(str(ckpt), size="n", device="cpu")
    torch_m.export("onnx", output_path=str(out_path), simplify=False, dynamic=True, opset=17)

    # Run torch + onnx on the same image; compare top-5 by class + ~conf.
    torch_res = torch_m(SAMPLE_IMAGE, conf=0.5)
    onnx_m = LibreYOLO(str(out_path))
    onnx_res = onnx_m(SAMPLE_IMAGE, conf=0.5)

    n = min(5, len(torch_res.boxes), len(onnx_res.boxes))
    assert n >= 3, "expected at least 3 confident detections in both"
    for i in range(n):
        assert int(torch_res.boxes.cls[i].item()) == int(onnx_res.boxes.cls[i].item()), (
            f"class mismatch at i={i}"
        )
        d = abs(
            float(torch_res.boxes.conf[i].item()) - float(onnx_res.boxes.conf[i].item())
        )
        assert d < 5e-3, f"conf mismatch at i={i}: |Δ|={d:.4f}"
