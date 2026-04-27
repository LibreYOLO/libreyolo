"""ONNX export + ONNX-backend round-trip tests for ECDet.

Skipped if ``onnx`` / ``onnxruntime`` are not installed, or if the ECDet-S
checkpoint is absent.
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


def test_ecdet_export_wrapper_returns_tuple():
    """Trace-friendly wrapper must return a 2-tuple, not a dict."""
    from libreyolo import LibreECDet
    from libreyolo.models.ecdet.nn import ECDetExportWrapper

    wrapper = LibreECDet(None, size="s", device="cpu")
    exp = ECDetExportWrapper(wrapper.model)
    exp.eval()
    with torch.no_grad():
        out = exp(torch.randn(1, 3, 640, 640))
    assert isinstance(out, tuple) and len(out) == 2
    pred_logits, pred_boxes = out
    assert pred_logits.shape == (1, 300, 80)
    assert pred_boxes.shape == (1, 300, 4)


CKPT_PATH = Path("weights/LibreECDetS.pt")


@pytest.mark.skipif(not CKPT_PATH.exists(), reason=f"{CKPT_PATH} not present")
def test_ecdet_onnx_export_s_roundtrip(tmp_path):
    """Export S to ONNX, run via onnxruntime, verify graph + numeric parity vs PyTorch."""
    import onnx
    import onnxruntime as ort

    from libreyolo import LibreECDet
    from libreyolo.models.ecdet.postprocess import preprocess_numpy

    m = LibreECDet(str(CKPT_PATH), size="s", device="cpu")
    out_path = tmp_path / "LibreECDetS.onnx"
    m.export("onnx", output_path=str(out_path), simplify=False, dynamic=False, opset=17)

    # Graph inspection
    proto = onnx.load(str(out_path))
    output_names = [o.name for o in proto.graph.output]
    assert output_names == ["pred_logits", "pred_boxes"], output_names

    metadata = {p.key: p.value for p in proto.metadata_props}
    assert metadata.get("model_family") == "ecdet"
    assert metadata.get("model_size") == "s"

    # Numeric round-trip: same input → same output between PyTorch and ONNX.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    chw, _ = preprocess_numpy(img, input_size=640)
    blob = chw[None].astype(np.float32)

    with torch.no_grad():
        pt_out = m.model(torch.from_numpy(blob))
    pt_logits = pt_out["pred_logits"].numpy()
    pt_boxes = pt_out["pred_boxes"].numpy()

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ox_logits, ox_boxes = sess.run(["pred_logits", "pred_boxes"], {"images": blob})

    # BN-fusion in deploy() reorders fp32 ops, and onnxruntime's CPU EP uses
    # MLAS kernels with different summation order. Raw-tensor tolerance is
    # ~5e-3; final detections (post sigmoid + top-K) are bit-equivalent in
    # practice — see test_ecdet_onnx_backend_predict.
    assert np.allclose(pt_logits, ox_logits, atol=1e-2), (
        f"max err {np.abs(pt_logits - ox_logits).max():.2e}"
    )
    assert np.allclose(pt_boxes, ox_boxes, atol=1e-2), (
        f"max err {np.abs(pt_boxes - ox_boxes).max():.2e}"
    )


@pytest.mark.skipif(not CKPT_PATH.exists(), reason=f"{CKPT_PATH} not present")
def test_ecdet_onnx_backend_predict(tmp_path):
    """Exported ONNX, loaded through the unified factory, produces matching detections."""
    from libreyolo import LibreYOLO, SAMPLE_IMAGE

    pt = LibreYOLO(str(CKPT_PATH), device="cpu")
    onnx_path = tmp_path / "LibreECDetS.onnx"
    pt.export("onnx", output_path=str(onnx_path), simplify=False, dynamic=False, opset=17)

    ox = LibreYOLO(str(onnx_path), nb_classes=80)
    assert ox.model_family == "ecdet"

    pt_r = pt.predict(SAMPLE_IMAGE, conf=0.3)
    ox_r = ox.predict(SAMPLE_IMAGE, conf=0.3)

    assert len(pt_r.boxes) == len(ox_r.boxes), (len(pt_r.boxes), len(ox_r.boxes))

    pt_top = sorted([float(pt_r.boxes.conf[i].item()) for i in range(len(pt_r.boxes))], reverse=True)
    ox_top = sorted([float(ox_r.boxes.conf[i].item()) for i in range(len(ox_r.boxes))], reverse=True)
    for p, o in zip(pt_top, ox_top):
        assert abs(p - o) < 1e-4, f"conf drift {abs(p - o):.2e}"
