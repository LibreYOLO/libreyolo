"""ONNX export + ONNX-backend round-trip tests for DEIM.

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

if (
    importlib.util.find_spec("onnx") is None
    or importlib.util.find_spec("onnxruntime") is None
):
    pytest.skip("onnx/onnxruntime not installed", allow_module_level=True)


def test_deim_onnx_export_n_roundtrip(tmp_path):
    """Export N to ONNX, run via onnxruntime, sanity-check output shapes + names."""
    import onnx
    import onnxruntime as ort

    from libreyolo import LibreDEIM

    ckpt = Path("weights/LibreDEIMn.pt")
    if not ckpt.exists():
        pytest.skip(f"{ckpt} not present")

    m = LibreDEIM(str(ckpt), size="n", device="cpu")
    out_path = tmp_path / "LibreDEIMn.onnx"
    m.export("onnx", output_path=str(out_path), simplify=False, dynamic=True, opset=17)

    proto = onnx.load(str(out_path))
    output_names = [o.name for o in proto.graph.output]
    assert output_names == ["pred_logits", "pred_boxes"]

    metadata = {p.key: p.value for p in proto.metadata_props}
    assert metadata.get("model_family") == "deim"
    assert metadata.get("model_size") == "n"
    assert metadata.get("nb_classes") == "80"

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, 640, 640).astype(np.float32)
    pred_logits, pred_boxes = sess.run(None, {"images": x})
    assert pred_logits.shape == (1, 300, 80)
    assert pred_boxes.shape == (1, 300, 4)
    assert ((pred_boxes >= -0.1) & (pred_boxes <= 1.1)).all(), (
        f"boxes should be ~normalized cxcywh; got range "
        f"[{pred_boxes.min():.3f}, {pred_boxes.max():.3f}]"
    )


def test_deim_torchscript_export_roundtrip(tmp_path):
    """TorchScript export traces cleanly + the saved module returns a 2-tuple."""
    from libreyolo import LibreDEIM

    ckpt = Path("weights/LibreDEIMn.pt")
    if not ckpt.exists():
        pytest.skip(f"{ckpt} not present")

    m = LibreDEIM(str(ckpt), size="n", device="cpu")
    out_path = tmp_path / "LibreDEIMn.torchscript"
    m.export("torchscript", output_path=str(out_path))

    ts = torch.jit.load(str(out_path), map_location="cpu")
    ts.eval()
    with torch.no_grad():
        out = ts(torch.randn(1, 3, 640, 640))
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == (1, 300, 80)
    assert out[1].shape == (1, 300, 4)


def test_deim_ncnn_export_is_blocked():
    """NCNN can't run DETR-style decoders; export must error early."""
    from libreyolo import LibreDEIM

    ckpt = Path("weights/LibreDEIMn.pt")
    if not ckpt.exists():
        pytest.skip(f"{ckpt} not present")

    m = LibreDEIM(str(ckpt), size="n", device="cpu")
    with pytest.raises(NotImplementedError, match="NCNN export is not supported for DEIM"):
        m.export("ncnn", output_path="/tmp/should_not_exist_ncnn")


def test_deim_openvino_export_backend_predict(tmp_path):
    """OpenVINO export should load through LibreYOLO and run inference."""
    if importlib.util.find_spec("openvino") is None:
        pytest.skip("openvino not installed")

    from libreyolo import LibreDEIM, LibreYOLO, SAMPLE_IMAGE

    ckpt = Path("weights/LibreDEIMn.pt")
    if not ckpt.exists():
        pytest.skip(f"{ckpt} not present")

    out_path = tmp_path / "LibreDEIMn_openvino"
    m = LibreDEIM(str(ckpt), size="n", device="cpu")
    exported = m.export(
        "openvino",
        output_path=str(out_path),
        simplify=False,
        dynamic=False,
        opset=17,
    )

    exported_dir = Path(exported)
    assert (exported_dir / "model.xml").exists()
    assert (exported_dir / "model.bin").exists()
    assert (exported_dir / "metadata.yaml").exists()

    ov_model = LibreYOLO(exported, device="cpu")
    assert ov_model.model_family == "deim"
    result = ov_model(SAMPLE_IMAGE, conf=0.5)
    assert len(result.boxes) >= 3


def test_deim_tensorrt_metadata_is_deim():
    """TensorRT sidecar metadata must preserve the DEIM family."""
    from libreyolo import LibreDEIM
    from libreyolo.export.exporter import TensorRTExporter

    m = LibreDEIM(None, size="n", device="cpu")
    metadata = TensorRTExporter(m)._build_metadata(
        precision="fp16",
        dynamic=False,
        onnx_path="LibreDEIMn.onnx",
    )

    assert metadata["model_family"] == "deim"
    assert metadata["model_size"] == "n"
    assert metadata["exported_from"] == "LibreDEIMn.onnx"


def test_deim_onnx_backend_matches_torch_inference(tmp_path):
    """LibreYOLO(onnx_path)(image) should match PyTorch top-K detections."""
    from libreyolo import LibreDEIM, LibreYOLO, SAMPLE_IMAGE

    ckpt = Path("weights/LibreDEIMn.pt")
    if not ckpt.exists():
        pytest.skip(f"{ckpt} not present")

    out_path = tmp_path / "LibreDEIMn.onnx"
    torch_m = LibreDEIM(str(ckpt), size="n", device="cpu")
    torch_m.export(
        "onnx", output_path=str(out_path), simplify=False, dynamic=True, opset=17
    )

    torch_res = torch_m(SAMPLE_IMAGE, conf=0.5)
    onnx_m = LibreYOLO(str(out_path))
    onnx_res = onnx_m(SAMPLE_IMAGE, conf=0.5)

    n = min(5, len(torch_res.boxes), len(onnx_res.boxes))
    assert n >= 3, "expected at least 3 confident detections in both"
    for i in range(n):
        assert int(torch_res.boxes.cls[i].item()) == int(
            onnx_res.boxes.cls[i].item()
        ), f"class mismatch at i={i}"
        d = abs(
            float(torch_res.boxes.conf[i].item()) - float(onnx_res.boxes.conf[i].item())
        )
        assert d < 5e-3, f"conf mismatch at i={i}: |delta|={d:.4f}"
