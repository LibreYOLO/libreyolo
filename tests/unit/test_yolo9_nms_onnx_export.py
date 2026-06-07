"""YOLO9 ONNX embedded-NMS export tests.

Validates that ``nms=True`` produces a self-contained detection model whose
single ``(1, max_det, 6)`` output (``[x1, y1, x2, y2, score, class]``) reproduces
the library's own NMS, across fp32 and int8 precisions.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.onnx, pytest.mark.export_backend]

_HAS_ORT = (
    importlib.util.find_spec("onnx") is not None
    and importlib.util.find_spec("onnxruntime") is not None
)

IMG = 96
NC = 3
MAX_DET = 50


def _reference_nms(raw, conf, iou, max_det, offset):
    """Best-class, class-aware NMS reference matching EmbeddedNMSDetector."""
    from torchvision.ops import nms as tv_nms

    pred = torch.from_numpy(raw[0]).transpose(0, 1).float()
    boxes, scores = pred[:, :4], pred[:, 4:]
    cf, cl = scores.max(dim=1)
    keep_mask = cf > conf
    boxes, cf, cl = boxes[keep_mask], cf[keep_mask], cl[keep_mask]
    keep = tv_nms(boxes + cl.float()[:, None] * offset, cf, iou)
    order = torch.argsort(cf[keep], descending=True)[:max_det]
    keep = keep[order]
    return boxes[keep].numpy(), cf[keep].numpy(), cl[keep].float().numpy()


@pytest.mark.skipif(not _HAS_ORT, reason="onnx/onnxruntime not installed")
def test_yolo9_detect_onnx_nms_fp32_matches_reference(tmp_path):
    import onnx
    import onnxruntime as ort

    from libreyolo import LibreYOLO9

    torch.manual_seed(0)
    model = LibreYOLO9(None, size="t", nb_classes=NC, device="cpu")

    path = tmp_path / "LibreYOLO9t_nms.onnx"
    exported = model.export(
        "onnx",
        output_path=str(path),
        imgsz=IMG,
        simplify=False,
        dynamic=False,
        nms=True,
        conf=0.0,
        iou=0.45,
        max_det=MAX_DET,
    )
    assert exported == str(path)

    proto = onnx.load(str(path))
    meta = {p.key: p.value for p in proto.metadata_props}
    assert meta["nms"] == "true"
    assert meta["max_det"] == str(MAX_DET)
    assert meta["model_family"] == "yolo9"
    # Static output shape (1, max_det, 6).
    out_dims = [d.dim_value for d in proto.graph.output[0].type.tensor_type.shape.dim]
    assert out_dims == [1, MAX_DET, 6]

    x = np.random.default_rng(1).random((1, 3, IMG, IMG), dtype=np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"images": x})[0]
    assert out.shape == (1, MAX_DET, 6)

    # Reference NMS on the raw export-mode tensor.
    model.model.eval()
    model.model.head.export = True
    with torch.no_grad():
        raw = model.model(torch.from_numpy(x)).numpy()
    model.model.head.export = False

    ref_b, ref_s, ref_c = _reference_nms(
        raw[None] if raw.ndim == 2 else raw,
        conf=0.0,
        iou=0.45,
        max_det=MAX_DET,
        offset=2.0 * IMG + 1.0,
    )

    valid = out[0][:, 4] > 0
    det = out[0][valid]
    det = det[np.argsort(-det[:, 4])]

    assert det.shape[0] == len(ref_b)
    if len(ref_b):
        order = np.argsort(-ref_s)
        np.testing.assert_allclose(det[:, :4], ref_b[order], atol=1e-3)
        np.testing.assert_allclose(det[:, 4], ref_s[order], atol=1e-4)
        assert (det[:, 5] == ref_c[order]).all()


@pytest.mark.skipif(not _HAS_ORT, reason="onnx/onnxruntime not installed")
def test_yolo9_detect_onnx_nms_requires_batch_one(tmp_path):
    from libreyolo import LibreYOLO9

    model = LibreYOLO9(None, size="t", nb_classes=NC, device="cpu")
    with pytest.raises(NotImplementedError):
        model.export(
            "onnx",
            output_path=str(tmp_path / "bad.onnx"),
            imgsz=IMG,
            simplify=False,
            dynamic=False,
            nms=True,
            batch=2,
        )


@pytest.mark.skipif(not _HAS_ORT, reason="onnx/onnxruntime not installed")
def test_yolo9_detect_onnx_nms_int8_runs(tmp_path):
    import onnx
    import onnxruntime as ort
    from PIL import Image

    from libreyolo import LibreYOLO9

    image_dir = tmp_path / "images" / "train"
    image_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for idx in range(3):
        img = rng.integers(0, 256, size=(IMG, IMG, 3), dtype=np.uint8)
        Image.fromarray(img).save(image_dir / f"{idx}.jpg")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        f"path: {tmp_path.as_posix()}\ntrain: images/train\nval: images/train\n"
        f"nc: {NC}\nnames:\n  0: a\n  1: b\n  2: c\n",
        encoding="utf-8",
    )

    torch.manual_seed(0)
    model = LibreYOLO9(None, size="t", nb_classes=NC, device="cpu")
    fp32 = tmp_path / "m32.onnx"
    int8 = tmp_path / "m8.onnx"
    model.export(
        "onnx",
        output_path=str(fp32),
        imgsz=IMG,
        simplify=False,
        dynamic=False,
        nms=True,
        conf=0.25,
        iou=0.45,
        max_det=MAX_DET,
    )
    exported = model.export(
        "onnx",
        output_path=str(int8),
        imgsz=IMG,
        simplify=False,
        dynamic=False,
        nms=True,
        conf=0.25,
        iou=0.45,
        max_det=MAX_DET,
        int8=True,
        data=str(data_yaml),
    )
    assert exported == str(int8)
    assert int8.stat().st_size < fp32.stat().st_size

    proto = onnx.load(str(int8))
    meta = {p.key: p.value for p in proto.metadata_props}
    assert meta["precision"] == "int8"
    assert meta["nms"] == "true"

    sess = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"images": np.zeros((1, 3, IMG, IMG), dtype=np.float32)})[0]
    assert out.shape == (1, MAX_DET, 6)
