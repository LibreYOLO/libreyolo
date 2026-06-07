"""YOLO9 ONNX embedded-NMS export tests.

Validates that ``nms=True`` produces a self-contained detection model whose
single ``(1, max_det, 6)`` output (``[x1, y1, x2, y2, score, class]``) reproduces
the library's own multi-label NMS, and that LibreYOLO loads it back correctly.
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

IMG = 128
NC = 4
MAX_DET = 100


def _set_unmatched(rows_a, rows_b, *, box_tol=1e-3, score_tol=1e-4):
    """Count rows in a with no exact (box, score, class) counterpart in b."""
    used = np.zeros(len(rows_b), dtype=bool)
    unmatched = 0
    for r in rows_a:
        if len(rows_b) == 0:
            unmatched += 1
            continue
        d = np.abs(rows_b[:, :4] - r[:4]).max(axis=1)
        d = np.where(used | (rows_b[:, 5] != r[5]), np.inf, d)
        j = int(np.argmin(d))
        if d[j] < box_tol and abs(rows_b[j, 4] - r[4]) < score_tol:
            used[j] = True
        else:
            unmatched += 1
    return unmatched


@pytest.mark.skipif(not _HAS_ORT, reason="onnx/onnxruntime not installed")
def test_yolo9_detect_onnx_nms_fp32_matches_postprocess(tmp_path):
    import onnx
    import onnxruntime as ort

    from libreyolo import LibreYOLO9
    from libreyolo.models.yolo9.utils import postprocess

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
    out_dims = [d.dim_value for d in proto.graph.output[0].type.tensor_type.shape.dim]
    assert out_dims == [1, MAX_DET, 6]

    x = np.random.default_rng(1).random((1, 3, IMG, IMG), dtype=np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"images": x})[0]
    assert out.shape == (1, MAX_DET, 6)

    # Reference: the library's own multi-label postprocess on the raw tensor.
    model.model.eval()
    model.model.head.export = True
    with torch.no_grad():
        raw = model.model(torch.from_numpy(x))
    model.model.head.export = False
    ref = postprocess(
        {"predictions": raw},
        conf_thres=0.0,
        iou_thres=0.45,
        input_size=IMG,
        original_size=None,
        max_det=MAX_DET,
    )
    ref_rows = np.concatenate(
        [
            np.asarray(ref["boxes"], np.float32).reshape(-1, 4),
            np.asarray(ref["scores"], np.float32).reshape(-1, 1),
            np.asarray(ref["classes"], np.float32).reshape(-1, 1),
        ],
        axis=1,
    )

    det = out[0][out[0][:, 4] > 0][:, :6]
    assert det.shape[0] == ref_rows.shape[0]
    # Same detection set (order may differ on score ties).
    assert _set_unmatched(det, ref_rows) == 0


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
def test_yolo9_detect_onnx_nms_backend_roundtrip(tmp_path):
    """LibreYOLO loads an embedded-NMS ONNX and parses it (no double-NMS)."""
    import onnxruntime as ort

    from libreyolo import LibreYOLO, LibreYOLO9
    from libreyolo.models.yolo9.utils import preprocess_numpy

    torch.manual_seed(0)
    model = LibreYOLO9(None, size="t", nb_classes=NC, device="cpu")
    # Distinct, confident detections so the comparison is unambiguous.
    for block in model.model.head.cv3:
        convs = [m for m in block.modules() if isinstance(m, torch.nn.Conv2d)]
        convs[-1].bias.data = torch.linspace(-1.0, 3.0, NC)

    conf, iou = 0.25, 0.45
    path = model.export(
        "onnx",
        output_path=str(tmp_path / "nms.onnx"),
        imgsz=IMG,
        simplify=False,
        dynamic=False,
        nms=True,
        conf=conf,
        iou=iou,
        max_det=MAX_DET,
    )

    img = np.random.default_rng(3).integers(0, 256, (IMG, IMG, 3), dtype=np.uint8)

    backend = LibreYOLO(path, device="cpu")
    assert backend.embedded_nms is True

    result = backend.predict(img, conf=conf, iou=iou, imgsz=IMG)
    assert result.boxes is not None
    nb = np.asarray(result.boxes.xyxy, np.float32).reshape(-1, 4)

    # Raw embedded output on the same preprocessed blob (square image, ratio 1).
    chw, _ = preprocess_numpy(img, IMG)
    raw = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(
        None, {"images": chw[None].astype(np.float32)}
    )[0]
    raw_valid = raw[0][raw[0][:, 4] > conf]

    # Backend must surface exactly the embedded detections (no re-NMS dropping).
    assert len(nb) == raw_valid.shape[0]
    assert len(nb) > 0
    # Boxes are real xyxy within the image, not transposed/garbled.
    assert (nb[:, 0] >= 0).all() and (nb[:, 2] <= IMG + 1).all()
    assert (nb[:, 1] >= 0).all() and (nb[:, 3] <= IMG + 1).all()


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
        f"nc: {NC}\nnames:\n  0: a\n  1: b\n  2: c\n  3: d\n",
        encoding="utf-8",
    )

    torch.manual_seed(0)
    model = LibreYOLO9(None, size="t", nb_classes=NC, device="cpu")
    fp32 = tmp_path / "m32.onnx"
    int8 = tmp_path / "m8.onnx"
    model.export(
        "onnx", output_path=str(fp32), imgsz=IMG, simplify=False, dynamic=False,
        nms=True, conf=0.25, iou=0.45, max_det=MAX_DET,
    )
    exported = model.export(
        "onnx", output_path=str(int8), imgsz=IMG, simplify=False, dynamic=False,
        nms=True, conf=0.25, iou=0.45, max_det=MAX_DET,
        int8=True, data=str(data_yaml),
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
