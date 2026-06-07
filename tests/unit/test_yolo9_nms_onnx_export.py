"""YOLO9 ONNX embedded-NMS export tests.

Validates that ``nms=True`` produces a self-contained detection model whose
first ``(1, max_det, 6)`` output (``[x1, y1, x2, y2, score, class]``) reproduces
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


class _RawExportModel(torch.nn.Module):
    def __init__(self, raw: torch.Tensor):
        super().__init__()
        self.register_buffer("raw", raw)

    def forward(self, x):
        return self.raw + x.sum() * 0.0


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


def test_embedded_nms_caps_candidates_like_native_postprocess(monkeypatch):
    from libreyolo.export import nms as nms_mod

    monkeypatch.setattr(nms_mod, "_YOLO9_MAX_NMS_CANDIDATES", 3)
    raw = torch.zeros(1, 7, 2)
    raw[0, :4, 0] = torch.tensor([0.0, 0.0, 10.0, 10.0])
    raw[0, :4, 1] = torch.tensor([20.0, 20.0, 30.0, 30.0])
    raw[0, 4:, :] = torch.tensor(
        [
            [0.9, 0.6],
            [0.8, 0.5],
            [0.7, 0.4],
        ]
    )

    wrapped = nms_mod.EmbeddedNMSDetector(
        _RawExportModel(raw), conf=0.1, iou=0.45, max_det=10
    )

    out = wrapped(torch.zeros(1, 3, 32, 32))[0][0].detach().numpy()
    det = out[out[:, 4] > 0]

    assert det.shape[0] == 6
    np.testing.assert_allclose(det[:, 4], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    np.testing.assert_array_equal(det[:, 5], [0.0, 1.0, 2.0, 0.0, 1.0, 2.0])


def test_embedded_nms_clips_to_input_canvas_before_suppression():
    from libreyolo.export.nms import EmbeddedNMSDetector

    raw = torch.zeros(1, 5, 2)
    raw[0, :4, 0] = torch.tensor([0.0, 0.0, 1000.0, 1000.0])
    raw[0, :4, 1] = torch.tensor([0.0, 0.0, 32.0, 32.0])
    raw[0, 4, :] = torch.tensor([0.9, 0.8])

    wrapped = EmbeddedNMSDetector(
        _RawExportModel(raw), conf=0.1, iou=0.45, max_det=10
    )

    out = wrapped(torch.zeros(1, 3, 32, 32))[0][0].detach().numpy()
    det = out[out[:, 4] > 0]

    assert det.shape[0] == 1
    np.testing.assert_allclose(det[0, :4], [0.0, 0.0, 32.0, 32.0])
    assert det[0, 4] == pytest.approx(0.9)


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
    assert meta["nms_raw_output"] == "true"
    assert meta["max_det"] == str(MAX_DET)
    assert meta["model_family"] == "yolo9"
    assert [out.name for out in proto.graph.output] == ["output", "raw"]
    out_dims = [
        [d.dim_value for d in out.type.tensor_type.shape.dim]
        for out in proto.graph.output
    ]
    assert out_dims[0] == [1, MAX_DET, 6]
    assert out_dims[1][0] == 1

    x = np.random.default_rng(1).random((1, 3, IMG, IMG), dtype=np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = sess.run(None, {"images": x})
    assert len(outputs) == 2
    out = outputs[0]
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
        original_size=(IMG, IMG),
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
    """LibreYOLO loads an embedded-NMS ONNX and matches native postprocess."""
    import onnxruntime as ort

    from libreyolo import LibreYOLO
    from libreyolo.export.nms import EmbeddedNMSDetector
    from libreyolo.export.onnx import export_onnx

    conf, iou, max_det = 0.1, 0.45, 10
    raw = torch.zeros(1, 5, 2)
    # These two boxes overlap enough on the 100x100 model canvas for graph NMS
    # to suppress the lower-scored one. After mapping to a 200x100 original
    # image and clipping, their IoU falls below 0.45, so native YOLO9 keeps both.
    raw[0, :4, 0] = torch.tensor([14.9, 27.7, 100.0, 100.0])
    raw[0, :4, 1] = torch.tensor([0.0, 0.0, 100.0, 100.0])
    raw[0, 4, :] = torch.tensor([0.9, 0.8])
    wrapped = EmbeddedNMSDetector(
        _RawExportModel(raw), conf=conf, iou=iou, max_det=max_det
    ).eval()
    path = export_onnx(
        wrapped,
        torch.zeros(1, 3, 100, 100),
        output_path=str(tmp_path / "nms.onnx"),
        opset=13,
        simplify=False,
        dynamic=False,
        half=False,
        metadata={
            "model_family": "yolo9",
            "task": "detect",
            "nb_classes": "1",
            "imgsz": "100",
            "imgsz_h": "100",
            "imgsz_w": "100",
            "nms": "true",
            "nms_conf": str(conf),
            "nms_iou": str(iou),
            "max_det": str(max_det),
            "nms_raw_output": "true",
        },
        nms=True,
    )

    graph_outputs = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(
        None, {"images": np.zeros((1, 3, 100, 100), dtype=np.float32)}
    )
    assert graph_outputs[0].shape == (1, max_det, 6)
    assert graph_outputs[1].shape == (1, 5, 2)
    assert graph_outputs[0][0][graph_outputs[0][0][:, 4] > conf].shape[0] == 1

    backend = LibreYOLO(path, device="cpu")
    assert backend.embedded_nms is True
    assert backend.embedded_nms_raw_output_index == 1

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    result = backend.predict(
        img,
        conf=conf,
        iou=iou,
        imgsz=100,
        max_det=max_det,
        color_format="rgb",
    )
    assert result.boxes is not None
    backend_rows = np.concatenate(
        [
            np.asarray(result.boxes.xyxy, np.float32).reshape(-1, 4),
            np.asarray(result.boxes.conf, np.float32).reshape(-1, 1),
            np.asarray(result.boxes.cls, np.float32).reshape(-1, 1),
        ],
        axis=1,
    )
    assert backend_rows.shape == (2, 6)
    np.testing.assert_allclose(backend_rows[:, 4], [0.9, 0.8], rtol=1e-6)
    np.testing.assert_array_equal(backend_rows[:, 5], [0.0, 0.0])
    np.testing.assert_allclose(
        backend_rows[:, :4],
        [[29.8, 55.4, 200.0, 100.0], [0.0, 0.0, 200.0, 100.0]],
        atol=1e-4,
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
        f"nc: {NC}\nnames:\n  0: a\n  1: b\n  2: c\n  3: d\n",
        encoding="utf-8",
    )

    torch.manual_seed(0)
    model = LibreYOLO9(None, size="t", nb_classes=NC, device="cpu")
    for block in model.model.head.cv3:
        convs = [m for m in block.modules() if isinstance(m, torch.nn.Conv2d)]
        convs[-1].bias.data.fill_(4.0)
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
    x = np.random.default_rng(1).random((1, 3, IMG, IMG), dtype=np.float32)
    out = sess.run(None, {"images": x})[0]
    assert out.shape == (1, MAX_DET, 6)
    assert float(out[0, :, 4].max()) > 0.25
