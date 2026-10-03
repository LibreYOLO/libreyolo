"""YOLO9 exported runtimes follow the checkpoint ``letterbox_pad``.

A center-pad checkpoint (official conversions, ``train(letterbox_pad="center")``)
must predict the same boxes through an exported runtime as through the ``.pt``.
Artifacts without the key keep the historical top-left pad.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.yolo9, pytest.mark.export_backend]

_HAS_ORT = (
    importlib.util.find_spec("onnx") is not None
    and importlib.util.find_spec("onnxruntime") is not None
)
# test_export_coreml.py stubs coremltools into sys.modules when it is missing,
# so look for the installed package rather than the module cache.
_HAS_COREML = importlib.machinery.PathFinder.find_spec("coremltools") is not None

IMG = 128
NC = 3
CONF = 0.8


def _image() -> np.ndarray:
    """Synthetic 1280x720 RGB frame with some structure."""
    yy, xx = np.mgrid[0:720, 0:1280]
    img = np.stack(
        [xx * 255 // 1279, yy * 255 // 719, (xx + yy) % 256], axis=-1
    ).astype(np.uint8)
    img[200:500, 300:700] = (255, 40, 40)
    img[100:300, 900:1200] = (20, 200, 60)
    return img


def _tiny_yolo9(letterbox_pad: str):
    """Scratch YOLO9-t whose head emits spread scores and compact boxes.

    Default scratch init gives near-constant scores, which would make any
    parity check vacuous. Rescale only the final head convs.
    """
    from libreyolo import LibreYOLO9
    from libreyolo.preprocess.yolo9 import preprocess_numpy

    torch.manual_seed(0)
    model = LibreYOLO9(None, size="t", nb_classes=NC, device="cpu")
    model.letterbox_pad = letterbox_pad
    model.model.eval()

    head = model.model.head
    finals = [seq[-1] for seq in [*head.anchor_convs, *head.class_convs]]
    feature_std = {}
    hooks = [
        conv.register_forward_hook(
            lambda mod, inp, out: feature_std.__setitem__(mod, float(inp[0].std()))
        )
        for conv in finals
    ]
    canvas, _ = preprocess_numpy(_image(), IMG, letterbox_pad=letterbox_pad)
    with torch.no_grad():
        model.model(torch.from_numpy(canvas)[None])
    for hook in hooks:
        hook.remove()

    gen = torch.Generator().manual_seed(0)
    for conv in finals:
        scale = 2.0 / (conv.in_channels**0.5 * feature_std[conv])
        conv.weight.data = torch.randn(conv.weight.shape, generator=gen) * scale
        conv.bias.data.zero_()
    for seq in head.anchor_convs:
        # Favor short DFL distances so boxes do not all clip to the frame.
        seq[-1].bias.data = (-0.5 * torch.arange(16.0)).repeat(4)
    return model


def _rows(result) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(result.boxes.xyxy, np.float32).reshape(-1, 4),
            np.asarray(result.boxes.conf, np.float32).reshape(-1, 1),
            np.asarray(result.boxes.cls, np.float32).reshape(-1, 1),
        ],
        axis=1,
    )


def _assert_same_detections(expected, actual, *, box_tol=1e-2, score_tol=1e-4):
    exp, act = _rows(expected), _rows(actual)
    assert len(exp) > 0
    assert exp.shape == act.shape
    used = np.zeros(len(act), dtype=bool)
    for row in exp:
        dist = np.abs(act[:, :4] - row[:4]).max(axis=1)
        dist[used | (act[:, 5] != row[5]) | (np.abs(act[:, 4] - row[4]) > score_tol)] = np.inf
        j = int(np.argmin(dist))
        assert dist[j] < box_tol, f"no exported match for {row}"
        used[j] = True


def _export(model, fmt, tmp_path):
    suffix = {"onnx": ".onnx", "torchscript": ".torchscript"}[fmt]
    return model.export(
        fmt,
        output_path=str(tmp_path / f"yolo9t{suffix}"),
        imgsz=IMG,
        simplify=False,
        dynamic=False,
    )


@pytest.mark.parametrize(
    "fmt",
    [
        pytest.param(
            "onnx",
            marks=[
                pytest.mark.onnx,
                pytest.mark.skipif(not _HAS_ORT, reason="onnx/onnxruntime not installed"),
            ],
        ),
        pytest.param("torchscript", marks=pytest.mark.torchscript),
    ],
)
def test_center_pad_export_matches_pt_on_non_square_image(tmp_path, fmt):
    from libreyolo import LibreYOLO

    model = _tiny_yolo9("center")
    image = _image()
    expected = model.predict(image, conf=CONF, imgsz=IMG, color_format="rgb")

    backend = LibreYOLO(_export(model, fmt, tmp_path), device="cpu")
    actual = backend.predict(image, conf=CONF, imgsz=IMG, color_format="rgb")

    _assert_same_detections(expected, actual)
    assert backend.letterbox_pad == "center"
    assert backend._get_val_preprocessor().letterbox_pad == "center"


@pytest.mark.coreml
@pytest.mark.skipif(sys.platform != "darwin", reason="CoreML inference needs macOS")
@pytest.mark.skipif(not _HAS_COREML, reason="coremltools not installed")
@pytest.mark.parametrize("pad", ["topleft", "center"])
def test_coreml_export_matches_pt_on_non_square_image(tmp_path, pad):
    from libreyolo import LibreYOLO

    model = _tiny_yolo9(pad)
    image = _image()
    expected = model.predict(image, conf=CONF, imgsz=IMG, color_format="rgb")

    path = model.export(
        "coreml", output_path=str(tmp_path / "yolo9t.mlpackage"), imgsz=IMG
    )
    backend = LibreYOLO(path, device="cpu")
    actual = backend.predict(image, conf=CONF, imgsz=IMG, color_format="rgb")

    assert backend.letterbox_pad == pad
    _assert_same_detections(expected, actual)


@pytest.mark.onnx
@pytest.mark.skipif(not _HAS_ORT, reason="onnx/onnxruntime not installed")
def test_onnx_metadata_records_letterbox_pad(tmp_path):
    import onnx

    for pad in ("center", "topleft"):
        path = _export(_tiny_yolo9(pad), "onnx", tmp_path)
        meta = {p.key: p.value for p in onnx.load(path).metadata_props}
        assert meta["letterbox_pad"] == pad


@pytest.mark.onnx
@pytest.mark.skipif(not _HAS_ORT, reason="onnx/onnxruntime not installed")
def test_export_without_letterbox_pad_metadata_stays_topleft(tmp_path):
    """Artifacts written before the key existed keep top-left geometry."""
    import onnx

    from libreyolo import LibreYOLO

    model = _tiny_yolo9("center")
    path = _export(model, "onnx", tmp_path)
    proto = onnx.load(path)
    kept = [p for p in proto.metadata_props if p.key != "letterbox_pad"]
    del proto.metadata_props[:]
    proto.metadata_props.extend(kept)
    legacy = tmp_path / "legacy.onnx"
    onnx.save(proto, str(legacy))

    backend = LibreYOLO(str(legacy), device="cpu")
    assert backend.letterbox_pad == "topleft"

    image = _image()
    model.letterbox_pad = "topleft"
    expected = model.predict(image, conf=CONF, imgsz=IMG, color_format="rgb")
    actual = backend.predict(image, conf=CONF, imgsz=IMG, color_format="rgb")
    _assert_same_detections(expected, actual)


def test_invalid_letterbox_pad_metadata_is_rejected():
    from libreyolo.backends.base import _read_runtime_metadata, _validate_letterbox_pad

    assert "letterbox_pad" not in _read_runtime_metadata({})
    assert _read_runtime_metadata({"letterbox_pad": "center"})["letterbox_pad"] == "center"
    assert _validate_letterbox_pad(None) == "topleft"
    with pytest.raises(ValueError, match="letterbox_pad"):
        _validate_letterbox_pad("middle")


def test_int8_calibration_uses_checkpoint_pad():
    from libreyolo.preprocess.yolo9 import preprocess_numpy

    model = _tiny_yolo9("center")
    image = _image()
    calibrated, _ = model._get_preprocess_numpy()(image, IMG)
    expected, _ = preprocess_numpy(image, IMG, letterbox_pad="center")
    np.testing.assert_array_equal(calibrated, expected)


def test_deepstream_sidecar_uses_symmetric_padding_for_center(tmp_path):
    from pathlib import Path

    from libreyolo.export.deepstream import write_deepstream_sidecars

    onnx_path = tmp_path / "yolo9.onnx"
    onnx_path.write_bytes(b"stub")
    configs = {}
    for pad in (None, "topleft", "center"):
        config_path, _ = write_deepstream_sidecars(
            str(onnx_path),
            model_family="yolo9",
            class_names=["a"],
            imgsz=(640, 640),
            batch=1,
            precision="fp32",
            letterbox_pad=pad,
        )
        configs[pad] = Path(config_path).read_text()

    assert "symmetric-padding=0" in configs[None]
    assert "symmetric-padding=0" in configs["topleft"]
    assert "symmetric-padding=1" in configs["center"]
    assert "maintain-aspect-ratio=1" in configs["center"]
