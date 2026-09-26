"""Input-equivalence checks for the flagship YOLO9 and RF-DETR families (#849).

The same picture must give the same detections no matter how it is handed to
``model()``: file path (str or Path), PIL image, RGB NumPy array, or BGR NumPy
array with ``color_format="bgr"``. A list of mixed-size images must give one
result per image, in input order, with boxes in each image's own pixel space,
and batching must not change what any single image returns.

Geometry is checked against transforms whose effect on boxes is known: a 2x
upscale doubles every coordinate and a horizontal flip mirrors x. Weights are
the smallest pretrained checkpoints, so the suite also runs on CPU.

Usage:
    pytest tests/e2e/test_input_equivalence.py -v -m e2e
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

from libreyolo import LibreYOLO

from .conftest import cuda_cleanup, requires_rfdetr

pytestmark = [pytest.mark.e2e, pytest.mark.flagship_nightly]

FLAGSHIP_CASES = [
    pytest.param("LibreYOLO9t.pt", marks=pytest.mark.yolo9, id="yolo9-t"),
    pytest.param(
        "LibreRFDETRn.pt",
        marks=[pytest.mark.rfdetr, requires_rfdetr],
        id="rfdetr-n",
    ),
]

# Lossless inputs (path, PIL, NumPy) must agree exactly on every confident
# detection. Lossy transforms (resampling, flipping, batching) may nudge a
# marginal score across a threshold, so there only strong detections must find
# a partner, and the partner may sit anywhere above the predict threshold.
CONF = 0.25
COMPARE_CONF = 0.5
STRONG_CONF = 0.7
MIN_IOU = 0.9


@pytest.fixture(scope="module", params=FLAGSHIP_CASES)
def model(request):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = LibreYOLO(request.param, device=device)
    yield m
    del m
    cuda_cleanup()


@pytest.fixture(scope="module")
def image(sample_image):
    return Image.open(sample_image).convert("RGB")


def _detections(result, min_conf=COMPARE_CONF):
    """Return (xyxy, cls) numpy arrays of detections at or above ``min_conf``."""
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,), dtype=np.int64)
    xyxy = np.asarray(result.boxes.xyxy.detach().cpu(), dtype=np.float64)
    conf = np.asarray(result.boxes.conf.detach().cpu(), dtype=np.float64)
    cls = np.asarray(result.boxes.cls.detach().cpu(), dtype=np.int64)
    keep = conf >= min_conf
    return xyxy[keep], cls[keep]


def _iou(a, b):
    x1, y1 = np.maximum(a[0], b[0]), np.maximum(a[1], b[1])
    x2, y2 = np.minimum(a[2], b[2]), np.minimum(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _assert_all_matched(src, dst, what, min_iou):
    """Match every ``src`` box to its own same-class ``dst`` box (one-to-one)."""
    unused = set(range(len(dst[0])))
    for box, cls in zip(*src):
        candidates = [(_iou(box, dst[0][j]), j) for j in unused if dst[1][j] == cls]
        best, j = max(candidates, default=(0.0, None))
        assert best >= min_iou, (
            f"{what}: class {cls} box {box.round(1).tolist()} has no unused match "
            f"(best IoU {best:.3f} < {min_iou})"
        )
        unused.remove(j)


def _assert_same_detections(expected, actual, what, min_iou=MIN_IOU):
    """Lossless check: identical confident detections, matched both ways."""
    exp, act = _detections(expected), _detections(actual)
    assert len(exp[0]) > 0, f"{what}: reference produced no confident detections"
    assert len(exp[0]) == len(act[0]), (
        f"{what}: {len(exp[0])} confident detections expected, got {len(act[0])}"
    )
    _assert_all_matched(exp, act, what, min_iou)
    _assert_all_matched(act, exp, what, min_iou)


def _assert_strong_detections_match(expected, actual, what, min_iou=MIN_IOU):
    """Lossy check: every strong detection on either side has a partner.

    ``expected`` is an ``(xyxy, cls)`` pair already mapped into the actual
    image's pixel space, taken at ``CONF``; ``actual`` is a Results object.
    """
    exp_xyxy, exp_cls, exp_conf = expected
    strong_exp = (exp_xyxy[exp_conf >= STRONG_CONF], exp_cls[exp_conf >= STRONG_CONF])
    assert len(strong_exp[0]) > 0, f"{what}: reference produced no strong detections"
    _assert_all_matched(strong_exp, _detections(actual, CONF), what, min_iou)
    _assert_all_matched(
        _detections(actual, STRONG_CONF), (exp_xyxy, exp_cls), what, min_iou
    )


def _raw(result):
    """All detections as (xyxy, cls, conf) numpy arrays."""
    return (
        np.asarray(result.boxes.xyxy.detach().cpu(), dtype=np.float64),
        np.asarray(result.boxes.cls.detach().cpu(), dtype=np.int64),
        np.asarray(result.boxes.conf.detach().cpu(), dtype=np.float64),
    )


def _assert_boxes_inside(result, width, height, what):
    xyxy, _ = _detections(result, min_conf=CONF)
    assert (xyxy[:, [0, 2]] >= -1).all() and (xyxy[:, [0, 2]] <= width + 1).all(), (
        f"{what}: x coordinates fall outside the {width}px-wide original image"
    )
    assert (xyxy[:, [1, 3]] >= -1).all() and (xyxy[:, [1, 3]] <= height + 1).all(), (
        f"{what}: y coordinates fall outside the {height}px-high original image"
    )


def test_input_types_give_same_detections(model, image, sample_image):
    """Path, Path object, PIL, RGB array and BGR array agree."""
    rgb = np.asarray(image)
    sources = {
        "str path": (sample_image, {}),
        "Path": (Path(sample_image), {}),
        "PIL": (image, {}),
        "NumPy RGB": (rgb.copy(), {}),
        "NumPy BGR": (rgb[..., ::-1].copy(), {"color_format": "bgr"}),
    }

    reference = model(image, conf=CONF)
    assert reference.orig_shape == (image.height, image.width)
    for name, (source, kwargs) in sources.items():
        result = model(source, conf=CONF, **kwargs)
        assert result.orig_shape == reference.orig_shape, name
        _assert_same_detections(reference, result, name)


def test_upscaled_image_boxes_scale_with_it(model, image):
    """Boxes come back in original-image pixels, not network-input pixels."""
    reference = model(image, conf=CONF)
    big = image.resize((image.width * 2, image.height * 2), Image.BICUBIC)
    result = model(big, conf=CONF)

    assert result.orig_shape == (big.height, big.width)
    xyxy, cls, conf = _raw(reference)
    _assert_strong_detections_match((xyxy * 2, cls, conf), result, "2x upscale")


def test_flipped_image_boxes_mirror(model, image):
    """A horizontal flip mirrors x without shifting y."""
    reference = model(image, conf=CONF)
    result = model(ImageOps.mirror(image), conf=CONF)

    xyxy, cls, conf = _raw(reference)
    mirrored = xyxy.copy()
    mirrored[:, 0] = image.width - xyxy[:, 2]
    mirrored[:, 2] = image.width - xyxy[:, 0]
    # Flipping changes what the network sees, so allow a looser IoU than for
    # resampling.
    _assert_strong_detections_match(
        (mirrored, cls, conf), result, "horizontal flip", min_iou=0.8
    )


def _mixed_size_images(image):
    return [
        image,
        image.resize((image.width // 2, image.height // 2), Image.BICUBIC),
        image.rotate(90, expand=True),
        ImageOps.pad(image, (image.width + 400, image.height), color=(114, 114, 114)),
    ]


@pytest.mark.parametrize("batch", [1, 4], ids=["batch1", "batch4"])
def test_mixed_size_list_matches_single_calls(model, image, batch):
    """A list returns one result per image, in order, each in its own pixel space."""
    images = _mixed_size_images(image)
    singles = [model(img, conf=CONF) for img in images]
    results = model(images, conf=CONF, batch=batch)

    assert len(results) == len(images)
    for i, (img, single, result) in enumerate(zip(images, singles, results)):
        what = f"image {i} ({img.width}x{img.height}), batch={batch}"
        assert result.orig_shape == (img.height, img.width), what
        _assert_boxes_inside(result, img.width, img.height, what)
        # batch>1 may pick different GPU kernels (TF32), so compare as lossy.
        _assert_strong_detections_match(_raw(single), result, what)


def test_list_order_is_preserved(model, image):
    """Reversing the input list reverses the results."""
    images = _mixed_size_images(image)
    forward = model(images, conf=CONF, batch=len(images))
    backward = model(images[::-1], conf=CONF, batch=len(images))

    assert len(forward) == len(backward) == len(images)
    for i, (fwd, bwd) in enumerate(zip(forward, backward[::-1])):
        what = f"image {i}"
        assert fwd.orig_shape == bwd.orig_shape, what
        _assert_strong_detections_match(_raw(fwd), bwd, what)
