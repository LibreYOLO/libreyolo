"""Regression test for the YOLO9 mixup label-drop bug (issue #484, complaint #2).

``YOLO9MosaicMixupDataset._mixup`` used to vstack the second image's labels
onto the already-padded ``(max_labels, 5)`` block of the first image and then
truncate back to ``max_labels`` — which kept only the first image's block and
silently dropped every object from the second image. Those ~50%-opacity
objects then trained as unlabeled background, poisoning the model whenever
mixup was enabled.

The check is semantic and platform-independent (rather than a byte-exact
golden fixture, which would depend on the beta-distribution RNG and the
platform): after mixup, labels from *both* images must be present.
"""

from __future__ import annotations

import numpy as np
import pytest

from libreyolo.data.augment.yolo9 import (
    YOLO9MosaicMixupDataset,
    YOLO9TrainTransform,
)

pytestmark = pytest.mark.unit

_B_CLASS = 7  # the "second image" always contributes exactly this class


class _StubBDataset:
    """Every image is a single box of class ``_B_CLASS`` — so whatever the
    second image contributes to a mixup is unambiguous."""

    def __init__(self, img_size=(64, 64)):
        self.img_size = img_size

    def __len__(self):
        return 4

    def _label(self):
        # [x1, y1, x2, y2, class] in pixel coords (YOLO9TrainTransform input).
        return np.array([[8.0, 8.0, 40.0, 40.0, float(_B_CLASS)]], dtype=np.float32)

    def load_anno(self, idx):
        return self._label()

    def pull_item(self, idx):
        img = np.full((self.img_size[0], self.img_size[1], 3), 120, dtype=np.uint8)
        return img, self._label(), (img.shape[0], img.shape[1]), idx


def _make_dataset():
    preproc = YOLO9TrainTransform(max_labels=50, flip_prob=0.0, hsv_prob=0.0)
    return YOLO9MosaicMixupDataset(
        _StubBDataset(),
        img_size=(64, 64),
        mosaic=True,
        preproc=preproc,
        enable_mixup=True,
        mosaic_prob=1.0,
        mixup_prob=1.0,
    )


def test_mixup_keeps_labels_from_both_images():
    """The first image's object (class 3) AND the second image's object
    (class ``_B_CLASS``) must both survive the mix."""
    ds = _make_dataset()

    a_class = 3.0
    # First image's labels as YOLO9TrainTransform emits them: padded to
    # max_labels rows, real rows front-packed, padding class = -1,
    # boxes normalized [class, x1, y1, x2, y2].
    labels = np.zeros((50, 5), dtype=np.float32)
    labels[:, 0] = -1
    labels[0] = [a_class, 0.10, 0.10, 0.50, 0.50]
    img = np.zeros((3, 64, 64), dtype=np.float32)

    _mixed_img, out = ds._mixup(img, labels)

    real = out[out[:, 0] >= 0]
    classes = set(np.round(real[:, 0]).astype(int).tolist())

    assert int(a_class) in classes, "first image's label was dropped by mixup"
    assert _B_CLASS in classes, (
        "second image's label was dropped by mixup — the issue #484 "
        "label-drop bug (its objects would train as unlabeled background)"
    )


def test_mixup_keeps_float_images_float():
    """``YOLO9TrainTransform`` hands mixup float32 images in [0, 1]. The blend
    must stay float32: a uint8 cast would truncate nearly every pixel to 0
    and silently wreck every mixup sample (yolo7 enables mixup by default)."""
    ds = _make_dataset()

    labels = np.zeros((50, 5), dtype=np.float32)
    labels[:, 0] = -1
    labels[0] = [3.0, 0.10, 0.10, 0.50, 0.50]
    img = np.ones((3, 64, 64), dtype=np.float32)

    mixed, _out = ds._mixup(img, labels)

    assert mixed.dtype == np.float32
    assert mixed.shape == img.shape
    # Both inputs are uniform (1.0 and the stub partner's 120/255), so the
    # blend is uniform and lies strictly between them.
    value = float(mixed.flat[0])
    assert np.allclose(mixed, value)
    assert 120.0 / 255.0 < value < 1.0


def test_mixup_output_is_padded_to_max_labels():
    """The result must still be a ``(max_labels, 5)`` block with class = -1
    padding, so the collate/loss path is unchanged."""
    ds = _make_dataset()

    labels = np.zeros((50, 5), dtype=np.float32)
    labels[:, 0] = -1
    labels[0] = [3.0, 0.10, 0.10, 0.50, 0.50]
    img = np.zeros((3, 64, 64), dtype=np.float32)

    _mixed_img, out = ds._mixup(img, labels)

    assert out.shape == (50, 5)
    # Exactly two real objects (one per image); the rest padding class = -1.
    assert int((out[:, 0] >= 0).sum()) == 2
    assert np.all(out[out[:, 0] < 0][:, 0] == -1)
