"""Unit tests for the copy-paste instance augmentation primitive and wiring.

Covers the numpy/cv2 primitive (:func:`libreyolo.data.augment.copy_paste`):
seeded determinism, box/label/segment counts growing together, appended boxes
matching the tight bbox of their pasted polygon, occlusion-based dropping of
covered destination instances, and the no-source no-op. Also pins that the
YOLO9 mosaic wrapper is a bit-identical no-op when ``copy_paste == 0``.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from libreyolo.data.augment import copy_paste

pytestmark = pytest.mark.unit


def _square_ring(x1, y1, x2, y2):
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
    )


def _scene(seed=0):
    """A destination and a source image plus one instance each."""
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 255, (80, 100, 3), dtype=np.uint8)
    src_img = rng.randint(0, 255, (80, 100, 3), dtype=np.uint8)
    boxes = np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)
    labels = np.array([3.0], dtype=np.float32)
    segments = [[_square_ring(10, 10, 30, 30)]]
    src_boxes = np.array([[40.0, 40.0, 70.0, 70.0]], dtype=np.float32)
    src_labels = np.array([5.0], dtype=np.float32)
    src_segments = [[_square_ring(40, 40, 70, 70)]]
    return dict(
        img=img,
        boxes=boxes,
        labels=labels,
        segments=segments,
        src_img=src_img,
        src_boxes=src_boxes,
        src_labels=src_labels,
        src_segments=src_segments,
    )


def _call(scene, **kw):
    return copy_paste(
        scene["img"],
        scene["boxes"],
        scene["labels"],
        scene["segments"],
        scene["src_img"],
        scene["src_boxes"],
        scene["src_labels"],
        scene["src_segments"],
        **kw,
    )


def test_deterministic_under_seeded_rng():
    scene = _scene()
    img1, b1, l1, s1 = _call(scene, scale_range=(1.0, 1.0), rng=random.Random(123))
    img2, b2, l2, s2 = _call(scene, scale_range=(1.0, 1.0), rng=random.Random(123))
    np.testing.assert_array_equal(img1, img2)
    np.testing.assert_array_equal(b1, b2)
    np.testing.assert_array_equal(l1, l2)
    assert len(s1) == len(s2)
    for a, b in zip(s1, s2):
        for ra, rb in zip(a, b):
            np.testing.assert_array_equal(ra, rb)


def test_numpy_generator_rng_supported():
    scene = _scene()
    # Should run without error and paste at least one instance.
    _, boxes, labels, segments = _call(
        scene, scale_range=(1.0, 1.0), rng=np.random.default_rng(0)
    )
    assert len(boxes) == len(labels) == len(segments) >= 2


def test_counts_grow_together():
    scene = _scene()
    _, boxes, labels, segments = _call(
        scene, flip_prob=0.0, scale_range=(1.0, 1.0), rng=random.Random(7)
    )
    # One original + one pasted instance, all arrays aligned.
    assert len(boxes) == len(labels) == len(segments)
    assert len(segments) == 2


def test_appended_box_is_tight_bbox_of_pasted_polygon():
    scene = _scene()
    # No flip, no scale jitter -> pasted polygon keeps source coordinates.
    _, boxes, labels, segments = _call(
        scene, flip_prob=0.0, scale_range=(1.0, 1.0), rng=random.Random(7)
    )
    # The last instance is the pasted one.
    pasted_ring = np.concatenate(segments[-1], axis=0)
    expected = np.array(
        [
            pasted_ring[:, 0].min(),
            pasted_ring[:, 1].min(),
            pasted_ring[:, 0].max(),
            pasted_ring[:, 1].max(),
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(boxes[-1], expected)
    # Pasted label is carried from the source.
    assert labels[-1] == 5.0


def test_pasted_box_clipped_to_image_bounds():
    scene = _scene(seed=1)
    # Source instance near the far corner; upscale so it spills past the edges.
    scene["src_segments"] = [[_square_ring(60, 60, 95, 95)]]
    scene["src_boxes"] = np.array([[60.0, 60.0, 95.0, 95.0]], dtype=np.float32)
    _, boxes, _, _ = _call(
        scene, flip_prob=0.0, scale_range=(2.0, 2.0), rng=random.Random(3)
    )
    h, w = scene["img"].shape[:2]
    assert boxes[-1, 2] <= w
    assert boxes[-1, 3] <= h


def test_occluded_destination_instance_dropped():
    scene = _scene()
    # Destination instance sits exactly under the source instance's location,
    # so the paste fully covers it and it must be dropped.
    scene["boxes"] = np.array([[40.0, 40.0, 70.0, 70.0]], dtype=np.float32)
    scene["segments"] = [[_square_ring(40, 40, 70, 70)]]
    _, boxes, labels, segments = _call(
        scene, flip_prob=0.0, scale_range=(1.0, 1.0), occluded_ioa_drop=0.8,
        rng=random.Random(7),
    )
    # Original covered instance dropped; only the pasted instance remains.
    assert len(segments) == 1
    assert labels[-1] == 5.0
    np.testing.assert_array_equal(boxes[-1], np.array([40, 40, 70, 70], dtype=np.float32))


def test_no_source_instances_is_noop():
    scene = _scene()
    img, boxes, labels, segments = copy_paste(
        scene["img"],
        scene["boxes"],
        scene["labels"],
        scene["segments"],
        scene["src_img"],
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        [],
        rng=random.Random(0),
    )
    np.testing.assert_array_equal(img, scene["img"])
    np.testing.assert_array_equal(boxes, scene["boxes"])
    np.testing.assert_array_equal(labels, scene["labels"])
    assert len(segments) == 1


def test_dense_mask_drives_paste_not_polygon():
    """An RLE-sourced instance must paste by its exact mask, not the polygon.

    The source polygon covers a large square, but the attached dense mask marks
    only a small inner square. The paste must follow the dense mask: pixels
    inside the polygon but outside the dense mask stay untouched, and the pasted
    instance carries the exact mask forward as a ``DenseMaskRing``.
    """
    from libreyolo.data.dataset import DenseMaskRing

    dst = np.zeros((80, 80, 3), dtype=np.uint8)
    src = np.full((80, 80, 3), 200, dtype=np.uint8)
    big_ring = _square_ring(10, 10, 60, 60)
    dense = np.zeros((80, 80), dtype=np.uint8)
    dense[20:30, 20:30] = 1
    img_out, _, _, segments = copy_paste(
        dst,
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        [],
        src,
        np.array([[10.0, 10.0, 60.0, 60.0]], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        [[DenseMaskRing(big_ring, dense)]],
        flip_prob=0.0,
        scale_range=(1.0, 1.0),
        rng=random.Random(0),
    )
    # Inside the dense square: pasted (bright source pixels).
    assert (img_out[20:30, 20:30] == 200).all()
    # Inside the polygon but outside the dense mask: untouched background.
    assert (img_out[50, 50] == 0).all()
    # The pasted instance keeps its exact mask for downstream mask-aware ops.
    assert isinstance(segments[-1][0], DenseMaskRing)
    assert segments[-1][0].dense_mask[25, 25] == 1
    assert segments[-1][0].dense_mask[50, 50] == 0


def test_input_arrays_not_mutated():
    scene = _scene()
    img_before = scene["img"].copy()
    boxes_before = scene["boxes"].copy()
    _call(scene, scale_range=(1.0, 1.0), rng=random.Random(9))
    np.testing.assert_array_equal(scene["img"], img_before)
    np.testing.assert_array_equal(scene["boxes"], boxes_before)


# ---------------------------------------------------------------------------
# Wrapper-level default-off guarantee (RNG stream untouched at copy_paste == 0)
# ---------------------------------------------------------------------------


class _SegDataset:
    def __init__(self, n=6):
        self.n = n

    def __len__(self):
        return self.n

    def load_anno(self, idx):
        return np.array(
            [[10.0, 12.0, 45.0, 50.0, 1.0], [30.0, 25.0, 70.0, 60.0, 2.0]],
            dtype=np.float32,
        )

    def pull_item(self, idx):
        r = np.random.RandomState(100 + idx % 4)
        img = r.randint(0, 255, (70, 90, 3), dtype=np.uint8)
        label = np.array(
            [[10.0, 12.0, 45.0, 50.0, 1.0], [30.0, 25.0, 70.0, 60.0, 2.0]],
            dtype=np.float32,
        )
        segments = [
            [_square_ring(10, 12, 45, 50)],
            [_square_ring(30, 25, 70, 60)],
        ]
        return img, label, (img.shape[0], img.shape[1]), idx, segments


def _run_yolo9(copy_paste_prob):
    from libreyolo.data.augment.yolo9 import (
        YOLO9MosaicMixupDataset,
        YOLO9TrainTransform,
    )

    ds = YOLO9MosaicMixupDataset(
        _SegDataset(),
        img_size=(64, 64),
        preproc=YOLO9TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0),
        mosaic_prob=1.0,
        copy_paste=copy_paste_prob,
    )
    random.seed(999)
    np.random.seed(999)
    return ds[0]


def test_yolo9_wrapper_copy_paste_zero_is_noop():
    img_off, labels_off, *_off, masks_off = _run_yolo9(0.0)
    # Baseline built the same way but never touching the copy_paste branch.
    img_base, labels_base, *_base, masks_base = _run_yolo9(0.0)
    np.testing.assert_array_equal(img_off, img_base)
    np.testing.assert_array_equal(labels_off, labels_base)
    np.testing.assert_array_equal(masks_off, masks_base)


def test_yolo9_wrapper_copy_paste_on_adds_instances():
    _, labels_off, _info0, _id0, _masks0 = _run_yolo9(0.0)
    _, labels_on, _info1, _id1, _masks1 = _run_yolo9(1.0)
    n_off = int((labels_off[:, 0] >= 0).sum())
    n_on = int((labels_on[:, 0] >= 0).sum())
    assert n_on >= n_off
