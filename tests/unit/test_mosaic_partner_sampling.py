"""Mosaic/mixup partner sampling must prefer annotated images (issue #768).

Partner draws used to be uniform over the whole dataset at three sites
(YOLO9 mosaic, YOLOX mosaic, YOLO9 mixup). On background-heavy datasets that
silently degraded mosaic tiles to unsupervised pixels. The fix retries the
draw (up to 20 times, matching the in-repo YOLOX/YOLO-NAS mixup idiom) until
an annotated partner is found, falling back to the last draw when the whole
dataset is background.

The retry consumes extra RNG draws ONLY after an empty candidate, so on a
fully annotated dataset the augmentation output must stay bitwise identical
to the pre-change behavior. That reference behavior (one uniform draw, no
annotation check) is reimplemented inside this test via a subclass override
and compared byte-for-byte.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from libreyolo.data.augment.yolo9 import (
    YOLO9MosaicMixupDataset,
    YOLO9TrainTransform,
)
from libreyolo.data.augment.yolox import MosaicMixupDataset, TrainTransform

pytestmark = pytest.mark.unit

_BOXES = np.array(
    [[8.0, 10.0, 40.0, 44.0, 0.0], [20.0, 18.0, 60.0, 52.0, 1.0]],
    dtype=np.float32,
)


def _seed():
    random.seed(1234)
    np.random.seed(1234)


class _FakeDataset:
    """Minimal pull_item/load_anno dataset with controllable empty images.

    Images come from per-index RandomState instances so they never touch the
    global RNG stream (bitwise comparisons below depend on that).
    """

    def __init__(self, annotated):
        self.annotated = list(annotated)
        self.pulled = []

    def __len__(self):
        return len(self.annotated)

    def load_anno(self, idx):
        if self.annotated[idx]:
            return _BOXES.copy()
        return np.zeros((0, 5), dtype=np.float32)

    def pull_item(self, idx):
        self.pulled.append(idx)
        rng = np.random.RandomState(1000 + idx)
        img = rng.randint(0, 255, (72, 96, 3), dtype=np.uint8)
        return img, self.load_anno(idx), (72, 96), idx


class _UniformPartnerYOLO9(YOLO9MosaicMixupDataset):
    """Reference (pre-change) sampling: one uniform draw, no annotation check."""

    def _rand_partner_index(self):
        return random.randint(0, len(self.dataset) - 1)


class _UniformPartnerYOLOX(MosaicMixupDataset):
    """Reference (pre-change) sampling: one uniform draw, no annotation check."""

    def _rand_partner_index(self):
        return random.randint(0, len(self.dataset) - 1)


def _make_yolo9(cls, dataset, enable_mixup=False):
    return cls(
        dataset,
        img_size=(64, 64),
        mosaic=True,
        preproc=YOLO9TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0),
        enable_mixup=enable_mixup,
        mosaic_prob=1.0,
        mixup_prob=1.0,
    )


def _make_yolox(cls, dataset, enable_mixup=False):
    return cls(
        dataset,
        img_size=(64, 64),
        mosaic=True,
        preproc=TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0),
        enable_mixup=enable_mixup,
        mosaic_prob=1.0,
        mixup_prob=1.0,
    )


# ---------------------------------------------------------------------------
# RNG parity: fully annotated datasets must be bitwise identical to the
# pre-change uniform sampling (the guard only draws extra RNG after an empty
# candidate).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "guarded_cls,reference_cls,factory",
    [
        (YOLO9MosaicMixupDataset, _UniformPartnerYOLO9, _make_yolo9),
        (MosaicMixupDataset, _UniformPartnerYOLOX, _make_yolox),
    ],
    ids=["yolo9", "yolox"],
)
def test_fully_annotated_dataset_is_bitwise_identical(
    guarded_cls, reference_cls, factory
):
    annotated = [True] * 12
    for item_idx in range(4):
        _seed()
        img_new, labels_new, *_ = factory(guarded_cls, _FakeDataset(annotated), True)[
            item_idx
        ]
        _seed()
        img_ref, labels_ref, *_ = factory(reference_cls, _FakeDataset(annotated), True)[
            item_idx
        ]
        np.testing.assert_array_equal(np.asarray(img_new), np.asarray(img_ref))
        np.testing.assert_array_equal(np.asarray(labels_new), np.asarray(labels_ref))


# ---------------------------------------------------------------------------
# Background-heavy datasets: the guard must (nearly) eliminate empty partners
# while the reference sampling picks them at roughly the dataset's empty rate.
# ---------------------------------------------------------------------------


def _empty_partner_rate(ds, dataset, n_items=30):
    """Fraction of partner pulls (everything after the item's own pull) that
    landed on an image without annotations."""
    empty = total = 0
    _seed()
    for i in range(n_items):
        dataset.pulled = []
        ds[i % len(dataset)]
        for partner_idx in dataset.pulled[1:]:
            total += 1
            if not dataset.annotated[partner_idx]:
                empty += 1
    assert total >= 3 * n_items  # 3 mosaic partners per item at minimum
    return empty / total


@pytest.mark.parametrize(
    "guarded_cls,reference_cls,factory",
    [
        (YOLO9MosaicMixupDataset, _UniformPartnerYOLO9, _make_yolo9),
        (MosaicMixupDataset, _UniformPartnerYOLOX, _make_yolox),
    ],
    ids=["yolo9", "yolox"],
)
def test_background_heavy_dataset_prefers_annotated_partners(
    guarded_cls, reference_cls, factory
):
    # 80% empty: every 5th image is annotated.
    annotated = [i % 5 == 0 for i in range(40)]

    dataset_ref = _FakeDataset(annotated)
    rate_ref = _empty_partner_rate(factory(reference_cls, dataset_ref), dataset_ref)
    dataset_new = _FakeDataset(annotated)
    rate_new = _empty_partner_rate(factory(guarded_cls, dataset_new), dataset_new)

    # Uniform sampling lands on background roughly 80% of the time.
    assert rate_ref > 0.5, f"reference empty-partner rate unexpectedly low: {rate_ref}"
    # With 20 retries at 20% annotated, the miss probability per partner is
    # 0.8**20 (about 1%). Allow a little slack over the sampled draws.
    assert rate_new <= 0.05, (
        f"guarded empty-partner rate {rate_new} did not drop to ~0 "
        f"(reference rate {rate_ref})"
    )


def test_yolo9_mixup_partner_prefers_annotated():
    """The mixup partner draw (yolo9's third guarded site) must pick
    annotated partners on a background-heavy dataset."""
    annotated = [i % 5 == 0 for i in range(40)]
    dataset = _FakeDataset(annotated)
    ds = _make_yolo9(YOLO9MosaicMixupDataset, dataset, enable_mixup=True)

    labels = np.zeros((50, 5), dtype=np.float32)
    labels[:, 0] = -1
    labels[0] = [0.0, 0.10, 0.10, 0.50, 0.50]
    img = np.zeros((3, 64, 64), dtype=np.float32)

    _seed()
    empty = 0
    n_draws = 50
    for _ in range(n_draws):
        dataset.pulled = []
        ds._mixup(img.copy(), labels.copy())
        assert len(dataset.pulled) == 1
        if not dataset.annotated[dataset.pulled[0]]:
            empty += 1
    # The 20-retry guard misses with probability 0.8**20 (about 1%) per draw
    # and then falls back to the last draw by design; uniform sampling would
    # land on background ~80% of the time.
    assert empty / n_draws <= 0.1, (
        f"mixup drew background partners at rate {empty / n_draws} although "
        "annotated images exist"
    )


# ---------------------------------------------------------------------------
# All-background datasets: the fallback keeps the last draw; no infinite
# loop, no crash, and the output carries no phantom labels.
# ---------------------------------------------------------------------------


def test_all_empty_dataset_yolo9_falls_back():
    dataset = _FakeDataset([False] * 8)
    ds = _make_yolo9(YOLO9MosaicMixupDataset, dataset, enable_mixup=True)
    _seed()
    img, labels, _info, _id = ds[0]
    assert np.asarray(img).shape[-2:] == (64, 64)
    labels = np.asarray(labels)
    assert int((labels[:, 0] >= 0).sum()) == 0


def test_all_empty_dataset_yolox_falls_back():
    dataset = _FakeDataset([False] * 8)
    ds = _make_yolox(MosaicMixupDataset, dataset, enable_mixup=True)
    _seed()
    img, labels, _info, _id = ds[0]
    assert np.asarray(img).shape[-2:] == (64, 64)
    # TrainTransform pads with zero rows; no real boxes may appear.
    assert not np.any(np.asarray(labels)[:, 1:])
