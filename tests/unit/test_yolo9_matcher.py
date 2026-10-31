"""YOLO9 label assignment, one image at a time (issue #935).

The task-aligned matcher assigns each image on its own boxes, so its memory
follows the densest image instead of the batch size times ``max_labels``. The
result must be exactly the batched match over the padded targets it replaces.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from libreyolo.data.augment import yolo9 as yolo9_augment
from libreyolo.models.yolo9.loss import YOLO9Loss
from libreyolo.models.yolo9.transforms import YOLO9TrainTransform

pytestmark = pytest.mark.unit

IMAGE = 160
NUM_CLASSES = 3


def _loss(strides=(8, 16, 32)):
    return YOLO9Loss(
        num_classes=NUM_CLASSES,
        reg_max=16,
        strides=list(strides),
        image_size=[IMAGE, IMAGE],
        device=torch.device("cpu"),
        distributed_normalize=False,
    )


def _targets(counts, max_labels, duplicate=False, seed=0):
    """Pixel targets ``[B, max_labels, 5]``, class -1 padding after the boxes."""
    g = torch.Generator().manual_seed(seed)
    targets = torch.zeros(len(counts), max_labels, 5)
    targets[..., 0] = -1
    for i, n in enumerate(counts):
        xy = torch.rand(n, 2, generator=g) * (IMAGE - 24)
        wh = 4 + torch.rand(n, 2, generator=g) * 20
        targets[i, :n, 0] = torch.randint(NUM_CLASSES, (n,), generator=g).float()
        targets[i, :n, 1:3] = xy
        targets[i, :n, 3:5] = xy + wh
        if duplicate and n > 1:
            targets[i, 1] = targets[i, 0]
    return targets


def _predictions(loss, batch, seed=1):
    g = torch.Generator().manual_seed(seed)
    anchors = loss.vec2box.anchor_grid
    half = 2 + torch.rand(batch, len(anchors), 2, generator=g) * 12
    boxes = torch.cat([anchors - half, anchors + half], dim=-1)
    logits = torch.randn(batch, len(anchors), NUM_CLASSES, generator=g)
    return logits, boxes


@pytest.mark.parametrize("strides", [(8, 16, 32), (4, 8, 16, 32)], ids=["p3", "p2"])
@pytest.mark.parametrize(
    "counts,duplicate",
    [
        ([5, 0, 37, 1], False),
        ([0, 0], False),
        ([12, 12, 12], True),
        ([60], False),
    ],
    ids=["mixed", "no-boxes", "duplicates", "full"],
)
def test_per_image_match_equals_the_batched_padded_match(strides, counts, duplicate):
    loss = _loss(strides)
    targets = _targets(counts, max_labels=60, duplicate=duplicate)
    predict = _predictions(loss, len(counts))

    per_image = loss.matcher(targets, predict, return_indices=True)
    # The pre-#935 path: one batched match over the padded rows.
    batched = loss.matcher._match_image(targets, predict)

    for got, expected in zip(per_image, batched):
        assert got.dtype == expected.dtype
        assert torch.equal(got, expected)
    assert per_image[1].any() == (sum(counts) > 0)


def test_each_image_is_matched_on_its_real_boxes_only():
    loss = _loss()
    targets = _targets([5, 0, 37], max_labels=300)
    shapes = []
    match_image = loss.matcher._match_image

    def _spy(target, predict):
        shapes.append(tuple(target.shape))
        return match_image(target, predict)

    loss.matcher._match_image = _spy
    loss.matcher(targets, _predictions(loss, 3))
    assert shapes == [(1, 5, 5), (1, 0, 5), (1, 37, 5)]


def test_loss_is_unchanged():
    loss = _loss()
    counts = [5, 0, 37, 1]
    targets = _targets(counts, max_labels=300) / torch.tensor(
        [1.0, IMAGE, IMAGE, IMAGE, IMAGE]
    )
    targets[..., 0] = _targets(counts, max_labels=300)[..., 0]
    g = torch.Generator().manual_seed(2)
    raw = [
        torch.randn(len(counts), 4 * 16 + NUM_CLASSES, IMAGE // s, IMAGE // s, generator=g)
        for s in (8, 16, 32)
    ]

    per_image = loss(raw, targets)
    matcher = loss.matcher
    loss.matcher = lambda target, predict: matcher._match_image(target, predict)[:2]
    batched = loss(raw, targets)

    assert float(per_image["num_fg"]) > 0
    for key in ("total_loss", "box_loss", "dfl_loss", "cls_loss", "num_fg"):
        assert torch.equal(per_image[key], batched[key])


def _transform_targets(n):
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 50, size=(n, 2))
    return np.hstack([xy, xy + 8, rng.integers(0, NUM_CLASSES, size=(n, 1))]).astype(
        np.float32
    )


def test_dropped_labels_warn_once(monkeypatch, caplog):
    monkeypatch.setattr(yolo9_augment, "_dropped_labels_warned", False)
    transform = YOLO9TrainTransform(max_labels=10, flip_prob=0.0, hsv_prob=0.0)
    image = np.full((64, 64, 3), 114, dtype=np.uint8)

    with caplog.at_level(logging.WARNING, logger=yolo9_augment.__name__):
        _, labels = transform(image, _transform_targets(25), (64, 64))
        transform(image, _transform_targets(25), (64, 64))

    assert (labels[:, 0] >= 0).sum() == 10
    warnings = [r for r in caplog.records if "max_labels=10" in r.getMessage()]
    assert len(warnings) == 1
    assert "25 labels" in warnings[0].getMessage()


def test_no_warning_within_the_cap(monkeypatch, caplog):
    monkeypatch.setattr(yolo9_augment, "_dropped_labels_warned", False)
    transform = YOLO9TrainTransform(max_labels=10, flip_prob=0.0, hsv_prob=0.0)
    image = np.full((64, 64, 3), 114, dtype=np.uint8)

    with caplog.at_level(logging.WARNING, logger=yolo9_augment.__name__):
        transform(image, _transform_targets(10), (64, 64))

    assert not [r for r in caplog.records if "max_labels" in r.getMessage()]
