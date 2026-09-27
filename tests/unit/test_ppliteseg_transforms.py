"""Geometry and photometric tests for the PP-LiteSeg training recipes.

Covers the two source crop recipes (512x1024 for the 50 sizes, 768x768 for the
75 sizes) on top of the rectangular ``SemanticDataset`` canvas, plus the
family-local colour jitter. The scalar no-regression cases live in
``test_semantic_dataset.py``.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from libreyolo.data.semantic_dataset import IGNORE_INDEX, SemanticDataset
from libreyolo.models.ppliteseg.transforms import SegColorJitter

pytestmark = [pytest.mark.unit, pytest.mark.ppliteseg]


@pytest.fixture()
def dataset_root(tmp_path):
    """Two repository-authored image/mask pairs, 64x128 (H x W)."""
    images = tmp_path / "images" / "train"
    masks = tmp_path / "masks" / "train"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for index in range(2):
        Image.fromarray(rng.integers(0, 256, (64, 128, 3), dtype=np.uint8)).save(
            images / f"sample{index}.png"
        )
        mask = np.zeros((64, 128), dtype=np.uint8)
        mask[:, 64:] = 1
        mask[:8, :8] = IGNORE_INDEX
        Image.fromarray(mask, mode="L").save(masks / f"sample{index}.png")
    return tmp_path


def _config(root):
    return {
        "root": str(root),
        "train": str(root / "images" / "train"),
        "val": str(root / "images" / "train"),
        "names": {0: "left", 1: "right"},
        "nc": 2,
        "masks_dir": "masks",
    }


def _build(root, imgsz, **kwargs):
    return SemanticDataset(_config(root), split="train", imgsz=imgsz, **kwargs)


def test_rectangular_canvas_is_stored_as_explicit_h_and_w(dataset_root):
    dataset = _build(dataset_root, (512, 1024))
    assert (dataset.canvas_h, dataset.canvas_w) == (512, 1024)
    scalar = _build(dataset_root, 512)
    assert (scalar.canvas_h, scalar.canvas_w) == (512, 512)


def test_scalar_imgsz_is_exactly_equivalent_to_the_equal_pair(dataset_root):
    random.seed(0)
    scalar_img, scalar_mask, scalar_info, _ = _build(dataset_root, 96)[0]
    random.seed(0)
    pair_img, pair_mask, pair_info, _ = _build(dataset_root, (96, 96))[0]
    assert scalar_img.shape == pair_img.shape == (3, 96, 96)
    assert (scalar_img == pair_img).all()
    assert (scalar_mask == pair_mask).all()
    assert scalar_info == pair_info


@pytest.mark.parametrize("crop", [(512, 1024), (768, 768)])
def test_rescale_crop_emits_the_requested_crop_for_both_recipes(dataset_root, crop):
    dataset = _build(
        dataset_root,
        crop,
        augment=True,
        resize_mode="rescale_crop",
        scale_jitter=(0.125, 1.5),
    )
    random.seed(3)
    for _ in range(5):
        img, mask, _, _ = dataset[0]
        assert img.shape == (3, crop[0], crop[1])
        assert mask.shape == (crop[0], crop[1])


def test_rescale_crop_pads_with_ignore_and_preserves_ignore_labels(dataset_root):
    # The source images are 64x128, far smaller than the crop, so almost the
    # whole canvas is ignore-padding.
    dataset = _build(
        dataset_root,
        (512, 1024),
        augment=True,
        resize_mode="rescale_crop",
        scale_jitter=(0.125, 0.125),
    )
    random.seed(1)
    _, mask, _, _ = dataset[0]
    values = set(mask.unique().tolist())
    assert IGNORE_INDEX in values
    assert values <= {0, 1, IGNORE_INDEX}


def test_rescale_crop_scale_bounds_change_content_extent(dataset_root):
    low = _build(
        dataset_root, (64, 64), augment=True, resize_mode="rescale_crop", scale_jitter=(0.25, 0.25)
    )
    high = _build(
        dataset_root, (64, 64), augment=True, resize_mode="rescale_crop", scale_jitter=(1.75, 1.75)
    )
    random.seed(5)
    _, low_mask, low_info, _ = low[0]
    random.seed(5)
    _, high_mask, high_info, _ = high[0]
    assert low_info["ratio"] == pytest.approx(0.25)
    assert high_info["ratio"] == pytest.approx(1.75)
    # At 0.25 the 64x128 source becomes 16x32 and the rest of the 64x64 canvas
    # is ignore padding; at 1.75 the crop is fully inside real content.
    assert (low_mask == IGNORE_INDEX).sum() > (high_mask == IGNORE_INDEX).sum()


def test_rescale_crop_validates_by_direct_resize(dataset_root):
    dataset = _build(dataset_root, (512, 1024), augment=False, resize_mode="rescale_crop")
    img, mask, info, _ = dataset[0]
    assert img.shape == (3, 512, 1024)
    assert mask.shape == (512, 1024)
    # A direct resize introduces no padding, so no ignore pixels are added
    # beyond the ones the source mask already carries.
    assert info["ratio"] == 1.0


def test_forced_flip_mirrors_image_and_mask_together(dataset_root):
    dataset = _build(
        dataset_root,
        (64, 128),
        augment=True,
        resize_mode="rescale_crop",
        scale_jitter=(1.0, 1.0),
        hsv_prob=0.0,
    )
    random.seed(0)
    flipped = None
    unflipped = None
    for seed in range(20):
        random.seed(seed)
        # random.random() < 0.5 drives the flip; probe both branches.
        state = random.Random(seed).random() < 0.5
        random.seed(seed)
        _, mask, _, _ = dataset[0]
        if state and flipped is None:
            flipped = mask
        if not state and unflipped is None:
            unflipped = mask
    assert flipped is not None and unflipped is not None
    assert (flipped.flip(-1) == unflipped).float().mean() > 0.99


def test_unknown_resize_mode_is_rejected(dataset_root):
    with pytest.raises(ValueError, match="rescale_crop"):
        _build(dataset_root, 64, resize_mode="nonsense")


def test_color_jitter_changes_pixels_but_not_shape_or_dtype():
    rng = np.random.default_rng(0)
    img = rng.integers(20, 200, (32, 48, 3), dtype=np.uint8)
    jitter = SegColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
    random.seed(0)
    out = jitter(img)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert not np.array_equal(out, img)


def test_color_jitter_at_zero_magnitude_is_the_identity():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    out = SegColorJitter(brightness=0.0, contrast=0.0, saturation=0.0)(img)
    assert np.array_equal(out, img)


def test_color_jitter_rejects_negative_magnitude():
    with pytest.raises(ValueError, match="non-negative"):
        SegColorJitter(brightness=-0.1)


def test_family_photometric_replaces_hsv_jitter(dataset_root):
    calls = []

    def photometric(img):
        calls.append(img.shape)
        return img

    dataset = _build(
        dataset_root,
        (64, 128),
        augment=True,
        resize_mode="rescale_crop",
        photometric=photometric,
        hsv_prob=1.0,
    )
    random.seed(0)
    dataset[0]
    assert calls, "the family transform must run in place of augment_hsv"
