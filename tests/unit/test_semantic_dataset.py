"""Unit tests for the semantic-segmentation dataset."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.data import (
    SemanticDataset,
    img2mask_paths,
    semantic_collate_fn,
)

pytestmark = pytest.mark.unit


def _write_image(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(30, 60, 90)).save(path)


def _write_mask(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def _make_mask_dataset(root: Path, n_images: int = 2, size: int = 32) -> dict:
    """Dense-mask dataset: left half class 0, right half class 1, one ignore row."""
    for i in range(n_images):
        _write_image(root / "images" / "train" / f"img{i}.jpg", size, size)
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[:, size // 2 :] = 1
        mask[0, :] = 255
        _write_mask(root / "masks" / "train" / f"img{i}.png", mask)
    return {
        "path": str(root),
        "train": str(root / "images" / "train"),
        "names": {0: "left", 1: "right"},
        "nc": 2,
        "masks_dir": "masks",
    }


class TestMaskPairing:
    def test_img2mask_paths_substitutes_images_dir(self):
        paths = img2mask_paths([Path("/data/images/train/001.jpg")])
        assert paths == [Path("/data/masks/train/001.png")]

    def test_img2mask_paths_custom_dir(self):
        paths = img2mask_paths([Path("/data/images/val/x.jpg")], masks_dir="ann")
        assert paths == [Path("/data/ann/val/x.png")]


class TestSemanticDatasetMasks:
    def test_loads_paired_masks(self, tmp_path):
        config = _make_mask_dataset(tmp_path)
        dataset = SemanticDataset(config, split="train", imgsz=32)

        assert len(dataset) == 2
        assert dataset.nc == 2
        img, mask, info, img_id = dataset[0]
        assert img.shape == (3, 32, 32)
        assert img.dtype == torch.float32
        assert float(img.max()) <= 1.0
        assert mask.shape == (32, 32)
        assert mask.dtype == torch.long
        assert set(torch.unique(mask).tolist()) <= {0, 1, 255}
        assert info["orig_shape"] == (32, 32)
        assert img_id == 0

    def test_letterbox_pads_mask_with_ignore(self, tmp_path):
        # 64x32 wide image letterboxed into a 64x64 square -> vertical ignore pad.
        _write_image(tmp_path / "images" / "train" / "wide.jpg", 64, 32)
        mask = np.zeros((32, 64), dtype=np.uint8)
        _write_mask(tmp_path / "masks" / "train" / "wide.png", mask)
        config = {
            "train": str(tmp_path / "images" / "train"),
            "names": {0: "thing"},
            "nc": 1,
            "masks_dir": "masks",
        }
        dataset = SemanticDataset(config, split="train", imgsz=64)
        _, mask_t, info, _ = dataset[0]

        assert mask_t.shape == (64, 64)
        assert int((mask_t == 255).sum()) == 64 * 32  # padded half is ignore
        assert info["pad"][0] == 16

    def test_stretch_mode_has_no_padding(self, tmp_path):
        _write_image(tmp_path / "images" / "train" / "wide.jpg", 64, 32)
        _write_mask(
            tmp_path / "masks" / "train" / "wide.png",
            np.zeros((32, 64), dtype=np.uint8),
        )
        config = {
            "train": str(tmp_path / "images" / "train"),
            "names": {0: "thing"},
            "nc": 1,
            "masks_dir": "masks",
        }
        dataset = SemanticDataset(config, split="train", imgsz=64, resize_mode="stretch")
        _, mask_t, _, _ = dataset[0]

        assert mask_t.shape == (64, 64)
        assert int((mask_t == 255).sum()) == 0

    def test_missing_mask_raises(self, tmp_path):
        config = _make_mask_dataset(tmp_path)
        (tmp_path / "masks" / "train" / "img1.png").unlink()

        with pytest.raises(FileNotFoundError, match="mask file"):
            SemanticDataset(config, split="train", imgsz=32)

    def test_out_of_range_class_raises(self, tmp_path):
        config = _make_mask_dataset(tmp_path, n_images=1)
        bad = np.full((32, 32), 9, dtype=np.uint8)
        _write_mask(tmp_path / "masks" / "train" / "img0.png", bad)
        dataset = SemanticDataset(config, split="train", imgsz=32)

        with pytest.raises(ValueError, match="outside 0..1"):
            dataset[0]

    def test_label_mapping_remaps_and_ignores_unmapped(self, tmp_path):
        _write_image(tmp_path / "images" / "train" / "a.jpg", 16, 16)
        source = np.full((16, 16), 7, dtype=np.uint8)
        source[:, 8:] = 26  # unmapped -> ignore
        _write_mask(tmp_path / "masks" / "train" / "a.png", source)
        config = {
            "train": str(tmp_path / "images" / "train"),
            "names": {0: "road"},
            "nc": 1,
            "masks_dir": "masks",
            "label_mapping": {7: 0},
        }
        dataset = SemanticDataset(config, split="train", imgsz=16)
        _, mask_t, _, _ = dataset[0]

        assert set(torch.unique(mask_t).tolist()) == {0, 255}

    def test_label_mapping_validates_train_ids(self, tmp_path):
        config = _make_mask_dataset(tmp_path, n_images=1)
        config["label_mapping"] = {7: 5}

        with pytest.raises(ValueError, match="outside 0..1"):
            SemanticDataset(config, split="train", imgsz=32)

    def test_augment_keeps_shapes_and_label_domain(self, tmp_path):
        config = _make_mask_dataset(tmp_path, n_images=1)
        dataset = SemanticDataset(config, split="train", imgsz=32, augment=True)

        for _ in range(5):
            img, mask, _, _ = dataset[0]
            assert img.shape == (3, 32, 32)
            assert mask.shape == (32, 32)
            assert set(torch.unique(mask).tolist()) <= {0, 1, 255}


class TestPolygonFallback:
    def test_polygons_rasterize_with_background(self, tmp_path):
        _write_image(tmp_path / "images" / "train" / "a.jpg", 32, 32)
        labels = tmp_path / "labels" / "train" / "a.txt"
        labels.parent.mkdir(parents=True, exist_ok=True)
        # Class-0 polygon covering the left half of the image.
        labels.write_text("0 0.0 0.0 0.5 0.0 0.5 1.0 0.0 1.0\n")
        config = {
            "train": str(tmp_path / "images" / "train"),
            "names": {0: "object"},
            "nc": 1,
        }
        dataset = SemanticDataset(config, split="train", imgsz=32)

        assert dataset.nc == 2  # object + appended background
        assert dataset.names[1] == "background"
        _, mask_t, _, _ = dataset[0]
        values = set(torch.unique(mask_t).tolist())
        assert 0 in values  # polygon area
        assert 1 in values  # background
        assert int((mask_t == 0).sum()) > 0

    def test_box_rows_fill_rectangles(self, tmp_path):
        _write_image(tmp_path / "images" / "train" / "b.jpg", 32, 32)
        labels = tmp_path / "labels" / "train" / "b.txt"
        labels.parent.mkdir(parents=True, exist_ok=True)
        labels.write_text("0 0.5 0.5 0.5 0.5\n")  # centered box
        config = {
            "train": str(tmp_path / "images" / "train"),
            "names": {0: "object"},
            "nc": 1,
        }
        dataset = SemanticDataset(config, split="train", imgsz=32)
        _, mask_t, _, _ = dataset[0]

        assert int(mask_t[16, 16]) == 0  # inside box
        assert int(mask_t[2, 2]) == 1  # background corner

    def test_missing_label_file_means_all_background(self, tmp_path):
        _write_image(tmp_path / "images" / "train" / "c.jpg", 16, 16)
        config = {
            "train": str(tmp_path / "images" / "train"),
            "names": {0: "object"},
            "nc": 1,
        }
        dataset = SemanticDataset(config, split="train", imgsz=16)
        _, mask_t, _, _ = dataset[0]

        assert set(torch.unique(mask_t).tolist()) == {1}


def test_semantic_collate_fn_stacks_batch(tmp_path):
    config = _make_mask_dataset(tmp_path)
    dataset = SemanticDataset(config, split="train", imgsz=32)
    imgs, masks, infos, ids = semantic_collate_fn([dataset[0], dataset[1]])

    assert imgs.shape == (2, 3, 32, 32)
    assert masks.shape == (2, 32, 32)
    assert masks.dtype == torch.long
    assert len(infos) == 2
    assert ids == [0, 1]
