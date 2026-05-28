"""Tests for the Ultralytics-parity image cache.

Covers:
- ``TrainConfig.from_kwargs(cache=...)`` accepts the new field without warning.
- ``YOLODataset(cache=...)`` populates RAM / disk caches and falls back
  silently when the pre-flight capacity check fails.
- The DataLoader-level data flow (a one-epoch iteration proxy) yields finite
  image tensors in all three modes.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from libreyolo.data.dataset import YOLODataset
from libreyolo.training.config import TrainConfig, YOLO9Config


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_tiny_yolo_dataset(root: Path, n: int = 10, size: int = 96) -> List[Path]:
    """Create ``n`` small RGB JPGs + matching YOLO label files."""
    image_dir = root / "images"
    label_dir = root / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    img_paths: List[Path] = []
    rng = np.random.default_rng(seed=0)
    for i in range(n):
        # Vary dimensions slightly so caching has to handle different aspect
        # ratios — keeps the resize path honest.
        w, h = size + i, size + (i % 3)
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        img_path = image_dir / f"sample_{i:02d}.jpg"
        Image.fromarray(arr, mode="RGB").save(img_path)
        # One bbox per image, centered.
        (label_dir / f"sample_{i:02d}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
        img_paths.append(img_path)
    return img_paths


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------


def test_train_config_accepts_cache_without_warning():
    """``cache`` must be a recognised TrainConfig field — no UserWarning."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        cfg = TrainConfig.from_kwargs(cache="ram")
    unknown_key_warnings = [
        w for w in captured if "Unknown training config keys" in str(w.message)
    ]
    assert unknown_key_warnings == [], (
        f"cache=ram triggered an unknown-key warning: {unknown_key_warnings}"
    )
    assert cfg.cache == "ram"


def test_train_config_default_cache_is_false():
    """Existing call sites that never pass ``cache`` must be unchanged."""
    cfg = TrainConfig.from_kwargs()
    assert cfg.cache is False


@pytest.mark.parametrize("value", [False, None, True, "ram", "disk"])
def test_train_config_cache_accepts_documented_values(value):
    cfg = TrainConfig.from_kwargs(cache=value)
    assert cfg.cache == value


def test_yolo9_config_inherits_cache_field():
    """Subclassed configs inherit the new field for free."""
    cfg = YOLO9Config.from_kwargs(cache="disk")
    assert cfg.cache == "disk"


# ---------------------------------------------------------------------------
# YOLODataset — no cache (default)
# ---------------------------------------------------------------------------


def test_dataset_no_cache_by_default(tmp_path):
    img_files = _build_tiny_yolo_dataset(tmp_path)
    ds = YOLODataset(img_files=img_files, img_size=(64, 64))
    assert ds.cache_mode is None
    assert ds.ims == []
    assert ds.npy_files == []


# ---------------------------------------------------------------------------
# YOLODataset — RAM cache
# ---------------------------------------------------------------------------


def test_dataset_cache_ram_populates_ims(tmp_path):
    img_files = _build_tiny_yolo_dataset(tmp_path)
    ds = YOLODataset(img_files=img_files, img_size=(64, 64), cache="ram")
    assert ds.cache_mode == "ram"
    assert len(ds.ims) == len(img_files)
    assert ds.ims[0] is not None
    assert all(im is not None for im in ds.ims)
    # Cached images are already resized to fit the long side within imgsz.
    for im in ds.ims:
        assert im.dtype == np.uint8
        assert im.ndim == 3 and im.shape[2] == 3
        assert max(im.shape[:2]) <= 64


def test_dataset_cache_true_is_alias_for_ram(tmp_path):
    img_files = _build_tiny_yolo_dataset(tmp_path)
    ds = YOLODataset(img_files=img_files, img_size=(64, 64), cache=True)
    assert ds.cache_mode == "ram"
    assert ds.ims[0] is not None


def test_load_resized_img_returns_ram_cached_array(tmp_path):
    img_files = _build_tiny_yolo_dataset(tmp_path)
    ds = YOLODataset(img_files=img_files, img_size=(64, 64), cache="ram")
    # ``load_resized_img`` must serve directly from RAM with zero IO.
    out = ds.load_resized_img(3)
    assert out is ds.ims[3]


def test_ram_cache_content_matches_no_cache(tmp_path):
    """RAM-cached images must equal the non-cached resize path byte-for-byte."""
    img_files = _build_tiny_yolo_dataset(tmp_path)
    ref = YOLODataset(img_files=img_files, img_size=(64, 64))
    cached = YOLODataset(img_files=img_files, img_size=(64, 64), cache="ram")
    for i in range(len(img_files)):
        np.testing.assert_array_equal(ref.load_resized_img(i), cached.load_resized_img(i))


# ---------------------------------------------------------------------------
# YOLODataset — disk cache
# ---------------------------------------------------------------------------


def test_dataset_cache_disk_writes_npy_files(tmp_path):
    img_files = _build_tiny_yolo_dataset(tmp_path)
    ds = YOLODataset(img_files=img_files, img_size=(64, 64), cache="disk")
    assert ds.cache_mode == "disk"
    assert len(ds.npy_files) == len(img_files)
    for npy in ds.npy_files:
        assert npy.exists()
        assert npy.suffix == ".npy"
        # ``.npy`` lives next to the source.
        assert npy.parent == img_files[0].parent


def test_disk_cache_load_image_reads_npy(tmp_path):
    img_files = _build_tiny_yolo_dataset(tmp_path)
    ds_disk = YOLODataset(img_files=img_files, img_size=(64, 64), cache="disk")
    ds_ref = YOLODataset(img_files=img_files, img_size=(64, 64))
    # ``load_image`` returns the raw decoded BGR — ``.npy`` and cv2.imread
    # must agree.
    for i in range(len(img_files)):
        np.testing.assert_array_equal(ds_disk.load_image(i), ds_ref.load_image(i))


def test_disk_cache_reuses_existing_npy(tmp_path):
    """A second dataset construction must not rewrite existing ``.npy`` files."""
    img_files = _build_tiny_yolo_dataset(tmp_path)
    ds1 = YOLODataset(img_files=img_files, img_size=(64, 64), cache="disk")
    mtimes = {p: p.stat().st_mtime_ns for p in ds1.npy_files}
    ds2 = YOLODataset(img_files=img_files, img_size=(64, 64), cache="disk")
    for p in ds2.npy_files:
        assert p.stat().st_mtime_ns == mtimes[p], f"{p} was rewritten"


# ---------------------------------------------------------------------------
# Capacity-check fallback
# ---------------------------------------------------------------------------


def test_ram_cache_falls_back_when_capacity_check_fails(tmp_path, monkeypatch):
    """When the RAM check returns False, the dataset must silently downgrade."""
    img_files = _build_tiny_yolo_dataset(tmp_path)
    monkeypatch.setattr(
        "libreyolo.data.dataset._check_cache_ram",
        lambda paths, img_size, safety_margin=0.5: (False, 9.9e9),
    )
    ds = YOLODataset(img_files=img_files, img_size=(64, 64), cache="ram")
    assert ds.cache_mode is None
    assert ds.ims == []
    # ``load_resized_img`` still works via the standard decode-and-resize path.
    img = ds.load_resized_img(0)
    assert img is not None and img.size > 0


def test_disk_cache_falls_back_when_capacity_check_fails(tmp_path, monkeypatch):
    img_files = _build_tiny_yolo_dataset(tmp_path)
    monkeypatch.setattr(
        "libreyolo.data.dataset._check_cache_disk",
        lambda paths, safety_margin=0.5: (False, 9.9e9),
    )
    ds = YOLODataset(img_files=img_files, img_size=(64, 64), cache="disk")
    assert ds.cache_mode is None
    assert ds.npy_files == []


def test_unknown_cache_mode_disables_cache(tmp_path):
    img_files = _build_tiny_yolo_dataset(tmp_path)
    ds = YOLODataset(img_files=img_files, img_size=(64, 64), cache="banana")
    assert ds.cache_mode is None


# ---------------------------------------------------------------------------
# One-epoch DataLoader iteration proxy
# ---------------------------------------------------------------------------


def _identity_preproc(img, target, input_dim):
    """Tiny preproc: pad to ``input_dim`` and return ``(CHW float32, 5-col targets)``."""
    target_h, target_w = input_dim
    h, w = img.shape[:2]
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[: min(h, target_h), : min(w, target_w)] = img[: min(h, target_h), : min(w, target_w)]
    # CHW float32 in [0, 1]
    out = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    padded_targets = np.zeros((10, 5), dtype=np.float32)
    if target.shape[0] > 0:
        m = min(target.shape[0], 10)
        padded_targets[:m] = target[:m]
    return out, padded_targets


@pytest.mark.parametrize("cache", [False, "ram", "disk"])
def test_one_epoch_iteration_produces_finite_loss(tmp_path, cache):
    """Proxy for "1 epoch of training": iterate the DataLoader once for each
    cache mode and compute a dummy mean-squared loss against zeros. The loss
    must be finite (no NaN/Inf) and the iteration must complete without raising.
    """
    img_files = _build_tiny_yolo_dataset(tmp_path, n=10)
    ds = YOLODataset(
        img_files=img_files,
        img_size=(64, 64),
        preproc=_identity_preproc,
        cache=cache,
    )

    def _collate(batch):
        imgs = torch.from_numpy(np.stack([item[0] for item in batch]))
        targets = torch.from_numpy(np.stack([item[1] for item in batch]))
        return imgs, targets

    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=_collate)
    total = torch.zeros(())
    n_batches = 0
    for imgs, _targets in loader:
        # Dummy "loss" — squared norm of the image batch against zeros.
        loss = (imgs * imgs).mean()
        assert torch.isfinite(loss), f"non-finite loss with cache={cache}"
        total = total + loss.detach()
        n_batches += 1
    assert n_batches == 5  # 10 imgs / batch 2
    assert torch.isfinite(total)
    assert total.item() > 0
