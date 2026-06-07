"""Tests for optional RAM/disk image caching (libreyolo.data.cache)."""

import time

import numpy as np
import pytest
from PIL import Image

from libreyolo.data.cache import normalize_cache
from libreyolo.data.dataset import YOLODataset

pytestmark = pytest.mark.unit


def _write_files(tmp_path, n=5):
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(n):
        arr = rng.randint(0, 255, (40 + i, 50 + i, 3), dtype=np.uint8)
        Image.fromarray(arr).save(image_dir / f"img{i}.png")
        (label_dir / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")


def _build(tmp_path):
    return YOLODataset(data_dir=str(tmp_path), split="train", img_size=(64, 64))


def _make_dataset(tmp_path, n=5):
    _write_files(tmp_path, n)
    return _build(tmp_path)


def test_normalize_cache_values():
    assert normalize_cache(False) is None
    assert normalize_cache(None) is None
    assert normalize_cache(True) == "ram"
    assert normalize_cache("ram") == "ram"
    assert normalize_cache("DISK") == "disk"
    assert normalize_cache("true") == "ram"
    assert normalize_cache("false") is None
    with pytest.raises(ValueError):
        normalize_cache("bogus")


def test_cache_disabled_by_default(tmp_path):
    ds = _make_dataset(tmp_path)
    assert ds.cache is None  # disabled until enabled
    ds.enable_image_cache(False)
    assert ds.cache is None


def test_ram_cache_matches_uncached_and_returns_copies(tmp_path):
    _write_files(tmp_path)
    ref = _build(tmp_path)
    ref.enable_image_cache(False)
    expected = [ref.load_image(i).copy() for i in range(len(ref))]

    ram = _build(tmp_path)
    ram.enable_image_cache("ram")
    for i in range(len(ram)):
        first = ram.load_image(i)
        second = ram.load_image(i)
        assert np.array_equal(first, expected[i])
        assert np.array_equal(second, expected[i])
        # Copy-on-read so in-place augmentation cannot corrupt the cache.
        assert first is not second


def test_disk_cache_creates_npy_and_reloads(tmp_path):
    _write_files(tmp_path)
    ref = _build(tmp_path)
    ref.enable_image_cache(False)
    expected = [ref.load_image(i).copy() for i in range(len(ref))]

    disk = _build(tmp_path)
    disk.enable_image_cache("disk")
    for i in range(len(disk)):
        assert np.array_equal(disk.load_image(i), expected[i])

    # .npy sidecars created with appended suffix (collision-free).
    for i in range(len(disk)):
        assert (tmp_path / "images" / "train" / f"img{i}.png.npy").exists()

    # Fresh dataset reads from the .npy cache and matches.
    reload = _build(tmp_path)
    reload.enable_image_cache("disk")
    for i in range(len(reload)):
        assert np.array_equal(reload.load_image(i), expected[i])


def test_disk_cache_invalidates_on_source_change(tmp_path):
    _write_files(tmp_path)
    disk = _build(tmp_path)
    disk.enable_image_cache("disk")
    _ = disk.load_image(0)  # populate cache

    time.sleep(1.1)  # ensure a newer mtime than the .npy
    new = np.full((40, 50, 3), 123, dtype=np.uint8)
    Image.fromarray(new).save(tmp_path / "images" / "train" / "img0.png")

    fresh = _build(tmp_path)
    fresh.enable_image_cache("disk")
    import cv2

    expected = cv2.imread(str(tmp_path / "images" / "train" / "img0.png"))
    assert np.array_equal(fresh.load_image(0), expected)
