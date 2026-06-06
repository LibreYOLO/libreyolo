"""Tests for dataset annotation loading."""

import logging
import math

import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from torch.utils.data import SubsetRandomSampler

from libreyolo.data.dataset import YOLODataset, create_dataloader

pytestmark = pytest.mark.unit


def test_yolo_annotation_loading_preserves_order_and_shape(tmp_path, monkeypatch):
    monkeypatch.setattr("libreyolo.data.dataset.os.cpu_count", lambda: 8)

    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    order = [3, 1, 4, 0, 2, 7, 5, 9, 6, 8]
    for index in order:
        width = 100 + index
        height = 80 + index
        Image.new("RGB", (width, height), color="white").save(
            image_dir / f"sample_{index}.jpg"
        )
        (label_dir / f"sample_{index}.txt").write_text("0 0.5 0.5 0.25 0.5\n")

    img_files = [image_dir / f"sample_{index}.jpg" for index in order]
    label_files = [label_dir / f"sample_{index}.txt" for index in order]

    dataset = YOLODataset(
        img_files=img_files,
        label_files=label_files,
        img_size=(64, 64),
    )

    assert [annotation[3] for annotation in dataset.annotations] == [
        image_path.name for image_path in img_files
    ]

    for index, annotation in zip(order, dataset.annotations):
        labels, img_info, resized_info, file_name = annotation
        width = 100 + index
        height = 80 + index
        scale = min(64 / height, 64 / width)

        assert isinstance(labels, np.ndarray)
        assert labels.shape == (1, 5)
        assert img_info == (height, width)
        assert resized_info == (int(height * scale), int(width * scale))
        assert file_name == f"sample_{index}.jpg"


def test_yolo_dataset_loads_obb_rows_as_proxy_box_and_angle(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    Image.new("RGB", (100, 100), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text(
        "0 0.10 0.20 0.50 0.20 0.50 0.40 0.10 0.40\n"
    )

    dataset = YOLODataset(
        img_files=[image_dir / "sample.jpg"],
        label_files=[label_dir / "sample.txt"],
        img_size=(64, 64),
        load_obb=True,
    )

    labels, _, _, _ = dataset.annotations[0]
    assert labels.shape == (1, 6)
    np.testing.assert_allclose(labels[0, :4], [6.4, 12.8, 32.0, 25.6], atol=1e-5)
    assert labels[0, 4] == 0
    assert labels[0, 5] == pytest.approx(0.0, abs=1e-6)


def test_yolo_dataset_obb_uses_pixel_geometry_for_rectangular_images(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    width, height = 200, 100
    Image.new("RGB", (width, height), color="white").save(image_dir / "sample.jpg")

    cx, cy = 100.0, 50.0
    box_w, box_h = 80.0, 20.0
    angle = math.radians(30.0)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    corners = []
    for dx, dy in [(-box_w / 2, -box_h / 2), (box_w / 2, -box_h / 2),
                   (box_w / 2, box_h / 2), (-box_w / 2, box_h / 2)]:
        x = cx + dx * cos_a - dy * sin_a
        y = cy + dx * sin_a + dy * cos_a
        corners.extend([x / width, y / height])

    (label_dir / "sample.txt").write_text(
        "0 " + " ".join(f"{value:.8f}" for value in corners) + "\n"
    )

    dataset = YOLODataset(
        img_files=[image_dir / "sample.jpg"],
        label_files=[label_dir / "sample.txt"],
        img_size=(height, width),
        load_obb=True,
    )

    labels, _, _, _ = dataset.annotations[0]
    np.testing.assert_allclose(labels[0, :4], [60.0, 40.0, 140.0, 60.0], atol=1e-3)
    assert labels[0, 5] == pytest.approx(angle, abs=1e-3)


def test_yolo_dataset_skips_invalid_obb_rows_with_warning(tmp_path, caplog):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    Image.new("RGB", (100, 100), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text(
        "\n".join(
            [
                "0 0.10 0.20 0.50 0.20 0.50 0.40 0.10 0.40",
                "0 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20",
                "0 -0.10 0.20 0.50 0.20 0.50 0.40 0.10 0.40",
                "1 0.10 0.20 0.50 0.20 0.50 0.40 0.10 0.40",
            ]
        )
        + "\n"
    )

    with caplog.at_level(logging.WARNING):
        dataset = YOLODataset(
            img_files=[image_dir / "sample.jpg"],
            label_files=[label_dir / "sample.txt"],
            img_size=(64, 64),
            load_obb=True,
            num_classes=1,
        )

    labels, _, _, _ = dataset.annotations[0]
    assert labels.shape == (2, 6)
    assert "Skipped 2 invalid YOLO OBB label rows" in caplog.text
    assert "sample.txt" in caplog.text


def test_yolo_dataset_rejects_segments_and_obb_together(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    Image.new("RGB", (100, 100), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text("")

    with pytest.raises(ValueError, match="segmentation and OBB"):
        YOLODataset(
            img_files=[image_dir / "sample.jpg"],
            label_files=[label_dir / "sample.txt"],
            img_size=(64, 64),
            load_segments=True,
            load_obb=True,
        )


def test_yolo_dataset_directory_mode_dedupes_case_insensitive_glob(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("libreyolo.data.dataset.os.cpu_count", lambda: 8)

    data_dir = tmp_path / "dataset"
    image_dir = data_dir / "images" / "train"
    label_dir = data_dir / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    Image.new("RGB", (32, 24), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.25 0.5\n")

    original_glob = Path.glob

    def case_insensitive_glob(self, pattern):
        if self == image_dir and pattern == "*.JPG":
            return original_glob(self, "*.jpg")
        return original_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", case_insensitive_glob)

    dataset = YOLODataset(data_dir=data_dir, split="train", img_size=(64, 64))

    assert dataset.num_imgs == 1
    assert dataset.img_files == [image_dir / "sample.jpg"]
    assert dataset.label_files == [label_dir / "sample.txt"]


@pytest.mark.parametrize(
    ("dataset_len", "batch_size", "expected_batches"),
    [(2, 4, 1), (5, 2, 2)],
)
def test_create_dataloader_drop_last_only_when_safe(
    dataset_len, batch_size, expected_batches
):
    loader = create_dataloader(
        [None] * dataset_len,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
    )

    assert len(loader) == expected_batches


def test_create_dataloader_uses_sampler_visible_size():
    sampler = SubsetRandomSampler([0, 1])
    loader = create_dataloader(
        [None] * 10,
        batch_size=4,
        num_workers=0,
        shuffle=True,
        sampler=sampler,
    )

    assert len(loader) == 1
