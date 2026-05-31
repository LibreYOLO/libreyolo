"""Tests for dataset annotation loading."""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from libreyolo.data.dataset import YOLODataset

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
