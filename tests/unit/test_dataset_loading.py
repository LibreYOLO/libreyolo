"""Tests for dataset annotation loading."""

import numpy as np
import pytest
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
