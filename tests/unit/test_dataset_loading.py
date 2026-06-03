"""Tests for dataset annotation loading."""

import logging
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


def _build_mixed_dataset(tmp_path, label_contents, **kwargs):
    """Create a YOLODataset from a list of label file bodies (empty string = unlabeled)."""
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)

    img_files, label_files = [], []
    for i, body in enumerate(label_contents):
        img = image_dir / f"img_{i}.jpg"
        lbl = label_dir / f"img_{i}.txt"
        Image.new("RGB", (64, 64), color="white").save(img)
        lbl.write_text(body)
        img_files.append(img)
        label_files.append(lbl)

    return YOLODataset(img_files=img_files, label_files=label_files, img_size=(64, 64), **kwargs)


def test_filter_empty_annotations_drops_unlabeled_images(tmp_path):
    labeled = "0 0.5 0.5 0.25 0.5\n"
    ds = _build_mixed_dataset(tmp_path, [labeled, "", labeled], filter_empty_annotations=True)

    assert ds.num_imgs == 2
    assert len(ds.annotations) == 2
    assert len(ds.img_files) == 2
    assert len(ds.label_files) == 2
    assert all(ann[0].shape[0] > 0 for ann in ds.annotations)
    assert [f.name for f in ds.img_files] == ["img_0.jpg", "img_2.jpg"]


def test_filter_empty_annotations_logs_warning(tmp_path, caplog):
    labeled = "0 0.5 0.5 0.25 0.5\n"
    with caplog.at_level(logging.WARNING, logger="libreyolo.data.dataset"):
        _build_mixed_dataset(tmp_path, [labeled, "", ""], filter_empty_annotations=True)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "2" in msg and "1" in msg  # 2 dropped, 1 remain


def test_filter_empty_annotations_raises_when_all_unlabeled(tmp_path):
    with pytest.raises(ValueError, match="No labeled images remain"):
        _build_mixed_dataset(tmp_path, ["", ""], filter_empty_annotations=True)


def test_filter_empty_annotations_false_keeps_unlabeled(tmp_path):
    labeled = "0 0.5 0.5 0.25 0.5\n"
    ds = _build_mixed_dataset(tmp_path, [labeled, ""], filter_empty_annotations=False)

    assert ds.num_imgs == 2


def test_filter_empty_annotations_syncs_segments(tmp_path):
    # Polygon label: class + 4 xy pairs (> 5 parts triggers segment parsing)
    poly = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
    ds = _build_mixed_dataset(
        tmp_path, [poly, "", poly], filter_empty_annotations=True, load_segments=True
    )

    assert ds.num_imgs == 2
    assert len(ds.segments) == 2


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
