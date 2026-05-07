"""Empty-label warning during dataset load (issue #188)."""

import logging

import pytest
from PIL import Image

from libreyolo.data.dataset import YOLODataset

pytestmark = pytest.mark.unit


def _build_dataset(tmp_path, label_contents):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    img_files, label_files = [], []
    for index, body in enumerate(label_contents):
        img = image_dir / f"sample_{index}.jpg"
        lbl = label_dir / f"sample_{index}.txt"
        Image.new("RGB", (64, 64), color="white").save(img)
        lbl.write_text(body)
        img_files.append(img)
        label_files.append(lbl)

    return YOLODataset(img_files=img_files, label_files=label_files, img_size=(64, 64))


def test_warns_when_all_label_files_empty(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="libreyolo.data.dataset"):
        _build_dataset(tmp_path, ["", "", ""])

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "no labels" in warnings[0].getMessage().lower()


def test_silent_when_any_labels_present(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="libreyolo.data.dataset"):
        _build_dataset(tmp_path, ["0 0.5 0.5 0.25 0.5\n", "", ""])

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
