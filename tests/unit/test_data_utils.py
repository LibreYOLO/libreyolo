"""Tests for dataset config loading safety and path resolution."""

from pathlib import Path

import pytest
import yaml
from PIL import Image

from libreyolo.data.utils import load_data_config

pytestmark = pytest.mark.unit


def _write_dataset_yaml(tmp_path: Path) -> Path:
    yaml_path = tmp_path / "scripted.yaml"
    marker_path = tmp_path / "marker.txt"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {tmp_path / 'dataset'}",
                "train: images/train",
                "val: images/val",
                "download: |",
                f'  Path(r"{marker_path}").write_text("ran")',
            ]
        )
    )
    return yaml_path


def test_load_data_config_blocks_embedded_scripts_by_default(tmp_path):
    yaml_path = _write_dataset_yaml(tmp_path)
    marker_path = tmp_path / "marker.txt"

    load_data_config(str(yaml_path), autodownload=True)

    assert not marker_path.exists()


def test_load_data_config_can_opt_in_to_embedded_scripts(tmp_path):
    yaml_path = _write_dataset_yaml(tmp_path)
    marker_path = tmp_path / "marker.txt"

    load_data_config(str(yaml_path), autodownload=True, allow_scripts=True)

    assert marker_path.read_text() == "ran"


def test_embedded_scripts_map_common_yolo_helper_imports(tmp_path, monkeypatch):
    import libreyolo.data.utils as data_utils

    marker_path = tmp_path / "marker.txt"
    yaml_path = tmp_path / "scripted.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {tmp_path / 'dataset'}",
                "train: images/train",
                "val: images/val",
                "download: |",
                "  from ultralytics.utils.downloads import download",
                "  from ultralytics.utils import ASSETS_URL",
                f'  Path(r"{marker_path}").write_text(ASSETS_URL)',
                "  download([ASSETS_URL + '/labels.zip'], dir=path)",
            ]
        )
    )

    monkeypatch.setattr(data_utils, "ASSETS_URL", "libreyolo-assets")
    called = {}

    def fake_download(urls, dir=".", unzip=True, delete=True, threads=1):
        called["urls"] = urls
        called["dir"] = dir

    monkeypatch.setattr(data_utils, "download", fake_download)

    load_data_config(str(yaml_path), autodownload=True, allow_scripts=True)

    assert marker_path.read_text() == "libreyolo-assets"
    assert called == {
        "urls": ["libreyolo-assets/labels.zip"],
        "dir": tmp_path / "dataset",
    }


def test_load_data_config_resolves_directory_test_split(tmp_path):
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "test" / "images"
    labels_dir = dataset_root / "test" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    image_path = images_dir / "sample.jpg"
    Image.new("RGB", (32, 32), color="white").save(image_path)
    label_path = labels_dir / "sample.txt"
    label_path.write_text("0 0.5 0.5 0.25 0.25\n")

    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "names": ["marble"],
                "nc": 1,
            }
        )
    )

    config = load_data_config(str(yaml_path), autodownload=False)

    assert config["test"] == str(images_dir)
    assert config["test_img_files"] == [image_path]
    assert config["test_label_files"] == [label_path]
