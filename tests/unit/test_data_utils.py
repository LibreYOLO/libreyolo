"""Tests for dataset config loading safety."""

from pathlib import Path

import pytest

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
                f"  Path(r\"{marker_path}\").write_text(\"ran\")",
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
