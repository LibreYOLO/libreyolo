"""val(split=...) on a dataset yaml whose entry for that split is empty."""

from __future__ import annotations

import pytest
import yaml
from PIL import Image

pytestmark = pytest.mark.unit


@pytest.fixture
def yaml_without_test_images(tmp_path):
    """Like coco8.yaml: ``test:`` is present but empty."""
    root = tmp_path / "data"
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(root / "images" / "val" / "a.jpg")
    (root / "labels" / "val" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    path = root / "data.yaml"
    path.write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "images/val", "val": "images/val",
             "test": None, "names": {0: "a"}}
        )
    )
    return str(path)


@pytest.mark.parametrize(
    "module,class_name,size,task",
    [("yolo9", "LibreYOLO9", "t", "detect"), ("yolonas", "LibreYOLONAS", "s", "obb")],
)
def test_empty_split_entry_names_the_split(
    yaml_without_test_images, module, class_name, size, task
):
    """This was TypeError: unsupported operand type(s) for /: 'PosixPath' and 'NoneType'."""
    import importlib

    cls = getattr(importlib.import_module(f"libreyolo.models.{module}.model"), class_name)
    model = cls(None, size=size, device="cpu", task=task)

    with pytest.raises(FileNotFoundError, match="no 'test' split"):
        model.val(data=yaml_without_test_images, split="test", workers=0)
