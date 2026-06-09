"""Synthetic-dataset builder for doctor tests."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image


class FakeDataset:
    """Builds a YOLO-format detection dataset under a temp directory."""

    def __init__(self, root: Path, nc: int, names: list[str] | dict) -> None:
        self.root = root
        self.nc = nc
        self.names = names
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        self.yaml_path = root / "data.yaml"
        self._write_yaml()

    def _write_yaml(self, **extra) -> None:
        config = {
            "path": str(self.root),
            "train": "images/train",
            "val": "images/val",
            "nc": self.nc,
            "names": self.names,
            **extra,
        }
        self.yaml_path.write_text(yaml.safe_dump(config))

    def set_yaml(self, **extra) -> None:
        self._write_yaml(**extra)

    def image(
        self,
        split: str,
        name: str,
        size: tuple[int, int] = (64, 64),
        color: tuple[int, int, int] | None = None,
        seed: int | None = None,
        exif_orientation: int | None = None,
        raw_bytes: bytes | None = None,
        mode: str = "RGB",
    ) -> Path:
        path = self.root / "images" / split / name
        if raw_bytes is not None:
            path.write_bytes(raw_bytes)
            return path
        if seed is not None:
            rng = np.random.default_rng(seed)
            arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            im = Image.fromarray(arr, "RGB")
        else:
            im = Image.new("RGB", size, color or (120, 60, 200))
        if mode != "RGB":
            im = im.convert(mode)
        kwargs = {}
        if exif_orientation is not None:
            exif = Image.Exif()
            exif[0x0112] = exif_orientation
            kwargs["exif"] = exif
        im.save(path, **kwargs)
        return path

    def label(self, split: str, name: str, text: str) -> Path:
        path = self.root / "labels" / split / name
        path.write_text(text)
        return path

    def sample(
        self, split: str, name: str, boxes: str = "0 0.5 0.5 0.2 0.2\n", **img_kwargs
    ) -> None:
        """One image with its label file in a single call."""
        stem = Path(name).stem
        self.image(split, name, **img_kwargs)
        self.label(split, f"{stem}.txt", boxes)


@pytest.fixture
def make_dataset(tmp_path):
    def factory(nc: int = 2, names=None) -> FakeDataset:
        if names is None:
            names = (["cat", "dog"] + [f"class{i}" for i in range(2, nc)])[:nc]
        return FakeDataset(tmp_path / "ds", nc, names)

    return factory


def finding_ids(report) -> set[str]:
    return {f.check_id for f in report.findings}


def findings_for(report, check_id: str):
    return [f for f in report.findings if f.check_id == check_id]
