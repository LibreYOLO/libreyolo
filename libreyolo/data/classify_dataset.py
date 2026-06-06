"""Image-classification dataset for LibreYOLO.

LibreYOLO classification follows the de-facto folder layout used across the
ecosystem: a dataset root holding ``train/`` and ``val/`` (optionally
``test/``) sub-directories, each with one sub-folder per class::

    imagenet10/
        train/
            n01440764/  *.JPEG
            n02102040/  *.JPEG
            ...
        val/
            n01440764/  *.JPEG
            ...

This mirrors the user-facing convention so ``model.train(data="imagenet10")``
behaves the way users expect. The class list is the sorted set of sub-folder
names, identical across splits.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlopen

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .utils import DATASETS_DIR

logger = logging.getLogger(__name__)

# ImageNet channel statistics — the standard normalization for ImageNet-style
# classification backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# Small classification datasets that can be fetched by bare name. These are the
# folder-format sets the wider ecosystem uses for CPU/CI runs and quick checks.
# ``imagenet10`` is a 10-image-per-split smoke set; ``imagenette160`` is a real
# 10-class subset (~9k train images at 160px) for accuracy validation.
_KNOWN_DATASETS: Dict[str, str] = {
    "imagenet10": "https://github.com/ultralytics/assets/releases/download/v0.0.0/imagenet10.zip",
    "imagenette160": "https://github.com/ultralytics/assets/releases/download/v0.0.0/imagenette160.zip",
}


def _find_train_root(base: Path) -> Path | None:
    """Locate the directory that holds the ``train`` split under ``base``."""
    if not base.is_dir():
        return None
    if (base / "train").is_dir():
        return base
    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / "train").is_dir():
            return child
    return None


def _download_and_extract(url: str, name: str) -> Path:
    """Download a ``.zip`` dataset into ``DATASETS_DIR/<name>`` and extract it.

    Returns the directory that contains the ``train``/``val`` split folders
    (which may be ``DATASETS_DIR/<name>`` or a wrapper directory inside it,
    depending on how the archive was packed).
    """
    dest_dir = DATASETS_DIR / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{name}.zip"

    if not zip_path.exists():
        logger.info("Downloading classification dataset %s -> %s", url, zip_path)
        with urlopen(url) as response, open(zip_path, "wb") as out:  # noqa: S310
            out.write(response.read())

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    root = _find_train_root(dest_dir)
    if root is None:
        raise FileNotFoundError(
            f"Downloaded {url} but could not locate a 'train' split under {dest_dir}."
        )
    return root


def resolve_classify_data(data: str | Path) -> Path:
    """Resolve a classification ``data`` argument to a dataset root directory.

    Accepts:
      - a path to a directory that already contains a ``train`` split,
      - a known dataset name (e.g. ``"imagenet10"``) that is auto-downloaded,
      - a ``.zip`` URL.

    Returns the dataset root directory (containing ``train``/``val``).
    """
    if data is None:
        raise ValueError("Classification training requires data= (a dataset root or name).")

    data_str = str(data)
    path = Path(data_str)

    # Already a local dataset root.
    if path.is_dir():
        if (path / "train").is_dir():
            return path
        # A bare split directory was passed (e.g. ".../train") — use its parent
        # only when it also exposes the split as a sibling layout.
        if path.name in ("train", "val", "test") and (path.parent / "train").is_dir():
            return path.parent
        raise FileNotFoundError(
            f"Classification data directory {path} has no 'train/' sub-folder. "
            "Expected an ImageFolder layout: <root>/train/<class>/*.jpg."
        )

    # Known name or URL -> download.
    name = data_str.lower()
    url = _KNOWN_DATASETS.get(name)
    if url is None and data_str.endswith(".zip") and "://" in data_str:
        url = data_str
        name = Path(data_str).stem
    if url is not None:
        cached = _find_train_root(DATASETS_DIR / name)
        if cached is not None:
            return cached
        return _download_and_extract(url, name)

    raise FileNotFoundError(
        f"Could not resolve classification dataset {data_str!r}. Pass a directory "
        f"with a train/ split, a .zip URL, or a known name ({', '.join(_KNOWN_DATASETS)})."
    )


def get_class_names(dataset_root: str | Path, split: str = "train") -> List[str]:
    """Return the sorted class-folder names for a dataset split."""
    split_dir = Path(dataset_root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    classes = sorted(
        entry.name for entry in split_dir.iterdir() if entry.is_dir()
    )
    if not classes:
        raise FileNotFoundError(f"No class sub-folders found under {split_dir}.")
    return classes


def build_classify_transforms(imgsz: int, augment: bool):
    """Build train/val image transforms for classification.

    Training uses a random-resized crop plus horizontal flip; validation uses a
    deterministic resize and center crop. Both normalize with ImageNet stats.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(imgsz, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    resize = int(round(imgsz / 0.875))
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(imgsz),
            transforms.ToTensor(),
            normalize,
        ]
    )


class ClassifyDataset(Dataset):
    """ImageFolder-backed classification dataset returning ``(image, label)``.

    The class-to-index mapping is fixed from the ``train`` split so train/val
    share identical label indices.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        imgsz: int,
        augment: bool,
        class_to_idx: Dict[str, int] | None = None,
    ):
        self.root = Path(dataset_root)
        self.split = split
        self.imgsz = imgsz
        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        transform = build_classify_transforms(imgsz, augment)
        self._impl = ImageFolder(str(split_dir), transform=transform)

        # Pin the label mapping to the train split when supplied so val labels
        # line up with the head's output indices.
        if class_to_idx is not None:
            remap = {
                old_idx: class_to_idx[name]
                for name, old_idx in self._impl.class_to_idx.items()
                if name in class_to_idx
            }
            self._impl.samples = [
                (p, remap[old]) for p, old in self._impl.samples if old in remap
            ]
            self._impl.targets = [t for _, t in self._impl.samples]
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = self._impl.class_to_idx

        self.classes = [
            name for name, _ in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])
        ]

    def __len__(self) -> int:
        return len(self._impl)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self._impl[idx]


def classify_collate_fn(batch):
    """Collate ``(image, label)`` pairs into the trainer's 4-tuple batch shape.

    Returns ``(imgs, labels, img_infos, img_ids)`` so the shared training loop
    (which unpacks a 4- or 5-tuple) can drive classification unchanged: ``imgs``
    is ``[B,3,H,W]`` float and ``labels`` is a ``[B]`` long tensor that the
    classification head consumes as cross-entropy targets.
    """
    imgs = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([int(item[1]) for item in batch], dtype=torch.long)
    img_infos = [{} for _ in batch]
    img_ids = list(range(len(batch)))
    return imgs, labels, img_infos, img_ids
