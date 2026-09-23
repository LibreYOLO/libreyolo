"""Image-classification dataset for LibreYOLO.

LibreYOLO classification uses the de-facto ImageFolder layout: a dataset root
holding ``train/`` and ``val/`` (optionally ``test/``) sub-directories, each
with one sub-folder per class::

    dataset_root/
        train/
            class_a/  *.jpg
            class_b/  *.jpg
            ...
        val/
            class_a/  *.jpg
            ...

The class list is the sorted set of sub-folder names, identical across splits,
so ``model.train(data="smoke10")`` behaves the way users expect.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlopen

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

# The transform / collate recipe lives in the augment package next to the
# detection recipes; these names are re-exported here for existing importers.
from .augment.classify import (  # noqa: F401
    AUTO_AUGMENT_POLICIES,
    DEFAULT_CROP_PCT,
    DEFAULT_CROP_SCALE,
    DEFAULT_FLIP_PROB,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ClassifyAugKnobs,
    ClassifyBatchMixer,
    _ClassifyBatchMixer,
    build_classify_collate,
    build_classify_transforms,
    classify_collate_fn,
    normalize_auto_augment,
    normalize_crop_scale,
)
from .utils import DATASETS_DIR

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# Small classification datasets that can be fetched by bare name, hosted under
# the LibreYOLO HF org and rebuilt from clean upstream sources (Apache-2.0
# Imagenette; see scripts/build_imagenette.py). ``smoke10`` is a tiny
# 2-image-per-class CI smoke set; ``imagenette160`` is the full 10-class subset
# (~9k train images at 160px) for accuracy validation.
_KNOWN_DATASETS: Dict[str, str] = {
    "smoke10": "https://huggingface.co/datasets/LibreYOLO/smoke10/resolve/main/smoke10.zip",
    "imagenette160": "https://huggingface.co/datasets/LibreYOLO/imagenette160/resolve/main/imagenette160.zip",
}


def _safe_extract_zip(zf: zipfile.ZipFile, dest_dir: Path) -> None:
    """Extract a zip, rejecting entries that escape ``dest_dir`` (zip-slip).

    Archives can be fetched from arbitrary URLs, so a crafted member with an
    absolute path or ``..`` components could otherwise write outside the
    dataset cache. Each resolved member path is verified to stay within
    ``dest_dir`` before extraction.
    """
    dest_root = dest_dir.resolve()
    for member in zf.namelist():
        target = (dest_dir / member).resolve()
        if target != dest_root and dest_root not in target.parents:
            raise ValueError(
                f"Unsafe path in archive (escapes dataset directory): {member!r}"
            )
    zf.extractall(dest_dir)


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
        _safe_extract_zip(zf, dest_dir)

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
      - a known dataset name (e.g. ``"smoke10"``) that is auto-downloaded,
      - a ``.zip`` URL.

    Returns the dataset root directory (containing ``train``/``val``).
    """
    if data is None:
        raise ValueError(
            "Classification training requires data= (a dataset root or name)."
        )

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
    classes = sorted(entry.name for entry in split_dir.iterdir() if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class sub-folders found under {split_dir}.")
    return classes


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
        transform_kwargs: Dict | None = None,
        transform=None,
    ):
        self.root = Path(dataset_root)
        self.split = split
        self.imgsz = imgsz
        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.augment = augment
        self._transform_kwargs: Dict = dict(transform_kwargs or {})
        if transform is not None:
            # An explicit eval transform (the model's own, #886) replaces the
            # generic one; augmentation has no meaning for it.
            if augment:
                raise ValueError("an explicit transform is for evaluation only")
            if transform_kwargs:
                raise ValueError("pass transform or transform_kwargs, not both")
        else:
            transform = build_classify_transforms(
                imgsz, augment, **self._transform_kwargs
            )
        self._impl = ImageFolder(str(split_dir), transform=transform)

        # Pin the label mapping to the train split when supplied so val labels
        # line up with the head's output indices.
        if class_to_idx is not None:
            expected = set(class_to_idx)
            actual = set(self._impl.class_to_idx)
            unknown = sorted(actual - expected)
            missing = sorted(expected - actual)
            if unknown or missing:
                details = []
                if unknown:
                    details.append(f"unknown classes: {unknown}")
                if missing:
                    details.append(f"missing classes: {missing}")
                raise ValueError(
                    f"Classification split '{split}' classes must match the "
                    "expected class set from training/checkpoint names "
                    f"({'; '.join(details)})."
                )
            remap = {
                old_idx: class_to_idx[name]
                for name, old_idx in self._impl.class_to_idx.items()
            }
            self._impl.samples = [(p, remap[old]) for p, old in self._impl.samples]
            self._impl.targets = [t for _, t in self._impl.samples]
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = self._impl.class_to_idx

        self.classes = [
            name for name, _ in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])
        ]

    @property
    def transform(self):
        """The active torchvision transform (train or eval pipeline)."""
        return self._impl.transform

    def close_strong_aug(self) -> None:
        """Switch off the strong training augmentations in place.

        Rebuilds the train transform with ``auto_augment`` and ``erasing`` off
        while keeping the crop and flip geometry, mirroring ``close_mosaic`` on
        the detection datasets. The trainer calls this at the ``no_aug_epochs``
        boundary; it is a no-op for eval datasets.
        """
        if not self.augment:
            return
        kwargs = dict(self._transform_kwargs)
        kwargs.update(
            {k: v for k, v in ClassifyAugKnobs.STRONG_OFF.items() if k in ("auto_augment", "erasing")}
        )
        self._transform_kwargs = kwargs
        self._impl.transform = build_classify_transforms(self.imgsz, True, **kwargs)

    def __len__(self) -> int:
        return len(self._impl)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self._impl[idx]
