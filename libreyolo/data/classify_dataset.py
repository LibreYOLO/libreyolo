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
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder

from .utils import DATASETS_DIR

logger = logging.getLogger(__name__)

# ImageNet channel statistics — the standard normalization for ImageNet-style
# classification backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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


def _interp_mode(interpolation) -> InterpolationMode:
    if isinstance(interpolation, InterpolationMode):
        return interpolation
    return {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST,
    }.get(str(interpolation).lower(), InterpolationMode.BILINEAR)


# Valid values for the ``auto_augment`` knob, mapped to their torchvision class.
AUTO_AUGMENT_POLICIES = ("randaugment", "autoaugment", "augmix")

#: Default ``RandomResizedCrop`` area range for classification training.
DEFAULT_CROP_SCALE = (0.5, 1.0)

#: Default shorter-side resize ratio for the deterministic eval crop.
DEFAULT_CROP_PCT = 0.875


def _build_auto_augment(name: str, mode: InterpolationMode):
    """Return the torchvision auto-augment transform for ``name``.

    ``name`` is validated against :data:`AUTO_AUGMENT_POLICIES`; unknown values
    raise a ``ValueError`` listing the accepted policies. These transforms
    operate on PIL / uint8 images, so they are inserted before ``ToTensor``.
    """
    key = str(name).lower()
    if key == "randaugment":
        return transforms.RandAugment(interpolation=mode)
    if key == "autoaugment":
        return transforms.AutoAugment(interpolation=mode)
    if key == "augmix":
        return transforms.AugMix(interpolation=mode)
    raise ValueError(
        f"Unknown auto_augment {name!r}. Valid values are "
        f"{', '.join(AUTO_AUGMENT_POLICIES)} or None."
    )


def normalize_crop_scale(scale) -> tuple[float, float]:
    """Normalize the classification ``scale`` knob to a ``(min, max)`` pair.

    Accepts the ecosystem spelling (a single float, the lower bound, upper
    bound implied 1.0) or an explicit two-value sequence. Raises ``ValueError``
    for anything outside ``0 < min <= max <= 1``.
    """
    if isinstance(scale, (int, float)):
        pair = (float(scale), 1.0)
    else:
        values = tuple(float(v) for v in scale)
        if len(values) != 2:
            raise ValueError(
                f"scale must be a float or two values (min, max), got {scale!r}"
            )
        pair = values
    lo, hi = pair
    if not 0.0 < lo <= hi <= 1.0:
        raise ValueError(
            f"scale must satisfy 0 < min <= max <= 1, got ({lo}, {hi})"
        )
    return pair


def build_classify_transforms(
    imgsz: int,
    augment: bool,
    *,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    crop_pct: float = DEFAULT_CROP_PCT,
    interpolation="bilinear",
    auto_augment: str | None = None,
    erasing: float = 0.0,
    square_resize: bool = False,
    scale=DEFAULT_CROP_SCALE,
):
    """Build train/val image transforms for classification.

    Training uses a random-resized crop plus horizontal flip; validation uses a
    deterministic shorter-side resize (``floor(imgsz / crop_pct)``) and center
    crop. ``crop_pct`` and ``interpolation`` let a model family match its native
    eval pipeline (e.g. bicubic + 0.95 crop) so ``model.val()`` agrees with
    ``model.predict()``. Normalization defaults to ImageNet stats; families with
    their own preprocessing (e.g. CLIP, which uses its own mean/std + bicubic and
    a 1.0 crop ratio) override ``mean``/``std``/``interpolation``/``crop_pct``.

    ``scale`` is the ``RandomResizedCrop`` area range for training. It accepts
    the ecosystem spelling (a single float, the lower bound) or an explicit
    ``(min, max)`` pair; the default ``(0.5, 1.0)`` is unchanged. Lower it for a
    more aggressive crop, raise it (e.g. ``scale=0.9``) when the subject fills
    the frame and cropping it away costs accuracy.

    Two optional training-only knobs strengthen the train pipeline (both default
    off, so the composition is unchanged unless requested):

    - ``auto_augment``: one of ``"randaugment"``, ``"autoaugment"``,
      ``"augmix"`` (or ``None``). Inserted after the horizontal flip and before
      ``ToTensor`` since these transforms act on PIL / uint8 images.
    - ``erasing``: probability for ``RandomErasing``, appended after
      ``Normalize`` (tensor space, the standard placement). Must satisfy
      ``0 <= erasing < 1``.

    Both only affect the ``augment=True`` branch; the val pipeline is untouched.
    """
    import math

    mode = _interp_mode(interpolation)
    normalize = transforms.Normalize(mean=mean, std=std)
    if augment and square_resize:
        # The square-resize path is a val-only pipeline; combining it with the
        # random-resized-crop train pipeline is not defined. Fail loudly rather
        # than silently ignoring square_resize (the augment branch returns first).
        raise ValueError(
            "square_resize=True is only supported with augment=False "
            "(it is a deterministic validation transform)."
        )
    if augment:
        ops = [
            transforms.RandomResizedCrop(
                imgsz, scale=normalize_crop_scale(scale), interpolation=mode
            ),
            transforms.RandomHorizontalFlip(),
        ]
        if auto_augment is not None:
            ops.append(_build_auto_augment(auto_augment, mode))
        ops.append(transforms.ToTensor())
        ops.append(normalize)
        if erasing:
            if not (0.0 <= erasing < 1.0):
                raise ValueError(
                    f"erasing must be in [0, 1), got {erasing}."
                )
            ops.append(transforms.RandomErasing(p=erasing, inplace=True))
        return transforms.Compose(ops)
    if square_resize:
        # Squash to a fixed square (no aspect-preserving resize + center crop).
        # SigLIP's native eval pipeline resizes directly to (imgsz, imgsz).
        return transforms.Compose(
            [
                transforms.Resize((imgsz, imgsz), interpolation=mode),
                transforms.ToTensor(),
                normalize,
            ]
        )
    resize = int(math.floor(imgsz / crop_pct))
    return transforms.Compose(
        [
            transforms.Resize(resize, interpolation=mode),
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
        transform_kwargs: Dict | None = None,
    ):
        self.root = Path(dataset_root)
        self.split = split
        self.imgsz = imgsz
        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        transform = build_classify_transforms(imgsz, augment, **(transform_kwargs or {}))
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


class _ClassifyBatchMixer:
    """Batch-level MixUp / CutMix wrapper for the classification collate path.

    Wraps :func:`classify_collate_fn`, then with the configured probability
    applies torchvision's ``v2.MixUp`` / ``v2.CutMix`` to the stacked batch.
    These ops need ``num_classes`` and emit soft (class-probability) label
    tensors of shape ``[B, num_classes]`` whose rows sum to 1, which the
    cross-entropy criterion consumes directly.

    Probability semantics: at most one op is applied per batch, from a single
    draw ``r``. MixUp is applied when ``r < mixup``; otherwise CutMix is applied
    when ``r < mixup + cutmix``. So ``mixup`` is honored exactly as MixUp's
    per-batch probability and ``cutmix`` as CutMix's (the two are additive and
    should sum to at most 1). With a single op enabled this reduces to applying
    that op with its own probability. When neither is enabled the plain collate
    is used (see :func:`build_classify_collate`), so default behavior is
    unchanged.
    """

    def __init__(self, num_classes: int, mixup: float = 0.0, cutmix: float = 0.0):
        from torchvision.transforms import v2

        self._mixup = v2.MixUp(num_classes=num_classes) if mixup > 0 else None
        self._cutmix = v2.CutMix(num_classes=num_classes) if cutmix > 0 else None
        if self._mixup is None and self._cutmix is None:
            raise ValueError("_ClassifyBatchMixer needs mixup>0 or cutmix>0.")
        self._mixup_p = float(mixup)
        self._cutmix_p = float(cutmix)

    def __call__(self, batch):
        imgs, labels, img_infos, img_ids = classify_collate_fn(batch)
        r = float(torch.rand(1).item())
        if self._mixup is not None and r < self._mixup_p:
            imgs, labels = self._mixup(imgs, labels)
        elif self._cutmix is not None and r < self._mixup_p + self._cutmix_p:
            imgs, labels = self._cutmix(imgs, labels)
        return imgs, labels, img_infos, img_ids


def build_classify_collate(num_classes: int, mixup: float = 0.0, cutmix: float = 0.0):
    """Return the classification collate function for the given mixing knobs.

    With ``mixup == 0`` and ``cutmix == 0`` this returns :func:`classify_collate_fn`
    unchanged (byte-identical batches, so default training is unaffected).
    Otherwise it returns a :class:`_ClassifyBatchMixer` that applies MixUp / CutMix
    at the batch level and produces soft labels.
    """
    for name, value in (("mixup", mixup), ("cutmix", cutmix)):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {value}.")
    if mixup == 0 and cutmix == 0:
        return classify_collate_fn
    return _ClassifyBatchMixer(num_classes, mixup=mixup, cutmix=cutmix)
