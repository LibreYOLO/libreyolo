"""Classification augmentation recipe.

This is the classification counterpart of the per-family detection recipe
modules in this package (``yolox``, ``yolo9``, ``rfdetr`` ...). Everything
that turns a ``TrainConfig`` into image transforms or a batch collate for the
ImageFolder classification pipeline lives here:

- :func:`build_classify_transforms`: the train / eval torchvision pipelines.
- :class:`ClassifyAugKnobs`: the one place that reads the classification
  augmentation knobs off a training config, validates them, and knows which
  of them count as *strong* augmentation for the ``no_aug_epochs`` tail.
- :func:`build_classify_collate` / :class:`ClassifyBatchMixer`: batch-level
  MixUp / CutMix with soft labels.

The knob names follow the de-facto YOLO CLI conventions (``scale``,
``fliplr``/``flip_prob``, ``flipud``, ``auto_augment``, ``erasing``,
``mixup``, ``cutmix``); which family honors which knob is declared in
:mod:`libreyolo.data.augment.spec` and pinned by the unit tests.

This module imports torchvision, so it is intentionally *not* re-exported
from ``libreyolo.data.augment`` (whose numpy core stays torch-free). Import it
explicitly: ``from libreyolo.data.augment.classify import ...``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ImageNet channel statistics: the standard normalization for ImageNet-style
# classification backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Valid values for the ``auto_augment`` knob, mapped to their torchvision class.
AUTO_AUGMENT_POLICIES = ("randaugment", "autoaugment", "augmix")

#: Default RandomResizedCrop area range for classification training.
DEFAULT_CROP_SCALE = (0.5, 1.0)
#: Default eval resize ratio before the center crop (``resize = imgsz / crop_pct``).
DEFAULT_CROP_PCT = 0.875
#: Default horizontal-flip probability (the historical fixed value).
DEFAULT_FLIP_PROB = 0.5

ScaleSpec = float | tuple[float, float]


def _interp_mode(interpolation) -> InterpolationMode:
    if isinstance(interpolation, InterpolationMode):
        return interpolation
    return {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST,
    }.get(str(interpolation).lower(), InterpolationMode.BILINEAR)


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


def normalize_auto_augment(value) -> str | None:
    """Normalize the ``auto_augment`` knob: ``None`` / ``""`` / ``"none"`` mean off."""
    if value is None:
        return None
    key = str(value).strip().lower()
    if key in ("", "none", "null", "false"):
        return None
    if key not in AUTO_AUGMENT_POLICIES:
        raise ValueError(
            f"Unknown auto_augment {value!r}. Valid values are "
            f"{', '.join(AUTO_AUGMENT_POLICIES)} or None."
        )
    return key


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


def _probability(value, name: str, *, exclusive_upper: bool = False) -> float:
    try:
        p = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}") from None
    if math.isnan(p) or p < 0.0 or p > 1.0 or (exclusive_upper and p >= 1.0):
        upper = "1)" if exclusive_upper else "1]"
        raise ValueError(f"{name} must be in [0, {upper}, got {value}")
    return p


def validate_mix_probabilities(mixup, cutmix) -> tuple[float, float]:
    """Validate the batch-mixing knobs together.

    The mixer takes one random draw per batch: MixUp fires when
    ``r < mixup``, otherwise CutMix when ``r < mixup + cutmix``. A pair whose
    sum exceeds 1 would silently run CutMix with less than the requested
    probability, so it is rejected here instead.
    """
    mixup_p = _probability(mixup, "mixup")
    cutmix_p = _probability(cutmix, "cutmix")
    if mixup_p + cutmix_p > 1.0 + 1e-9:
        raise ValueError(
            "mixup + cutmix must be <= 1 (one op runs per batch, so CutMix "
            f"would only fire with probability {max(0.0, 1.0 - mixup_p):.3g}); "
            f"got mixup={mixup_p}, cutmix={cutmix_p}"
        )
    return mixup_p, cutmix_p


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
    scale: ScaleSpec = DEFAULT_CROP_SCALE,
    flip_prob: float = DEFAULT_FLIP_PROB,
    flipud: float = 0.0,
):
    """Build train/val image transforms for classification.

    Training (``augment=True``)::

        RandomResizedCrop(imgsz, scale)      # always
        RandomHorizontalFlip(flip_prob)      # when flip_prob > 0
        RandomVerticalFlip(flipud)           # when flipud > 0
        <auto_augment policy>                # when set (PIL space)
        ToTensor, Normalize(mean, std)
        RandomErasing(erasing)               # when erasing > 0 (tensor space)

    Evaluation (``augment=False``) is deterministic: a shorter-side resize to
    ``floor(imgsz / crop_pct)`` followed by ``CenterCrop(imgsz)``, or a plain
    square ``Resize((imgsz, imgsz))`` when ``square_resize`` is set. ``crop_pct``
    and ``interpolation`` let a model family match its native eval pipeline
    (e.g. bicubic + 0.95 crop) so ``model.val()`` agrees with ``model.predict()``.
    Normalization defaults to ImageNet stats; families with their own
    preprocessing (e.g. CLIP) override ``mean``/``std``/``interpolation``/``crop_pct``.

    At their defaults the training knobs reproduce the historical pipeline
    exactly (crop + 0.5 horizontal flip + normalize), so existing runs are
    unchanged unless a knob is set.
    """
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
        flip_p = _probability(flip_prob, "flip_prob")
        flipud_p = _probability(flipud, "flipud")
        erasing_p = _probability(erasing, "erasing", exclusive_upper=True)
        ops: list = [
            transforms.RandomResizedCrop(
                imgsz, scale=normalize_crop_scale(scale), interpolation=mode
            ),
        ]
        if flip_p > 0:
            ops.append(transforms.RandomHorizontalFlip(p=flip_p))
        if flipud_p > 0:
            ops.append(transforms.RandomVerticalFlip(p=flipud_p))
        if auto_augment is not None:
            ops.append(_build_auto_augment(auto_augment, mode))
        ops.append(transforms.ToTensor())
        ops.append(normalize)
        if erasing_p > 0:
            ops.append(transforms.RandomErasing(p=erasing_p, inplace=True))
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
    if not 0.0 < float(crop_pct) <= 1.0:
        raise ValueError(f"crop_pct must be in (0, 1], got {crop_pct}")
    resize = math.floor(imgsz / crop_pct)
    return transforms.Compose(
        [
            transforms.Resize(resize, interpolation=mode),
            transforms.CenterCrop(imgsz),
            transforms.ToTensor(),
            normalize,
        ]
    )


@dataclass(frozen=True)
class ClassifyAugKnobs:
    """The classification augmentation knobs, read once off a training config.

    This is the single place that knows which ``TrainConfig`` fields drive the
    classification pipeline and what their off-values are. The trainer builds
    one of these and passes :meth:`transform_kwargs` to the dataset and
    :meth:`collate_kwargs` to the collate builder, so adding a knob means
    adding a field here, not another ``getattr`` in the trainer.

    Two groups:

    - *geometry* (``scale``, ``flip_prob``, ``flipud``): cheap, always on.
    - *strong* (``auto_augment``, ``erasing``, ``mixup``, ``cutmix``): the
      regularizers that ``no_aug_epochs`` switches off for the final epochs,
      mirroring how detection closes mosaic/mixup (see :meth:`weak`).
    """

    scale: tuple[float, float] = DEFAULT_CROP_SCALE
    flip_prob: float = DEFAULT_FLIP_PROB
    flipud: float = 0.0
    auto_augment: str | None = None
    erasing: float = 0.0
    mixup: float = 0.0
    cutmix: float = 0.0

    #: Knobs that ``no_aug_epochs`` disables, with their off-values.
    STRONG_OFF: ClassVar[dict[str, Any]] = {
        "auto_augment": None,
        "erasing": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
    }

    @classmethod
    def from_config(cls, config: Any) -> ClassifyAugKnobs:
        """Read and validate the knobs from a ``TrainConfig``-like object.

        Missing attributes fall back to the defaults, so plain objects and
        older configs work. Invalid values raise ``ValueError`` here, before
        any data is loaded.
        """
        mixup, cutmix = validate_mix_probabilities(
            getattr(config, "mixup", 0.0), getattr(config, "cutmix", 0.0)
        )
        return cls(
            scale=normalize_crop_scale(getattr(config, "scale", DEFAULT_CROP_SCALE)),
            flip_prob=_probability(getattr(config, "flip_prob", DEFAULT_FLIP_PROB), "flip_prob"),
            flipud=_probability(getattr(config, "flipud", 0.0), "flipud"),
            auto_augment=normalize_auto_augment(getattr(config, "auto_augment", None)),
            erasing=_probability(getattr(config, "erasing", 0.0), "erasing", exclusive_upper=True),
            mixup=mixup,
            cutmix=cutmix,
        )

    def transform_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :func:`build_classify_transforms` (train side)."""
        return {
            "scale": self.scale,
            "flip_prob": self.flip_prob,
            "flipud": self.flipud,
            "auto_augment": self.auto_augment,
            "erasing": self.erasing,
        }

    def collate_kwargs(self) -> dict[str, float]:
        """Keyword arguments for :func:`build_classify_collate`."""
        return {"mixup": self.mixup, "cutmix": self.cutmix}

    @property
    def has_strong(self) -> bool:
        """Whether any strong (``no_aug_epochs``-gated) knob is active."""
        return any(getattr(self, k) != off for k, off in self.STRONG_OFF.items())

    def weak(self) -> ClassifyAugKnobs:
        """The same knobs with every strong regularizer switched off."""
        return replace(self, **self.STRONG_OFF)


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


class ClassifyBatchMixer:
    """Batch-level MixUp / CutMix wrapper for the classification collate path.

    Wraps :func:`classify_collate_fn`, then with the configured probability
    applies torchvision's ``v2.MixUp`` / ``v2.CutMix`` to the stacked batch.
    These ops need ``num_classes`` and emit soft (class-probability) label
    tensors of shape ``[B, num_classes]`` whose rows sum to 1, which the
    cross-entropy criterion consumes directly.

    Probability semantics: at most one op is applied per batch, from a single
    draw ``r``. MixUp is applied when ``r < mixup``; otherwise CutMix is applied
    when ``r < mixup + cutmix``. So ``mixup`` is honored exactly as MixUp's
    per-batch probability and ``cutmix`` as CutMix's. The two are additive, so
    :func:`validate_mix_probabilities` rejects pairs whose sum exceeds 1 rather
    than silently truncating CutMix. With a single op enabled this reduces to
    applying that op with its own probability.

    :meth:`close_strong_aug` turns mixing off in place (plain hard-label
    batches from then on); the trainer calls it at the ``no_aug_epochs``
    boundary through the same worker-safe path detection uses for
    ``close_mosaic``.
    """

    def __init__(self, num_classes: int, mixup: float = 0.0, cutmix: float = 0.0):
        from torchvision.transforms import v2

        self._mixup = v2.MixUp(num_classes=num_classes) if mixup > 0 else None
        self._cutmix = v2.CutMix(num_classes=num_classes) if cutmix > 0 else None
        if self._mixup is None and self._cutmix is None:
            raise ValueError("ClassifyBatchMixer needs mixup>0 or cutmix>0.")
        self._mixup_p = float(mixup)
        self._cutmix_p = float(cutmix)
        self.enabled = True

    def close_strong_aug(self) -> None:
        """Disable MixUp / CutMix; subsequent batches carry hard labels."""
        self.enabled = False

    def __call__(self, batch):
        imgs, labels, img_infos, img_ids = classify_collate_fn(batch)
        if not self.enabled:
            return imgs, labels, img_infos, img_ids
        r = float(torch.rand(1).item())
        if self._mixup is not None and r < self._mixup_p:
            imgs, labels = self._mixup(imgs, labels)
        elif self._cutmix is not None and r < self._mixup_p + self._cutmix_p:
            imgs, labels = self._cutmix(imgs, labels)
        return imgs, labels, img_infos, img_ids


# Historical private name, kept for callers that imported it.
_ClassifyBatchMixer = ClassifyBatchMixer


def build_classify_collate(num_classes: int, mixup: float = 0.0, cutmix: float = 0.0):
    """Return the classification collate function for the given mixing knobs.

    With ``mixup == 0`` and ``cutmix == 0`` this returns :func:`classify_collate_fn`
    unchanged (byte-identical batches, so default training is unaffected).
    Otherwise it returns a :class:`ClassifyBatchMixer` that applies MixUp / CutMix
    at the batch level and produces soft labels.
    """
    mixup, cutmix = validate_mix_probabilities(mixup, cutmix)
    if mixup == 0 and cutmix == 0:
        return classify_collate_fn
    return ClassifyBatchMixer(num_classes, mixup=mixup, cutmix=cutmix)


__all__ = [
    "AUTO_AUGMENT_POLICIES",
    "DEFAULT_CROP_PCT",
    "DEFAULT_CROP_SCALE",
    "DEFAULT_FLIP_PROB",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "ClassifyAugKnobs",
    "ClassifyBatchMixer",
    "build_classify_collate",
    "build_classify_transforms",
    "classify_collate_fn",
    "normalize_auto_augment",
    "normalize_crop_scale",
    "validate_mix_probabilities",
]
