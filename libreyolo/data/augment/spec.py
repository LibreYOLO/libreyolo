"""Declarative augmentation-knob spec: which family honors which knob.

This module is the single source of truth for how each trainable model family
treats the augmentation knobs on :class:`~libreyolo.training.config.TrainConfig`.
It is a plain-Python table (no torch/cv2 imports) derived by reading each
family's trainer + transform + dataset-wrapper code, and it is pinned to
reality by ``tests/unit/test_augment_spec.py``, which asserts the claimed
statuses against the actual pipeline plumbing.

Consumers:
  - ``libreyolo.cli.config.get_unsupported_train_params`` warns when a user
    explicitly sets a CLI training parameter that the selected family ignores
    (previously this existed only for RF-DETR).
  - ``BaseTrainer._setup_data`` (and FOMO's data setup) warn when
    ``mixup_prob > 0`` can never fire because the family only applies mixup to
    mosaic samples and ``mosaic_prob == 0``.
  - Docs generation can render the support matrix instead of hand-writing it.

Statuses:
  - ``USED``: the knob reaches the family's train pipeline and changes samples.
  - ``GATED_BY_MOSAIC``: the knob is only applied to samples that took the
    mosaic branch (probability ``mosaic_prob``); with ``mosaic_prob == 0`` it
    never fires. This is how the YOLOX-style wrappers apply affine and mixup.
  - ``IGNORED``: the knob never reaches the family's train pipeline; setting
    it silently does nothing.

Keys are TrainConfig field names (``mosaic_prob``, not the CLI alias
``mosaic``); the CLI consumer maps to CLI spellings via
``libreyolo.cli.aliases.TRAIN_ALIASES``.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Set

USED = "used"
GATED_BY_MOSAIC = "gated_by_mosaic"
IGNORED = "ignored"

_VALID_STATUSES = frozenset({USED, GATED_BY_MOSAIC, IGNORED})


class Support(NamedTuple):
    """How one family treats one augmentation knob."""

    status: str
    note: str = ""


# Base TrainConfig augmentation knobs covered by the matrix, with one-line
# descriptions (config-field names, not CLI aliases).
AUG_KNOBS: Dict[str, str] = {
    "mosaic_prob": "Probability of building a 4-image mosaic sample.",
    "mixup_prob": "Probability of blending in a second sample (MixUp).",
    "hsv_prob": "Probability of HSV color jitter.",
    "flip_prob": "Horizontal-flip probability.",
    "degrees": "Random-rotation range for the affine warp, in degrees.",
    "translate": "Random-translation fraction for the affine warp.",
    "mosaic_scale": "Random-scale range for the affine warp.",
    "mixup_scale": "Jitter-scale range applied to the MixUp partner image.",
    "shear": "Random-shear range for the affine warp, in degrees.",
    "perspective": "Projective (perspective) warp magnitude for the affine warp.",
    "flipud": "Vertical-flip probability.",
    "no_aug_epochs": "Final epochs trained with strong augmentation disabled.",
    # Classification-only pack (ImageFolder pipeline; detection families
    # ignore these).
    "auto_augment": "Classification AutoAugment policy (randaugment/autoaugment/augmix).",
    "erasing": "Classification RandomErasing probability.",
    "mixup": "Classification batch-MixUp probability (soft labels).",
    "cutmix": "Classification batch-CutMix probability (soft labels).",
}

# Human-facing family names for warnings and docs.
FAMILY_DISPLAY_NAMES: Dict[str, str] = {
    "yolox": "YOLOX",
    "yolo7": "YOLOv7",
    "yolo9": "YOLOv9",
    "yolo9_e2e": "YOLOv9-E2E",
    "yolo9_p2": "YOLOv9-P2",
    "yolonas": "YOLO-NAS",
    "rtdetr": "RT-DETR",
    "rtdetrv2": "RT-DETRv2",
    "rtdetrv4": "RT-DETRv4",
    "dfine": "D-FINE",
    "domedetr": "Dome-DETR",
    "deim": "DEIM",
    "deimv2": "DEIMv2",
    "tinyformer": "TinyFormer",
    "ec": "EC",
    "rtmdet": "RTMDet",
    "picodet": "PICODET",
    "ppyoloe": "PP-YOLOE",
    "rfdetr": "RF-DETR",
    "fomo": "FOMO",
    "resnet": "ResNet",
    "convnext": "ConvNeXt",
    "mobilenetv4": "MobileNetV4",
    "efficientnetv2": "EfficientNetV2",
    "dinov2": "DINOv2",
    "segformer": "SegFormer",
    "unet": "U-Net",
    "nafnet": "NAFNet",
}


def _u(note: str = "") -> Support:
    return Support(USED, note)


def _g(note: str = "") -> Support:
    return Support(GATED_BY_MOSAIC, note)


def _i(note: str = "") -> Support:
    return Support(IGNORED, note)


_CLS_KNOBS_IGNORED = {
    "auto_augment": _i("Classification-only knob."),
    "erasing": _i("Classification-only knob."),
    "mixup": _i("Classification-only knob (API-only; the CLI --mixup is the detection mixup_prob alias)."),
    "cutmix": _i("Classification-only knob."),
}

# YOLOX-style pipeline: per-sample preproc applies HSV + flips; affine and
# MixUp run only inside the mosaic branch (probability mosaic_prob).
_YOLOX_STYLE = {
    "mosaic_prob": _u(),
    "mixup_prob": _g("MixUp only applies to mosaic samples."),
    "hsv_prob": _u(),
    "flip_prob": _u(),
    "degrees": _g("Affine runs on the mosaic canvas only."),
    "translate": _g("Affine runs on the mosaic canvas only."),
    "mosaic_scale": _g("Affine runs on the mosaic canvas only."),
    "mixup_scale": _g("MixUp only applies to mosaic samples."),
    "shear": _g("Affine runs on the mosaic canvas only."),
    "perspective": _g("Affine runs on the mosaic canvas only."),
    "flipud": _u(),
    "no_aug_epochs": _u("Disables mosaic/mixup for the final epochs."),
    **_CLS_KNOBS_IGNORED,
}

# DETR-style pass-through pipeline (torchvision-v2 transform, no mosaic):
# horizontal flip is configurable; photometric distortion / zoom-out /
# IoU-crop are family recipe constants, so hsv_prob and the geometry knobs
# never reach the pipeline.
_DETR_STYLE = {
    "mosaic_prob": _i("DETR-style pipeline has no mosaic."),
    "mixup_prob": _i("DETR-style pipeline has no MixUp."),
    "hsv_prob": _i("Color jitter comes from RandomPhotometricDistort with a fixed recipe probability."),
    "flip_prob": _u(),
    "degrees": _i("DETR-style pipeline has no affine warp."),
    "translate": _i("DETR-style pipeline has no affine warp."),
    "mosaic_scale": _i("DETR-style pipeline has no affine warp."),
    "mixup_scale": _i("DETR-style pipeline has no MixUp."),
    "shear": _i("DETR-style pipeline has no affine warp."),
    "perspective": _i("DETR-style pipeline has no affine warp."),
    "flipud": _i("DETR-style pipeline has no vertical flip."),
    "no_aug_epochs": _u("Stops the strong augs (photometric/zoom-out/IoU-crop) and shapes the LR tail."),
    **_CLS_KNOBS_IGNORED,
}

# Classification ImageFolder pipeline: detection knobs never apply; the
# horizontal flip is a fixed RandomHorizontalFlip(0.5), not flip_prob.
_CLASSIFY_STYLE = {
    "mosaic_prob": _i("Classification pipeline has no mosaic."),
    "mixup_prob": _i("Detection MixUp; classification uses the 'mixup' field instead."),
    "hsv_prob": _i("Classification pipeline has no HSV jitter."),
    "flip_prob": _i("Classification flip is a fixed RandomHorizontalFlip(0.5)."),
    "degrees": _i("Classification pipeline has no affine warp."),
    "translate": _i("Classification pipeline has no affine warp."),
    "mosaic_scale": _i("Classification pipeline has no affine warp."),
    "mixup_scale": _i("Classification pipeline has no detection MixUp."),
    "shear": _i("Classification pipeline has no affine warp."),
    "perspective": _i("Classification pipeline has no affine warp."),
    "flipud": _i("Classification pipeline has no vertical flip."),
    "no_aug_epochs": _u("Shapes the scheduler's plateau tail."),
    "auto_augment": _u(),
    "erasing": _u(),
    "mixup": _u("API-only: the CLI --mixup is the detection mixup_prob alias."),
    "cutmix": _u(),
}

FAMILY_AUG_SUPPORT: Dict[str, Dict[str, Support]] = {
    # --- YOLOX-style mosaic pipelines -----------------------------------
    "yolox": dict(_YOLOX_STYLE),
    "yolo7": dict(_YOLOX_STYLE),
    "yolo9": dict(_YOLOX_STYLE),
    "yolo9_e2e": dict(_YOLOX_STYLE),
    "yolo9_p2": dict(_YOLOX_STYLE),
    "rtmdet": {
        **_YOLOX_STYLE,
        "flipud": _i("RTMDet's transform has no vertical flip."),
    },
    "picodet": {
        **_YOLOX_STYLE,
        "flipud": _i("PICODET's transform has no vertical flip."),
    },
    "rtdetr": {
        **_YOLOX_STYLE,
        "flipud": _i("RT-DETR's transform has no vertical flip."),
    },
    "rtdetrv2": {
        **_YOLOX_STYLE,
        "flipud": _i("RT-DETRv2's transform has no vertical flip."),
    },
    "fomo": {
        **_YOLOX_STYLE,
        "perspective": _i("FOMO's mosaic wrapper is built without perspective."),
        "flipud": _i("FOMO's transform has no vertical flip."),
    },
    # --- YOLO-NAS: affine always on, mosaic ignored, mixup independent --
    "yolonas": {
        "mosaic_prob": _i("YOLO-NAS trains with a per-sample affine instead of mosaic."),
        "mixup_prob": _u("MixUp is independent of mosaic here."),
        "hsv_prob": _u(),
        "flip_prob": _u(),
        "degrees": _u("Applied by the always-on per-sample affine."),
        "translate": _u("Applied by the always-on per-sample affine."),
        "mosaic_scale": _u("Reused as the per-sample affine scale range."),
        "mixup_scale": _u(),
        "shear": _u("Applied by the always-on per-sample affine."),
        "perspective": _u("Applied by the always-on per-sample affine."),
        "flipud": _u(),
        "no_aug_epochs": _u("Disables the affine and MixUp for the final epochs."),
        **_CLS_KNOBS_IGNORED,
    },
    # PP-YOLOE reuses YOLO-NAS's affine + MixUp wrapper, so the geometric
    # knobs behave the same. Its own transform adds the source recipe's
    # rot90 and RGB-to-BGR swap, which have their own config fields rather
    # than shared knobs.
    "ppyoloe": {
        "mosaic_prob": _i("The source recipe uses a per-sample affine, not mosaic."),
        "mixup_prob": _u("MixUp is independent of mosaic here."),
        "hsv_prob": _u(),
        "flip_prob": _u(),
        "degrees": _u("Applied by the always-on per-sample affine."),
        "translate": _u("Applied by the always-on per-sample affine."),
        "mosaic_scale": _u("Reused as the per-sample affine scale range."),
        "mixup_scale": _u(),
        "shear": _u("Applied by the always-on per-sample affine."),
        "perspective": _u("Applied by the always-on per-sample affine."),
        "flipud": _i(
            "The source recipe has no vertical flip; it uses a random "
            "90-degree rotation (rot90_prob) instead."
        ),
        "no_aug_epochs": _u("Disables the affine and MixUp for the final epochs."),
        **_CLS_KNOBS_IGNORED,
    },
    # --- DETR-style pass-through pipelines ------------------------------
    "dfine": dict(_DETR_STYLE),
    # Dome-DETR inherits DFINETrainer.create_transforms unchanged, so its
    # knob support is D-FINE's. Multi-scale is the one thing it cannot
    # take (MWAS needs the stride-8 map divisible by the window size), and
    # that is enforced by DOMEDETRConfig.multi_scale=False, not here.
    "domedetr": dict(_DETR_STYLE),
    "deim": dict(_DETR_STYLE),
    "deimv2": dict(_DETR_STYLE),
    # TinyFormer trains through DEIMv2's transform path unchanged.
    "tinyformer": dict(_DETR_STYLE),
    # EC is multi-task (detect/segment/pose). Its pose pipeline honors
    # hsv_prob/degrees/translate, so those cannot be marked IGNORED for the
    # family as a whole; the CLI warning must never be wrong for any task.
    "ec": {
        **_DETR_STYLE,
        "hsv_prob": _u("task='pose' only; detect/segment use fixed photometric recipes."),
        "degrees": _u("task='pose' only (keypoint-aware affine)."),
        "translate": _u("task='pose' only (keypoint-aware affine)."),
    },
    "rtdetrv4": dict(_DETR_STYLE),
    # --- RF-DETR: native pipeline (flip + scale jitter + crop) ----------
    "rfdetr": {
        **_DETR_STYLE,
        "hsv_prob": _i("RF-DETR's native pipeline has no HSV jitter."),
        "no_aug_epochs": _u("Shapes the scheduler's plateau tail."),
    },
    # --- DINOv2 (multi-task: semantic/classify/detect). Only knobs that
    # are ignored across ALL of its trainable tasks are marked IGNORED, so
    # the CLI warning is never wrong for any task. flip_prob reaches the
    # RF-DETR-style detect pipeline; the classification pack applies to
    # task='classify'.
    "dinov2": {
        **_DETR_STYLE,
        "hsv_prob": _i("No HSV jitter in any DINOv2 task pipeline."),
        "no_aug_epochs": _u("Shapes the scheduler's plateau tail."),
        "auto_augment": _u("task='classify' only."),
        "erasing": _u("task='classify' only."),
        "mixup": _u("task='classify' only (API-only knob)."),
        "cutmix": _u("task='classify' only."),
    },
    # --- Classification-only families -----------------------------------
    "resnet": dict(_CLASSIFY_STYLE),
    "convnext": dict(_CLASSIFY_STYLE),
    "mobilenetv4": dict(_CLASSIFY_STYLE),
    "efficientnetv2": dict(_CLASSIFY_STYLE),
    # --- Semantic-only: the semantic pipeline reads family class
    # attributes (semantic_scale_jitter / semantic_hsv_prob), not config
    # knobs; horizontal flip is a fixed 0.5.
    "segformer": {
        "mosaic_prob": _i("Semantic pipeline has no mosaic."),
        "mixup_prob": _i("Semantic pipeline has no MixUp."),
        "hsv_prob": _i("Semantic HSV comes from the family's semantic_hsv_prob class attribute."),
        "flip_prob": _i("Semantic flip is a fixed 0.5."),
        "degrees": _i("Semantic pipeline has no affine warp."),
        "translate": _i("Semantic pipeline has no affine warp."),
        "mosaic_scale": _i("Semantic scale jitter comes from the family's semantic_scale_jitter class attribute."),
        "mixup_scale": _i("Semantic pipeline has no MixUp."),
        "shear": _i("Semantic pipeline has no affine warp."),
        "perspective": _i("Semantic pipeline has no affine warp."),
        "flipud": _i("Semantic pipeline has no vertical flip."),
        "no_aug_epochs": _u("Shapes the scheduler's tail."),
        **_CLS_KNOBS_IGNORED,
    },
    "unet": {
        "mosaic_prob": _i("Semantic pipeline has no mosaic."),
        "mixup_prob": _i("Semantic pipeline has no MixUp."),
        "hsv_prob": _i("Semantic HSV comes from the family's semantic_hsv_prob class attribute."),
        "flip_prob": _i("Semantic flip is a fixed 0.5."),
        "degrees": _i("Semantic pipeline has no affine warp."),
        "translate": _i("Semantic pipeline has no affine warp."),
        "mosaic_scale": _i("Semantic scale jitter comes from the family's semantic_scale_jitter class attribute."),
        "mixup_scale": _i("Semantic pipeline has no MixUp."),
        "shear": _i("Semantic pipeline has no affine warp."),
        "perspective": _i("Semantic pipeline has no affine warp."),
        "flipud": _i("Semantic pipeline has no vertical flip."),
        "no_aug_epochs": _u("Shapes the scheduler's tail."),
        **_CLS_KNOBS_IGNORED,
    },
    # --- Restoration: coupled crop/flip/rot90 with fixed probabilities --
    "nafnet": {
        "mosaic_prob": _i("Restore pipeline has no mosaic."),
        "mixup_prob": _i("Restore pipeline has no MixUp."),
        "hsv_prob": _i("Restore pipeline has no HSV jitter."),
        "flip_prob": _i("Restore flips are coupled input/target ops with fixed 0.5 probability."),
        "degrees": _i("Restore pipeline has no affine warp."),
        "translate": _i("Restore pipeline has no affine warp."),
        "mosaic_scale": _i("Restore pipeline has no affine warp."),
        "mixup_scale": _i("Restore pipeline has no MixUp."),
        "shear": _i("Restore pipeline has no affine warp."),
        "perspective": _i("Restore pipeline has no affine warp."),
        "flipud": _i("Restore vertical flip is a coupled op with fixed 0.5 probability."),
        "no_aug_epochs": _u("Shapes the scheduler's tail."),
        **_CLS_KNOBS_IGNORED,
    },
}

# Family-specific augmentation knobs that live on the family's TrainConfig
# subclass rather than the base TrainConfig. Documentation-oriented; the CLI
# does not expose these.
FAMILY_EXTRA_AUG_KNOBS: Dict[str, Dict[str, str]] = {
    "yolo9": {
        "copy_paste": "Copy-paste instance augmentation probability (task='segment' only).",
        "copy_paste_mode": "Copy-paste source: 'flip' (same sample mirrored) or 'mixup' (second sample).",
        "rot90": "Random 90-degree rotation probability (detect/OBB).",
    },
    "yolo9_e2e": {
        "copy_paste": "Copy-paste instance augmentation probability (task='segment' only).",
        "copy_paste_mode": "Copy-paste source: 'flip' or 'mixup'.",
        "rot90": "Random 90-degree rotation probability.",
    },
    "yolo9_p2": {
        "copy_paste": "Copy-paste instance augmentation probability (task='segment' only).",
        "copy_paste_mode": "Copy-paste source: 'flip' or 'mixup'.",
        "rot90": "Random 90-degree rotation probability.",
    },
    "rfdetr": {
        "copy_paste": "Copy-paste probability for task='segment' ('flip' mode only).",
        "copy_paste_mode": "Copy-paste source mode for task='segment'.",
        "crop_resize_prob": "Random crop-resize probability in the native pipeline.",
    },
    "dfine": {
        "crop_resize_prob": "Random crop-resize probability (task='segment').",
    },
    "ec": {
        "crop_resize_prob": "Random crop-resize probability (task='segment').",
        "brightness_contrast_prob": "Brightness/contrast jitter probability (task='pose').",
        "affine_prob": "Keypoint-aware affine probability (task='pose').",
    },
    "yolonas": {
        "brightness_contrast_prob": "Brightness/contrast jitter probability (task='pose').",
        "affine_prob": "Keypoint-aware affine probability (task='pose').",
    },
}


def aug_support(family: Optional[str]) -> Optional[Dict[str, Support]]:
    """Return the knob-support table for a family, or None if unknown."""
    if family is None:
        return None
    return FAMILY_AUG_SUPPORT.get(family)


def ignored_aug_params(family: Optional[str]) -> Set[str]:
    """TrainConfig field names of augmentation knobs a family ignores.

    Returns an empty set for unknown families (no spurious warnings for
    families the spec does not cover yet).
    """
    table = aug_support(family)
    if table is None:
        return set()
    return {knob for knob, support in table.items() if support.status == IGNORED}


def uses_mosaic_gating(family: Optional[str]) -> bool:
    """Whether the family's MixUp only fires on mosaic samples."""
    table = aug_support(family)
    if table is None:
        return False
    return table.get("mixup_prob", Support(IGNORED)).status == GATED_BY_MOSAIC


def display_name(family: Optional[str]) -> str:
    """Human-facing family name for warnings and docs."""
    if not family:
        return "This model family"
    return FAMILY_DISPLAY_NAMES.get(family, family)


def mixup_gating_warning(
    family: Optional[str],
    mosaic_prob: float,
    mixup_prob: float,
) -> Optional[str]:
    """Warning text when mixup can never fire, or None when it can.

    For families whose MixUp only applies to mosaic samples, ``mixup_prob > 0``
    with ``mosaic_prob <= 0`` silently disables mixup. Callers log the returned
    message; this stays a pure function so it is trivially testable and never
    touches the RNG stream.
    """
    if not uses_mosaic_gating(family):
        return None
    if float(mixup_prob or 0.0) > 0.0 and float(mosaic_prob or 0.0) <= 0.0:
        return (
            f"mixup_prob={mixup_prob} has no effect for {display_name(family)}: "
            "mixup only applies to mosaic samples and mosaic_prob=0. "
            "Set mosaic_prob > 0 to enable mixup."
        )
    return None
