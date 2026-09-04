"""Model coverage groups used by cross-family tests and tooling.

Groups select representative coverage sets; they do not grant or restrict a
user-facing capability. Support is determined by each family's implemented
API and by format-specific capability checks. Groups classify families, never
tasks. ``tests/unit/test_model_registry.py`` fails when a registered family is
not enrolled here, so porting a new model requires adding exactly one line to
``MODEL_GROUPS``.
"""

from __future__ import annotations

GROUPS: dict[str, str] = {
    "g0": "Flagship anchors required in shared-feature coverage.",
    "g1": "Trainable detector coverage set.",
    "g2": "Additional trainable-family coverage set.",
    "g3": "Families without a training implementation.",
    "g4": "Historical families with inference coverage.",
    "s": "Sibling APIs (SAM, open-vocab, VLM, grounding, zero-shot) covered separately.",
}

MODEL_GROUPS: dict[str, str] = {
    # g0 - flagship
    "yolo9": "g0",
    "rfdetr": "g0",
    # g1 - core trainable detectors
    "yolo9_e2e": "g1",
    "yolo9_p2": "g1",
    "ec": "g1",
    "rtdetr": "g1",
    "rtdetrv2": "g1",
    "rtdetrv4": "g1",
    "dfine": "g1",
    "deim": "g1",
    "deimv2": "g1",
    "tinyformer": "g1",
    "yolonas": "g1",
    # g2 - supporting trainables
    "yolox": "g2",
    "yolo7": "g2",
    "rtmdet": "g2",
    "picodet": "g2",
    "ppyoloe": "g2",
    "fomo": "g2",
    "ppliteseg": "g2",
    "unet": "g2",
    "segformer": "g2",
    "lingbotvision": "g2",
    "dinov2": "g2",
    "nafnet": "g2",
    "resnet": "g2",
    "convnext": "g2",
    "mobilenetv4": "g2",
    "efficientnetv2": "g2",
    "domedetr": "g2",
    "vjepa2": "g2",
    # s - sibling / zero-shot APIs covered separately
    "pe": "s",
    # g3 - inference-only specialists
    "lwdetr": "g3",
    "detr": "g3",
    "deformable_detr": "g3",
    "mask_rcnn": "g3",
    "dinodetr": "g3",
    "fcos": "g3",
    "faster_rcnn": "g3",
    "vit": "g3",
    "retinanet": "g3",
    "ssd": "g3",
    "fcn": "g3",
    "centernet": "g3",
    "alexnet": "g3",
    "deeplabv3": "g3",
    "efficientdet": "g3",
    "eomt": "g3",
    "pidnet": "g3",
    "depth_anything": "g3",
    "depth_anything3": "g3",
    "zipdepth": "g3",
    "midas": "g3",
    "moge2": "g3",
    "dexined": "g3",
    "teed": "g3",
    "swinir": "g3",
    "realesrgan": "g3",
    "quicksrnet": "g3",
    "ddcolor": "g3",
    "hvi_cidnet": "g3",
    "lama": "g3",
    "birefnet": "g3",
    "feynobg": "g3",
    "ben2": "g3",
    "vitmatte": "g3",
    "ppocr": "g3",
    "l2cs": "g3",
    "vgg": "g3",
    "hrnet": "g3",
    "dekr": "g3",
    "facerec": "g3",
    "sam3dbody": "g3",
    "swin": "g3",
    # g4 - museum
    "yolo1": "g4",
    "yolo2": "g4",
    "yolo3": "g4",
    "yolo4": "g4",
    "deit": "g4",
    # s - sibling tiers
    "sam": "s",
    "sam2": "s",
    "sam3": "s",
    "edgetam": "s",
    "mobilesam": "s",
    "picosam3": "s",
    "grounding_dino": "s",
    "owlv2": "s",
    "omdet_turbo": "s",
    "ov_deim": "s",
    "florence2": "s",
    "kosmos2": "s",
    "internvl3": "s",
    "lfm2vl": "s",
    "locateanything": "s",
    "northmicrovision": "s",
    "gemma4": "s",
    "moondream": "s",
    "qwen3vl": "s",
    "ground_florence2": "s",
    "ground_qwen3vl": "s",
    "showui": "s",
    "smolvlm2": "s",
    "clip": "s",
    "siglip2": "s",
    "libremodus": "s",
    "sensenovavision": "s",
}


def group_of(family: str) -> str | None:
    """Return the coverage group for ``family``, or ``None`` if unenrolled."""
    return MODEL_GROUPS.get(family)


def families_in(group: str) -> tuple[str, ...]:
    """Return the enrolled families of ``group``, in registry order."""
    if group not in GROUPS:
        raise KeyError(f"Unknown model group {group!r}. Known: {sorted(GROUPS)}")
    return tuple(f for f, g in MODEL_GROUPS.items() if g == group)


__all__ = ["GROUPS", "MODEL_GROUPS", "group_of", "families_in"]
