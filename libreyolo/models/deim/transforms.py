"""DEIM training transform recipes (shared DETR pipeline + DEIM defaults)."""

from ...data.augment.detr import (
    DETRMultiScaleCollate,
    DETRPassThroughDataset,
    DETRTrainTransform,
    generate_scales,
    labels_at_index_2,
)

# Historical private names kept importable for compatibility.
_labels_at_index_2 = labels_at_index_2
_generate_scales = generate_scales


class DEIMTrainTransform(DETRTrainTransform):
    """DEIM per-sample transform recipe."""

    # DEIM datasets hand over the original image so the single Resize here
    # avoids a letterbox-then-stretch double resize.
    wants_unresized_image = True


class DEIMPassThroughDataset(DETRPassThroughDataset):
    """DEIM dataset wrapper recipe."""

    transform_cls = DEIMTrainTransform


class DEIMMultiScaleCollate(DETRMultiScaleCollate):
    """DEIM multi-scale collate recipe."""


__all__ = [
    "DEIMMultiScaleCollate",
    "DEIMPassThroughDataset",
    "DEIMTrainTransform",
]
