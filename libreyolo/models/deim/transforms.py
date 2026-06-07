"""DEIM training transform recipes."""

from ...data.augment.detr import (
    DETRMultiScaleCollate,
    DETRPassThroughDataset,
    DETRTrainTransform,
    generate_scales,
    labels_at_index_2,
)

_labels_at_index_2 = labels_at_index_2
_generate_scales = generate_scales


class DEIMTrainTransform(DETRTrainTransform):
    """DEIM per-sample transform recipe."""

    wants_unresized_image = True


class DEIMPassThroughDataset(DETRPassThroughDataset):
    """DEIM dataset wrapper recipe."""

    transform_cls = DEIMTrainTransform


class DEIMMultiScaleCollate(DETRMultiScaleCollate):
    """DEIM multi-scale collate recipe."""


__all__ = [
    "DEIMTrainTransform",
    "DEIMPassThroughDataset",
    "DEIMMultiScaleCollate",
    "_labels_at_index_2",
    "_generate_scales",
]
