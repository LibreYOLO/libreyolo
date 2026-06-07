"""D-FINE training transform recipes."""

from ...data.augment.detr import (
    DETRMultiScaleCollate,
    DETRPassThroughDataset,
    DETRTrainTransform,
    generate_scales,
    labels_at_index_2,
)

_labels_at_index_2 = labels_at_index_2
_generate_scales = generate_scales


class DFINETrainTransform(DETRTrainTransform):
    """D-FINE per-sample transform recipe."""


class DFINEPassThroughDataset(DETRPassThroughDataset):
    """D-FINE dataset wrapper recipe."""

    transform_cls = DFINETrainTransform


class DFINEMultiScaleCollate(DETRMultiScaleCollate):
    """D-FINE multi-scale collate recipe."""


__all__ = [
    "DFINETrainTransform",
    "DFINEPassThroughDataset",
    "DFINEMultiScaleCollate",
    "_labels_at_index_2",
    "_generate_scales",
]
