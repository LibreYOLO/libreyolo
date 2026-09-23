"""Zero-shot classification validator for LibreSigLIP2.

SigLIP's preprocessing (mean/std 0.5, square bilinear resize) comes from the
model's ``eval_transform`` (#886), the same transform ``predict()`` uses. Like
:class:`CLIPClassifyValidator`, this class adds only:

* **Open-vocabulary indexing** - the model's display names are the humanized
  prompt labels (e.g. ``"tench"``), which deliberately differ from wnid folder
  names. ``LibreSigLIP2.val`` calls ``set_classes`` on the train-split folder
  names *in sorted order*, so the model's logit index ``i`` already lines up with
  the dataset's sorted-folder label ``i``; we let the dataset drive the label
  indices (return ``None`` from ``_model_class_names``).
"""

from __future__ import annotations

from .classify_validator import ClassifyValidator


class SigLIP2ClassifyValidator(ClassifyValidator):
    """Top-1/top-5 zero-shot validator for LibreSigLIP2 (preprocessing from the model)."""

    def _model_class_names(self):
        # Indices come from the sorted train folders, which LibreSigLIP2.val
        # mirrored via set_classes; the humanized display names intentionally
        # differ from wnid folder names, so do not enforce a name match.
        return None
