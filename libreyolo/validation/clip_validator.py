"""Zero-shot classification validator for LibreCLIP.

CLIP's own mean/std + bicubic preprocessing comes from the model's
``eval_transform`` (#886), the same transform ``predict()`` uses. This class
adds only open-vocabulary indexing:

* **Open-vocabulary indexing** — the model's display names are the humanized
  prompt labels (e.g. ``"tench"``), which deliberately differ from wnid folder
  names (``"n01440764"``). ``LibreCLIP.val`` calls ``set_classes`` on the
  train-split folder names *in sorted order*, so the model's logit index ``i``
  already lines up with the dataset's sorted-folder label ``i``. We therefore
  let the dataset drive the label indices (return ``None`` from
  ``_model_class_names``) instead of demanding a name match.
"""

from __future__ import annotations

from .classify_validator import ClassifyValidator


class CLIPClassifyValidator(ClassifyValidator):
    """Top-1/top-5 zero-shot validator for LibreCLIP."""

    def _model_class_names(self):
        # Indices come from the sorted train folders, which LibreCLIP.val
        # mirrored via set_classes; the humanized display names intentionally
        # differ from wnid folder names, so do not enforce a name match.
        return None
