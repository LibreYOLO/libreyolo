"""Zero-shot classification validator for LibreCLIP.

Extends :class:`ClassifyValidator` in two ways:

* **CLIP preprocessing** — CLIP's own mean/std + bicubic resize (a 1.0 crop
  ratio), not the ImageNet stats the trained classifiers use. Getting this
  wrong silently lowers zero-shot accuracy.
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
    """Top-1/top-5 zero-shot validator with CLIP preprocessing."""

    def _model_class_names(self):
        # Indices come from the sorted train folders, which LibreCLIP.val
        # mirrored via set_classes; the humanized display names intentionally
        # differ from wnid folder names, so do not enforce a name match.
        return None

    def _dataset_transform_kwargs(self) -> dict:
        from torchvision.transforms import InterpolationMode

        from ..models.clip.model import CLIP_MEAN, CLIP_STD

        return {
            "mean": CLIP_MEAN,
            "std": CLIP_STD,
            "interpolation": InterpolationMode.BICUBIC,
            "crop_pct": self._resolve_crop_pct(1.0),
        }
