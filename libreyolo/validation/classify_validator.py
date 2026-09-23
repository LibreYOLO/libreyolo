"""Image-classification validator for LibreYOLO.

Computes top-1 and top-5 accuracy over an ImageFolder-style validation split,
reusing the :class:`BaseValidator` template (setup -> iterate -> finalize).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from ..data.classify_dataset import (
    ClassifyDataset,
    classify_collate_fn,
    get_class_names,
    resolve_classify_data,
)
from ..data.imagenet import imagenet1k_class_list, imagenet1k_synset_to_index
from ..utils.general import COCO_CLASSES
from .base import BaseValidator
from .config import wants_more_plot_errors
from .loss import ValidationLossMixin

if TYPE_CHECKING:
    from .config import ValidationConfig
    from .loss import ValidationLossAdapter

logger = logging.getLogger(__name__)


class ClassifyValidator(ValidationLossMixin, BaseValidator):
    """Top-1/top-5 accuracy validator for the classification task."""

    task = "classify"
    supports_plot_errors = True

    def __init__(
        self,
        model,
        config: Optional["ValidationConfig"] = None,
        *,
        loss_adapter: Optional["ValidationLossAdapter"] = None,
        **kwargs,
    ) -> None:
        super().__init__(model, config, **kwargs)
        self._init_validation_loss(loss_adapter)

    def _model_class_names(self) -> list[str] | None:
        names = getattr(self.model, "names", None)
        if not isinstance(names, dict) or not names:
            return None

        num_classes = int(getattr(self.model, "nb_classes", len(names)))
        if any(i not in names for i in range(num_classes)):
            return None

        ordered = [str(names[i]) for i in range(num_classes)]
        if ordered == [f"class_{i}" for i in range(num_classes)]:
            return None
        if num_classes == len(COCO_CLASSES) and ordered == list(COCO_CLASSES):
            return None
        return ordered

    def _resolve_crop_pct(self, family_default):
        """The eval crop ratio: an explicit ``config.crop_pct`` beats the family.

        Subclasses that pin a family-specific eval pipeline must route their
        crop ratio through this, or ``val(crop_pct=...)`` would be accepted and
        silently ignored for those families (#878).
        """
        override = getattr(self.config, "crop_pct", None)
        if override is None:
            return family_default
        if not 0.0 < override <= 1.0:
            raise ValueError(f"crop_pct must be in (0, 1], got {override}")
        return float(override)

    def _dataset_transform_kwargs(self) -> dict:
        """Extra kwargs for ``build_classify_transforms`` (mean/std/interp/crop).

        Defaults to the model's native eval pipeline (``crop_pct`` +
        ``interpolation``) so ``val()`` matches ``predict()`` with ImageNet
        normalization; subclasses (e.g. the CLIP validator) override to inject
        family-specific normalization (mean/std).
        """
        kwargs: dict = {}
        crop_pct = self._resolve_crop_pct(getattr(self.model, "crop_pct", None))
        if crop_pct is not None:
            kwargs["crop_pct"] = crop_pct
        interpolation = getattr(self.model, "interpolation", None)
        if interpolation is not None:
            kwargs["interpolation"] = interpolation
        return kwargs

    @staticmethod
    def _imagenet_synset_mapping(
        model_classes: list[str], dataset_classes: list[str]
    ) -> dict[str, int] | None:
        """Map a full or subset WNID ImageFolder to a canonical ImageNet head."""
        if model_classes != imagenet1k_class_list():
            return None
        synset_to_index = imagenet1k_synset_to_index()
        if not dataset_classes or any(
            name not in synset_to_index for name in dataset_classes
        ):
            return None
        return {name: synset_to_index[name] for name in dataset_classes}

    @staticmethod
    def _format_class_delta(expected: set[str], actual: set[str]) -> str:
        details = []
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        if extra:
            details.append(f"unknown classes: {extra}")
        if missing:
            details.append(f"missing classes: {missing}")
        return "; ".join(details)

    def _setup_dataloader(self) -> DataLoader:
        dataset_root = resolve_classify_data(self.config.data)
        split = self.config.split or "val"
        train_classes = get_class_names(dataset_root, split="train")
        model_classes = self._model_class_names()
        class_to_idx = None
        if model_classes is None:
            class_names = train_classes
        else:
            class_to_idx = self._imagenet_synset_mapping(model_classes, train_classes)
            if class_to_idx is not None:
                class_names = train_classes
            else:
                expected = set(model_classes)
                actual = set(train_classes)
                if expected != actual:
                    raise ValueError(
                        "Classification train classes must match the model class names "
                        f"({self._format_class_delta(expected, actual)})."
                    )
                class_names = model_classes

        # Label indices are pinned to the model/checkpoint class order when it is
        # explicit, otherwise to the train split order shared across splits.
        if class_to_idx is None:
            class_to_idx = {name: i for i, name in enumerate(class_names)}

        dataset = ClassifyDataset(
            dataset_root=dataset_root,
            split=split,
            imgsz=self.config.imgsz,
            augment=False,
            class_to_idx=class_to_idx,
            # Match the model's native eval pipeline so val() agrees with predict().
            transform_kwargs=self._dataset_transform_kwargs(),
        )
        self._num_classes = len(model_classes or class_names)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=classify_collate_fn,
        )

    def _init_metrics(self) -> None:
        self._top1_correct = 0
        self._top5_correct = 0
        self._total = 0
        self._error_samples: list[dict] = []
        self._reset_validation_loss()

    def _preprocess_batch(self, batch: Any) -> tuple:
        images, targets, img_info, img_ids = batch
        return images, targets, img_info, img_ids

    def _postprocess_predictions(self, preds: Any, batch: Any) -> Any:
        # ``preds`` are raw logits [B, num_classes]; nothing to decode.
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            preds = preds[0]
        if isinstance(preds, dict) or isinstance(preds, torch.Tensor):
            return preds
        return torch.as_tensor(preds)

    def _update_metrics(
        self, preds: Any, targets: Any, img_info: Any, img_ids: Any = None
    ) -> None:
        # Accumulate before the .cpu() below so the adapter sees the logits on
        # the device the criterion expects.
        self._accumulate_validation_loss(preds, targets, image_size=None)

        logits = preds
        if isinstance(logits, dict):
            logits = logits.get("logits", logits.get("predictions"))
        logits = logits.detach().float().cpu()
        targets = targets.detach().cpu().view(-1)

        # NOTE: "top5" is really top-min(5, num_classes). For nc < 5 it
        # degrades to top-nc, so accuracy_top5 == 1.0 trivially when nc <= 5.
        num_classes = logits.shape[1]
        k = min(5, num_classes)
        topk = logits.topk(k, dim=1).indices  # [B, k]
        correct = topk == targets.unsqueeze(1)

        self._top1_correct += int(correct[:, 0].sum().item())
        self._top5_correct += int(correct.any(dim=1).sum().item())
        self._total += int(targets.numel())
        self._track_errors(logits, targets, topk[:, 0], correct[:, 0])

    def _track_errors(self, logits, targets, top1, top1_correct) -> None:
        """Keep wrong top-1 images for the error-analysis plot (#887).

        Plotting only: nothing here feeds the metrics.
        """
        if getattr(self, "_error_samples", None) is None:
            return
        if not wants_more_plot_errors(self.config, len(self._error_samples)):
            return
        wrong = torch.nonzero(~top1_correct).view(-1).tolist()
        if not wrong:
            return
        # ``predict()`` reports softmax probabilities of the same logits.
        scores = logits.softmax(dim=1)
        samples = getattr(getattr(self.dataloader, "dataset", None), "_impl", None)
        samples = getattr(samples, "samples", None)
        for i in wrong:
            if not wants_more_plot_errors(self.config, len(self._error_samples)):
                return
            index = self.seen + i
            pred = int(top1[i])
            self._error_samples.append({
                "img_path": samples[index][0] if samples and index < len(samples) else None,
                "target": int(targets[i]),
                "pred": pred,
                "score": float(scores[i, pred]),
            })

    def _class_display_name(self, index: int) -> str:
        # Label indices follow the model's explicit class order when it has
        # one (including the full ImageNet head), else the train split order.
        classes = self._model_class_names() or getattr(
            getattr(self.dataloader, "dataset", None), "classes", None
        )
        if classes and 0 <= index < len(classes):
            return str(classes[index])
        return str(index)

    def _save_plots(self, metrics: Dict[str, float]) -> None:
        """Write the wrong top-1 images to ``plots/errors/`` (#887)."""
        if not getattr(self, "_error_samples", None):
            return
        import cv2  # noqa: PLC0415

        from .val_plotter import ValPlotter  # noqa: PLC0415

        errors_dir = self.save_dir / "plots" / "errors"
        for idx, sample in enumerate(self._error_samples):
            if sample["img_path"] is None:
                continue
            img_bgr = cv2.imread(str(sample["img_path"]))
            if img_bgr is None:
                continue
            try:
                ValPlotter.plot_classify_error(
                    img_bgr,
                    self._class_display_name(sample["target"]),
                    self._class_display_name(sample["pred"]),
                    sample["score"],
                    errors_dir / f"error_{idx:03d}.jpg",
                )
            except Exception as exc:
                logger.warning("Plot failed (plot_classify_error): %s", exc)

    def _compute_metrics(self) -> Dict[str, float]:
        total = max(self._total, 1)
        top1 = self._top1_correct / total
        top5 = self._top5_correct / total
        return {
            "metrics/accuracy_top1": top1,
            "metrics/accuracy_top5": top5,
            "fitness": top1,
            **self._validation_loss_metrics(),
        }

    def _print_results(self, metrics: Dict[str, float]) -> None:
        logger.info("=" * 50)
        logger.info("Classification Validation Results")
        logger.info("=" * 50)
        logger.info("  top-1 accuracy: %.4f", metrics.get("metrics/accuracy_top1", 0.0))
        logger.info("  top-5 accuracy: %.4f", metrics.get("metrics/accuracy_top5", 0.0))
        logger.info("  images: %d", self._total)
        logger.info("=" * 50)
