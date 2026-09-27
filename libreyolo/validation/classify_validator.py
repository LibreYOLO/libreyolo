"""Image-classification validator for LibreYOLO.

Computes top-1 and top-5 accuracy, plus macro-averaged precision, recall and
F1 from per-class confusion counts (the confusion matrix's diagonal and
marginals), over an ImageFolder-style validation split, reusing the
:class:`BaseValidator` template (setup -> iterate -> finalize).
"""

from __future__ import annotations

import logging
from pathlib import Path
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
from .loss import ValidationLossMixin

if TYPE_CHECKING:
    from .config import ValidationConfig
    from .loss import ValidationLossAdapter

logger = logging.getLogger(__name__)


class ClassifyValidator(ValidationLossMixin, BaseValidator):
    """Top-1/top-5 accuracy validator for the classification task.

    Also reports macro-averaged precision, recall and F1 from per-class
    confusion counts (the confusion matrix's diagonal and marginals)
    accumulated over the validation split.
    """

    task = "classify"
    supports_visualize = True
    # Per-class confusion counts: the confusion matrix's diagonal and its
    # marginals. Kept as three length-nc vectors so memory stays linear in the
    # class count. Class-level defaults keep metric code safe on instances
    # that skip _init_metrics.
    _class_tp: torch.Tensor | None = None
    _class_pred: torch.Tensor | None = None
    _class_target: torch.Tensor | None = None

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

    def _dataset_transform(self) -> dict:
        """The eval transform for the validation images.

        Taken from the model (``_get_eval_transform``), so the family's own
        eval pipeline scores the model exactly as ``predict()`` preprocesses
        (#886); ``config.crop_pct`` is the only override. Models without the
        hook fall back to :meth:`_dataset_transform_kwargs`.
        """
        get_transform = getattr(self.model, "_get_eval_transform", None)
        if callable(get_transform):
            return {
                "transform": get_transform(
                    self.config.imgsz, crop_pct=self._resolve_crop_pct(None)
                )
            }
        return {"transform_kwargs": self._dataset_transform_kwargs()}

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
            **self._dataset_transform(),
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
        self._class_tp = None
        self._class_pred = None
        self._class_target = None
        if getattr(self.config, "visualize", False):
            from .val_plotter import reset_visualize_dir  # noqa: PLC0415

            reset_visualize_dir(self.save_dir)
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
        logits = preds
        if isinstance(logits, dict):
            logits = logits.get("logits", logits.get("predictions"))
        logits = logits.detach().float().cpu()
        target_idx = targets.detach().cpu().view(-1)

        # NOTE: "top5" is really top-min(5, num_classes). For nc < 5 it
        # degrades to top-nc, so accuracy_top5 == 1.0 trivially when nc <= 5.
        num_classes = logits.shape[1]
        invalid = (target_idx < 0) | (target_idx >= num_classes)
        if bool(invalid.any()):
            invalid_labels = target_idx[invalid].unique().tolist()
            raise ValueError(
                "Classification dataset/model class mismatch: "
                f"target labels {invalid_labels[:8]} are outside [0, {num_classes - 1}] "
                f"for a {num_classes}-class model. "
                "Check the dataset label mapping and checkpoint."
            )
        # Reject invalid labels before invoking a loss kernel. Keep the original
        # tensors here so the adapter receives them on the expected device.
        self._accumulate_validation_loss(preds, targets, image_size=None)
        targets = target_idx
        k = min(5, num_classes)
        topk = logits.topk(k, dim=1).indices  # [B, k]
        correct = topk == targets.unsqueeze(1)

        self._top1_correct += int(correct[:, 0].sum().item())
        self._top5_correct += int(correct.any(dim=1).sum().item())
        self._total += int(targets.numel())
        # Sized lazily from the logits width. Every validated target and its
        # prediction contribute, including false positives for absent classes.
        pred = topk[:, 0]
        if self._class_tp is None:
            self._class_tp = torch.zeros(num_classes, dtype=torch.long)
            self._class_pred = torch.zeros(num_classes, dtype=torch.long)
            self._class_target = torch.zeros(num_classes, dtype=torch.long)
        target_idx = targets.long()
        pred_idx = pred.long()
        ones = torch.ones_like(target_idx)
        self._class_target.index_add_(0, target_idx, ones)
        self._class_pred.index_add_(0, pred_idx, ones)
        hit = target_idx == pred_idx
        self._class_tp.index_add_(0, target_idx[hit], ones[hit])
        if getattr(self.config, "visualize", False):
            self._visualize_batch(logits, targets, pred)

    def _visualize_batch(self, logits, targets, top1) -> None:
        """Draw each image with its label and top-1 prediction (#887).

        LibreYOLO's classification counterpart of detection ``visualize``:
        written as the images are validated, framed green in
        ``visualize/correct/`` when top-1 is right and red in
        ``visualize/errors/`` when it is wrong. Drawing only: it
        never feeds the metrics.
        """
        import cv2  # noqa: PLC0415

        from .val_plotter import ValPlotter, visualize_subdir  # noqa: PLC0415

        # predict() reports softmax probabilities of the same logits.
        scores = logits.softmax(dim=1)
        dataset = getattr(self.dataloader, "dataset", None)
        samples = getattr(getattr(dataset, "_impl", None), "samples", None) or []
        out_dir = self.save_dir / "visualize"
        for i in range(len(targets)):
            index = self.seen + i
            if index >= len(samples):
                continue
            img_path = samples[index][0]
            try:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                pred = int(top1[i])
                correct = pred == int(targets[i])
                ValPlotter.plot_classify_visualize(
                    img_bgr,
                    self._class_display_name(int(targets[i])),
                    self._class_display_name(pred),
                    float(scores[i, pred]),
                    out_dir
                    / visualize_subdir(correct)
                    / f"{index:06d}_{Path(str(img_path)).stem}.jpg",
                    show_labels=self.config.show_labels,
                    show_conf=self.config.show_conf,
                    correct=correct,
                )
            except Exception as exc:
                logger.warning("visualize failed for image %d: %s", index, exc)

    def _class_display_name(self, index: int) -> str:
        # Label indices follow the model's explicit class order when it has
        # one (including the full ImageNet head), else the train split order.
        classes = self._model_class_names() or getattr(
            getattr(self.dataloader, "dataset", None), "classes", None
        )
        if classes and 0 <= index < len(classes):
            return str(classes[index])
        return str(index)

    def _compute_metrics(self) -> Dict[str, float]:
        total = max(self._total, 1)
        top1 = self._top1_correct / total
        top5 = self._top5_correct / total
        precision, recall, f1 = self._macro_precision_recall_f1()
        return {
            "metrics/accuracy_top1": top1,
            "metrics/accuracy_top5": top5,
            "metrics/precision": precision,
            "metrics/recall": recall,
            "metrics/f1": f1,
            "fitness": top1,
            **self._validation_loss_metrics(),
        }

    def _macro_precision_recall_f1(self) -> tuple[float, float, float]:
        """Macro-averaged precision, recall and F1 from per-class confusion counts.

        Per class: precision = tp / (tp + fp) (0 when the class was never
        predicted), recall = tp / (tp + fn), f1 = 2PR / (P + R) (0 when both
        are 0). Classes with no ground-truth samples are excluded from the
        mean. Returns zeros when nothing was accumulated.
        """
        if self._class_tp is None:
            return 0.0, 0.0, 0.0
        tp = self._class_tp.double()
        predicted = self._class_pred.double()
        support = self._class_target.double()
        present = support > 0
        if not bool(present.any()):
            return 0.0, 0.0, 0.0
        zeros = torch.zeros_like(tp)
        precision = torch.where(predicted > 0, tp / predicted.clamp(min=1), zeros)
        recall = torch.where(support > 0, tp / support.clamp(min=1), zeros)
        denom = precision + recall
        f1 = torch.where(
            denom > 0, 2 * precision * recall / denom.clamp(min=1e-12), zeros
        )
        return (
            float(precision[present].mean()),
            float(recall[present].mean()),
            float(f1[present].mean()),
        )

    def _print_results(self, metrics: Dict[str, float]) -> None:
        logger.info("=" * 50)
        logger.info("Classification Validation Results")
        logger.info("=" * 50)
        logger.info("  top-1 accuracy: %.4f", metrics.get("metrics/accuracy_top1", 0.0))
        logger.info("  top-5 accuracy: %.4f", metrics.get("metrics/accuracy_top5", 0.0))
        logger.info(
            "  macro precision: %.4f  recall: %.4f  f1: %.4f",
            metrics.get("metrics/precision", 0.0),
            metrics.get("metrics/recall", 0.0),
            metrics.get("metrics/f1", 0.0),
        )
        logger.info("  images: %d", self._total)
        logger.info("=" * 50)
