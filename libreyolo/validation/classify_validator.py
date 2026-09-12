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
from .loss import ValidationLossMixin

if TYPE_CHECKING:
    from .config import ValidationConfig
    from .loss import ValidationLossAdapter

logger = logging.getLogger(__name__)


class ClassifyValidator(ValidationLossMixin, BaseValidator):
    """Top-1/top-5 accuracy validator for the classification task."""

    task = "classify"
    # Confusion matrix, rows = targets, cols = top-1 predictions. Sized
    # lazily from the first batch's logits width. Declared as a class-level
    # default so validators built via ``object.__new__`` without a call to
    # ``_init_metrics`` still see ``None`` instead of raising AttributeError.
    _confusion: Optional[torch.Tensor] = None

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

    def _dataset_transform_kwargs(self) -> dict:
        """Extra kwargs for ``build_classify_transforms`` (mean/std/interp/crop).

        Defaults to the model's native eval pipeline (``crop_pct`` +
        ``interpolation``) so ``val()`` matches ``predict()`` with ImageNet
        normalization; subclasses (e.g. the CLIP validator) override to inject
        family-specific normalization (mean/std).
        """
        kwargs: dict = {}
        crop_pct = getattr(self.model, "crop_pct", None)
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
        # Confusion matrix, rows = targets, cols = top-1 predictions. Sized
        # lazily from the first batch's logits width.
        self._confusion: Optional[torch.Tensor] = None
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

        pred = topk[:, 0]
        if self._confusion is None:
            self._confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        valid = (targets >= 0) & (targets < num_classes)
        flat = targets[valid].long() * num_classes + pred[valid].long()
        self._confusion += torch.bincount(
            flat, minlength=num_classes * num_classes
        ).view(num_classes, num_classes)

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
        """Macro-averaged P/R/F1 over classes present in the targets.

        Per class: precision = tp / (tp + fp) (0 when the class was never
        predicted), recall = tp / (tp + fn), f1 = 2PR / (P + R) (0 when both
        are 0). Classes with no ground-truth samples are excluded from the
        mean. Returns zeros when nothing was accumulated.
        """
        if self._confusion is None:
            return 0.0, 0.0, 0.0
        confusion = self._confusion.double()
        tp = confusion.diag()
        fp = confusion.sum(dim=0) - tp
        fn = confusion.sum(dim=1) - tp
        support = confusion.sum(dim=1)
        present = support > 0
        if not bool(present.any()):
            return 0.0, 0.0, 0.0
        precision = torch.where(
            tp + fp > 0, tp / (tp + fp).clamp(min=1), torch.zeros_like(tp)
        )
        recall = torch.where(
            tp + fn > 0, tp / (tp + fn).clamp(min=1), torch.zeros_like(tp)
        )
        denom = precision + recall
        f1 = torch.where(
            denom > 0,
            2 * precision * recall / denom.clamp(min=1e-12),
            torch.zeros_like(tp),
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
        logger.info("  images: %d", self._total)
        logger.info("=" * 50)
