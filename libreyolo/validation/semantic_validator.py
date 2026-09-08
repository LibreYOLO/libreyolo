"""Semantic-segmentation validator for LibreYOLO.

Computes mean IoU and pixel accuracy from a confusion matrix accumulated over
a dense-mask validation split, reusing the :class:`BaseValidator` template
(setup -> iterate -> finalize). Predictions and targets are compared at the
model input resolution; ignore pixels (255 by default) are excluded.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.semantic_dataset import (
    IGNORE_INDEX,
    SemanticDataset,
    resolve_semantic_data,
    semantic_collate_fn,
    valid_content_hw,
)
from .base import BaseValidator
from .loss import ValidationLossMixin

if TYPE_CHECKING:
    from .config import ValidationConfig
    from .loss import ValidationLossAdapter

logger = logging.getLogger(__name__)


class SemanticValidator(ValidationLossMixin, BaseValidator):
    """mIoU / pixel-accuracy validator for the semantic task."""

    task = "semantic"

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

    def _setup_dataloader(self) -> DataLoader:
        if not self.config.data:
            raise ValueError("Semantic validation requires data= (a dataset YAML).")
        data_config = resolve_semantic_data(
            self.config.data,
            allow_scripts=getattr(self.config, "allow_download_scripts", False),
        )
        split = self.config.split or "val"

        divisor = getattr(self.model, "semantic_imgsz_divisor", None)
        if divisor:
            imgsz = self.config.imgsz
            sides = imgsz if isinstance(imgsz, (tuple, list)) else (imgsz,)
            if any(int(side) % int(divisor) for side in sides):
                raise ValueError(
                    f"Semantic validation imgsz={self.config.imgsz} must be "
                    f"divisible by {int(divisor)} for this model family."
                )
        resize_mode = getattr(self.model, "semantic_resize_mode", "letterbox")
        self._resize_mode = resize_mode
        dataset = SemanticDataset(
            data_config,
            split=split,
            imgsz=self.config.imgsz,
            augment=False,
            resize_mode=resize_mode,
        )

        model_nc = getattr(self.model, "nb_classes", None)
        if model_nc is not None and int(model_nc) != dataset.nc:
            raise ValueError(
                f"Semantic dataset has {dataset.nc} classes but the model "
                f"predicts {int(model_nc)}. Use a matching dataset/checkpoint."
            )

        self._num_classes = dataset.nc
        self._class_names = dict(dataset.names)
        self._ignore_index = dataset.ignore_index
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=semantic_collate_fn,
        )

    def _init_metrics(self) -> None:
        nc = getattr(self, "_num_classes", 0)
        self._confusion = torch.zeros((nc, nc), dtype=torch.int64)
        self._reset_validation_loss()

    def _preprocess_batch(self, batch: Any) -> tuple:
        images, targets, img_info, img_ids = batch
        return images, targets, img_info, img_ids

    def _extract_logits(self, preds: Any, target_hw: tuple) -> torch.Tensor:
        """Unwrap raw model output into ``[B, C, H, W]`` logits at ``target_hw``."""
        logits = preds
        if isinstance(logits, dict):
            logits = logits.get(
                "semantic_logits",
                logits.get("logits", logits.get("out")),
            )
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        logits = torch.as_tensor(logits)

        if logits.ndim != 4:
            raise ValueError(
                f"Semantic validation expects [B, C, H, W] logits, got shape "
                f"{tuple(logits.shape)}."
            )
        # Upcast before, not inside, the interpolate branch: under half=True a
        # model whose logits already sit at target_hw would otherwise stay fp16
        # all the way into the softmax merge. (.float() on an fp32 tensor is a
        # no-op that returns self, so the common path allocates nothing.)
        logits = logits.float()
        if tuple(logits.shape[-2:]) != target_hw:
            logits = F.interpolate(
                logits, size=target_hw, mode="bilinear", align_corners=False
            )
        return logits

    def _postprocess_predictions(self, preds: Any, batch: Any) -> Any:
        """Decode raw model output into ``[B, H, W]`` class maps."""
        targets = batch[1]
        target_hw = tuple(targets.shape[-2:])
        logits = self._extract_logits(preds, target_hw)
        # Fed the logits already upsampled to the mask resolution, which
        # is exactly what the families' own forward scores in training.
        self._accumulate_validation_loss(logits, targets, image_size=None)
        return logits.argmax(dim=1)

    def _flip_content(
        self, tensor: torch.Tensor, img_info: Any, inplace: bool = False
    ) -> torch.Tensor:
        """Flip each item's real (unpadded) content along width.

        ``SemanticDataset`` anchors real content at the top-left origin and
        pads only the bottom/right (see ``valid_content_hw``). A naive
        ``torch.flip`` on the whole padded canvas would relocate that
        padding to the top-left for the flipped view — an input layout the
        model never saw in training or non-augmented validation. Flipping
        only the valid content window, in its original position, keeps the
        padding exactly where the model expects it in both views. Applying
        this same operation to a logits tensor undoes the mirroring
        (flip-back), since flipping a fixed window twice restores its
        original order.

        ``stretch`` resize mode has no padding (the whole canvas is
        content), so a plain full-width flip is exact there. ``rescale_crop``
        only defines a training sampler; ``SemanticDataset._resize`` validates
        it by direct resize to the canvas, so it shares the stretch geometry.

        ``inplace`` mutates ``tensor`` and is for the flip-back of a logits
        tensor the caller solely owns — at ADE20K scale the defensive copy is
        1.2 GB. Never pass it for the dataloader's image batch.
        """
        if self._resize_mode in ("stretch", "rescale_crop"):
            return tensor.flip(-1)
        if self._resize_mode not in ("letterbox", "resize_crop"):
            # Anything else pads somewhere this window math does not model
            # (EoMT's "split", say). Unreachable today only because
            # SemanticDataset rejects such modes outright; fail loudly rather
            # than silently mirroring the wrong region if that ever changes.
            raise ValueError(
                f"Flip TTA does not know how to place letterbox padding for "
                f"semantic_resize_mode={self._resize_mode!r}."
            )

        canvas_hw = tuple(tensor.shape[-2:])
        flipped = tensor if inplace else tensor.clone()
        for i, info in enumerate(img_info):
            new_h, new_w = valid_content_hw(info["orig_shape"], info["ratio"], canvas_hw)
            window = tensor[i, ..., :new_h, :new_w]
            # .flip() already materializes a copy, so this stays correct even
            # when source and destination are the same storage.
            flipped[i, ..., :new_h, :new_w] = window.flip(-1)
        return flipped

    def _run_validation_augmented(self) -> None:
        """Flip TTA: average softmax(original) and softmax(h-flip), then argmax.

        Runs directly on the dataloader's already-letterboxed batch tensors
        (unlike DetectionValidator, which reloads each image from disk to
        recover true-original-resolution boxes). Semantic targets are dense
        per-pixel masks already aligned 1:1 with the input tensor, so
        flipping the batch tensor and merging in that same space is both
        correct and far cheaper than re-deriving full-resolution masks per
        image and resizing them back down — as long as the flip only
        mirrors the real content and leaves letterbox padding in place
        (see ``_flip_content``).
        """
        import sys
        import time

        from tqdm import tqdm

        from ..utils.tta import average_flip_softmax

        # BaseValidator._run_validation already put the model in eval() before
        # dispatching here.
        pbar = tqdm(
            self.dataloader,
            desc="Validating (TTA x2)",
            total=len(self.dataloader),
            disable=not self.config.verbose or not sys.stderr.isatty(),
            file=sys.stderr,
        )

        total_start = time.time()

        with torch.no_grad():
            for batch in pbar:
                t1 = time.time()
                images, targets, img_info, img_ids = self._preprocess_batch(batch)
                self.speed["preprocess"] += time.time() - t1

                t2 = time.time()
                preds_orig = self._inference(images)
                preds_flip = self._inference(self._flip_content(images, img_info))
                self.speed["inference"] += time.time() - t2

                t3 = time.time()
                target_hw = tuple(targets.shape[-2:])
                logits_orig = self._extract_logits(preds_orig, target_hw)
                # inplace: the extracted flipped logits are a private buffer
                # here (nothing else reads preds_flip afterwards), and at
                # ADE20K scale each [B, C, H, W] float copy is ~1.2 GB.
                logits_flip = self._flip_content(
                    self._extract_logits(preds_flip, target_hw), img_info, inplace=True
                )
                avg_probs = average_flip_softmax(logits_orig, logits_flip)
                del logits_orig, logits_flip, preds_orig, preds_flip
                detections = avg_probs.argmax(dim=1)
                del avg_probs
                self.speed["postprocess"] += time.time() - t3

                self._update_metrics(detections, targets, img_info, img_ids)
                self.seen += len(images)

        self.speed["total"] = time.time() - total_start

    def _update_metrics(
        self, preds: Any, targets: Any, img_info: Any, img_ids: Any = None
    ) -> None:
        pred_maps = preds.detach().cpu().long().view(-1)
        target_maps = targets.detach().cpu().long().view(-1)

        valid = target_maps != self._ignore_index
        if not bool(valid.any()):
            return
        target_valid = target_maps[valid]
        pred_valid = pred_maps[valid].clamp_(0, self._num_classes - 1)

        index = target_valid * self._num_classes + pred_valid
        counts = torch.bincount(index, minlength=self._num_classes**2)
        self._confusion += counts.reshape(self._num_classes, self._num_classes)

    def _per_class_iou(self) -> torch.Tensor:
        confusion = self._confusion.double()
        true_positive = confusion.diag()
        union = confusion.sum(dim=0) + confusion.sum(dim=1) - true_positive
        iou = torch.full_like(true_positive, float("nan"))
        present = union > 0
        iou[present] = true_positive[present] / union[present]
        return iou

    def _compute_metrics(self) -> Dict[str, float]:
        total = self._confusion.sum()
        iou = self._per_class_iou()
        observed = ~torch.isnan(iou)
        miou = float(iou[observed].mean()) if bool(observed.any()) else 0.0
        accuracy = float(self._confusion.diag().sum() / total) if total > 0 else 0.0
        return {
            "metrics/mIoU": miou,
            "metrics/pixel_accuracy": accuracy,
            "fitness": miou,
            **self._validation_loss_metrics(),
        }

    def _print_results(self, metrics: Dict[str, float]) -> None:
        logger.info("=" * 50)
        logger.info("Semantic Segmentation Validation Results")
        logger.info("=" * 50)
        iou = self._per_class_iou()
        for class_id in range(self._num_classes):
            value = iou[class_id]
            if torch.isnan(value):
                continue
            name = self._class_names.get(class_id, str(class_id))
            logger.info("  IoU %-20s %.4f", name, float(value))
        logger.info("  mIoU:           %.4f", metrics.get("metrics/mIoU", 0.0))
        logger.info(
            "  pixel accuracy: %.4f", metrics.get("metrics/pixel_accuracy", 0.0)
        )
        logger.info("=" * 50)


__all__ = ["SemanticValidator", "IGNORE_INDEX"]
