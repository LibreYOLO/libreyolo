"""Frozen attentive-probe trainer for V-JEPA 2 video classification.

The encoder is frozen by default and kept in eval mode; only the attentive
pooler and the linear classifier are optimized. This is the bounded, supported
training story for this family -- self-supervised encoder pretraining is
rejected in ``LibreVJEPA2.train`` before any dataset is constructed.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Type

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...training.config import TrainConfig
from ...training.optim import build_optimizer
from ...training.scheduler import WarmupCosineScheduler
from ...training.trainer import BaseTrainer
from ..base.classify_validation_loss import ClassifyValidationLossMixin
from .dataset import VideoClipDataset, collate_clips, load_video_dataset

logger = logging.getLogger(__name__)


class VJEPA2Config(TrainConfig):
    """Probe-training defaults.

    The encoder is frozen, so the trainable head is tiny and tolerates a
    higher LR than a full fine-tune would.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("lr0", 1e-3)
        kwargs.setdefault("weight_decay", 0.05)
        kwargs.setdefault("warmup_epochs", 1)
        kwargs.setdefault("batch", 2)
        super().__init__(**kwargs)


class VJEPA2Trainer(ClassifyValidationLossMixin, BaseTrainer):
    """Cross-entropy training of the attentive pooler + classifier only.

    The probe is an ordinary cross-entropy classifier over clip logits, so it
    reuses the shared classify validation-loss gate rather than reimplementing
    it; ``val_loss=True`` is therefore supported.
    """

    best_metric_key = "metrics/accuracy_top1"

    def __init__(self, *args, **kwargs):
        # V-JEPA2 has no working per-epoch validation path yet, so a selected
        # metric could never be honored. Reject rather than accept and ignore.
        if str(kwargs.get("best_metric") or "").strip():
            raise NotImplementedError(
                "best_metric is not supported for V-JEPA2; its training run "
                "does not validate, so no metric can drive best.pt or patience."
            )
        super().__init__(*args, **kwargs)

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return VJEPA2Config

    def get_model_family(self) -> str:
        return "vjepa2"

    def get_model_tag(self) -> str:
        return f"VJEPA2-{self.config.size}"

    def create_transforms(self):
        # Video clips come from the family-local dataset below, so the
        # detection transform factory is never used.
        return None, None

    def create_scheduler(self, iters_per_epoch: int):
        return WarmupCosineScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            plateau_epochs=getattr(self.config, "no_aug_epochs", 0),
            min_lr_ratio=getattr(self.config, "min_lr_ratio", 0.05),
        )

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> List[str]:
        """Freeze the encoder and keep it in eval mode.

        Returns the names of the parameters that remain trainable, so the
        caller (and the tests) can assert exactly what is being optimized
        rather than trusting that the freeze happened.
        """
        model = self.model
        encoder = getattr(model, "encoder", None)
        if encoder is None:
            raise RuntimeError(
                "V-JEPA 2 probe training needs a classifier model with an "
                "'encoder' submodule."
            )
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        return [n for n, p in model.named_parameters() if p.requires_grad]

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        trainable = self.freeze_encoder()
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("no trainable parameters after freezing the encoder")
        logger.info(
            "V-JEPA 2 probe: training %d tensors (%s...), encoder frozen",
            len(trainable),
            ", ".join(trainable[:3]),
        )
        return build_optimizer(
            torch.optim.AdamW, params, lr=self.config.lr0, weight_decay=self.config.weight_decay
        )

    def _setup_data(self):
        """Build video clip loaders; never the image ImageFolder path."""
        wrapper = getattr(self, "wrapper_model", None)
        clip_frames = int(getattr(wrapper, "clip_frames", 16))
        frame_stride = int(getattr(wrapper, "frame_stride", 2))
        crop_size = int(getattr(wrapper, "crop_size", 256))

        data = load_video_dataset(self.config.data)
        self.data_names = data["names"]

        train_dataset = VideoClipDataset(
            data["train"], clip_frames, frame_stride, crop_size, train=True
        )
        batch = max(1, int(self.config.batch))
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch,
            shuffle=True,
            num_workers=self.config.workers,
            collate_fn=collate_clips,
            drop_last=len(train_dataset) >= batch,
        )
        if "val" in data:
            self.val_loader = DataLoader(
                VideoClipDataset(
                    data["val"], clip_frames, frame_stride, crop_size, train=False
                ),
                batch_size=batch,
                shuffle=False,
                num_workers=self.config.workers,
                collate_fn=collate_clips,
            )
        logger.info(
            "V-JEPA 2 video dataset: %d train clips, %d classes, %d frames/clip",
            len(train_dataset),
            data["nc"],
            clip_frames,
        )
        return train_dataset

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        # imgs is (B, F, C, H, W): time must still be there.
        if imgs.ndim != 5:
            raise ValueError(
                f"probe training expects (B, F, C, H, W) clips, got {tuple(imgs.shape)}"
            )
        logits = self.model(imgs)
        loss = F.cross_entropy(logits, targets)
        return {"total_loss": loss, "loss_ce": loss.detach()}

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        return {"ce": float(outputs.get("loss_ce", outputs["total_loss"]))}
