"""
YOLOX Trainer for LibreYOLO.

Thin subclass of BaseTrainer with YOLOX-specific transforms, scheduler,
loss extraction, and bias initialisation.
"""

import torch
from typing import Dict, Type

from libreyolo.training.trainer import BaseTrainer, ensure_mutation_reaches_workers
from libreyolo.training.config import TrainConfig, YOLOXConfig
from ...training.scheduler import WarmupCosineScheduler
from ...training.augment import TrainTransform, MosaicMixupDataset


class YOLOXTrainer(BaseTrainer):
    """YOLOX-specific trainer."""

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return YOLOXConfig

    def get_model_family(self) -> str:
        return "yolox"

    def get_model_tag(self) -> str:
        return f"YOLOX-{self.config.size}"

    def validate_validation_loss_config(self) -> None:
        if not getattr(self.config, "val_loss", False):
            return
        task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        if task != "detect":
            raise ValueError(
                f"val_loss=True currently supports {self.get_model_family()} "
                "detection only; other tasks are not supported"
            )

    def build_validation_loss_adapter(self, model: torch.nn.Module):
        from .validation_loss import YOLOXValidationLoss

        head = getattr(getattr(model, "head", None), "num_classes", None)
        return YOLOXValidationLoss(
            model,
            num_classes=head if head is not None else self.config.num_classes,
            device=self.device,
            family=self.get_model_family(),
        )

    def create_transforms(self):
        preproc = TrainTransform(
            max_labels=50,
            flip_prob=self.config.flip_prob,
            hsv_prob=self.config.hsv_prob,
            flipud=getattr(self.config, "flipud", 0.0),
        )
        return preproc, MosaicMixupDataset

    def create_scheduler(self, iters_per_epoch: int):
        return WarmupCosineScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            warmup_lr_start=self.config.warmup_lr_start,
            plateau_epochs=self.config.no_aug_epochs,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        return {
            "iou": outputs.get("iou_loss", 0),
            "obj": outputs.get("obj_loss", 0),
            "cls": outputs.get("cls_loss", 0),
            "l1": outputs.get("l1_loss", 0),
        }

    def on_setup(self):
        # Only seed focal-loss bias priors for a fresh (from-scratch) head.
        # on_setup runs after any pretrained/resume checkpoint has been loaded
        # in __init__, so unconditionally calling initialize_biases would wipe
        # the learned cls/obj priors on every warm-start.
        if getattr(self.wrapper_model, "model_path", None):
            return
        raw = getattr(self.model, "module", self.model)
        if hasattr(raw, "head") and hasattr(raw.head, "initialize_biases"):
            raw.head.initialize_biases(0.01)

    def on_mosaic_disable(self):
        ensure_mutation_reaches_workers(
            self.train_loader, self.train_loader.dataset, "close_mosaic"
        )
        self.train_loader.dataset.close_mosaic()
        raw = getattr(self.model, "module", self.model)
        raw.head.use_l1 = True
        # use_l1 adds the origin_preds tensors to the captured region's
        # output, so any graph taken before this point no longer matches
        # what the loss needs.
        self.invalidate_cuda_graph("YOLOX enabled the L1 branch at mosaic close")

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        return self.model(imgs, targets)

    def cuda_graph_train_spec(self):
        """Capture spec: graph backbone + head convolutions, SimOTA eager.

        ``YOLOXHead.forward_train_maps`` is the boundary: everything it
        returns is a pure function of the input images at a fixed input
        shape, while ``get_losses`` runs SimOTA, which loops per ground-truth
        box on the host and is neither capturable nor worth capturing.

        The head's ``use_l1`` flag changes what ``forward_train_maps``
        returns, so ``on_mosaic_disable`` invalidates the capture when it
        flips; a graph is never replayed across that switch.
        """
        from libreyolo.training.cuda_graph import (
            CudaGraphTrainSpec,
            GraphableNetwork,
        )
        from .nn import LibreYOLOXModel

        task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        if task != "detect":
            return None
        raw = getattr(self.model, "module", self.model)
        if not isinstance(raw, LibreYOLOXModel):
            return None

        class _YOLOXNetwork(torch.nn.Module):
            """Backbone + head convolutions + decode, no loss."""

            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, imgs):
                return self.model.head.forward_train_maps(self.model.backbone(imgs))

        network = GraphableNetwork(_YOLOXNetwork(raw))

        def assemble(flat, imgs, targets, polygons=None):
            (
                outputs,
                x_shifts,
                y_shifts,
                expanded_strides,
                origin_preds,
            ) = network.rebuild(flat)
            return raw.head.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                targets,
                outputs,
                origin_preds,
                dtype=outputs.dtype,
            )

        return CudaGraphTrainSpec(network=network, assemble=assemble)
