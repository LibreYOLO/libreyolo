"""YOLOv9 E2E trainer."""

import torch

from .config import YOLO9E2EConfig
from ..yolo9.trainer import YOLO9Trainer


class YOLO9E2ETrainer(YOLO9Trainer):
    """Thin trainer subclass for yolo9_e2e family metadata and defaults."""

    @classmethod
    def _config_class(cls):
        return YOLO9E2EConfig

    def get_model_family(self) -> str:
        return "yolo9_e2e"

    def get_model_tag(self) -> str:
        return f"YOLOv9-E2E-{self.config.size}"

    def validate_validation_loss_config(self) -> None:
        if not getattr(self.config, "val_loss", False):
            return

        from .nn import LibreYOLO9E2EModel, YOLO9E2EDetect

        task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        standard_model = (
            type(self.model) is LibreYOLO9E2EModel
            and type(self.model.head) is YOLO9E2EDetect
        )
        if task != "detect" or not standard_model:
            raise ValueError(
                "val_loss=True currently supports YOLO9-E2E detection only; "
                "non-detect tasks are not supported"
            )

    def build_validation_loss_adapter(self, model: torch.nn.Module):
        from .validation_loss import YOLO9E2EValidationLoss

        return YOLO9E2EValidationLoss(
            model,
            max_labels=int(getattr(self.config, "max_labels", 300)),
        )

    def cuda_graph_train_spec(self):
        """Capture spec: graph both branches, keep the dual TAL loss eager.

        The base YOLO9 spec is restricted to the plain ``DDetect`` head, so
        E2E needs its own: a train-mode forward without targets returns
        ``{"one2many": [...], "one2one": [...]}`` — both branches' raw maps,
        including the detach that blocks one-to-one gradients from reaching
        the backbone — and ``assemble`` replays the dual-assignment loss over
        them exactly as ``YOLO9E2EDetect.forward`` does with targets.
        """
        from libreyolo.training.cuda_graph import (
            CudaGraphTrainSpec,
            GraphableNetwork,
        )
        from .nn import LibreYOLO9E2EModel, YOLO9E2EDetect

        task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        if task != "detect":
            return None
        if type(self.model) is not LibreYOLO9E2EModel:
            return None
        if type(self.model.head) is not YOLO9E2EDetect:
            return None

        network = GraphableNetwork(self.model)

        def assemble(flat, imgs, targets, polygons=None):
            branches = network.rebuild(flat)
            loss_fn = self.model.head._get_loss_fn(imgs.device)
            loss_fn.update_anchors([imgs.shape[3], imgs.shape[2]])
            return loss_fn(branches["one2many"], branches["one2one"], targets)

        return CudaGraphTrainSpec(network=network, assemble=assemble)
