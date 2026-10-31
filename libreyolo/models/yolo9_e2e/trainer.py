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

        from .nn import LibreYOLO9E2EModel, YOLO9E2EHead

        task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        standard_model = (
            type(self.model) is LibreYOLO9E2EModel
            and type(self.model.head) is YOLO9E2EHead
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
        """Capture spec: graph both branches, keep the two-branch loss eager.

        The base YOLO9 spec is restricted to the plain ``YOLO9Head``, so E2E
        needs its own: a train-mode forward without targets returns
        ``{"one_to_many": [...], "one_to_one": [...]}`` (both branches' raw
        maps, including the detach that keeps one-to-one gradients out of the
        backbone) and ``assemble`` applies the two-branch loss to them
        exactly as ``YOLO9E2EHead.forward`` does with targets.
        """
        from libreyolo.training.cuda_graph import (
            CudaGraphTrainSpec,
            GraphableNetwork,
        )
        from .nn import LibreYOLO9E2EModel, YOLO9E2EHead

        task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        if task != "detect":
            return None
        if type(self.model) is not LibreYOLO9E2EModel:
            return None
        if type(self.model.head) is not YOLO9E2EHead:
            return None

        network = GraphableNetwork(self.model)

        def assemble(flat, imgs, targets, polygons=None):
            branches = network.rebuild(flat)
            loss_fn = self.model.head._get_loss_fn(imgs.device)
            loss_fn.update_anchors([imgs.shape[3], imgs.shape[2]])
            return loss_fn(branches["one_to_many"], branches["one_to_one"], targets)

        return CudaGraphTrainSpec(network=network, assemble=assemble)
