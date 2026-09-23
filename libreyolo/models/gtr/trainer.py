"""GTR fine-tuning through the shared D-FINE training infrastructure.

Uses upstream grouped matching and MAL/FGL losses. The shared DETR transform
receives original-resolution images for square resize, normalization and strong
augmentation. Upstream mosaic/mixup reproduction remains separate work.
"""

from ..base.detr_validation_loss import DETRValidationLoss
from ..deim.transforms import DEIMPassThroughDataset, DEIMTrainTransform
from ..dfine.trainer import DFINETrainer
from .config import GTRConfig
from .criterion import GTRCriterion
from .matcher import HungarianMatcher
from .scheduler import GTRScheduler


class GTRTrainer(DFINETrainer):
    @classmethod
    def _config_class(cls):
        return GTRConfig

    def get_model_family(self):
        return "gtr"

    def get_model_tag(self):
        return f"GTR-{self.config.size}"

    def create_scheduler(self, iters_per_epoch):
        return GTRScheduler(self.effective_lr, iters_per_epoch, self.config)

    def resume(self, checkpoint_path):
        # Restore optimizer moments but keep the resolved recipe (saved config
        # plus explicit overrides). Otherwise optimizer.load_state_dict also
        # overwrites caller-specified weight decay and backbone LR multipliers.
        groups = (
            [
                {
                    key: group[key]
                    for key in ("lr", "lr_mult", "weight_decay")
                    if key in group
                }
                for group in self.optimizer.param_groups
            ]
            if self.optimizer is not None
            else []
        )
        super().resume(checkpoint_path)
        if self.optimizer is not None:
            for group, settings in zip(self.optimizer.param_groups, groups):
                group.update(settings)

    def create_transforms(self):
        return DEIMTrainTransform(
            max_labels=300,
            flip_prob=self.config.flip_prob,
            imgsz=self.config.imgsz,
            imagenet_norm=True,
        ), DEIMPassThroughDataset

    def on_setup(self):
        if self.config.lora:
            raise ValueError(
                "GTR LoRA is not implemented; use freeze for partial fine-tuning"
            )
        self.criterion = self.build_criterion()

    def build_criterion(self, *, distributed_normalize=True):
        matcher = HungarianMatcher(
            weight_dict={"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0,
        )
        return GTRCriterion(
            matcher=matcher,
            weight_dict={
                "loss_mal": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
            },
            losses=["mal", "boxes", "local"],
            num_classes=self.config.num_classes,
            alpha=0.75,
            gamma=1.5,
            reg_max=32,
            group_detr=3,
            distributed_normalize=distributed_normalize,
        ).to(self.device)

    def build_validation_loss_adapter(self, model):
        return DETRValidationLoss(
            model, self.build_criterion(distributed_normalize=False)
        )

    def get_loss_components(self, outputs):
        return {
            name: sum(
                float(v.detach())
                for k, v in outputs.items()
                if k == f"loss_{name}" or k.startswith(f"loss_{name}_")
            )
            for name in ("mal", "bbox", "giou", "fgl", "ddf")
        }
