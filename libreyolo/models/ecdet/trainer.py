"""ECDetTrainer — D-FINE-style trainer adapted for ECDet (EXPERIMENTAL).

Subclasses ``DFINETrainer`` and overrides only the points where ECDet's recipe
diverges from D-FINE's:

* ``on_setup`` swaps the criterion to ``ECCriterion`` with MAL classification
  loss (upstream's default), the ECDet weight_dict, and the ECDet loss list
  ``["mal", "boxes", "local"]``.
* ``get_loss_components`` reports ``mal`` instead of ``vfl``.
* ``get_model_family`` / ``get_model_tag`` / ``_config_class`` updated.

Training has not been validated on a real fine-tune run; this is shipped
behind an explicit ``allow_experimental`` gate on ``LibreECDet.train()``.
"""

from __future__ import annotations

from typing import Dict, Type

import torch

from ...training.config import ECDetConfig, TrainConfig
from ..dfine.matcher import HungarianMatcher
from ..dfine.trainer import DFINETrainer
from ..dfine.transforms import DFINEPassThroughDataset, DFINETrainTransform
from .loss import ECCriterion


class ECDetTrainer(DFINETrainer):
    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return ECDetConfig

    def get_model_family(self) -> str:
        return "ecdet"

    def get_model_tag(self) -> str:
        return f"ECDet-{self.config.size}"

    def create_transforms(self):
        # ECDet's pretrained ViT backbone expects ImageNet-normalized inputs at
        # both train and eval time; the inference path applies the same norm
        # (see commit cc14dd20). Without this, the train/eval input distribution
        # diverges and fine-tuning silently corrupts the model.
        preproc = DFINETrainTransform(
            max_labels=120,
            flip_prob=self.config.flip_prob,
            imgsz=self.config.imgsz,
            imagenet_norm=True,
        )
        return preproc, DFINEPassThroughDataset

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        # FGL/DDF are emitted only by the aux/dn paths (no main-loss key);
        # bare ``outputs.get("loss_ddf")`` was always 0. Aggregate over every
        # variant key so the tqdm display reflects the actual loss magnitude.
        def _sum_with_prefix(prefix: str) -> float:
            total = 0.0
            for k, v in outputs.items():
                if k == prefix or k.startswith(prefix + "_"):
                    total += v.item() if isinstance(v, torch.Tensor) else float(v)
            return total

        return {
            "mal": _sum_with_prefix("loss_mal"),
            "bbox": _sum_with_prefix("loss_bbox"),
            "giou": _sum_with_prefix("loss_giou"),
            "fgl": _sum_with_prefix("loss_fgl"),
            "ddf": _sum_with_prefix("loss_ddf"),
        }

    def on_setup(self):
        matcher = HungarianMatcher(
            weight_dict={"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0,
        )
        self.criterion = ECCriterion(
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
            gamma=2.0,
            reg_max=32,
        ).to(self.device)

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Same target-format translation as D-FINE; only the loss key names differ."""
        B = targets.shape[0]
        H, W = imgs.shape[-2], imgs.shape[-1]
        scale = torch.tensor([W, H, W, H], device=targets.device, dtype=targets.dtype)

        target_list = []
        for b in range(B):
            t = targets[b]
            valid = (t[:, 3] > 0) & (t[:, 4] > 0)
            t_valid = t[valid]
            if t_valid.numel() == 0:
                target_list.append(
                    {
                        "labels": torch.zeros(0, dtype=torch.int64, device=self.device),
                        "boxes": torch.zeros(0, 4, dtype=torch.float32, device=self.device),
                    }
                )
            else:
                target_list.append(
                    {
                        "labels": t_valid[:, 0].long(),
                        "boxes": (t_valid[:, 1:] / scale).clamp(0.0, 1.0),
                    }
                )

        outputs = self.model(imgs, targets=target_list)
        losses = self.criterion(outputs, target_list)
        total = sum(losses.values())

        # Expose every named loss (including aux_/dn_/pre/enc variants) so
        # ``get_loss_components`` can aggregate by prefix.
        result = {"total_loss": total}
        result.update(losses)
        return result
