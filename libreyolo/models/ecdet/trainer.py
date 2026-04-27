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
from .loss import ECCriterion


class ECDetTrainer(DFINETrainer):
    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return ECDetConfig

    def get_model_family(self) -> str:
        return "ecdet"

    def get_model_tag(self) -> str:
        return f"ECDet-{self.config.size}"

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        def _scalar(v):
            if isinstance(v, torch.Tensor):
                return v.item()
            return float(v)

        return {
            "mal": _scalar(outputs.get("loss_mal", 0)),
            "bbox": _scalar(outputs.get("loss_bbox", 0)),
            "giou": _scalar(outputs.get("loss_giou", 0)),
            "fgl": _scalar(outputs.get("loss_fgl", 0)),
            "ddf": _scalar(outputs.get("loss_ddf", 0)),
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

        zero = torch.tensor(0.0, device=self.device)
        return {
            "total_loss": total,
            "loss_mal": losses.get("loss_mal", zero),
            "loss_bbox": losses.get("loss_bbox", zero),
            "loss_giou": losses.get("loss_giou", zero),
            "loss_fgl": losses.get("loss_fgl", zero),
            "loss_ddf": losses.get("loss_ddf", zero),
        }
