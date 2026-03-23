"""
Model-agnostic distillation orchestrator.

Wires together:
    1. A frozen teacher model
    2. FeatureHookManagers on both teacher and student
    3. Per-scale loss modules (MGD or CWD)

The Distiller is architecture-agnostic — it receives distillation configs
(tap points + channel dims) from the model wrappers and handles the rest.

Usage::

    from libreyolo.distillation import Distiller

    distiller = Distiller(
        teacher_model=teacher.model,      # nn.Module
        student_model=student.model,      # nn.Module
        teacher_config=teacher.get_distill_config(),
        student_config=student.get_distill_config(),
        loss_type="mgd",
    )

    # In training loop:
    teacher_out = distiller.teacher_forward(images)  # no_grad internally
    student_out = model(images, targets)              # normal forward, hooks capture features
    distill_loss = distiller.compute_loss()
    total_loss = task_loss + distill_loss
    distiller.step()  # clear features for next iteration
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .hooks import FeatureHookManager
from .losses import MGDLoss, CWDLoss, DISTILL_LOSSES

logger = logging.getLogger(__name__)


class Distiller(nn.Module):
    """Model-agnostic knowledge distillation orchestrator.

    Manages the teacher model, feature extraction hooks, channel adaptation,
    and distillation loss computation. Works with any architecture that
    provides a ``get_distill_config()`` method.

    Args:
        teacher_model: The teacher's ``nn.Module`` (will be frozen).
        student_model: The student's ``nn.Module`` (hooks are read-only).
        teacher_config: Dict from ``teacher.get_distill_config()`` with keys:
            - tap_points: list of module path strings
            - channels: list of int channel dimensions
            - strides: list of int spatial strides
        student_config: Dict from ``student.get_distill_config()`` (same format).
        loss_type: ``"mgd"`` or ``"cwd"`` (default: ``"mgd"``).
        loss_weight: Global distillation loss weight (alpha). Default: 2e-5 for MGD, 10.0 for CWD.
        mask_ratio: MGD mask ratio (default: 0.65). Ignored for CWD.
        tau: CWD temperature (default: 1.0). Ignored for MGD.
        per_scale_weight: Optional list of per-scale weights. If None, uniform.

    Example::

        distiller = Distiller(
            teacher_model=teacher_nn,
            student_model=student_nn,
            teacher_config={"tap_points": ["neck.elan_down2"], "channels": [512], "strides": [32]},
            student_config={"tap_points": ["neck.elan_down2"], "channels": [128], "strides": [32]},
            loss_type="mgd",
        )
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        teacher_config: Dict,
        student_config: Dict,
        loss_type: str = "mgd",
        loss_weight: Optional[float] = None,
        mask_ratio: float = 0.65,
        tau: float = 1.0,
        per_scale_weight: Optional[List[float]] = None,
    ):
        super().__init__()

        self.loss_type = loss_type.lower()
        self.loss_weight = loss_weight if loss_weight is not None else self._default_weight()

        # Validate configs
        t_strides = teacher_config["strides"]
        s_strides = student_config["strides"]
        if t_strides != s_strides:
            raise ValueError(
                f"Teacher and student must have matching strides. "
                f"Teacher: {t_strides}, Student: {s_strides}"
            )

        self.num_scales = len(t_strides)
        t_channels = teacher_config["channels"]
        s_channels = student_config["channels"]

        # Freeze teacher
        self.teacher = teacher_model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Register hooks on both models
        self.t_hooks = FeatureHookManager(self.teacher, teacher_config["tap_points"])
        self.s_hooks = FeatureHookManager(student_model, student_config["tap_points"])

        # Per-scale weights
        if per_scale_weight is not None:
            assert len(per_scale_weight) == self.num_scales
            self._scale_weights = per_scale_weight
        else:
            self._scale_weights = [1.0] * self.num_scales

        # Build loss modules (one per feature scale)
        self.loss_modules = nn.ModuleList()
        for i in range(self.num_scales):
            loss_fn = self._build_loss(
                s_channels[i], t_channels[i],
                mask_ratio=mask_ratio, tau=tau,
                scale_weight=self._scale_weights[i],
            )
            self.loss_modules.append(loss_fn)

        # Log configuration
        logger.info("Distiller initialized:")
        logger.info(f"  Loss type: {self.loss_type}")
        logger.info(f"  Global weight (alpha): {self.loss_weight}")
        logger.info(f"  Num scales: {self.num_scales}")
        for i, (sc, tc) in enumerate(zip(s_channels, t_channels)):
            logger.info(
                f"  Scale {i} (stride {t_strides[i]}): "
                f"student={sc}ch -> teacher={tc}ch"
            )

    def _default_weight(self) -> float:
        """Return sensible default loss weight for the chosen loss type."""
        if self.loss_type == "mgd":
            return 2e-5
        elif self.loss_type == "cwd":
            return 1.0  # CWD has per-scale weights built in
        return 1.0

    def _build_loss(
        self,
        student_ch: int,
        teacher_ch: int,
        mask_ratio: float,
        tau: float,
        scale_weight: float,
    ) -> nn.Module:
        """Construct a loss module for one feature scale."""
        if self.loss_type == "mgd":
            return MGDLoss(
                student_channels=student_ch,
                teacher_channels=teacher_ch,
                mask_ratio=mask_ratio,
                loss_weight=scale_weight,
            )
        elif self.loss_type == "cwd":
            return CWDLoss(
                student_channels=student_ch,
                teacher_channels=teacher_ch,
                tau=tau,
                loss_weight=scale_weight,
            )
        else:
            raise ValueError(
                f"Unknown loss type: '{self.loss_type}'. "
                f"Available: {list(DISTILL_LOSSES.keys())}"
            )

    @torch.no_grad()
    def teacher_forward(self, images: torch.Tensor) -> Any:
        """Run the teacher model in eval mode with no gradients.

        The forward hooks automatically capture the teacher's features.
        Call this BEFORE the student forward pass.

        Args:
            images: Input batch of shape (N, 3, H, W).

        Returns:
            Teacher model output (usually ignored — we only need the hooks).
        """
        was_training = self.teacher.training
        self.teacher.eval()
        try:
            return self.teacher(images)
        finally:
            if was_training:
                self.teacher.train()

    def compute_loss(self) -> torch.Tensor:
        """Compute total distillation loss across all feature scales.

        Must be called AFTER both teacher_forward() and the student forward
        pass have been executed (so that hooks have captured features).

        Returns:
            Scalar distillation loss, scaled by ``self.loss_weight``.

        Raises:
            RuntimeError: If features haven't been captured yet.
        """
        t_feats = self.t_hooks.get_feature_list()
        s_feats = self.s_hooks.get_feature_list()

        if len(t_feats) != self.num_scales:
            raise RuntimeError(
                f"Expected {self.num_scales} teacher features, got {len(t_feats)}. "
                f"Did you call teacher_forward() before compute_loss()?"
            )
        if len(s_feats) != self.num_scales:
            raise RuntimeError(
                f"Expected {self.num_scales} student features, got {len(s_feats)}. "
                f"Did the student forward pass run before compute_loss()?"
            )

        total = torch.tensor(0.0, device=s_feats[0].device)
        for i, (loss_fn, s_feat, t_feat) in enumerate(
            zip(self.loss_modules, s_feats, t_feats)
        ):
            scale_loss = loss_fn(s_feat, t_feat.detach())
            total = total + scale_loss

        return self.loss_weight * total

    def step(self):
        """Clear captured features. Call at the end of each training step."""
        self.t_hooks.clear()
        self.s_hooks.clear()

    def cleanup(self):
        """Remove all hooks and free resources. Call when training ends."""
        self.t_hooks.remove()
        self.s_hooks.remove()
        logger.info("Distiller cleaned up")

    def __repr__(self) -> str:
        return (
            f"Distiller(loss_type='{self.loss_type}', "
            f"loss_weight={self.loss_weight}, "
            f"num_scales={self.num_scales})"
        )
