"""
Distillation loss functions: MGD and CWD.

Both losses are model-agnostic — they operate on pairs of feature tensors
(student, teacher) of shape [N, C, H, W] and handle channel mismatches
internally via 1x1 convolution adapters.

References:
    - MGD: Yang et al., "Masked Generative Distillation", ECCV 2022
    - CWD: Shu et al., "Channel-Wise Knowledge Distillation for Dense Prediction", ICCV 2021
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MGDLoss(nn.Module):
    """Masked Generative Distillation loss.

    Randomly masks spatial positions of the student's (channel-aligned)
    features and trains a lightweight generator to reconstruct the
    teacher's full feature map from the remaining pixels.

    Args:
        student_channels: Number of channels in the student feature map.
        teacher_channels: Number of channels in the teacher feature map.
        mask_ratio: Fraction of spatial positions to mask (default: 0.65).
        loss_weight: Scalar multiplier for this loss term (default: 1.0).
            Note: The overall distillation weight (alpha) is applied in the
            Distiller, so this is per-scale weight.

    Shape:
        - student_feat: (N, student_channels, H, W)
        - teacher_feat: (N, teacher_channels, H, W)
        - output: scalar loss

    Example::

        loss_fn = MGDLoss(student_channels=128, teacher_channels=256)
        loss = loss_fn(student_feat, teacher_feat)
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        mask_ratio: float = 0.65,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.loss_weight = loss_weight

        # 1x1 conv to align student channels to teacher channels
        self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)

        # Two-layer generation block (from MGD paper)
        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
        )

    def forward(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor
    ) -> torch.Tensor:
        # Align student channels to teacher channels
        aligned = self.align(student_feat)

        # Generate random spatial mask (same mask for all channels)
        N, C, H, W = aligned.shape
        mat = torch.rand(N, 1, H, W, device=aligned.device)
        mask = (mat >= self.mask_ratio).float()

        # Reconstruct teacher features from masked student features
        generated = self.generation(aligned * mask)

        # MSE loss between generated and actual teacher features
        loss = F.mse_loss(generated, teacher_feat)
        return self.loss_weight * loss


class CWDLoss(nn.Module):
    """Channel-Wise Knowledge Distillation loss.

    Converts each channel's spatial activations into a probability
    distribution via softmax, then minimizes KL divergence between
    teacher and student distributions channel by channel.

    If student and teacher have different channel counts, an optional
    1x1 conv adapter aligns them.

    Args:
        student_channels: Number of channels in the student feature map.
            If None, no channel adapter is created (assumes channels match).
        teacher_channels: Number of channels in the teacher feature map.
            If None, no channel adapter is created.
        tau: Temperature for softmax (default: 1.0).
        loss_weight: Scalar multiplier for this loss term (default: 1.0).

    Shape:
        - student_feat: (N, student_channels, H, W)
        - teacher_feat: (N, teacher_channels, H, W)
        - output: scalar loss

    Example::

        loss_fn = CWDLoss(student_channels=128, teacher_channels=256)
        loss = loss_fn(student_feat, teacher_feat)
    """

    def __init__(
        self,
        student_channels: int | None = None,
        teacher_channels: int | None = None,
        tau: float = 1.0,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.tau = tau
        self.loss_weight = loss_weight

        # Optional channel adapter
        if (
            student_channels is not None
            and teacher_channels is not None
            and student_channels != teacher_channels
        ):
            self.adapter = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
        else:
            self.adapter = None

    def forward(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor
    ) -> torch.Tensor:
        # Adapt student channels if needed
        if self.adapter is not None:
            student_feat = self.adapter(student_feat)

        N, C, H, W = student_feat.shape
        assert student_feat.shape == teacher_feat.shape, (
            f"Feature shape mismatch after adaptation: "
            f"student={student_feat.shape}, teacher={teacher_feat.shape}"
        )

        # Reshape to (N*C, H*W) — treat each channel as an independent distribution
        s_flat = student_feat.reshape(N * C, -1) / self.tau
        t_flat = teacher_feat.reshape(N * C, -1) / self.tau

        # Softmax over spatial dimension -> probability distributions
        soft_t = F.softmax(t_flat, dim=1)
        log_soft_s = F.log_softmax(s_flat, dim=1)

        # KL divergence, summed over spatial, averaged over (N, C)
        loss = F.kl_div(log_soft_s, soft_t, reduction="sum")
        loss = loss * (self.tau ** 2) / (N * C)

        return self.loss_weight * loss


# Registry of available loss classes
DISTILL_LOSSES = {
    "mgd": MGDLoss,
    "cwd": CWDLoss,
}
