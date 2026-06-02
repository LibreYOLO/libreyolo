"""LibreFOMO training loss."""

from __future__ import annotations

import torch
import torch.nn as nn


class LibreFOMOLoss(nn.Module):
    """Weighted pixel-wise CrossEntropyLoss for FOMO grid targets.

    Args:
        num_classes: Number of foreground classes (not counting background).
        fg_weight: Loss weight applied to all foreground channels.
            Background channel always has weight 1.0.
        device: Tensor device for the weight vector.
    """

    def __init__(
        self,
        num_classes: int = 1,
        fg_weight: float = 100.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        # weights shape: (1 + num_classes,) — index 0 is background
        weights = torch.ones(1 + num_classes, dtype=torch.float32, device=device)
        weights[1:] = fg_weight
        self.register_buffer("weights", weights)
        self.criterion = nn.CrossEntropyLoss(weight=self.weights)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # Rebuild criterion with updated weights buffer after device move
        result.criterion = nn.CrossEntropyLoss(weight=result.weights)
        return result

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> dict:
        """Compute FOMO loss.

        Args:
            logits: Shape ``(B, 1+nc, H_grid, W_grid)``.
            targets: Shape ``(B, H_grid, W_grid)``, dtype int64, value = class_id
                (0 = background, 1..nc = foreground classes).

        Returns:
            Dict with keys ``total_loss`` (tensor) and ``ce`` (float).
        """
        loss = self.criterion(logits, targets)
        return {"total_loss": loss, "ce": float(loss.item())}
