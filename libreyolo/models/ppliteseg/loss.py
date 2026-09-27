"""PP-LiteSeg compound loss: Dice + cross-entropy + edge attention.

Adapted from the Apache-2.0 SuperGradients ``DiceCEEdgeLoss`` and its
``DiceLoss`` / ``MaskAttentionLoss`` / ``target_to_binary_edge`` dependencies
(see ``NOTICE`` in this directory).

One deliberate difference from the source: the released recipe encodes ignore
as class index 19 -- one past the 19 real classes -- and builds a 20-channel
one-hot it then slices back down. LibreYOLO's semantic contract uses 255, and
the model output width stays ``nc`` with no ignore logit, so ignore is handled
by masking instead of by an extra channel. The two are numerically identical:
in the source the ignore channel is dropped and ignore pixels are zeroed out
of both Dice terms anyway.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255


def to_one_hot(target: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    """One-hot ``(B, H, W)`` class IDs into ``(B, C, H, W)``, ignore rows zeroed.

    Ignore pixels become an all-zero vector rather than a class of their own,
    so they contribute to neither the numerator nor the denominator downstream.
    """
    valid = target.ne(ignore_index)
    safe = torch.where(valid, target, torch.zeros_like(target))
    one_hot = F.one_hot(safe.long(), num_classes).permute(0, 3, 1, 2)
    return one_hot * valid.unsqueeze(1)


def one_hot_to_binary_edge(one_hot: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Morphological dilation-minus-erosion edge map, flattened over classes."""
    if kernel_size < 0 or kernel_size % 2 == 0:
        raise ValueError(f"edge kernel must be an odd positive value, got {kernel_size}")
    channels = one_hot.size(1)
    kernel = torch.ones(
        channels, 1, kernel_size, kernel_size, dtype=torch.float32, device=one_hot.device
    )
    padding = (kernel_size - 1) // 2
    # Replicate padding keeps the frame from reading as an edge everywhere.
    padded = F.pad(one_hot.float(), mode="replicate", pad=[padding] * 4)
    dilation = torch.clamp(F.conv2d(padded, kernel, groups=channels), 0, 1)
    erosion = 1 - torch.clamp(F.conv2d(1 - padded, kernel, groups=channels), 0, 1)
    edge = dilation - erosion
    return edge.max(dim=1, keepdim=True)[0]


def target_to_binary_edge(
    target: torch.Tensor, num_classes: int, kernel_size: int, ignore_index: int
) -> torch.Tensor:
    return one_hot_to_binary_edge(
        to_one_hot(target, num_classes, ignore_index), kernel_size=kernel_size
    )


class DiceLoss(nn.Module):
    """Multi-class soft Dice over softmax probabilities, averaged over classes."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = IGNORE_INDEX,
        smooth: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.eps = eps

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(predict, dim=1)
        labels = to_one_hot(target, self.num_classes, self.ignore_index).to(probs.dtype)

        numerator = labels * probs
        denominator = labels + probs
        # False positives predicted on ignore pixels must not count either.
        valid = target.ne(self.ignore_index).unsqueeze(1).expand_as(denominator)
        numerator = numerator * valid
        denominator = denominator * valid

        reduce_dims = [0] + list(range(2, predict.dim()))
        numerator = numerator.sum(dim=reduce_dims)
        denominator = denominator.sum(dim=reduce_dims)
        losses = 1.0 - ((2.0 * numerator + self.smooth) / (denominator + self.eps + self.smooth))
        return losses.mean()


class MaskAttentionCELoss(nn.Module):
    """``w0 * CE + w1 * CE restricted to edge pixels``."""

    def __init__(
        self, ignore_index: int = IGNORE_INDEX, loss_weights: Sequence[float] = (0.5, 0.5)
    ) -> None:
        super().__init__()
        if len(loss_weights) != 2:
            raise ValueError(f"loss_weights must have 2 values, got {len(loss_weights)}")
        self.criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
        self.loss_weights = tuple(float(w) for w in loss_weights)

    def forward(
        self, predict: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        per_pixel = self.criterion(predict, target)
        mask = mask.view(per_pixel.size()) if mask.dim() == 4 else mask
        masked = per_pixel * mask
        selected = masked[mask == 1]
        if selected.numel() == 0:
            # No edge pixels in this batch (e.g. an all-ignore or uniform
            # target): the edge term is defined as zero rather than NaN.
            mask_loss = per_pixel.sum() * 0.0
        else:
            mask_loss = selected.mean()
        return per_pixel.mean() * self.loss_weights[0] + mask_loss * self.loss_weights[1]


class PPLiteSegLoss(nn.Module):
    """Source ``DiceCEEdgeLoss`` over the main head plus three aux heads.

    ``forward`` accepts the training 4-tuple ``(main, aux_s8, aux_s16,
    aux_s32)`` -- or a single tensor, which is treated as main-only -- and
    returns a dict of scalar tensors keyed for the trainer's log line, with
    ``"loss"`` as the total.
    """

    def __init__(
        self,
        num_classes: int,
        num_aux_heads: int = 3,
        weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
        dice_ce_weights: Sequence[float] = (1.0, 1.0),
        ce_edge_weights: Sequence[float] = (0.5, 0.5),
        edge_kernel: int = 5,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        if len(weights) != num_aux_heads + 1:
            raise ValueError(
                f"weights must hold {num_aux_heads + 1} values (main + {num_aux_heads} aux), "
                f"got {len(weights)}"
            )
        self.num_classes = num_classes
        self.num_aux_heads = num_aux_heads
        self.weights = tuple(float(w) for w in weights)
        self.dice_ce_weights = tuple(float(w) for w in dice_ce_weights)
        self.edge_kernel = edge_kernel
        self.ignore_index = ignore_index
        self.ce_edge = MaskAttentionCELoss(ignore_index=ignore_index, loss_weights=ce_edge_weights)
        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def component_names(self) -> List[str]:
        return ["main"] + [f"aux{i}" for i in range(self.num_aux_heads)] + ["loss"]

    def forward(
        self, preds: Sequence[torch.Tensor] | torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(preds):
            preds = (preds,)
        preds = tuple(preds)
        expected = self.num_aux_heads + 1
        if len(preds) not in (1, expected):
            raise ValueError(
                f"PP-LiteSeg loss expects 1 or {expected} prediction tensors, got {len(preds)}"
            )
        target = target.long()
        edge_target = target_to_binary_edge(
            target,
            num_classes=self.num_classes,
            kernel_size=self.edge_kernel,
            ignore_index=self.ignore_index,
        )

        components: Dict[str, torch.Tensor] = {}
        total = None
        names = ["main"] + [f"aux{i}" for i in range(self.num_aux_heads)]
        for index, pred in enumerate(preds):
            if pred.shape[-2:] != target.shape[-2:]:
                raise ValueError(
                    f"PP-LiteSeg loss got logits {tuple(pred.shape[-2:])} against target "
                    f"{tuple(target.shape[-2:])}; every head upsamples to the input canvas, "
                    "so a mismatch means the head scale factors are wrong."
                )
            ce_loss = self.ce_edge(pred, target, edge_target)
            dice_loss = self.dice_loss(pred, target)
            loss = ce_loss * self.dice_ce_weights[0] + dice_loss * self.dice_ce_weights[1]
            components[names[index]] = loss
            weighted = self.weights[index] * loss
            total = weighted if total is None else total + weighted

        components["loss"] = total
        return components


__all__ = [
    "IGNORE_INDEX",
    "DiceLoss",
    "MaskAttentionCELoss",
    "PPLiteSegLoss",
    "one_hot_to_binary_edge",
    "target_to_binary_edge",
    "to_one_hot",
]
