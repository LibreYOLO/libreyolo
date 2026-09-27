"""Shared image-classification training loss (first-party implementation)."""

import torch
import torch.nn.functional as F


def classification_loss(logits, targets, weights=None):
    """Sample-mean CE with optional train-set inverse-frequency weights.

    A fixed batch denominator keeps hard labels and MixUp/CutMix probabilities
    equivalent and lets DDP average local means without world-size scaling.
    """
    if weights is None:
        return F.cross_entropy(logits, targets)
    # Rare-class weights can exceed float16's finite range under AMP.
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    return F.cross_entropy(
        logits, targets, weight=weights.to(logits), reduction="none"
    ).mean()
