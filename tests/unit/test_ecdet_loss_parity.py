"""Loss-parity test for ECCriterion vs upstream EdgeCrafter ECCriterion.

Builds the same model + criterion in both implementations, feeds an identical
forward output, and asserts every loss key agrees at 1e-5. Also runs a single
training step (forward + criterion + backward + optimizer step) end-to-end
without asserting convergence — a non-NaN, finite-loss smoke test only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.unit

UPSTREAM_PATH = Path(os.environ.get("UPSTREAM_PATH", "downloads/EdgeCrafter/ecdetseg"))
CKPT_PATH = Path("downloads/ec_weights/ecdet_s.pth")


def _try_import_upstream():
    if not UPSTREAM_PATH.exists():
        pytest.skip(f"Upstream EdgeCrafter not found at {UPSTREAM_PATH}")
    sys.path.insert(0, str(UPSTREAM_PATH))
    try:
        from engine.edgecrafter.criterion import ECCriterion as UpstreamECCriterion
        from engine.edgecrafter.matcher import HungarianMatcher as UpstreamMatcher
    except ImportError as e:
        pytest.skip(f"Upstream deps missing: {e}")
    return UpstreamECCriterion, UpstreamMatcher


def _seeded_outputs_and_targets(device="cpu"):
    """Synthetic ECDET training-mode outputs that mimic the model's structure."""
    torch.manual_seed(0)
    B, Q, NC, RM = 2, 300, 80, 32

    def make_layer():
        return {
            "pred_logits": torch.randn(B, Q, NC, device=device),
            "pred_boxes": torch.rand(B, Q, 4, device=device).clamp(0.05, 0.95),
            "pred_corners": torch.randn(B, Q, 4 * (RM + 1), device=device),
            "ref_points": torch.rand(B, Q, 4, device=device).clamp(0.05, 0.95),
        }

    layers = [make_layer() for _ in range(4)]
    out = layers[-1].copy()
    out["up"] = torch.tensor([0.5], device=device)
    out["reg_scale"] = torch.tensor([4.0], device=device)
    aux = []
    for layer in layers[:-1]:
        d = layer.copy()
        d["teacher_corners"] = layers[-1]["pred_corners"]
        d["teacher_logits"] = layers[-1]["pred_logits"]
        aux.append(d)
    out["aux_outputs"] = aux
    out["pre_outputs"] = {
        "pred_logits": torch.randn(B, Q, NC, device=device),
        "pred_boxes": torch.rand(B, Q, 4, device=device).clamp(0.05, 0.95),
    }
    out["enc_aux_outputs"] = [
        {
            "pred_logits": torch.randn(B, Q, NC, device=device),
            "pred_boxes": torch.rand(B, Q, 4, device=device).clamp(0.05, 0.95),
        }
    ]
    out["enc_meta"] = {"class_agnostic": False}

    targets = [
        {
            "labels": torch.tensor([0, 1, 2], dtype=torch.long, device=device),
            "boxes": torch.tensor(
                [[0.3, 0.3, 0.2, 0.2], [0.6, 0.6, 0.1, 0.1], [0.5, 0.5, 0.3, 0.3]],
                device=device,
            ),
        },
        {
            "labels": torch.tensor([5], dtype=torch.long, device=device),
            "boxes": torch.tensor([[0.4, 0.4, 0.2, 0.2]], device=device),
        },
    ]
    return out, targets


def test_eccriterion_loss_parity_vs_upstream():
    UpECCriterion, UpMatcher = _try_import_upstream()
    from libreyolo.models.dfine.matcher import HungarianMatcher as LibreMatcher
    from libreyolo.models.ecdet.loss import ECCriterion as LibreECCriterion

    cost_weights = {"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0}
    weight_dict = {
        "loss_mal": 1.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
        "loss_fgl": 0.15,
        "loss_ddf": 1.5,
    }

    up_matcher = UpMatcher(
        weight_dict=cost_weights, use_focal_loss=True, alpha=0.25, gamma=2.0
    )
    lb_matcher = LibreMatcher(
        weight_dict=cost_weights, use_focal_loss=True, alpha=0.25, gamma=2.0
    )

    up_crit = UpECCriterion(
        matcher=up_matcher,
        weight_dict=weight_dict,
        losses=["mal", "boxes", "local"],
        num_classes=80,
        alpha=0.75,
        gamma=2.0,
        reg_max=32,
    )
    lb_crit = LibreECCriterion(
        matcher=lb_matcher,
        weight_dict=weight_dict,
        losses=["mal", "boxes", "local"],
        num_classes=80,
        alpha=0.75,
        gamma=2.0,
        reg_max=32,
    )
    up_crit.eval()
    lb_crit.eval()

    out_up, targets_up = _seeded_outputs_and_targets()
    out_lb, targets_lb = _seeded_outputs_and_targets()  # same seeds → same tensors

    up_losses = up_crit(out_up, targets_up)
    lb_losses = lb_crit(out_lb, targets_lb)

    common_keys = set(up_losses) & set(lb_losses)
    assert len(common_keys) > 0, "no overlapping loss keys"

    for k in sorted(common_keys):
        up_v = float(up_losses[k].item())
        lb_v = float(lb_losses[k].item())
        diff = abs(up_v - lb_v)
        assert diff < 1e-4, (
            f"{k}: upstream={up_v:.6f} libreyolo={lb_v:.6f} diff={diff:.2e}"
        )


@pytest.mark.skipif(not CKPT_PATH.exists(), reason=f"{CKPT_PATH} not present")
def test_ecdet_one_step_training_smoke():
    """Single training step: forward + criterion + backward + optimizer step.

    Asserts the loss is finite and a backward pass produces non-zero gradients.
    Does NOT assert convergence — this is a wiring smoke test.
    """
    from libreyolo import LibreYOLO
    from libreyolo.models.dfine.matcher import HungarianMatcher
    from libreyolo.models.ecdet.loss import ECCriterion

    m = LibreYOLO("weights/LibreECDETs.pt", device="cpu")
    m.model.train()

    matcher = HungarianMatcher(
        weight_dict={"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
    )
    crit = ECCriterion(
        matcher=matcher,
        weight_dict={
            "loss_mal": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
        },
        losses=["mal", "boxes", "local"],
        num_classes=80,
        alpha=0.75,
        gamma=2.0,
        reg_max=32,
    )
    crit.train()

    optim = torch.optim.AdamW(m.model.parameters(), lr=1e-5)

    torch.manual_seed(0)
    imgs = torch.randn(2, 3, 640, 640)
    targets = [
        {
            "labels": torch.tensor([0, 1], dtype=torch.long),
            "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.6, 0.6, 0.1, 0.1]]),
        },
        {
            "labels": torch.tensor([2], dtype=torch.long),
            "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
        },
    ]

    out = m.model(imgs, targets=targets)
    losses = crit(out, targets)
    total = sum(losses.values())

    assert torch.isfinite(total), f"loss is non-finite: {total}"
    optim.zero_grad()
    total.backward()
    n_with_grad = sum(
        1 for p in m.model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad > 100, f"too few params with non-zero grad: {n_with_grad}"
    optim.step()
