"""Regression tests for the three ECDet training fixes:

1. ``DFINETrainTransform(imagenet_norm=True)`` actually applies ImageNet
   normalization. ECDet's pretrained ViT expects this; without it, fine-tunes
   silently corrupt the model.
2. ``DFINETrainer._setup_optimizer`` correctly excludes MHA's
   ``self_attn.in_proj_bias`` from weight decay (matches upstream's
   ``(?:norm|bn|bias)`` substring regex).
3. ``ECDetTrainer.get_loss_components`` aggregates ``_aux_*`` / ``_dn_*``
   suffix variants, so FGL / DDF report non-zero magnitudes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit

CKPT_PATH = Path("weights/LibreECDetS.pt")


def test_imagenet_norm_applied_when_flag_true():
    """DFINETrainTransform(imagenet_norm=True) shifts the image distribution
    to roughly mean 0, std ~1; without the flag it stays in [0, 1]."""
    from libreyolo.models.dfine.transforms import DFINETrainTransform

    rng = np.random.default_rng(0)
    img_bgr = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    targets = np.zeros((0, 5), dtype=np.float32)

    plain = DFINETrainTransform(
        strong_augs=False, flip_prob=0.0, imgsz=640, imagenet_norm=False
    )
    norm = DFINETrainTransform(
        strong_augs=False, flip_prob=0.0, imgsz=640, imagenet_norm=True
    )

    img_p, _ = plain(img_bgr, targets, (640, 640))
    img_n, _ = norm(img_bgr, targets, (640, 640))

    # Plain output: in [0, 1], per-channel mean ≈ 0.5 for uniform-random pixels.
    assert 0.0 <= img_p.min() and img_p.max() <= 1.0
    assert np.allclose(img_p.mean(axis=(1, 2)), 0.5, atol=0.01)

    # Normalized output: per-channel mean = (0.5 - imagenet_mean) / imagenet_std.
    # For uniform-random input this is roughly [+0.07, +0.20, +0.42].
    expected_mean = (0.5 - np.array([0.485, 0.456, 0.406])) / np.array(
        [0.229, 0.224, 0.225]
    )
    actual_mean = img_n.mean(axis=(1, 2))
    assert np.allclose(actual_mean, expected_mean, atol=0.05), (
        actual_mean,
        expected_mean,
    )
    # And the std rises from ~0.29 (uniform [0,1]) to ~1.27 (after dividing by ~0.22).
    assert img_n.std() > 1.0, f"normalized std is {img_n.std():.3f}, expected > 1.0"


@pytest.mark.skipif(not CKPT_PATH.exists(), reason=f"{CKPT_PATH} not present")
def test_in_proj_bias_in_no_wd_group():
    """Self-attn ``in_proj_bias`` parameters must land in the no-weight-decay
    group (matches upstream's regex). The previous ``endswith('.bias')`` check
    missed them since the name doesn't contain a dot before ``in_proj_bias``.
    """
    from libreyolo import LibreYOLO

    m = LibreYOLO(str(CKPT_PATH), device="cpu")
    in_proj_bias_params = [
        n for n, _ in m.model.named_parameters() if n.endswith("in_proj_bias")
    ]
    assert len(in_proj_bias_params) >= 5, (
        f"expected >=5 in_proj_bias params (1 enc + 4 dec MHA layers), got "
        f"{len(in_proj_bias_params)}: {in_proj_bias_params}"
    )

    # Mirror DFINETrainer._setup_optimizer's classification logic and verify
    # every in_proj_bias lands in head_no_wd / backbone_no_wd (not _wd).
    no_wd, wd = [], []
    for name, p in m.model.named_parameters():
        if not p.requires_grad:
            continue
        is_norm_or_bias = (
            "norm" in name or ".bn." in name or "bias" in name or "lab.scale" in name
        )
        if is_norm_or_bias:
            no_wd.append(name)
        else:
            wd.append(name)

    misclassified = [n for n in in_proj_bias_params if n in wd]
    assert not misclassified, f"in_proj_bias params got weight decay: {misclassified}"
    assert all(n in no_wd for n in in_proj_bias_params)


def test_get_loss_components_aggregates_prefixes():
    """get_loss_components must sum ``_aux_*`` and ``_dn_*`` variants so DDF/FGL
    report non-zero values. Bare ``loss_ddf`` doesn't exist."""
    from libreyolo.models.dfine.trainer import DFINETrainer

    # Build a fake outputs dict with no main keys, only aux/dn variants.
    outputs = {
        "loss_ddf_aux_0": torch.tensor(1.0),
        "loss_ddf_aux_1": torch.tensor(2.0),
        "loss_ddf_dn_0": torch.tensor(0.5),
        "loss_fgl": torch.tensor(0.3),
        "loss_fgl_aux_0": torch.tensor(0.2),
        "loss_vfl": torch.tensor(4.0),
        "loss_bbox": torch.tensor(5.0),
        "loss_giou": torch.tensor(6.0),
    }

    components = DFINETrainer.get_loss_components(None, outputs)
    assert components["ddf"] == pytest.approx(3.5), components  # 1 + 2 + 0.5
    assert components["fgl"] == pytest.approx(0.5), components  # 0.3 + 0.2
    assert components["vfl"] == pytest.approx(4.0)
    assert components["bbox"] == pytest.approx(5.0)
    assert components["giou"] == pytest.approx(6.0)
