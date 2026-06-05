"""Regression tests for two EC training fixes:

1. ``DFINETrainTransform(imagenet_norm=True)`` actually applies ImageNet
   normalization. EC's pretrained ViT expects this; without it, fine-tunes
   silently corrupt the model.
2. ``DFINETrainer._setup_optimizer`` correctly excludes MHA's
   ``self_attn.in_proj_bias`` from weight decay (matches upstream's
   ``(?:norm|bn|bias)`` substring regex).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

CKPT_PATH = Path("weights/LibreECs.pt")


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


def test_ec_train_applies_dataset_class_names_before_trainer(monkeypatch, tmp_path):
    from libreyolo.models.ec import trainer as trainer_module
    from libreyolo.models.ec.model import LibreEC

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {tmp_path}",
                "train: images/train",
                "val: images/val",
                "nc: 2",
                "names: [red, white]",
            ]
        )
    )
    captured = {}

    class FakeTrainer:
        def __init__(self, *, wrapper_model, num_classes, data, **kwargs):
            captured["wrapper_names"] = dict(wrapper_model.names)
            captured["wrapper_nc"] = wrapper_model.nb_classes
            captured["num_classes"] = num_classes
            captured["data"] = data

        def train(self):
            return {"best_checkpoint": ""}

    monkeypatch.setattr(trainer_module, "ECTrainer", FakeTrainer)

    model = LibreEC(model_path=None, size="s", nb_classes=80, device="cpu")
    model.train(data=str(data_yaml), allow_experimental=True, seed=0)

    assert model.nb_classes == 2
    assert model.names == {0: "red", 1: "white"}
    assert captured == {
        "wrapper_names": {0: "red", 1: "white"},
        "wrapper_nc": 2,
        "num_classes": 2,
        "data": str(data_yaml),
    }


@pytest.mark.external_data
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

