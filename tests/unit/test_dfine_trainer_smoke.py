"""DFINETrainer smoke tests — wiring only, no data."""

from __future__ import annotations

import pytest
import torch

from libreyolo import LibreDFINE

pytestmark = pytest.mark.unit


def test_trainer_target_translation_smoke():
    """Drive on_forward manually with synthetic padded targets.

    Goal: the (B, max_labels, 5) → list[dict] translation works, model+criterion
    run, and a backward pass produces gradients on trainable params.
    """
    from libreyolo.models.dfine.trainer import DFINETrainer

    wrapper = LibreDFINE(None, size="n", device="cpu")
    wrapper.model.train()

    trainer = DFINETrainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="n",
        num_classes=80,
        data=None,
        epochs=1,
        batch=2,
        imgsz=640,
        device="cpu",
        amp=False,
        ema=False,
        no_aug_epochs=0,
        warmup_epochs=0,
        eval_interval=-1,
    )
    trainer.on_setup()  # build criterion

    # Synthetic batch: 2 images, padded targets in pixel cxcywh on a 640×640 grid.
    imgs = torch.randn(2, 3, 640, 640)
    targets = torch.zeros(2, 120, 5)
    # Image 0: 2 boxes
    targets[0, 0] = torch.tensor([3.0, 320.0, 240.0, 100.0, 80.0])
    targets[0, 1] = torch.tensor([17.0, 200.0, 200.0, 60.0, 40.0])
    # Image 1: 1 box
    targets[1, 0] = torch.tensor([1.0, 400.0, 320.0, 120.0, 100.0])

    out = trainer.on_forward(imgs, targets)
    assert "total_loss" in out
    assert torch.isfinite(out["total_loss"]), "total_loss must be finite"
    assert out["total_loss"].item() > 0
    # Each loss family must appear in some form. DDF and FGL only emit
    # aux/dn variants (no bare key) — match by prefix instead of exact name.
    for prefix in ("loss_vfl", "loss_bbox", "loss_giou", "loss_fgl", "loss_ddf"):
        assert any(k == prefix or k.startswith(prefix + "_") for k in out), (
            f"no key matching {prefix}* in output: {sorted(out)}"
        )

    out["total_loss"].backward()

    # Some non-frozen params must have nonzero grad.
    nonzero_grads = sum(
        1
        for p in wrapper.model.encoder.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    assert nonzero_grads > 0, "encoder must have at least one param with nonzero grad"


def test_trainer_handles_empty_targets():
    """A batch where one image has zero GT boxes still works (no NaN, finite loss)."""
    from libreyolo.models.dfine.trainer import DFINETrainer

    wrapper = LibreDFINE(None, size="n", device="cpu")
    wrapper.model.train()

    trainer = DFINETrainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="n",
        num_classes=80,
        data=None,
        epochs=1,
        batch=2,
        imgsz=640,
        device="cpu",
        amp=False,
        ema=False,
        no_aug_epochs=0,
        warmup_epochs=0,
        eval_interval=-1,
    )
    trainer.on_setup()

    imgs = torch.randn(2, 3, 640, 640)
    targets = torch.zeros(2, 120, 5)
    targets[0, 0] = torch.tensor([3.0, 320.0, 240.0, 100.0, 80.0])
    # Image 1: all padding (no boxes)

    out = trainer.on_forward(imgs, targets)
    assert torch.isfinite(out["total_loss"])


def test_optimizer_setup_groups_params_correctly():
    """Norm + bias params should land in the no-WD group; conv weights in WD group."""
    from libreyolo.models.dfine.trainer import DFINETrainer

    wrapper = LibreDFINE(None, size="n", device="cpu")
    trainer = DFINETrainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="n",
        num_classes=80,
        data=None,
        epochs=1,
        batch=2,
        imgsz=640,
        device="cpu",
        amp=False,
        ema=False,
        eval_interval=-1,
    )
    optimizer = trainer._setup_optimizer()
    groups = optimizer.param_groups
    # 4 groups: head/backbone × wd/no-wd, all populated for D-FINE-N.
    assert len(groups) == 4
    wd_groups = [g for g in groups if g["weight_decay"] > 0]
    no_wd_groups = [g for g in groups if g["weight_decay"] == 0]
    assert len(wd_groups) == 2 and len(no_wd_groups) == 2
    for g in groups:
        assert sum(p.numel() for p in g["params"]) > 0


def test_optimizer_layernorm_weights_in_no_wd_group():
    """Regression: transformer norm1/norm2/norm3 weights MUST land in no-WD group.

    A bug in the original pattern (``.norm.`` only) silently put 8 LayerNorm
    weights into the weight-decay group, mis-regularizing decoder/encoder norms.
    """
    from libreyolo.models.dfine.trainer import DFINETrainer

    wrapper = LibreDFINE(None, size="n", device="cpu")
    trainer = DFINETrainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="n",
        num_classes=80,
        data=None,
        epochs=1,
        batch=2,
        imgsz=640,
        device="cpu",
        amp=False,
        ema=False,
        eval_interval=-1,
    )
    optimizer = trainer._setup_optimizer()
    name_by_id = {id(p): n for n, p in wrapper.model.named_parameters()}
    no_wd_names = set()
    for g in optimizer.param_groups:
        if g["weight_decay"] == 0:
            no_wd_names.update(name_by_id[id(p)] for p in g["params"])

    expected_in_no_wd = [
        "encoder.encoder.0.layers.0.norm1.weight",
        "encoder.encoder.0.layers.0.norm2.weight",
        "decoder.decoder.layers.0.norm1.weight",
        "decoder.decoder.layers.0.norm3.weight",
        "decoder.decoder.layers.1.norm1.weight",
        "decoder.decoder.layers.1.norm3.weight",
        "decoder.decoder.layers.2.norm1.weight",
        "decoder.decoder.layers.2.norm3.weight",
    ]
    missing = [n for n in expected_in_no_wd if n not in no_wd_names]
    assert not missing, (
        f"These transformer norm weights should be no-WD but aren't: {missing}"
    )


def test_scheduler_warmup_then_flat():
    """FlatCosineScheduler: warmup → flat → cosine, no NaNs."""
    from libreyolo.training.scheduler import FlatCosineScheduler

    sched = FlatCosineScheduler(
        lr=0.001,
        iters_per_epoch=10,
        total_epochs=10,
        warmup_epochs=2,
        warmup_lr_start=1e-6,
        no_aug_epochs=2,
        min_lr_ratio=0.05,
    )
    # iter 0 → warmup_lr_start
    assert abs(sched.update_lr(0) - 1e-6) < 1e-7
    # iter 20 → end of warmup → ~lr
    assert abs(sched.update_lr(20) - 0.001) < 1e-6
    # iter 40 → flat
    assert abs(sched.update_lr(40) - 0.001) < 1e-6
    # iter 100 → end of cosine → min_lr (0.001 * 0.05)
    final = sched.update_lr(100)
    assert abs(final - 0.001 * 0.05) < 1e-5


def test_optimizer_backbone_lr_mult():
    """4 param groups with backbone groups carrying lr_mult=backbone_lr_mult."""
    from libreyolo.models.dfine.trainer import DFINETrainer

    wrapper = LibreDFINE(None, size="n", device="cpu")
    trainer = DFINETrainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="n",
        num_classes=80,
        data=None,
        epochs=1,
        batch=2,
        imgsz=640,
        device="cpu",
        amp=False,
        ema=False,
        eval_interval=-1,
        backbone_lr_mult=0.5,
    )
    optimizer = trainer._setup_optimizer()
    by_mult = {}
    for pg in optimizer.param_groups:
        by_mult.setdefault(pg["lr_mult"], []).append(pg)
    # Expect exactly two distinct lr_mult values: 1.0 (head) and 0.5 (backbone).
    assert sorted(by_mult.keys()) == [0.5, 1.0]
    # And both head + backbone should have non-empty groups.
    assert sum(len(pg["params"]) for pg in by_mult[1.0]) > 0
    assert sum(len(pg["params"]) for pg in by_mult[0.5]) > 0


def test_train_transform_strong_augs_toggle():
    """Strong augs run by default; disable_strong_augs() takes them off."""
    import numpy as np

    from libreyolo.models.dfine.transforms import DFINETrainTransform

    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    targets = np.array([[100, 100, 200, 200, 0]], dtype=np.float32)

    t = DFINETrainTransform(strong_augs=True, imgsz=640)
    img_out, padded = t(img, targets, (640, 640))
    assert img_out.shape == (3, 640, 640)
    assert img_out.dtype == np.float32
    # At least one valid box should remain after augs.
    assert int((padded[:, 3] > 0).sum()) >= 1

    t.disable_strong_augs()
    img_out, padded = t(img, targets, (640, 640))
    assert img_out.shape == (3, 640, 640)
    assert int((padded[:, 3] > 0).sum()) == 1  # weak ops alone preserve the single box


def test_multi_scale_collate_resize_and_stop_epoch():
    """Random resize before stop_epoch, fixed base_size after."""
    import numpy as np

    from libreyolo.models.dfine.transforms import (
        DFINEMultiScaleCollate,
        _generate_scales,
    )

    scales = _generate_scales(640, 3)
    assert 480 in scales and 800 in scales and scales.count(640) == 3

    collate = DFINEMultiScaleCollate(base_size=640, base_size_repeat=3, stop_epoch=10)
    batch = [
        (
            np.random.rand(3, 640, 640).astype(np.float32),
            np.zeros((120, 5), dtype=np.float32),
            {},
            i,
        )
        for i in range(2)
    ]
    # Pre-stop_epoch: shape may be any of the 13 scales.
    collate.set_epoch(0)
    seen_sizes = set()
    for _ in range(50):
        imgs, *_ = collate(batch)
        seen_sizes.add(imgs.shape[-1])
    assert seen_sizes.issubset(set(scales))

    # Post-stop_epoch: shape stays 640.
    collate.set_epoch(20)
    imgs, *_ = collate(batch)
    assert imgs.shape[-2:] == (640, 640)


def test_set_decay_changes_ema_behavior():
    """ModelEMA.set_decay swaps the schedule lambda."""
    from libreyolo.training.ema import ModelEMA

    m = LibreDFINE(None, size="n", device="cpu").model
    ema = ModelEMA(m, decay=0.9999)
    pre = ema.decay(1000)
    ema.set_decay(0.99, ramp=False)
    post = ema.decay(1000)
    assert abs(post - 0.99) < 1e-9
    assert post != pre
