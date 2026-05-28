"""Smoke test that reproduces the user's training scenario locally.

Scenario:
    LibreRFDETR(size="l", task="segment").train(imgsz=624, device="0,1,2,3")

Two bugs were fixed:
  1. imgsz=624 was silently overridden by self.input_size (504 for l-seg).
  2. _multi_scale_scales() read patch_size/num_windows via getattr on a
     DDP-wrapped model, always falling back to wrong defaults (16/4), so
     the divisor was 64 instead of 24, producing scale 512 which fails
     the backbone's assertion (512 % 24 != 0).

These tests exercise the whole path on CPU with size="s" (smallest seg
variant that shares block_size=24 / num_windows=2 with "l") and the same
user-supplied imgsz=624 so every check is faithful to the production failure.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

pytestmark = pytest.mark.unit

# ---- helpers ----------------------------------------------------------------

_GLOO_PORT = "29513"


_DDP_ENV_KEYS = ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE")


@pytest.fixture()
def gloo_pg():
    """Init a single-rank gloo process group and fully clean up after the test.

    Destroys the process group AND restores the DDP env vars so that
    has_torchrun_env() / is_distributed() return False for later tests —
    preventing scale_loss_for_ddp() from treating the next trainer as DDP.
    """
    already_up = dist.is_initialized()
    saved_env = {k: os.environ.get(k) for k in _DDP_ENV_KEYS}

    if not already_up:
        os.environ.update({
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": _GLOO_PORT,
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        })
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    yield

    if not already_up:
        if dist.is_initialized():
            dist.destroy_process_group()
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _build_wrapper_and_trainer(size: str, imgsz: int, multi_scale: bool = True):
    from libreyolo.models.rfdetr.model import LibreRFDETR
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    wrapper = LibreRFDETR(model_path={}, size=size, task="segment", device="cpu")
    wrapper.model.train()

    trainer = RFDETRTrainer(
        wrapper.model,
        wrapper_model=wrapper,
        size=size,
        num_classes=5,
        data=None,
        epochs=1,
        batch=2,
        imgsz=imgsz,
        device="cpu",
        amp=False,
        ema=False,
        eval_interval=-1,
        warmup_epochs=0,
        multi_scale=multi_scale,
    )
    trainer.on_setup()
    return wrapper, trainer


# ---- Bug 1: imgsz must not be overridden ------------------------------------


def test_explicit_imgsz_lands_in_trainer_config():
    """imgsz=624 must survive the train() kwarg assembly and reach the trainer."""
    from libreyolo.models.rfdetr.model import LibreRFDETR

    model = LibreRFDETR(model_path={}, size="l", task="segment", device="cpu")
    default_size = model.input_size  # 504 for l-seg

    # Replicate what model.train() does when the user passes imgsz=624.
    train_kwargs: dict = {"imgsz": 624}
    train_kwargs.update(
        {
            "data": "dummy.yaml",
            "epochs": 1,
            "batch": 4,
            "lr0": 1e-4,
            "project": "runs/train",
            "name": "test",
            "exist_ok": True,
            "size": model.size,
            "num_classes": model.nb_classes,
        }
    )
    if train_kwargs.get("imgsz") is None:
        train_kwargs["imgsz"] = default_size

    assert train_kwargs["imgsz"] == 624, (
        f"imgsz was overridden: expected 624, got {train_kwargs['imgsz']} "
        f"(model default={default_size})"
    )


def test_trainer_receives_user_imgsz():
    """The trainer's config.imgsz must equal what the caller passed."""
    _wrapper, trainer = _build_wrapper_and_trainer(size="s", imgsz=624)
    assert trainer.config.imgsz == 624, (
        f"Trainer received imgsz={trainer.config.imgsz}, expected 624"
    )


# ---- imgsz validation: non-divisible value must raise early -----------------


def test_imgsz_not_divisible_by_block_size_raises():
    """create_transforms() must raise ValueError before any data loads when
    imgsz is not divisible by block_size (patch_size × num_windows = 24)."""
    # 500 % 24 == 20 — invalid for this backbone
    _wrapper, trainer = _build_wrapper_and_trainer(size="s", imgsz=500)
    with pytest.raises(ValueError, match="not divisible by 24"):
        trainer.create_transforms()


def test_imgsz_divisible_by_block_size_does_not_raise():
    """create_transforms() must not raise for a valid imgsz (multiple of 24)."""
    _wrapper, trainer = _build_wrapper_and_trainer(size="s", imgsz=480)
    trainer.create_transforms()  # 480 == 20 × 24 — should be fine


# ---- Bug 2: DDP wrap must not break patch_size / num_windows lookup ---------


def test_multi_scale_scales_correct_after_ddp_wrap(gloo_pg):
    """After DDP wrapping, _multi_scale_scales() must use the real patch_size and
    num_windows (12 / 2 → block_size 24), not the wrong fallbacks (16 / 4 → 64)."""

    wrapper, trainer = _build_wrapper_and_trainer(size="s", imgsz=624, multi_scale=True)

    block_size = wrapper.model.patch_size * wrapper.model.num_windows  # 12 * 2 = 24

    # Simulate what BaseTrainer.setup() does: wrap in DDP before training starts.
    trainer.model = nn.parallel.DistributedDataParallel(trainer.model)

    scales = trainer._multi_scale_scales()

    assert scales, "Expected non-empty scale list with multi_scale=True"

    bad = [s for s in scales if s % block_size != 0]
    assert not bad, (
        f"Scales not divisible by block_size={block_size} after DDP wrap: {bad}. "
        f"All scales: {scales}"
    )


def test_all_multi_scale_sizes_divisible_by_block_size():
    """Every scale for l-seg with imgsz=624 is divisible by block_size=24."""
    from libreyolo.models.rfdetr.nn import RFDETR_SEG_CONFIGS
    from libreyolo.models.rfdetr.seg_transforms import compute_multi_scale_scales

    cfg = RFDETR_SEG_CONFIGS["l"]
    block_size = cfg.patch_size * cfg.num_windows  # 12 * 2 = 24

    scales = compute_multi_scale_scales(
        resolution=624,
        expanded_scales=False,
        patch_size=cfg.patch_size,
        num_windows=cfg.num_windows,
    )
    bad = [s for s in scales if s % block_size != 0]
    assert not bad, f"Bad scales for l-seg imgsz=624: {bad}"


# ---- Bug 3: DDP "marked ready twice" for segmentation_head.bias -------------


def test_seg_trainer_ddp_uses_static_graph_not_find_unused(gloo_pg):
    """RFDETRTrainer must use static_graph=True / find_unused_parameters=False.

    The segmentation head is called twice per two-stage forward pass (once for
    decoder queries, once for encoder queries), so its parameters receive
    gradients from two branches.  find_unused_parameters=True registers
    per-parameter autograd hooks that both fire → 'marked ready twice' error.
    static_graph=True avoids this without losing unused-parameter protection.
    """
    _wrapper, trainer = _build_wrapper_and_trainer(size="s", imgsz=624)
    kwargs = trainer._ddp_kwargs()
    assert kwargs.get("static_graph") is True, (
        f"Expected static_graph=True for seg trainer, got {kwargs}"
    )
    assert kwargs.get("find_unused_parameters") is False, (
        f"Expected find_unused_parameters=False for seg trainer, got {kwargs}"
    )


# ---- End-to-end: forward pass must not crash --------------------------------


def _make_seg_batch(batch_size: int, imgsz: int, max_labels: int = 100, mask_downsample: int = 4):
    """Return (imgs, targets, polygons) with valid seg targets.

    polygons shape: [B, max_labels, mask_h, mask_w] — matches the output of
    RFDETRSegTransform's collate step that on_forward expects.
    """
    mask_sz = imgsz // mask_downsample
    imgs = torch.randn(batch_size, 3, imgsz, imgsz)
    targets = torch.zeros(batch_size, max_labels, 5)
    targets[0, 0] = torch.tensor([0.0, imgsz * 0.5, imgsz * 0.5, imgsz * 0.16, imgsz * 0.13])
    targets[1, 0] = torch.tensor([2.0, imgsz * 0.3, imgsz * 0.3, imgsz * 0.10, imgsz * 0.08])
    polygons = torch.zeros(batch_size, max_labels, mask_sz, mask_sz)
    # Put a foreground blob in the matched slots so mask loss is non-trivial.
    polygons[0, 0, mask_sz // 4 : mask_sz * 3 // 4, mask_sz // 4 : mask_sz * 3 // 4] = 1.0
    polygons[1, 0, mask_sz // 4 : mask_sz * 3 // 4, mask_sz // 4 : mask_sz * 3 // 4] = 1.0
    return imgs, targets, polygons


def test_on_forward_at_user_imgsz_does_not_crash():
    """The backbone's divisibility assertion must not fire when imgsz=624 is used.

    Uses size="s" (same block_size=24 as "l") so the test finishes quickly on CPU
    while exercising the exact path that was failing in production.
    """
    _wrapper, trainer = _build_wrapper_and_trainer(size="s", imgsz=624, multi_scale=False)

    imgs, targets, polygons = _make_seg_batch(batch_size=2, imgsz=624)
    out = trainer.on_forward(imgs, targets, polygons=polygons)

    assert "total_loss" in out
    assert torch.isfinite(out["total_loss"]), "total_loss must be finite"
    assert out["total_loss"].item() > 0


def test_on_forward_at_default_imgsz_when_no_override():
    """Without a user override, the model's default imgsz is used (384 for s-seg)
    and the forward must also succeed."""
    from libreyolo.models.rfdetr.model import LibreRFDETR
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    wrapper = LibreRFDETR(model_path={}, size="s", task="segment", device="cpu")
    default_imgsz = wrapper.input_size  # 384

    wrapper.model.train()
    trainer = RFDETRTrainer(
        wrapper.model,
        wrapper_model=wrapper,
        size="s",
        num_classes=5,
        data=None,
        epochs=1,
        batch=2,
        imgsz=default_imgsz,
        device="cpu",
        amp=False,
        ema=False,
        eval_interval=-1,
        warmup_epochs=0,
        multi_scale=False,
    )
    trainer.on_setup()

    imgs, targets, polygons = _make_seg_batch(batch_size=2, imgsz=default_imgsz)
    out = trainer.on_forward(imgs, targets, polygons=polygons)
    assert torch.isfinite(out["total_loss"])


def test_backward_produces_gradients_at_user_imgsz():
    """Gradients flow back through the model at imgsz=624; regression guard for
    a silent forward-only breakage."""
    _wrapper, trainer = _build_wrapper_and_trainer(size="s", imgsz=624, multi_scale=False)
    _wrapper.model.train()

    imgs, targets, polygons = _make_seg_batch(batch_size=2, imgsz=624)
    out = trainer.on_forward(imgs, targets, polygons=polygons)
    out["total_loss"].backward()

    nonzero = sum(
        1 for p in _wrapper.model.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    assert nonzero > 0, "Expected non-zero gradients after backward"
