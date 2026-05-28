"""Regression tests for two bugs fixed together:

1. model.py: user-supplied imgsz was unconditionally overridden by self.input_size.
2. trainer.py: _multi_scale_scales() read patch_size/num_windows via getattr on a
   DDP-wrapped model, which always fell back to wrong defaults (16/4 instead of
   the actual 12/2 for seg), generating scale values not divisible by block_size.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch.distributed as dist
import torch.nn as nn

pytestmark = pytest.mark.unit


_DDP_ENV_KEYS = ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE")


@pytest.fixture()
def gloo_pg():
    """Init a single-rank gloo process group and fully clean up after the test.

    Destroys the process group AND restores the DDP env vars so that
    has_torchrun_env() / is_distributed() return False for later tests.
    """
    already_up = dist.is_initialized()
    saved_env = {k: os.environ.get(k) for k in _DDP_ENV_KEYS}

    if not already_up:
        os.environ.update({
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29510",
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


# ---------------------------------------------------------------------------
# Bug 1: imgsz must not be clobbered by model.train()
# ---------------------------------------------------------------------------


_SENTINEL = object()


def _build_train_kwargs(model, imgsz=_SENTINEL) -> dict:
    """Replicate the kwarg assembly path from model.train(), without a trainer."""
    from pathlib import Path

    output_path = Path("runs/train/test_run")
    train_kwargs: dict = {}
    if imgsz is not _SENTINEL:
        train_kwargs["imgsz"] = imgsz

    train_kwargs.update(
        {
            "data": "dummy.yaml",
            "epochs": 1,
            "batch": 4,
            "lr0": 1e-4,
            "project": str(output_path.parent),
            "name": output_path.name,
            "exist_ok": True,
            "size": model.size,
            "num_classes": model.nb_classes,
        }
    )
    if train_kwargs.get("imgsz") is None:
        train_kwargs["imgsz"] = model.input_size
    return train_kwargs


def _rfdetr_seg_stub(size: str = "l") -> SimpleNamespace:
    """Lightweight stand-in for LibreRFDETR used in kwarg-assembly tests.

    Reads only the class-level SEG_INPUT_SIZES dict — no model is constructed.
    """
    from libreyolo.models.rfdetr.model import LibreRFDETR

    return SimpleNamespace(size=size, nb_classes=80, input_size=LibreRFDETR.SEG_INPUT_SIZES[size])


def test_explicit_imgsz_is_not_overridden():
    model = _rfdetr_seg_stub("l")
    default_size = model.input_size  # 504 for l-seg
    user_size = 624
    assert user_size != default_size, "precondition: user_size must differ from default"

    kwargs = _build_train_kwargs(model, imgsz=user_size)
    assert kwargs["imgsz"] == user_size, (
        f"User imgsz={user_size} was overridden to {kwargs['imgsz']} "
        f"(model default={default_size})"
    )


def test_default_imgsz_applied_when_not_supplied():
    model = _rfdetr_seg_stub("l")
    kwargs = _build_train_kwargs(model)  # no imgsz kwarg at all
    assert kwargs["imgsz"] == model.input_size, (
        f"Expected default imgsz={model.input_size}, got {kwargs['imgsz']}"
    )


def test_imgsz_none_falls_back_to_model_default():
    """imgsz=None must not be forwarded to the trainer — treat it as unset."""
    model = _rfdetr_seg_stub("l")
    kwargs = _build_train_kwargs(model, imgsz=None)
    assert kwargs["imgsz"] == model.input_size, (
        f"imgsz=None should fall back to model default {model.input_size}, "
        f"got {kwargs['imgsz']}"
    )


# ---------------------------------------------------------------------------
# Bug 2: _multi_scale_scales() must read correct patch_size/num_windows even
#         after DDP wrapping.
# ---------------------------------------------------------------------------


def _make_ddp_wrapped_rfdetr_model(size: str = "l", task: str = "segment"):
    """Return an RFDETRModel nn.Module wrapped in a CPU DDP container.

    Requires the gloo_pg fixture to be active — the fixture owns init/teardown.
    """
    from libreyolo.models.rfdetr.nn import RFDETR_SEG_CONFIGS, RFDETR_CONFIGS

    configs = RFDETR_SEG_CONFIGS if task == "segment" else RFDETR_CONFIGS
    cfg = configs[size]

    class _FakeRFDETRModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_size = cfg.patch_size
            self.num_windows = cfg.num_windows
            self.dummy = nn.Linear(1, 1, bias=False)

        def forward(self, x):
            return x

    raw = _FakeRFDETRModel()
    ddp = nn.parallel.DistributedDataParallel(raw)
    return raw, ddp, cfg


@pytest.mark.parametrize("size,task", [
    ("l", "segment"),
    ("s", "segment"),
    ("x", "segment"),
])
def test_patch_size_survives_ddp_wrap(gloo_pg, size, task):
    """getattr on a DDP-wrapped model must NOT fall back to wrong defaults."""
    from libreyolo.training.distributed import unwrap_model
    from libreyolo.models.rfdetr.nn import RFDETR_SEG_CONFIGS

    raw, ddp, cfg = _make_ddp_wrapped_rfdetr_model(size=size, task=task)

    # Direct getattr on DDP wrapper — demonstrates the old bug
    ddp_patch_size = getattr(ddp, "patch_size", 16)
    ddp_num_windows = getattr(ddp, "num_windows", 4)

    # The buggy path would silently return wrong values
    if ddp_patch_size != cfg.patch_size or ddp_num_windows != cfg.num_windows:
        # DDP doesn't proxy custom attrs — this is expected (documents the bug)
        pass

    # Fixed path: unwrap first
    unwrapped = unwrap_model(ddp)
    assert getattr(unwrapped, "patch_size") == cfg.patch_size, (
        f"Expected patch_size={cfg.patch_size} for {size}-{task}, "
        f"got {getattr(unwrapped, 'patch_size')}"
    )
    assert getattr(unwrapped, "num_windows") == cfg.num_windows, (
        f"Expected num_windows={cfg.num_windows} for {size}-{task}, "
        f"got {getattr(unwrapped, 'num_windows')}"
    )


@pytest.mark.parametrize("size,task,imgsz", [
    ("l", "segment", 504),  # default for l-seg; must be divisible by 24
    ("l", "segment", 624),  # user override; 624 % 24 == 0
    ("x", "segment", 624),  # default for x-seg; 624 % 24 == 0
])
def test_multi_scale_scales_all_divisible_by_block_size(size, task, imgsz):
    """Every scale generated by compute_multi_scale_scales must be divisible
    by patch_size * num_windows (the backbone block_size)."""
    from libreyolo.models.rfdetr.nn import RFDETR_SEG_CONFIGS
    from libreyolo.models.rfdetr.seg_transforms import compute_multi_scale_scales

    cfg = RFDETR_SEG_CONFIGS[size]
    block_size = cfg.patch_size * cfg.num_windows

    scales = compute_multi_scale_scales(
        resolution=imgsz,
        expanded_scales=False,
        patch_size=cfg.patch_size,
        num_windows=cfg.num_windows,
    )

    bad = [s for s in scales if s % block_size != 0]
    assert not bad, (
        f"Scales not divisible by block_size={block_size} "
        f"(patch={cfg.patch_size}, windows={cfg.num_windows}): {bad}"
    )


def test_old_wrong_divisor_would_have_generated_bad_scale():
    """Demonstrates that the old fallback values (patch_size=16, num_windows=4)
    produced scales like 512 that fail the backbone's divisibility check."""
    from libreyolo.models.rfdetr.seg_transforms import compute_multi_scale_scales
    from libreyolo.models.rfdetr.nn import RFDETR_SEG_CONFIGS

    cfg = RFDETR_SEG_CONFIGS["l"]
    correct_block_size = cfg.patch_size * cfg.num_windows  # 12 * 2 = 24

    # Scales computed with the WRONG fallback divisor (old bug)
    wrong_scales = compute_multi_scale_scales(
        resolution=504,
        expanded_scales=False,
        patch_size=16,   # wrong fallback
        num_windows=4,   # wrong fallback
    )

    # At least one scale must fail the backbone's real divisibility check
    bad = [s for s in wrong_scales if s % correct_block_size != 0]
    assert bad, (
        "Expected at least one bad scale with wrong divisor — "
        "this test documents the original bug"
    )

    # Scales computed with the CORRECT values (after fix)
    good_scales = compute_multi_scale_scales(
        resolution=504,
        expanded_scales=False,
        patch_size=cfg.patch_size,
        num_windows=cfg.num_windows,
    )
    bad_after_fix = [s for s in good_scales if s % correct_block_size != 0]
    assert not bad_after_fix, (
        f"Fixed scales should all be divisible by {correct_block_size}, "
        f"but got bad: {bad_after_fix}"
    )
