"""Unit tests for RF-DETR DDP patch_size/num_windows unwrapping in _multi_scale_scales().

Bug: before this fix, _multi_scale_scales() called getattr(self.model, "patch_size", 16)
after DDP wrapping.  DDP does not proxy custom attributes, so both reads returned the
fallback defaults (16 and 4) instead of the real values (12 and 2 for l/s/m/x-seg).
The generated scales used divisor 16×4=64 instead of 12×2=24, introducing scales
that the backbone then rejected at runtime.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

pytestmark = pytest.mark.unit

rfdetr_trainer = pytest.importorskip("libreyolo.models.rfdetr.trainer")

from libreyolo.models.rfdetr.seg_transforms import compute_multi_scale_scales  # noqa: E402
from libreyolo.training.distributed import unwrap_model  # noqa: E402


# ---------------------------------------------------------------------------
# Non-DDP: scale generation correctness
# ---------------------------------------------------------------------------

def test_multi_scale_scales_all_divisible_by_block_size():
    """compute_multi_scale_scales with correct attrs produces only block-24-aligned scales."""
    block_size = 12 * 2  # = 24 (real values for l/s/m/x-seg)
    scales = compute_multi_scale_scales(504, False, patch_size=12, num_windows=2)
    assert scales, "expected at least one scale"
    for s in scales:
        assert s % block_size == 0, f"scale {s} not divisible by {block_size}"


def test_old_wrong_divisor_produces_bad_scales():
    """Regression doc: old DDP fallbacks (16×4=64) generate scales not divisible by 24."""
    block_size = 24  # backbone requirement for l/s/m/x-seg
    scales = compute_multi_scale_scales(504, False, patch_size=16, num_windows=4)
    bad = [s for s in scales if s % block_size != 0]
    assert bad, (
        "Expected at least one scale not divisible by 24 with old wrong fallbacks; "
        f"all scales were clean: {scales}"
    )


# ---------------------------------------------------------------------------
# Wrapper/attribute proxying (uses DataParallel — same .module structure as DDP,
# no process group required)
# ---------------------------------------------------------------------------

def test_patch_size_survives_dp_wrap():
    """unwrap_model(DataParallel(module)) returns the original module with its custom attributes.

    DataParallel and DistributedDataParallel share the same .module structure and
    are both handled identically by unwrap_model. This test uses DataParallel so
    the suite can run without a distributed process group.
    """
    class _Model(nn.Module):
        patch_size = 12
        num_windows = 2

        def forward(self, x):
            return x

    m = _Model()
    dp = nn.parallel.DataParallel(m)
    assert not hasattr(dp, "patch_size"), "DataParallel should not proxy patch_size (pre-condition)"
    raw = unwrap_model(dp)
    assert raw.patch_size == 12
    assert raw.num_windows == 2


def test_multi_scale_scales_correct_after_dp_wrap():
    """_multi_scale_scales() returns block-24-aligned scales when model is DataParallel-wrapped.

    DataParallel and DistributedDataParallel share the same .module structure; the
    unwrap_model call in _multi_scale_scales() handles both identically.
    """
    class _FakeModel(nn.Module):
        patch_size = 12
        num_windows = 2

        def forward(self, x):
            return x

    trainer = rfdetr_trainer.RFDETRTrainer.__new__(rfdetr_trainer.RFDETRTrainer)
    trainer.config = rfdetr_trainer.RFDETRConfig(
        data=None, imgsz=504, multi_scale=True,
        do_random_resize_via_padding=False,
    )
    trainer.model = nn.parallel.DataParallel(_FakeModel())
    trainer.wrapper_model = None

    block_size = 12 * 2  # = 24
    scales = trainer._multi_scale_scales()
    assert scales, "expected at least one multi-scale size"
    for s in scales:
        assert s % block_size == 0, f"scale {s} not divisible by {block_size}"
