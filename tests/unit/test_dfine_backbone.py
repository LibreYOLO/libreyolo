"""HGNetV2 backbone shape tests for the D-FINE port."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit

from libreyolo.models.dfine.backbone import HGNetv2


@pytest.mark.parametrize(
    "name,return_idx,expected_out_channels",
    [
        # D-FINE N uses a 2-level pyramid (strides 16 and 32) off B0.
        ("B0", (2, 3), [512, 1024]),
        # S uses 3-level off B0.
        ("B0", (1, 2, 3), [256, 512, 1024]),
        # M uses 3-level off B2.
        ("B2", (1, 2, 3), [384, 768, 1536]),
        # L uses 3-level off B4.
        ("B4", (1, 2, 3), [512, 1024, 2048]),
        # X uses 3-level off B5.
        ("B5", (1, 2, 3), [512, 1024, 2048]),
    ],
)
def test_forward_shapes(name, return_idx, expected_out_channels):
    net = HGNetv2(name=name, return_idx=return_idx, freeze_at=-1, freeze_norm=False)
    net.eval()
    x = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        outs = net(x)
    assert len(outs) == len(return_idx)
    strides = [4, 8, 16, 32]
    for out, idx, expected_ch in zip(outs, return_idx, expected_out_channels):
        assert out.shape[0] == 1
        assert out.shape[1] == expected_ch, (
            f"{name}[{idx}] channels: expected {expected_ch}, got {out.shape[1]}"
        )
        assert out.shape[2] == 640 // strides[idx]
        assert out.shape[3] == 640 // strides[idx]


def test_use_lab_adds_scale_bias_parameters():
    """use_lab=True should introduce LearnableAffineBlock params after activations."""
    without = HGNetv2(name="B0", use_lab=False, return_idx=(2, 3), freeze_at=-1, freeze_norm=False)
    with_lab = HGNetv2(name="B0", use_lab=True, return_idx=(2, 3), freeze_at=-1, freeze_norm=False)
    params_without = sum(p.numel() for p in without.parameters())
    params_with = sum(p.numel() for p in with_lab.parameters())
    assert params_with > params_without


def test_freeze_norm_replaces_batchnorm():
    """freeze_norm=True should replace BatchNorm2d with FrozenBatchNorm2d."""
    from libreyolo.models.dfine.common import FrozenBatchNorm2d

    net = HGNetv2(name="B4", return_idx=(1, 2, 3), freeze_norm=True, freeze_at=-1)
    has_frozen = any(isinstance(m, FrozenBatchNorm2d) for m in net.modules())
    has_plain_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in net.modules())
    assert has_frozen
    assert not has_plain_bn
