"""YOLOv7 raw-head export contract: implicit heads folded and split, host decode parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.export.exporter import _yolo7_raw_heads_wrapper
from libreyolo.postprocess.yolonas_raw import decode_raw_heads, raw_head_output_names

pytestmark = pytest.mark.unit

_LAYOUT = "yolo7_raw_heads_v1"


def _build(nb_classes: int = 5, seed: int = 0):
    from libreyolo import LibreYOLO7

    torch.manual_seed(seed)
    model = LibreYOLO7(None, size="b", nb_classes=nb_classes, device="cpu")
    net = model.model.eval()
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        # Non-trivial implicit knowledge, so the fold into the split convs is exercised.
        for idet in net.layers[net._head_index].heads:
            idet.implicit_a.implicit.copy_(torch.randn(idet.implicit_a.implicit.shape, generator=generator) * 0.1)
            idet.implicit_m.implicit.copy_(1 + torch.randn(idet.implicit_m.implicit.shape, generator=generator) * 0.1)
            idet.head_conv.weight.copy_(torch.randn(idet.head_conv.weight.shape, generator=generator) * 0.02)
            idet.head_conv.bias.copy_(torch.randn(idet.head_conv.bias.shape, generator=generator))
    return model, net


def _in_graph_decode(net, sample):
    from libreyolo.models.yolo7.export import YOLO7ExportWrapper

    with torch.no_grad():
        return YOLO7ExportWrapper(net).eval()(sample).numpy()  # (B, 4 + nc, N)


def test_raw_wrapper_output_contract():
    _, net = _build(nb_classes=5)
    wrapper = _yolo7_raw_heads_wrapper(net).eval()
    with torch.no_grad():
        outputs = wrapper(torch.rand(1, 3, 64, 96))
    assert wrapper.output_layout == _LAYOUT and wrapper.strides == (8, 16, 32)
    assert wrapper.output_names == raw_head_output_names(_LAYOUT)
    assert wrapper.scale_x_y is None and [len(a) for a in wrapper.anchors] == [3, 3, 3]
    for name, output in zip(wrapper.output_names, outputs):
        level, part = name.split("_")
        stride = 2 ** int(level[1:])
        channels = {"reg": 12, "obj": 3, "cls": 15}[part]
        assert tuple(output.shape) == (1, channels, 64 // stride, 96 // stride)


def test_repconv_fusion_is_exact():
    from libreyolo.export.exporter import _fuse_yolo7_repconvs

    _, net = _build()
    fused = __import__("copy").deepcopy(net)
    with torch.no_grad():  # non-trivial BatchNorm statistics in every branch
        for module in net.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.uniform_(-0.2, 0.2)
                module.running_var.uniform_(0.5, 1.5)
    fused.load_state_dict(net.state_dict())
    assert _fuse_yolo7_repconvs(fused) > 0
    sample = torch.rand(1, 3, 96, 96, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        for a, b in zip(net(sample), fused(sample)):
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-4, atol=1e-4)


def test_raw_wrapper_rejects_other_tasks():
    _, net = _build()
    with pytest.raises(ValueError, match="detect only"):
        _yolo7_raw_heads_wrapper(net, "segment")


@pytest.mark.parametrize("hw", [(128, 128), (96, 160)])
def test_host_decode_matches_in_graph_decode(hw):
    _, net = _build()
    wrapper = _yolo7_raw_heads_wrapper(net).eval()
    sample = torch.rand(2, 3, *hw, generator=torch.Generator().manual_seed(7))
    expected = _in_graph_decode(net, sample)
    with torch.no_grad():
        raw = [t.numpy() for t in wrapper(sample)]
    (decoded,) = decode_raw_heads(_LAYOUT, raw, strides=wrapper.strides, anchors=wrapper.anchors)
    assert decoded.shape == expected.shape and decoded.dtype == np.float32
    np.testing.assert_allclose(decoded[:, :4], expected[:, :4], rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(decoded[:, 4:], expected[:, 4:], atol=1e-5)
