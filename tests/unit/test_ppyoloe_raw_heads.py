"""PP-YOLOE raw-head export contract: fused RepVGG, YOLO-NAS-style host decode parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.export.exporter import _PPYOLOERawHeadsWrapper
from libreyolo.postprocess.yolonas_raw import decode_raw_heads, raw_head_output_names

pytestmark = pytest.mark.unit

_LAYOUT = "ppyoloe_raw_heads_v1"


def _build(nb_classes: int = 5, seed: int = 0):
    from libreyolo import LibrePPYOLOE

    torch.manual_seed(seed)
    model = LibrePPYOLOE(None, size="s", nb_classes=nb_classes, device="cpu")
    net = model.model.eval()
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                n = module.num_features
                module.running_mean.copy_(torch.randn(n, generator=generator) * 0.1)
                module.running_var.copy_(torch.rand(n, generator=generator) + 0.5)
        for conv in (*net.head.pred_cls, *net.head.pred_reg):
            conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator) * 0.02)
            conv.bias.copy_(torch.randn(conv.bias.shape, generator=generator))
    return model, net


def test_raw_wrapper_output_contract():
    _, net = _build(nb_classes=5)
    wrapper = _PPYOLOERawHeadsWrapper(net).eval()
    with torch.no_grad():
        outputs = wrapper(torch.rand(1, 3, 64, 96))
    assert wrapper.output_layout == _LAYOUT and wrapper.strides == (32, 16, 8)
    assert wrapper.reg_max == 16 and wrapper.grid_cell_offset == 0.5
    assert wrapper.output_names == raw_head_output_names(_LAYOUT, (32, 16, 8))
    for name, output in zip(wrapper.output_names, outputs):
        level, part = name.split("_")
        stride = 2 ** int(level[1:])
        channels = {"reg": 4 * 17, "cls": 5}[part]
        assert tuple(output.shape) == (1, channels, 64 // stride, 96 // stride)


def test_raw_wrapper_rejects_other_tasks():
    _, net = _build()
    with pytest.raises(ValueError, match="detect only"):
        _PPYOLOERawHeadsWrapper(net, "segment")


@pytest.mark.parametrize("hw", [(128, 128), (96, 160)])
def test_host_decode_matches_in_graph_decode(hw):
    _, net = _build()
    wrapper = _PPYOLOERawHeadsWrapper(net).eval()  # fuses RepVGG on a copy
    sample = torch.rand(2, 3, *hw, generator=torch.Generator().manual_seed(7))
    net.head.export = True
    try:
        with torch.no_grad():
            boxes, scores = (t.numpy() for t in net(sample))
            raw = [t.numpy() for t in wrapper(sample)]
    finally:
        net.head.export = False
    decoded_boxes, decoded_scores = decode_raw_heads(
        _LAYOUT, raw, strides=wrapper.strides, reg_max=wrapper.reg_max,
        grid_cell_offset=wrapper.grid_cell_offset,
    )
    np.testing.assert_allclose(decoded_boxes, boxes, rtol=1e-5, atol=2e-3)
    np.testing.assert_allclose(decoded_scores, scores, atol=1e-5)
