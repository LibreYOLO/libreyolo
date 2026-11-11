"""RTMDet raw-head export contract: graph cut, output windows and host decode parity."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from libreyolo.export.exporter import _RTMDetRawHeadsWrapper
from libreyolo.postprocess.yolonas_raw import (
    decode_raw_heads,
    raw_head_output_names,
)

pytestmark = pytest.mark.unit

_LAYOUT = "rtmdet_raw_heads_v1"


def _build(size: str = "t", nb_classes: int = 5, seed: int = 0, exp_on_reg: bool | None = None):
    from libreyolo import LibreRTMDet

    torch.manual_seed(seed)
    model = LibreRTMDet(None, size=size, nb_classes=nb_classes, device="cpu")
    net = model.model.eval()
    if exp_on_reg is not None:
        net.head.exp_on_reg = exp_on_reg
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        # Non-trivial BatchNorm statistics; distances and logits around zero
        # (small enough for exp_on_reg) so boxes and scores are both exercised.
        for module in net.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                n = module.num_features
                module.running_mean.copy_(torch.randn(n, generator=generator) * 0.1)
                module.running_var.copy_(torch.rand(n, generator=generator) + 0.5)
        head = net.head
        for conv in (*head.rtm_reg, *head.rtm_cls):
            conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator) * 0.02)
            conv.bias.copy_(torch.randn(conv.bias.shape, generator=generator))
    return model, net


def _in_graph_decode(net, sample):
    net.head.export = True
    try:
        with torch.no_grad():
            return net(sample).numpy()  # (B, N, 4 + nc)
    finally:
        net.head.export = False


def test_raw_wrapper_output_contract():
    _, net = _build(nb_classes=5)
    wrapper = _RTMDetRawHeadsWrapper(net, output_windows={}).eval()
    with torch.no_grad():
        outputs = wrapper(torch.rand(1, 3, 64, 96))

    assert wrapper.output_layout == _LAYOUT
    assert wrapper.strides == (8, 16, 32)
    assert wrapper.reg_max is None and wrapper.grid_cell_offset == 0.0
    assert wrapper.output_names == raw_head_output_names(_LAYOUT)
    assert wrapper.output_names[:2] == ["p3_reg", "p3_cls"]
    for name, output in zip(wrapper.output_names, outputs):
        level, part = name.split("_")
        stride = 2 ** int(level[1:])
        channels = {"reg": 4, "cls": 5}[part]
        assert tuple(output.shape) == (1, channels, 64 // stride, 96 // stride)


def test_raw_wrapper_rejects_other_tasks():
    _, net = _build()
    with pytest.raises(ValueError, match="detect only"):
        _RTMDetRawHeadsWrapper(net, "segment")


@pytest.mark.parametrize(("size", "hw"), [("t", (160, 160)), ("t", (192, 320)), ("s", (128, 128))])
@pytest.mark.parametrize("exp_on_reg", [False, True])
def test_host_decode_matches_in_graph_decode(size, hw, exp_on_reg):
    _, net = _build(size, exp_on_reg=exp_on_reg)
    wrapper = _RTMDetRawHeadsWrapper(net, output_windows={}).eval()
    assert wrapper.exp_on_reg is exp_on_reg
    sample = torch.rand(2, 3, *hw, generator=torch.Generator().manual_seed(7)) * 255
    expected = _in_graph_decode(net, sample)
    with torch.no_grad():
        raw = [t.numpy() for t in wrapper(sample)]

    (decoded,) = decode_raw_heads(
        _LAYOUT,
        raw,
        strides=wrapper.strides,
        grid_cell_offset=wrapper.grid_cell_offset,
        exp_on_reg=wrapper.exp_on_reg,
        output_windows=wrapper.output_windows,
    )
    assert decoded.shape == expected.shape and decoded.dtype == np.float32
    np.testing.assert_allclose(decoded[..., :4], expected[..., :4], rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(decoded[..., 4:], expected[..., 4:], atol=1e-5)


def test_windowed_cls_matches_unwindowed_inside_the_window():
    _, net = _build()
    plain = _RTMDetRawHeadsWrapper(net, output_windows={}).eval()
    windowed = _RTMDetRawHeadsWrapper(net, output_windows={"cls": (-8.0, 8.0)}).eval()
    sample = torch.rand(1, 3, 128, 128, generator=torch.Generator().manual_seed(3)) * 255
    with torch.no_grad():
        raw_plain = [t.numpy() for t in plain(sample)]
        raw_win = [t.numpy() for t in windowed(sample)]
    kwargs = dict(strides=plain.strides, grid_cell_offset=0.0, exp_on_reg=plain.exp_on_reg)
    (a,) = decode_raw_heads(_LAYOUT, raw_plain, output_windows={}, **kwargs)
    (b,) = decode_raw_heads(_LAYOUT, raw_win, output_windows={"cls": (-8.0, 8.0)}, **kwargs)
    # The random head keeps every class logit inside +-8, so the window is a no-op.
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-4)


def test_wrapper_on_deep_copy_leaves_the_model_unchanged():
    model, net = _build()
    before = copy.deepcopy(net.state_dict())
    wrapper = _RTMDetRawHeadsWrapper(copy.deepcopy(net).eval(), output_windows={"cls": (-8.0, 8.0)})
    with torch.no_grad():
        wrapper(torch.rand(1, 3, 64, 64))
    after = net.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert net.head.export is False


def test_decode_rejects_wrong_output_count_and_strides():
    _, net = _build()
    wrapper = _RTMDetRawHeadsWrapper(net, output_windows={}).eval()
    with torch.no_grad():
        raw = [t.numpy() for t in wrapper(torch.rand(1, 3, 64, 64))]
    with pytest.raises(ValueError, match="expects 6 outputs"):
        decode_raw_heads(_LAYOUT, raw[:-1])
    with pytest.raises(ValueError, match="do not match strides"):
        decode_raw_heads(_LAYOUT, raw, strides=(8, 16, 16))


@pytest.mark.parametrize(("mode", "method"), [("conv", "act"), ("mul", "act"), ("conv", "cle")])
def test_pointwise_equalization_keeps_fp32_outputs(mode, method):
    from libreyolo.export.exporter import _equalize_cspnext_pointwise_inputs
    from libreyolo.models.rtmdet.nn import DepthwiseSeparableConv

    _, net = _build()
    before = copy.deepcopy(net.state_dict())
    eq = copy.deepcopy(net).eval()
    # Make the per-channel ranges uneven, as in the trained weights.
    with torch.no_grad():
        for block in eq.modules():
            if isinstance(block, DepthwiseSeparableConv):
                bn = block.depthwise_conv.bn
                bn.weight.mul_(torch.linspace(0.2, 5.0, bn.weight.numel()))
    reference = copy.deepcopy(eq)
    samples = [torch.rand(1, 3, 64, 64) * 255 for _ in range(3)]
    blocks = _equalize_cspnext_pointwise_inputs(eq, samples, mode, method)
    assert blocks == sum(isinstance(m, DepthwiseSeparableConv) for m in reference.modules()) > 0
    plain = _RTMDetRawHeadsWrapper(reference, output_windows={}).eval()
    equalized = _RTMDetRawHeadsWrapper(eq, output_windows={}).eval()
    x = torch.rand(1, 3, 96, 128, generator=torch.Generator().manual_seed(5)) * 255
    with torch.no_grad():
        for a, b in zip(plain(x), equalized(x)):
            torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-4)
    convs = lambda m: sum(isinstance(c, torch.nn.Conv2d) for c in m.modules())  # noqa: E731
    assert convs(eq) == convs(reference) + (blocks if mode == "conv" else 0)
    # The caller's model is never touched (equalization runs on its own copy).
    after = net.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_pointwise_equalization_skips_without_samples():
    from libreyolo.export.exporter import _equalize_cspnext_pointwise_inputs

    _, net = _build()
    eq = copy.deepcopy(net)
    assert _equalize_cspnext_pointwise_inputs(eq, [], "conv") == 0
    assert _equalize_cspnext_pointwise_inputs(eq, [torch.rand(1, 3, 64, 64)], None) == 0
