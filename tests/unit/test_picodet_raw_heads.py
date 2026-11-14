"""PicoDet raw-head export contract: split GFL convs and host decode parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.export.exporter import _PicoDetRawHeadsWrapper
from libreyolo.postprocess.yolonas_raw import decode_raw_heads, raw_head_output_names

pytestmark = pytest.mark.unit

_LAYOUT = "picodet_raw_heads_v1"


def _build(size: str = "s", nb_classes: int = 5, seed: int = 0):
    from libreyolo.models.picodet.nn import LibrePICODETModel

    torch.manual_seed(seed)
    model = LibrePICODETModel(size=size, nb_classes=nb_classes).eval()
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():  # informative head outputs instead of the init biases
        for conv in model.head.gfl_cls:
            conv.weight.copy_(
                torch.randn(conv.weight.shape, generator=generator) * 0.05
            )
            conv.bias.copy_(torch.randn(conv.bias.shape, generator=generator))
    return model


def _in_graph_decode(model, sample):
    model.head.export = True
    try:
        with torch.no_grad():
            return model(sample).numpy()  # (B, N, 4 + nc)
    finally:
        model.head.export = False


def test_raw_wrapper_output_contract():
    model = _build(nb_classes=5)
    wrapper = _PicoDetRawHeadsWrapper(model).eval()
    with torch.no_grad():
        outputs = wrapper(torch.rand(1, 3, 320, 320))

    assert wrapper.output_layout == _LAYOUT
    assert wrapper.strides == (8, 16, 32, 64)
    assert wrapper.reg_max == 7 and wrapper.grid_cell_offset == 0.5
    assert wrapper.output_names == raw_head_output_names(_LAYOUT, (8, 16, 32, 64))
    assert wrapper.output_names[:2] == ["p3_reg", "p3_cls"]
    for name, output in zip(wrapper.output_names, outputs):
        level, part = name.split("_")
        stride = 2 ** int(level[1:])
        channels = {"reg": 32, "cls": 5}[part]
        assert tuple(output.shape) == (1, channels, 320 // stride, 320 // stride)
    assert model.head.gfl_cls[0].out_channels == 37  # the live model is untouched


@pytest.mark.parametrize(("size", "hw"), [("s", 320), ("m", 416)])
def test_host_decode_matches_in_graph_decode(size, hw):
    model = _build(size)
    wrapper = _PicoDetRawHeadsWrapper(model).eval()
    sample = torch.rand(2, 3, hw, hw, generator=torch.Generator().manual_seed(7))
    expected = _in_graph_decode(model, sample)
    with torch.no_grad():
        raw = [t.numpy() for t in wrapper(sample)]

    (decoded,) = decode_raw_heads(
        _LAYOUT,
        raw,
        strides=wrapper.strides,
        reg_max=wrapper.reg_max,
        grid_cell_offset=wrapper.grid_cell_offset,
    )
    assert decoded.shape == expected.shape and decoded.dtype == np.float32
    np.testing.assert_allclose(
        decoded[..., :4], expected[..., :4], rtol=1e-5, atol=1e-3
    )
    np.testing.assert_allclose(decoded[..., 4:], expected[..., 4:], atol=1e-6)


def test_decode_rejects_mismatched_levels():
    model = _build()
    wrapper = _PicoDetRawHeadsWrapper(model).eval()
    with torch.no_grad():
        raw = [t.numpy() for t in wrapper(torch.rand(1, 3, 320, 320))]
    raw[-2], raw[-1] = (
        raw[-2][:, :, :4, :4],
        raw[-1][:, :, :4, :4],
    )  # stride-64 level too small
    with pytest.raises(ValueError, match="do not match strides"):
        decode_raw_heads(_LAYOUT, raw, strides=wrapper.strides, reg_max=7)


def test_windowed_outputs_decode_like_the_plain_ones():
    model = _build()
    plain = _PicoDetRawHeadsWrapper(model).eval()
    windows = {"reg": (-40.0, 40.0), "cls": (-40.0, 40.0)}  # wide: nothing clips
    windowed = _PicoDetRawHeadsWrapper(model, output_windows=windows).eval()
    sample = torch.rand(1, 3, 320, 320, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        raw_plain = [t.numpy() for t in plain(sample)]
        raw_windowed = [t.numpy() for t in windowed(sample)]
    assert max(float(r.max()) for r in raw_windowed) <= 6.0
    params = {"strides": plain.strides, "reg_max": 7}
    (expected,) = decode_raw_heads(_LAYOUT, raw_plain, **params)
    (decoded,) = decode_raw_heads(
        _LAYOUT, raw_windowed, output_windows=windows, **params
    )
    np.testing.assert_allclose(decoded, expected, rtol=1e-4, atol=1e-3)
