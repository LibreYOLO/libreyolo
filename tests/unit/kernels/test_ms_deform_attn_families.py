"""Per-family adapters onto the ``ms_deform_attn`` op slot.

Two things are checked for every family that was rewired:

1. With the slot active, the family core hands it the classic Deformable-DETR
   layout (a mock implementation records what it received).
2. With the slot inactive - the CPU / no-kernel-installed case, which is the
   default everywhere these tests run - the family core takes its own
   untouched grid_sample path and produces the same attention output as the
   reference classic core.

Together those pin the accelerated path's input contract and prove the
portable path is still the one that runs by default.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo import kernels
from libreyolo.kernels.attention.ms_deform_attn import (
    maybe_ms_deform_attn_v2,
    ms_deform_attn_available,
    spatial_shapes_tensor,
)
from libreyolo.models.deformable_detr.ms_deform_attn import (
    ms_deform_attn_core_pytorch as classic_core,
)
from libreyolo.models.deim.ms_deform import (
    deformable_attention_core_func_v2 as deim_core,
)
from libreyolo.models.dfine.ms_deform import (
    deformable_attention_core_func_v2 as dfine_core,
)
from libreyolo.models.ec.decoder import (
    _ms_deform_attn_core_pytorch_pose as ec_pose_core,
)
from libreyolo.models.ec.utils import deformable_attention_core_func_v2 as ec_core
from libreyolo.models.grounding_dino.nn import _msda as gdino_core
from libreyolo.models.lwdetr.nn import ms_deform_attn_core_pytorch as lwdetr_core
from libreyolo.models.openvocab.ovdeim.decoder import (
    deformable_attention_core_func_v2 as ovdeim_core,
)
from libreyolo.models.rtdetr.utils import (
    deformable_attention_core_func as rtdetr_core,
)
from libreyolo.models.rtdetrv2.utils import (
    deformable_attention_core_func_v2 as rtdetrv2_core,
)

pytestmark = pytest.mark.unit

BATCH, LEN_Q, HEADS, CHANNELS = 2, 3, 2, 4
SHAPES = [(4, 6), (2, 3)]
LEVELS, POINTS = len(SHAPES), 2
LEN_IN = sum(h * w for h, w in SHAPES)
NUM_POINTS_LIST = [POINTS] * LEVELS

CLASSIC_CALL = (
    (BATCH, LEN_IN, HEADS, CHANNELS),
    (LEVELS, 2),
    (BATCH, LEN_Q, HEADS, LEVELS, POINTS, 2),
    (BATCH, LEN_Q, HEADS, LEVELS, POINTS),
)
MOCK_OUTPUT = torch.full((BATCH, LEN_Q, HEADS * CHANNELS), 7.0)


@pytest.fixture(autouse=True)
def _clean_registry_env(monkeypatch):
    monkeypatch.delenv("LIBREYOLO_KERNELS", raising=False)
    monkeypatch.delenv("LIBREYOLO_QUANT_KERNELS", raising=False)
    # Accelerated providers are on by default when their extras/runtime
    # exist; pin them off so these tests behave the same on any machine.
    monkeypatch.setenv("LIBREYOLO_HUB_KERNELS", "0")
    monkeypatch.setenv("LIBREYOLO_TRITON_MSDA", "0")
    kernels.clear_cache()
    yield
    kernels.unregister("ms_deform_attn", "mock")
    kernels.clear_cache()


@pytest.fixture
def recorded(monkeypatch):
    """Register a mock slot implementation and collect the shapes it sees."""
    calls: list[tuple] = []

    def mock_impl(value, spatial_shapes, sampling_locations, attention_weights):
        calls.append(
            (
                tuple(value.shape),
                tuple(spatial_shapes.shape),
                tuple(sampling_locations.shape),
                tuple(attention_weights.shape),
            )
        )
        return MOCK_OUTPUT.clone()

    kernels.register("ms_deform_attn", mock_impl, name="mock")
    monkeypatch.setenv("LIBREYOLO_KERNELS", "mock")
    kernels.clear_cache()
    return calls


def _classic_inputs():
    generator = torch.Generator().manual_seed(0)
    value = torch.randn(BATCH, LEN_IN, HEADS, CHANNELS, generator=generator)
    sampling_locations = torch.rand(
        BATCH, LEN_Q, HEADS, LEVELS, POINTS, 2, generator=generator
    )
    weights = torch.rand(BATCH, LEN_Q, HEADS, LEVELS, POINTS, generator=generator)
    weights = weights / weights.sum(dim=(-2, -1), keepdim=True)
    return value, sampling_locations, weights


def _split_value(value):
    """Classic layout -> the per-level ``(bs, n_head, c, H*W)`` list D-FINE uses."""
    permuted = value.permute(0, 2, 3, 1)
    return permuted.split([h * w for h, w in SHAPES], dim=-1)


def _reference_msda(value, spatial_shapes, sampling_locations, attention_weights):
    """Standalone MSDA oracle in the slot's signature.

    Written out here rather than delegating to a family core, both so it is
    independent of the code under test and because a family core would route
    straight back into the slot.
    """
    batch, _, heads, channels = value.shape
    _, queries, _, levels, points, _ = sampling_locations.shape
    shapes = [(int(h), int(w)) for h, w in spatial_shapes.tolist()]
    values = value.split([h * w for h, w in shapes], dim=1)
    grids = 2 * sampling_locations - 1
    sampled = [
        torch.nn.functional.grid_sample(
            values[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch * heads, channels, height, width),
            grids[:, :, :, level].transpose(1, 2).flatten(0, 1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        for level, (height, width) in enumerate(shapes)
    ]
    weights = attention_weights.transpose(1, 2).reshape(
        batch * heads, 1, queries, levels * points
    )
    out = (torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1)
    return out.view(batch, heads * channels, queries).transpose(1, 2).contiguous()


# --- one callable per family, in that family's native argument layout -------


def _call_lwdetr(value, locations, weights):
    # (bs, Len_in, heads, c) -> lwdetr's (bs, heads, c, Len_in); flat weights.
    return lwdetr_core(
        value.permute(0, 2, 3, 1).contiguous(),
        SHAPES,
        locations,
        weights.flatten(-2),
    )


def _call_gdino(value, locations, weights):
    return gdino_core(value, SHAPES, locations, weights)


def _call_rtdetr(value, locations, weights):
    return rtdetr_core(value, SHAPES, locations, weights)


def _call_rtdetrv2(value, locations, weights):
    return rtdetrv2_core(
        value, SHAPES, locations.flatten(3, 4), weights.flatten(3, 4), NUM_POINTS_LIST
    )


def _call_dfine(value, locations, weights):
    return dfine_core(
        _split_value(value),
        SHAPES,
        locations.flatten(3, 4),
        weights.flatten(3, 4),
        NUM_POINTS_LIST,
    )


def _call_deim(value, locations, weights):
    return deim_core(
        _split_value(value),
        SHAPES,
        locations.flatten(3, 4),
        weights.flatten(3, 4),
        NUM_POINTS_LIST,
    )


def _call_ec(value, locations, weights):
    return ec_core(
        _split_value(value),
        SHAPES,
        locations.flatten(3, 4),
        weights.flatten(3, 4),
        NUM_POINTS_LIST,
    )


def _call_ec_reshape(value, locations, weights):
    return ec_core(
        value,
        SHAPES,
        locations.flatten(3, 4),
        weights.flatten(3, 4),
        NUM_POINTS_LIST,
        value_shape="reshape",
    )


def _call_ovdeim(value, locations, weights):
    return ovdeim_core(
        value, SHAPES, locations.flatten(3, 4), weights.flatten(3, 4), NUM_POINTS_LIST
    )


def _pose_split_value(value):
    """Classic layout -> pose ``(bs * heads, c, hw)`` per level."""
    return tuple(
        value.permute(0, 2, 3, 1)
        .flatten(0, 1)
        .split([height * width for height, width in SHAPES], dim=-1)
    )


def _call_ec_pose(value, locations, weights):
    return ec_pose_core(_pose_split_value(value), SHAPES, locations, weights)


FAMILY_CORES = {
    "lwdetr": _call_lwdetr,
    "grounding_dino": _call_gdino,
    "rtdetr": _call_rtdetr,
    "rtdetrv2": _call_rtdetrv2,
    "dfine": _call_dfine,
    "deim": _call_deim,
    "ec": _call_ec,
    "ec_reshape": _call_ec_reshape,
    "ec_pose": _call_ec_pose,
    "ovdeim": _call_ovdeim,
}


@pytest.mark.parametrize("family", sorted(FAMILY_CORES))
def test_family_routes_through_slot_in_classic_layout(family, recorded):
    """Whatever the family's native layout, the slot must see the classic one."""
    value, locations, weights = _classic_inputs()
    out = FAMILY_CORES[family](value, locations, weights)
    assert torch.equal(out, MOCK_OUTPUT)
    assert recorded == [CLASSIC_CALL]


@pytest.mark.parametrize("family", sorted(FAMILY_CORES))
def test_family_falls_back_to_portable_path(family, recorded):
    """The wire-up must be inert when the slot has no eligible implementation."""
    kernels.unregister("ms_deform_attn", "mock")
    kernels.clear_cache()
    value, locations, weights = _classic_inputs()
    out = FAMILY_CORES[family](value, locations, weights)
    assert recorded == []
    torch.testing.assert_close(
        out, classic_core(value, SHAPES, locations, weights), rtol=0, atol=0
    )


@pytest.mark.parametrize("family", sorted(FAMILY_CORES))
def test_family_portable_path_matches_classic_reference(family):
    """The layout adaptation expresses the same attention problem as classic."""
    value, locations, weights = _classic_inputs()
    torch.testing.assert_close(
        FAMILY_CORES[family](value, locations, weights),
        classic_core(value, SHAPES, locations, weights),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("family", sorted(FAMILY_CORES))
def test_family_accelerated_result_matches_portable(family, monkeypatch):
    """Numerical closure: a correct slot implementation gives the right answer.

    The shape test above pins what the slot receives; this one pins that those
    arguments, run through a reference multi-scale deformable attention,
    reproduce what the family computes itself. Registering the classic core as
    the implementation stands in for the compiled Hub kernel, which is
    Linux-only and cannot be exercised here.
    """
    kernels.register("ms_deform_attn", _reference_msda, name="mock")
    monkeypatch.setenv("LIBREYOLO_KERNELS", "mock")
    kernels.clear_cache()

    value, locations, weights = _classic_inputs()
    accelerated = FAMILY_CORES[family](value, locations, weights)

    kernels.unregister("ms_deform_attn", "mock")
    kernels.clear_cache()
    monkeypatch.delenv("LIBREYOLO_KERNELS")
    portable = FAMILY_CORES[family](value, locations, weights)

    torch.testing.assert_close(accelerated, portable, rtol=1e-5, atol=1e-6)


# --- guards that must keep the slot out of the way -------------------------

V2_CORES = {
    "rtdetrv2": rtdetrv2_core,
    "ovdeim": ovdeim_core,
}


@pytest.mark.parametrize("family", sorted(V2_CORES))
def test_v2_discrete_method_never_reaches_slot(family, recorded):
    """``method='discrete'`` is a different equation: it must stay portable."""
    value, locations, weights = _classic_inputs()
    V2_CORES[family](
        value,
        SHAPES,
        locations.flatten(3, 4),
        weights.flatten(3, 4),
        NUM_POINTS_LIST,
        method="discrete",
    )
    assert recorded == []


def test_dfine_discrete_method_never_reaches_slot(recorded):
    value, locations, weights = _classic_inputs()
    dfine_core(
        _split_value(value),
        SHAPES,
        locations.flatten(3, 4),
        weights.flatten(3, 4),
        NUM_POINTS_LIST,
        method="discrete",
    )
    assert recorded == []


def test_dfine_forced_manual_grid_sample_never_reaches_slot(recorded, monkeypatch):
    """The exporters' manual-grid_sample escape hatch outranks the slot."""
    from libreyolo.models.dfine import ms_deform

    value, locations, weights = _classic_inputs()
    args = (
        _split_value(value),
        SHAPES,
        locations.flatten(3, 4),
        weights.flatten(3, 4),
        NUM_POINTS_LIST,
    )

    monkeypatch.setattr(ms_deform, "_FORCE_MANUAL_GRID_SAMPLE_EXPORT", True)
    dfine_core(*args)
    assert recorded == []
    monkeypatch.setattr(ms_deform, "_FORCE_MANUAL_GRID_SAMPLE_EXPORT", False)

    token = ms_deform._FORCE_MANUAL_GRID_SAMPLE.set(True)
    try:
        dfine_core(*args)
    finally:
        ms_deform._FORCE_MANUAL_GRID_SAMPLE.reset(token)
    assert recorded == []


def test_ec_pose_malformed_value_does_not_adapt(recorded):
    """A value tuple that cannot reshape onto the slot must not be offered."""
    from libreyolo.models.ec.decoder import _pose_value_to_slot

    value, locations, _weights = _classic_inputs()
    assert _pose_value_to_slot(_pose_split_value(value)[:1], locations) is None
    assert recorded == []


def test_ragged_num_points_never_reaches_slot(recorded):
    """A per-level point count does not reshape onto the slot's (L, P) layout."""
    generator = torch.Generator().manual_seed(0)
    ragged = [1, 3]
    total = sum(ragged)
    value = torch.randn(BATCH, LEN_IN, HEADS, CHANNELS, generator=generator)
    locations = torch.rand(BATCH, LEN_Q, HEADS, total, 2, generator=generator)
    weights = torch.rand(BATCH, LEN_Q, HEADS, total, generator=generator)
    out = rtdetrv2_core(value, SHAPES, locations, weights, ragged)
    assert recorded == []
    assert out.shape == (BATCH, LEN_Q, HEADS * CHANNELS)


def test_v2_adapter_rejects_ragged_and_mismatched_levels(recorded):
    value, locations, weights = _classic_inputs()
    shapes = spatial_shapes_tensor(SHAPES, value.device)
    flat_locations, flat_weights = locations.flatten(3, 4), weights.flatten(3, 4)
    assert (
        maybe_ms_deform_attn_v2(value, shapes, flat_locations, flat_weights, [1, 3])
        is None
    )
    assert (
        maybe_ms_deform_attn_v2(value, shapes, flat_locations, flat_weights, []) is None
    )
    # More point groups than levels: the split cannot be a per-level one.
    assert (
        maybe_ms_deform_attn_v2(
            value, shapes, flat_locations, flat_weights, [1, 1, 1, 1]
        )
        is None
    )
    assert recorded == []


def test_export_never_reaches_slot(recorded, monkeypatch):
    """ONNX export must not capture a runtime-fetched kernel."""
    monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
    assert not ms_deform_attn_available()
    value, locations, weights = _classic_inputs()
    for call in FAMILY_CORES.values():
        call(value, locations, weights)
    assert recorded == []


# --- spatial_shapes_tensor -------------------------------------------------


def test_spatial_shapes_tensor_from_pairs():
    shapes = spatial_shapes_tensor(SHAPES, torch.device("cpu"))
    assert shapes.dtype == torch.int64
    assert torch.equal(shapes, torch.tensor(SHAPES, dtype=torch.int64))
    # Repeated lookups reuse the cached tensor rather than re-copying.
    assert spatial_shapes_tensor(tuple(SHAPES), torch.device("cpu")) is shapes


def test_spatial_shapes_tensor_passes_through_matching_tensor():
    existing = torch.tensor(SHAPES, dtype=torch.int64)
    assert spatial_shapes_tensor(existing, existing.device) is existing
    converted = spatial_shapes_tensor(
        torch.tensor(SHAPES, dtype=torch.int32), torch.device("cpu")
    )
    assert converted.dtype == torch.int64
    assert torch.equal(converted, existing)
