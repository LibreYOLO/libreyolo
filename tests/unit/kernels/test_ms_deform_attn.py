"""CPU tests for the ``ms_deform_attn`` op slot and its call-site adapters."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from libreyolo import kernels
from libreyolo.kernels.attention.ms_deform_attn import (
    hub_ms_deform_attn,
    level_start_index,
    maybe_ms_deform_attn,
)
from libreyolo.models.deformable_detr.ms_deform_attn import (
    ms_deform_attn_core_pytorch as classic_core,
)
from libreyolo.models.rfdetr.transformer import (
    MSDeformAttn as RFDETRMSDeformAttn,
)
from libreyolo.models.rfdetr.transformer import (
    ms_deform_attn_core_pytorch as rfdetr_core,
)

pytestmark = pytest.mark.unit

BATCH, LEN_Q, HEADS, CHANNELS = 2, 3, 2, 4
SHAPES = [(4, 6), (2, 3)]
LEVELS, POINTS = len(SHAPES), 2
LEN_IN = sum(h * w for h, w in SHAPES)


class _CudaValue:
    is_cuda = True


@pytest.fixture(autouse=True)
def _clean_registry_env(monkeypatch):
    from libreyolo.kernels.attention import ms_deform_attn as module

    monkeypatch.delenv("LIBREYOLO_KERNELS", raising=False)
    monkeypatch.delenv("LIBREYOLO_QUANT_KERNELS", raising=False)
    # Accelerated providers are on by default when their extras/runtime
    # exist; pin them off so these tests behave the same on any machine.
    monkeypatch.setenv("LIBREYOLO_HUB_KERNELS", "0")
    monkeypatch.setenv("LIBREYOLO_TRITON_MSDA", "0")
    monkeypatch.setattr(module, "_missing_hub_hint_emitted", False)
    kernels.clear_cache()
    yield
    kernels.unregister("ms_deform_attn", "mock")
    kernels.clear_cache()


def _classic_inputs():
    generator = torch.Generator().manual_seed(0)
    value = torch.randn(BATCH, LEN_IN, HEADS, CHANNELS, generator=generator)
    spatial_shapes = torch.tensor(SHAPES, dtype=torch.int64)
    sampling_locations = torch.rand(
        BATCH, LEN_Q, HEADS, LEVELS, POINTS, 2, generator=generator
    )
    attention_weights = torch.rand(
        BATCH, LEN_Q, HEADS, LEVELS, POINTS, generator=generator
    )
    attention_weights = attention_weights / attention_weights.sum(
        dim=(-2, -1), keepdim=True
    )
    return value, spatial_shapes, sampling_locations, attention_weights


def test_slot_resolves_to_none_when_disabled():
    assert kernels.resolve("ms_deform_attn") is None
    value, shapes, locations, weights = _classic_inputs()
    assert maybe_ms_deform_attn(value, shapes, locations, weights) is None


def test_hub_default_on_and_env_opt_out(monkeypatch):
    from libreyolo.kernels.attention import ms_deform_attn as module

    monkeypatch.delenv("LIBREYOLO_HUB_KERNELS", raising=False)
    assert module._hub_enabled()
    for value in ("0", "false", "off", "no"):
        monkeypatch.setenv("LIBREYOLO_HUB_KERNELS", value)
        assert not module._hub_enabled()


def test_missing_provider_cuda_hint_warns_once(monkeypatch, caplog):
    from libreyolo.kernels.attention import ms_deform_attn as module

    monkeypatch.delenv("LIBREYOLO_HUB_KERNELS", raising=False)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _name: None)
    kernels.clear_cache()

    with caplog.at_level("WARNING", logger=module.__name__):
        assert not module.ms_deform_attn_available(_CudaValue())
        assert not module.ms_deform_attn_available(_CudaValue())

    hints = [
        record.getMessage()
        for record in caplog.records
        if "libreyolo[hub-kernels]" in record.getMessage()
    ]
    assert len(hints) == 1
    assert "LIBREYOLO_HUB_KERNELS=0" in hints[0]


def test_missing_provider_cuda_hint_after_provider_rejects(monkeypatch, caplog):
    from libreyolo.kernels.attention import ms_deform_attn as module

    monkeypatch.delenv("LIBREYOLO_HUB_KERNELS", raising=False)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _name: None)
    kernels.register("ms_deform_attn", lambda *_args: None, name="mock")
    kernels.clear_cache()

    with caplog.at_level("WARNING", logger=module.__name__):
        assert maybe_ms_deform_attn(_CudaValue(), None, None, None) is None
        assert maybe_ms_deform_attn(_CudaValue(), None, None, None) is None

    hints = [
        record.getMessage()
        for record in caplog.records
        if "libreyolo[hub-kernels]" in record.getMessage()
    ]
    assert len(hints) == 1


def test_missing_provider_hint_respects_opt_out(monkeypatch, caplog):
    from libreyolo.kernels.attention import ms_deform_attn as module

    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _name: None)
    kernels.clear_cache()

    with caplog.at_level("WARNING", logger=module.__name__):
        assert not module.ms_deform_attn_available(_CudaValue())

    assert not any(
        "libreyolo[hub-kernels]" in record.getMessage()
        for record in caplog.records
    )


def test_missing_provider_hint_when_hub_is_forced(monkeypatch, caplog):
    from libreyolo.kernels.attention import ms_deform_attn as module

    monkeypatch.delenv("LIBREYOLO_HUB_KERNELS", raising=False)
    monkeypatch.setenv("LIBREYOLO_KERNELS", "hub")
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _name: None)
    kernels.clear_cache()

    with caplog.at_level("WARNING", logger=module.__name__):
        assert not module.ms_deform_attn_available(_CudaValue())

    assert any(
        "libreyolo[hub-kernels]" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize("forced", ["off", "reference"])
def test_missing_provider_hint_respects_global_portable_override(
    monkeypatch, caplog, forced
):
    from libreyolo.kernels.attention import ms_deform_attn as module

    monkeypatch.delenv("LIBREYOLO_HUB_KERNELS", raising=False)
    monkeypatch.setenv("LIBREYOLO_KERNELS", forced)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _name: None)
    kernels.clear_cache()

    with caplog.at_level("WARNING", logger=module.__name__):
        assert not module.ms_deform_attn_available(_CudaValue())

    assert not any(
        "libreyolo[hub-kernels]" in record.getMessage()
        for record in caplog.records
    )


def test_missing_provider_hint_ignores_installed_hub_client(monkeypatch, caplog):
    from libreyolo.kernels.attention import ms_deform_attn as module

    monkeypatch.delenv("LIBREYOLO_HUB_KERNELS", raising=False)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(module, "_hub_failed", True)
    kernels.clear_cache()

    with caplog.at_level("WARNING", logger=module.__name__):
        assert not module.ms_deform_attn_available(_CudaValue())

    assert not any(
        "libreyolo[hub-kernels]" in record.getMessage()
        for record in caplog.records
    )


def test_missing_provider_hint_does_not_propagate_importer_errors(
    monkeypatch, caplog
):
    from libreyolo.kernels.attention import ms_deform_attn as module

    def broken_finder(_name):
        raise RuntimeError("broken meta path finder")

    monkeypatch.delenv("LIBREYOLO_HUB_KERNELS", raising=False)
    monkeypatch.setattr(module.importlib.util, "find_spec", broken_finder)
    kernels.clear_cache()

    with caplog.at_level("WARNING"):
        assert not module.ms_deform_attn_available(_CudaValue())

    assert not any(
        "libreyolo[hub-kernels]" in record.getMessage()
        for record in caplog.records
    )


def test_missing_provider_hint_ignores_cpu_calls(monkeypatch, caplog):
    from libreyolo.kernels.attention import ms_deform_attn as module

    monkeypatch.delenv("LIBREYOLO_HUB_KERNELS", raising=False)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _name: None)
    kernels.clear_cache()

    value, *_ = _classic_inputs()
    with caplog.at_level("WARNING", logger=module.__name__):
        assert not module.ms_deform_attn_available(value)

    assert not any(
        "libreyolo[hub-kernels]" in record.getMessage()
        for record in caplog.records
    )


def test_hub_impl_rejects_cpu_inputs():
    value, shapes, locations, weights = _classic_inputs()
    assert hub_ms_deform_attn(value, shapes, locations, weights) is None


def test_level_start_index():
    shapes = torch.tensor(SHAPES, dtype=torch.int64)
    expected = torch.tensor([0, SHAPES[0][0] * SHAPES[0][1]], dtype=torch.int64)
    assert torch.equal(level_start_index(shapes), expected)


def _register_mock(monkeypatch, recorded):
    def mock_impl(value, spatial_shapes, sampling_locations, attention_weights):
        recorded.append(
            (
                tuple(value.shape),
                tuple(spatial_shapes.shape),
                tuple(sampling_locations.shape),
                tuple(attention_weights.shape),
            )
        )
        heads_times_c = value.shape[2] * value.shape[3]
        return torch.full(
            (value.shape[0], sampling_locations.shape[1], heads_times_c), 7.0
        )

    kernels.register("ms_deform_attn", mock_impl, name="mock")
    monkeypatch.setenv("LIBREYOLO_KERNELS", "mock")
    kernels.clear_cache()


def test_classic_call_site_routes_through_slot(monkeypatch):
    recorded = []
    _register_mock(monkeypatch, recorded)
    value, shapes, locations, weights = _classic_inputs()
    out = classic_core(value, shapes, locations, weights)
    assert torch.equal(
        out, torch.full((BATCH, LEN_Q, HEADS * CHANNELS), 7.0)
    )
    assert recorded == [
        (
            (BATCH, LEN_IN, HEADS, CHANNELS),
            (LEVELS, 2),
            (BATCH, LEN_Q, HEADS, LEVELS, POINTS, 2),
            (BATCH, LEN_Q, HEADS, LEVELS, POINTS),
        )
    ]


def test_rfdetr_layout_is_numerically_equivalent():
    """The rfdetr core's layout must express the same attention problem."""
    value, shapes, locations, weights = _classic_inputs()
    classic_out = classic_core(value, shapes, locations, weights)
    rfdetr_out = rfdetr_core(
        value.permute(0, 2, 3, 1).contiguous(),
        shapes,
        locations,
        weights.flatten(-2),
    )
    torch.testing.assert_close(rfdetr_out, classic_out, rtol=1e-5, atol=1e-5)


D_MODEL = HEADS * CHANNELS


def _rfdetr_attention_module():
    torch.manual_seed(0)
    return RFDETRMSDeformAttn(
        d_model=D_MODEL, n_levels=LEVELS, n_heads=HEADS, n_points=POINTS
    )


def _rfdetr_attention_inputs():
    generator = torch.Generator().manual_seed(1)
    query = torch.randn(BATCH, LEN_Q, D_MODEL, generator=generator)
    reference_points = torch.rand(BATCH, LEN_Q, LEVELS, 2, generator=generator)
    input_flatten = torch.randn(BATCH, LEN_IN, D_MODEL, generator=generator)
    spatial_shapes = torch.tensor(SHAPES, dtype=torch.int64)
    return query, reference_points, input_flatten, spatial_shapes


def test_rfdetr_attention_forward_routes_through_slot(monkeypatch):
    """The real RF-DETR attention module must consult the slot in eager mode.

    This is the regression test for the slot being wired at a call site the
    model actually reaches: RF-DETR always threads ``input_spatial_shapes_hw``
    through its decoder, so gating the slot on that argument being None
    (an earlier revision of this PR) made the kernel unreachable.
    """
    recorded = []
    _register_mock(monkeypatch, recorded)
    module = _rfdetr_attention_module()
    query, reference_points, input_flatten, spatial_shapes = (
        _rfdetr_attention_inputs()
    )
    out = module(
        query,
        reference_points,
        input_flatten,
        spatial_shapes,
        level_start_index(spatial_shapes),
        input_spatial_shapes_hw=SHAPES,
    )
    # The slot receives the classic Deformable-DETR layout...
    assert recorded == [
        (
            (BATCH, LEN_IN, HEADS, CHANNELS),
            (LEVELS, 2),
            (BATCH, LEN_Q, HEADS, LEVELS, POINTS, 2),
            (BATCH, LEN_Q, HEADS, LEVELS, POINTS),
        )
    ]
    # ...and its output feeds output_proj: mock returns all-7s, so the
    # module output equals output_proj of an all-7s tensor.
    expected = module.output_proj(
        torch.full((BATCH, LEN_Q, D_MODEL), 7.0)
    )
    torch.testing.assert_close(out, expected)


def test_rfdetr_attention_export_mode_skips_slot(monkeypatch):
    recorded = []
    _register_mock(monkeypatch, recorded)
    module = _rfdetr_attention_module()
    module.export()
    query, reference_points, input_flatten, spatial_shapes = (
        _rfdetr_attention_inputs()
    )
    module(
        query,
        reference_points,
        input_flatten,
        spatial_shapes,
        level_start_index(spatial_shapes),
        input_spatial_shapes_hw=SHAPES,
    )
    assert recorded == []


def test_rfdetr_attention_slot_matches_portable(monkeypatch):
    """A slot impl wrapping the portable core must reproduce module output."""

    def portable_impl(value, spatial_shapes, sampling_locations, attention_weights):
        # Wrap the rfdetr portable core (slot-free) rather than classic_core,
        # which consults the slot itself and would recurse into this impl.
        return rfdetr_core(
            value.permute(0, 2, 3, 1).contiguous(),
            spatial_shapes,
            sampling_locations,
            attention_weights.flatten(-2),
        )

    module = _rfdetr_attention_module()
    query, reference_points, input_flatten, spatial_shapes = (
        _rfdetr_attention_inputs()
    )
    args = (
        query,
        reference_points,
        input_flatten,
        spatial_shapes,
        level_start_index(spatial_shapes),
    )
    baseline = module(*args, input_spatial_shapes_hw=SHAPES)

    kernels.register("ms_deform_attn", portable_impl, name="mock")
    monkeypatch.setenv("LIBREYOLO_KERNELS", "mock")
    kernels.clear_cache()
    routed = module(*args, input_spatial_shapes_hw=SHAPES)
    torch.testing.assert_close(routed, baseline, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("kernels") is None,
    reason="needs CUDA and the `kernels` package (libreyolo[hub-kernels])",
)
def test_hub_matches_portable_on_cuda():
    """Forward/backward parity of the pinned Hub kernel vs the portable core.

    This is the GPU smoke for the provider: run it on any CUDA box with the
    ``hub-kernels`` extra installed before bumping ``_HUB_REVISION``.
    """
    value, shapes, locations, weights = _classic_inputs()
    value = value.cuda().requires_grad_(True)
    shapes = shapes.cuda()
    locations = locations.cuda().requires_grad_(True)
    weights = weights.cuda().requires_grad_(True)

    hub_out = hub_ms_deform_attn(value, shapes, locations, weights)
    if hub_out is None:
        pytest.skip("hub kernel unavailable on this box (load failed)")
    hub_out.sum().backward()
    hub_grads = (value.grad.clone(), locations.grad.clone(), weights.grad.clone())

    value.grad = locations.grad = weights.grad = None
    # classic_core consults the slot itself; the autouse fixture pins
    # LIBREYOLO_HUB_KERNELS=0 for this file, so it runs the portable path.
    ref_out = classic_core(value, shapes, locations, weights)
    ref_out.sum().backward()
    ref_grads = (value.grad, locations.grad, weights.grad)

    torch.testing.assert_close(hub_out, ref_out, rtol=1e-4, atol=1e-5)
    for hub_grad, ref_grad in zip(hub_grads, ref_grads):
        torch.testing.assert_close(hub_grad, ref_grad, rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("half_dtype", [torch.float16, torch.bfloat16])
def test_hub_accepts_autocast_half_inputs_on_cuda(half_dtype):
    """Half inputs must take the fused path via the fp32 upcast boundary.

    Autocast hands the slot a fp16/bf16 ``value`` with fp32 sampling
    locations and softmax weights (softmax stays fp32 under autocast); the
    provider upcasts everything, runs the fp32 kernel, and returns the
    value's dtype. The output must be bit-identical to feeding the same
    quantized inputs through the fp32 path and casting the result.
    """
    value, shapes, locations, weights = _classic_inputs()
    value_half = value.cuda().to(half_dtype).requires_grad_(True)
    shapes = shapes.cuda()
    locations = locations.cuda().requires_grad_(True)
    weights = weights.cuda().requires_grad_(True)

    out = hub_ms_deform_attn(value_half, shapes, locations, weights)
    if out is None:
        pytest.skip("hub kernel unavailable on this box (load failed)")
    assert out.dtype == half_dtype

    # Reference: the same quantized inputs through the pure-fp32 path, on
    # fresh leaves so both paths' gradients can be compared.
    ref_value = value_half.detach().float().requires_grad_(True)
    ref_locations = locations.detach().clone().requires_grad_(True)
    ref_weights = weights.detach().clone().requires_grad_(True)
    ref = hub_ms_deform_attn(ref_value, shapes, ref_locations, ref_weights)
    torch.testing.assert_close(out, ref.to(half_dtype), rtol=0, atol=0)

    # Backward parity through the cast boundary: the fp32 kernel sees
    # identical inputs and an identical (exactly representable) grad seed,
    # so the fp32 grads must match bitwise; the value grad additionally
    # crosses the .float() cast node, which casts it back to the input
    # dtype.
    out.sum().backward()
    ref.sum().backward()
    assert value_half.grad is not None
    assert value_half.grad.dtype == half_dtype
    torch.testing.assert_close(
        value_half.grad, ref_value.grad.to(half_dtype), rtol=0, atol=0
    )
    torch.testing.assert_close(locations.grad, ref_locations.grad, rtol=0, atol=0)
    torch.testing.assert_close(weights.grad, ref_weights.grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_supported_inputs_dtype_gate_on_cuda():
    from libreyolo.kernels.attention.ms_deform_attn import _supported_inputs

    value, shapes, locations, weights = _classic_inputs()
    value = value.cuda()
    shapes = shapes.cuda()
    locations = locations.cuda()
    weights = weights.cuda()
    assert _supported_inputs(value, shapes, locations, weights)
    # The dtype mix autocast produces: half value, fp32 locations/weights.
    assert _supported_inputs(value.half(), shapes, locations, weights)
    assert _supported_inputs(value.bfloat16(), shapes, locations, weights)
    assert not _supported_inputs(value.double(), shapes, locations, weights)


# =============================================================================
# Pinned-snapshot fallback loader (the ``kernels``-resolver compatibility path)
# =============================================================================


def test_pinned_variant_name_maps_platform(monkeypatch):
    from libreyolo.kernels.attention import ms_deform_attn as module

    monkeypatch.setattr(torch, "__version__", "2.11.0+cu128")
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(module._platform, "system", lambda: "Linux")
    monkeypatch.setattr(module._platform, "machine", lambda: "x86_64")
    assert module._pinned_variant_name() == "torch211-cxx11-cu128-x86_64-linux"

    monkeypatch.setattr(module._platform, "system", lambda: "Windows")
    monkeypatch.setattr(module._platform, "machine", lambda: "AMD64")
    assert module._pinned_variant_name() == "torch211-cu128-x86_64-windows"

    monkeypatch.setattr(module._platform, "machine", lambda: "arm64")
    monkeypatch.setattr(module._platform, "system", lambda: "Linux")
    assert module._pinned_variant_name() == "torch211-cxx11-cu128-aarch64-linux"

    # CPU-only torch has no CUDA build to match.
    monkeypatch.setattr(torch.version, "cuda", None)
    assert module._pinned_variant_name() is None


def test_load_hub_kernel_falls_back_when_resolver_rejects_pin(monkeypatch):
    """A ``kernels`` release that cannot resolve the SHA pin must not kill the
    provider: the direct snapshot loader is tried before giving up."""
    import sys
    import types

    from libreyolo.kernels.attention import ms_deform_attn as module

    monkeypatch.setattr(module, "_hub_kernel", None)
    monkeypatch.setattr(module, "_hub_failed", False)

    fake_kernels = types.ModuleType("kernels")

    def rejects_sha(*args, **kwargs):
        raise ValueError("Invalid rev id")

    fake_kernels.get_kernel = rejects_sha
    monkeypatch.setitem(sys.modules, "kernels", fake_kernels)

    sentinel = object()
    monkeypatch.setattr(module, "_load_pinned_snapshot", lambda: sentinel)
    assert module._load_hub_kernel() is sentinel
    assert module._hub_failed is False


def test_load_hub_kernel_disables_when_both_paths_fail(monkeypatch):
    import sys
    import types

    from libreyolo.kernels.attention import ms_deform_attn as module

    monkeypatch.setattr(module, "_hub_kernel", None)
    monkeypatch.setattr(module, "_hub_failed", False)

    fake_kernels = types.ModuleType("kernels")

    def rejects_sha(*args, **kwargs):
        raise ValueError("Invalid rev id")

    fake_kernels.get_kernel = rejects_sha
    monkeypatch.setitem(sys.modules, "kernels", fake_kernels)

    def no_snapshot():
        raise OSError("offline")

    monkeypatch.setattr(module, "_load_pinned_snapshot", no_snapshot)
    assert module._load_hub_kernel() is None
    assert module._hub_failed is True


# =============================================================================
# In-tree Triton provider
# =============================================================================


def test_triton_selected_when_hub_disabled(monkeypatch):
    """No hub extra: resolve should land on Triton when CUDA+Triton exist."""
    if not torch.cuda.is_available() or importlib.util.find_spec("triton") is None:
        pytest.skip("needs CUDA and Triton")
    monkeypatch.delenv("LIBREYOLO_TRITON_MSDA", raising=False)
    monkeypatch.setenv("LIBREYOLO_HUB_KERNELS", "0")
    kernels.clear_cache()
    # Force the lazy providers to load against the new env.
    importlib.import_module("libreyolo.kernels.attention.ms_deform_attn_triton")
    assert kernels.active().get("ms_deform_attn") == "triton"


def test_triton_default_on_and_env_opt_out(monkeypatch):
    from libreyolo.kernels.attention import ms_deform_attn_triton as module

    monkeypatch.delenv("LIBREYOLO_TRITON_MSDA", raising=False)
    assert module._env_enabled()
    for value in ("0", "false", "off", "no"):
        monkeypatch.setenv("LIBREYOLO_TRITON_MSDA", value)
        assert not module._env_enabled()


def test_triton_impl_rejects_cpu_inputs():
    from libreyolo.kernels.attention.ms_deform_attn_triton import triton_ms_deform_attn

    value, shapes, locations, weights = _classic_inputs()
    assert triton_ms_deform_attn(value, shapes, locations, weights) is None


def test_triton_impl_rejects_mismatched_metadata():
    from libreyolo.kernels.attention.ms_deform_attn_triton import triton_ms_deform_attn

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    value, shapes, locations, weights = _classic_inputs()
    value = value.cuda()
    shapes = shapes.cuda()
    locations = locations.cuda()
    weights = weights.cuda()
    # Len_in does not match the spatial areas: must not launch.
    bad_value = value[:, :-1]
    assert triton_ms_deform_attn(bad_value, shapes, locations, weights) is None
    # Level count disagrees across tensors.
    assert triton_ms_deform_attn(value, shapes[:1], locations, weights) is None


def test_triton_impl_rejects_grad_inputs():
    from libreyolo.kernels.attention.ms_deform_attn_triton import triton_ms_deform_attn

    value, shapes, locations, weights = _classic_inputs()
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    value = value.cuda().requires_grad_(True)
    shapes = shapes.cuda()
    locations = locations.cuda()
    weights = weights.cuda()
    assert triton_ms_deform_attn(value, shapes, locations, weights) is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="needs CUDA and Triton",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_triton_matches_portable_on_cuda(dtype):
    """Forward parity of the in-tree Triton kernel vs the portable core."""
    from libreyolo.kernels.attention.ms_deform_attn_triton import triton_ms_deform_attn

    value, shapes, locations, weights = _classic_inputs()
    value = value.cuda().to(dtype)
    shapes = shapes.cuda()
    locations = locations.cuda().to(
        dtype if dtype == torch.float32 else torch.float32
    )
    weights = weights.cuda().to(dtype if dtype == torch.float32 else torch.float32)
    if dtype != torch.float32:
        value = value.to(dtype)

    out = triton_ms_deform_attn(value, shapes, locations, weights)
    assert out is not None
    assert out.dtype == value.dtype

    ref = rfdetr_core(
        value.float().permute(0, 2, 3, 1).contiguous(),
        shapes,
        locations.float(),
        weights.float().flatten(-2),
    )
    rtol, atol = (1e-4, 1e-5) if dtype == torch.float32 else (2e-3, 2e-3)
    torch.testing.assert_close(out.float(), ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="needs CUDA and Triton",
)
def test_triton_rfdetr_shapes_on_cuda():
    """RF-DETR detect is 1 level, 2 points, 16-wide heads, 300 queries."""
    from libreyolo.kernels.attention.ms_deform_attn_triton import triton_ms_deform_attn

    generator = torch.Generator(device="cuda").manual_seed(2)
    batch, queries, heads, channels, levels, points = 1, 300, 16, 16, 1, 2
    height = width = 24
    value = torch.randn(
        batch, height * width, heads, channels, generator=generator, device="cuda"
    )
    shapes = torch.tensor([(height, width)], dtype=torch.int64, device="cuda")
    locations = torch.rand(
        batch,
        queries,
        heads,
        levels,
        points,
        2,
        generator=generator,
        device="cuda",
    )
    weights = torch.rand(
        batch, queries, heads, levels, points, generator=generator, device="cuda"
    )
    weights = weights / weights.sum(dim=(-2, -1), keepdim=True)

    out = triton_ms_deform_attn(value, shapes, locations, weights)
    assert out is not None
    ref = rfdetr_core(
        value.permute(0, 2, 3, 1).contiguous(),
        shapes,
        locations,
        weights.flatten(-2),
    )
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_maybe_walks_next_provider_when_first_returns_none(monkeypatch):
    """Hub (or any newer impl) returning None must not hide the next provider."""
    recorded = []

    def first_impl(value, spatial_shapes, sampling_locations, attention_weights):
        recorded.append("first")
        return None

    def second_impl(value, spatial_shapes, sampling_locations, attention_weights):
        recorded.append("second")
        return torch.full(
            (value.shape[0], sampling_locations.shape[1], value.shape[2] * value.shape[3]),
            7.0,
        )

    kernels.register("ms_deform_attn", second_impl, name="second")
    kernels.register("ms_deform_attn", first_impl, name="first")
    kernels.clear_cache()
    try:
        value, shapes, locations, weights = _classic_inputs()
        out = maybe_ms_deform_attn(value, shapes, locations, weights)
        assert recorded == ["first", "second"]
        assert torch.equal(out, torch.full((BATCH, LEN_Q, HEADS * CHANNELS), 7.0))
    finally:
        kernels.unregister("ms_deform_attn", "first")
        kernels.unregister("ms_deform_attn", "second")
        kernels.clear_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_triton_supported_inputs_skips_item_during_capture(monkeypatch):
    from libreyolo.kernels.attention.ms_deform_attn_triton import _supported_inputs

    value, shapes, locations, weights = _classic_inputs()
    value = value.cuda()
    shapes = shapes.cuda()
    locations = locations.cuda()
    weights = weights.cuda()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert _supported_inputs(value, shapes, locations, weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_triton_rejects_mixed_devices():
    from libreyolo.kernels.attention.ms_deform_attn_triton import (
        _supported_inputs,
        triton_ms_deform_attn,
    )

    value, shapes, locations, weights = _classic_inputs()
    value = value.cuda()
    locations = locations.cuda()
    weights = weights.cuda()
    # shapes left on CPU: must not launch.
    assert not _supported_inputs(value, shapes, locations, weights)
    assert triton_ms_deform_attn(value, shapes, locations, weights) is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="needs CUDA and Triton",
)
def test_triton_disables_after_launch_failure(monkeypatch):
    from libreyolo.kernels.attention import ms_deform_attn_triton as module

    value, shapes, locations, weights = _classic_inputs()
    value = value.cuda()
    shapes = shapes.cuda()
    locations = locations.cuda()
    weights = weights.cuda()

    monkeypatch.setattr(module, "_triton_failed", False)
    calls = []

    class _Boom:
        def __getitem__(self, _grid):
            calls.append(1)
            raise RuntimeError("compile failed")

    monkeypatch.setattr(module, "_kernel", lambda: _Boom())
    assert module.triton_ms_deform_attn(value, shapes, locations, weights) is None
    assert module._triton_failed
    assert module.triton_ms_deform_attn(value, shapes, locations, weights) is None
    assert calls == [1]
