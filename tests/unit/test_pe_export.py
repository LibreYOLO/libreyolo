"""Export-graph parity for LibrePE.

These tests build the export wrappers directly on a randomly initialized tower,
so they need no downloaded weights: what is under test is the *graph* -- that
ONNX and TorchScript reproduce the native PyTorch computation, including the
frozen-class head and the fixed-frame video pooling.

Tolerances are asserted rather than hard-coded to zero for ONNX because its
kernels legitimately differ in accumulation order; TorchScript runs the same
kernels and must match exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.models.pe.export import (
    _FrozenPEClassifier,
    _PEImageEmbedder,
    _PEVideoEmbedder,
)
from libreyolo.models.pe.nn import PE_CONFIGS, build_pe_model

pytestmark = [pytest.mark.unit, pytest.mark.pe]

SIZE = "t16"
RES = PE_CONFIGS[SIZE].image_size
DIM = PE_CONFIGS[SIZE].projection_dim
# ONNX fp32 graphs of this depth land well inside 1e-4; the classify graph is
# amplified by logit_scale (~100x), so it gets the same absolute budget on a
# much larger signal.
ONNX_ATOL = 1e-4


@pytest.fixture(scope="module")
def visual():
    torch.manual_seed(0)
    return build_pe_model(SIZE).eval().visual


def _onnx_run(module, dummy, input_name, tmp_path, name):
    onnxruntime = pytest.importorskip("onnxruntime")
    path = str(tmp_path / f"{name}.onnx")
    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy,
            path,
            input_names=[input_name],
            output_names=["out"],
            opset_version=17,
            dynamic_axes={input_name: {0: "batch"}, "out": {0: "batch"}},
            dynamo=False,
        )
    session = onnxruntime.InferenceSession(path)
    return session.run(None, {input_name: dummy.numpy()})[0]


# =============================================================================
# Image embedding graph
# =============================================================================


def test_image_embed_graph_is_unit_norm(visual):
    module = _PEImageEmbedder(visual).eval()
    with torch.no_grad():
        out = module(torch.randn(2, 3, RES, RES))
    assert out.shape == (2, DIM)
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(2), rtol=0, atol=1e-5)


@pytest.mark.onnx
def test_image_embed_onnx_matches_native(visual, tmp_path):
    module = _PEImageEmbedder(visual).eval()
    x = torch.randn(1, 3, RES, RES)
    with torch.no_grad():
        native = module(x).numpy()
    got = _onnx_run(module, x, "images", tmp_path, "embed")
    assert np.abs(got - native).max() < ONNX_ATOL


@pytest.mark.torchscript
def test_image_embed_torchscript_is_exact(visual, tmp_path):
    module = _PEImageEmbedder(visual).eval()
    x = torch.randn(1, 3, RES, RES)
    with torch.no_grad():
        native = module(x)
        traced = torch.jit.trace(module, x, strict=False)
    torch.testing.assert_close(traced(x), native, rtol=0, atol=0)


# =============================================================================
# Frozen-class classify graph
# =============================================================================


@pytest.fixture
def frozen(visual):
    torch.manual_seed(1)
    weight = torch.nn.functional.normalize(torch.randn(3, DIM), dim=-1) * 100.0
    return _FrozenPEClassifier(visual, weight).eval()


def test_frozen_classifier_shape(frozen):
    with torch.no_grad():
        assert frozen(torch.randn(2, 3, RES, RES)).shape == (2, 3)


@pytest.mark.onnx
def test_frozen_classify_onnx_matches_native(frozen, tmp_path):
    x = torch.randn(1, 3, RES, RES)
    with torch.no_grad():
        native = frozen(x).numpy()
    got = _onnx_run(frozen, x, "images", tmp_path, "cls")
    # Logits are logit_scale-amplified, so compare on the same absolute budget
    # relative to the signal magnitude.
    assert np.abs(got - native).max() < ONNX_ATOL * max(1.0, np.abs(native).max())


@pytest.mark.torchscript
def test_frozen_classify_torchscript_is_exact(frozen):
    x = torch.randn(1, 3, RES, RES)
    with torch.no_grad():
        native = frozen(x)
        traced = torch.jit.trace(frozen, x, strict=False)
    torch.testing.assert_close(traced(x), native, rtol=0, atol=0)


# =============================================================================
# Fixed-frame video graph
# =============================================================================


def test_video_graph_matches_encode_video(visual):
    """The exported clip graph must agree with the eager encode_video path."""
    model = build_pe_model(SIZE).eval()
    module = _PEVideoEmbedder(model.visual).eval()
    clips = torch.randn(1, 3, 3, RES, RES)
    with torch.no_grad():
        torch.testing.assert_close(
            module(clips), model.encode_video(clips), rtol=0, atol=0
        )


@pytest.mark.onnx
def test_video_onnx_matches_native(visual, tmp_path):
    module = _PEVideoEmbedder(visual).eval()
    clips = torch.randn(1, 2, 3, RES, RES)
    with torch.no_grad():
        native = module(clips).numpy()
    got = _onnx_run(module, clips, "clip", tmp_path, "vembed")
    assert np.abs(got - native).max() < ONNX_ATOL


@pytest.mark.torchscript
def test_video_torchscript_is_exact(visual):
    module = _PEVideoEmbedder(visual).eval()
    clips = torch.randn(1, 2, 3, RES, RES)
    with torch.no_grad():
        native = module(clips)
        traced = torch.jit.trace(module, clips, strict=False)
    torch.testing.assert_close(traced(clips), native, rtol=0, atol=0)


# =============================================================================
# Declared support must match what is actually implemented
# =============================================================================


def test_declared_validated_formats_are_onnx_and_torchscript():
    from libreyolo.export.support import validated_alternatives

    for task in ("classify", "embed"):
        assert set(validated_alternatives("pe", task)) == {"onnx", "torchscript"}


def test_unvalidated_runtimes_are_blocked_not_advertised():
    from libreyolo.export.support import get_support

    for fmt in ("ncnn", "coreml", "coreai", "tflite", "paddle", "rknn"):
        assert get_support("pe", "embed", fmt).tier == "blocked", fmt
