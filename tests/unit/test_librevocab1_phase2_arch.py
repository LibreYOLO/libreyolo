"""LibreVocab1 Phase 2 architecture-level unit tests.

Three layers of verification:
    1. Each component (SPMv2Neck / HybridEncoder / OpenVocabDecoder) runs
       a real forward pass on synthetic inputs and produces correctly
       shaped tensors.
    2. The top-level Network composes them all without errors *at __init__
       time*. Running its forward needs the real CRADIOv4 backbone (which
       downloads ~1.6GB of weights) — that's covered by the e2e gated test.
    3. BaseModel-style isolation invariants from Phase 1 still hold
       (can_load=False, no factory leak on plain import, filename detection).
"""

from __future__ import annotations

import importlib.util

import pytest


_REQUIRED = ("einops", "timm", "huggingface_hub")
_MISSING = [m for m in _REQUIRED if importlib.util.find_spec(m) is None]
pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        bool(_MISSING),
        reason=f"libreyolo[vocab1] extras missing: {_MISSING!r}",
    ),
]


# ---------------------------------------------------------------------------
# Backbone wrapper construction (no weight download)
# ---------------------------------------------------------------------------


def test_backbone_wrapper_constructs_without_weights():
    from libreyolo.models.librevocab1.phase2.nn import CRADIOv4Backbone

    bb = CRADIOv4Backbone(version="c-radio_v4-so400m", device="cpu")
    assert bb.version == "c-radio_v4-so400m"
    assert bb.embed_dim == 1152
    assert bb.patch_size == 16
    assert bb.indices == [13, 19, 25]
    assert bb._radio is None  # lazy: no weights loaded yet

    bb_h = CRADIOv4Backbone(version="c-radio_v4-h", device="cpu")
    assert bb_h.embed_dim == 1280
    assert bb_h.indices == [15, 23, 31]


# ---------------------------------------------------------------------------
# Neck
# ---------------------------------------------------------------------------


def test_spmv2_neck_produces_three_scale_pyramid():
    """Synthetic stride-16 ViT intermediates → stride 8 / 16 / 32 outputs."""
    import torch
    from libreyolo.models.librevocab1.phase2.nn import SPMv2Neck

    embed_dim = 1152
    hidden_dim = 256
    neck = SPMv2Neck(embed_dim=embed_dim, hidden_dim=hidden_dim)
    neck.eval()

    B, H, W = 2, 256, 256
    image = torch.randn(B, 3, H, W)
    inter = [torch.randn(B, embed_dim, H // 16, W // 16) for _ in range(3)]
    out = neck(inter, image)

    assert isinstance(out, list) and len(out) == 3
    expected_sizes = [(H // 8, W // 8), (H // 16, W // 16), (H // 32, W // 32)]
    for tensor, (eh, ew) in zip(out, expected_sizes):
        assert tensor.shape == (B, hidden_dim, eh, ew), (
            f"neck output shape {tuple(tensor.shape)} != "
            f"expected {(B, hidden_dim, eh, ew)}"
        )
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()


def test_spmv2_neck_rejects_wrong_intermediate_count():
    import torch
    from libreyolo.models.librevocab1.phase2.nn import SPMv2Neck

    neck = SPMv2Neck(embed_dim=1152, hidden_dim=256)
    image = torch.randn(1, 3, 64, 64)
    with pytest.raises(ValueError, match="3 intermediates"):
        neck([torch.randn(1, 1152, 4, 4)], image)


# ---------------------------------------------------------------------------
# Hybrid encoder
# ---------------------------------------------------------------------------


def test_hybrid_encoder_preserves_pyramid_shape():
    import torch
    from libreyolo.models.librevocab1.phase2.nn import HybridEncoder

    enc = HybridEncoder(hidden_dim=256, num_levels=3, n_heads=4, dim_ff=512)
    enc.eval()
    B = 2
    pyramid = [
        torch.randn(B, 256, 32, 32),  # stride 8
        torch.randn(B, 256, 16, 16),  # stride 16
        torch.randn(B, 256, 8, 8),    # stride 32
    ]
    out = enc(pyramid)
    assert len(out) == 3
    for orig, fused in zip(pyramid, out):
        assert fused.shape == orig.shape
        assert torch.isfinite(fused).all()


def test_hybrid_encoder_self_attention_changes_smallest_scale():
    """The s32 output should *not* be a verbatim copy — AIFI mixed it."""
    import torch
    from libreyolo.models.librevocab1.phase2.nn import HybridEncoder

    torch.manual_seed(0)
    enc = HybridEncoder(hidden_dim=128, n_heads=4, dim_ff=256, n_intra_layers=1)
    enc.eval()
    pyramid = [
        torch.randn(1, 128, 16, 16),
        torch.randn(1, 128, 8, 8),
        torch.randn(1, 128, 4, 4),
    ]
    s32_in = pyramid[2]
    out = enc(pyramid)
    s32_out = out[2]
    # AIFI plus FPN-fusion downstream means s32 is heavily transformed.
    assert not torch.allclose(s32_in, s32_out), "AIFI did not change s32"


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def test_open_vocab_decoder_outputs_have_correct_shapes():
    import torch
    from libreyolo.models.librevocab1.phase2.nn import OpenVocabDecoder

    text_dim = 1152
    hidden = 256
    Q = 50
    K = 7

    dec = OpenVocabDecoder(
        hidden_dim=hidden, num_queries=Q, num_layers=2,
        text_dim=text_dim, n_heads=4, dim_ff=512,
    )
    dec.eval()

    B = 2
    pyramid = [
        torch.randn(B, hidden, 16, 16),
        torch.randn(B, hidden, 8, 8),
        torch.randn(B, hidden, 4, 4),
    ]
    text_emb = torch.randn(K, text_dim)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    out = dec(pyramid, text_emb)
    assert set(out.keys()) == {"pred_logits", "pred_boxes"}
    assert out["pred_logits"].shape == (B, Q, K)
    assert out["pred_boxes"].shape == (B, Q, 4)
    # boxes must be in (0, 1) (cxcywh sigmoid range).
    assert (out["pred_boxes"] > 0).all()
    assert (out["pred_boxes"] < 1).all()
    assert torch.isfinite(out["pred_logits"]).all()


def test_open_vocab_decoder_handles_dynamic_K():
    """Same model, different K at successive calls — open-vocab requirement."""
    import torch
    from libreyolo.models.librevocab1.phase2.nn import OpenVocabDecoder

    dec = OpenVocabDecoder(hidden_dim=128, num_queries=20, num_layers=1, text_dim=64)
    dec.eval()
    pyramid = [torch.randn(1, 128, h, h) for h in (8, 4, 2)]

    for K in (1, 3, 12):
        text_emb = torch.randn(K, 64)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        out = dec(pyramid, text_emb)
        assert out["pred_logits"].shape == (1, 20, K)


def test_open_vocab_decoder_backward_works():
    """End-to-end backward through decoder + cosine logits + boxes."""
    import torch
    from libreyolo.models.librevocab1.phase2.nn import OpenVocabDecoder

    dec = OpenVocabDecoder(hidden_dim=64, num_queries=8, num_layers=1, text_dim=32)
    pyramid = [torch.randn(1, 64, h, h, requires_grad=True) for h in (8, 4, 2)]
    text_emb = torch.randn(3, 32, requires_grad=True)
    text_emb = text_emb / (text_emb.norm(dim=-1, keepdim=True) + 1e-6)

    out = dec(pyramid, text_emb)
    loss = out["pred_logits"].sum() + out["pred_boxes"].sum()
    loss.backward()
    # At least one decoder param must have a grad.
    assert any(p.grad is not None for p in dec.parameters())


# ---------------------------------------------------------------------------
# Top-level network
# ---------------------------------------------------------------------------


def test_phase2_network_assembles_without_weights():
    """Network composes (lazy backbone) without downloading anything."""
    from libreyolo.models.librevocab1.phase2.nn import LibreVocab1Phase2Network

    net = LibreVocab1Phase2Network(size="s", device="cpu")
    assert net.backbone.embed_dim == 1152
    assert net.neck.hidden_dim == 256
    assert net.encoder.num_levels == 3
    assert net.decoder.num_queries == 300
    assert net.decoder.num_layers == 6
    assert net.decoder.text_dim == 1152

    net_h = LibreVocab1Phase2Network(size="h", device="cpu")
    assert net_h.backbone.embed_dim == 1280


def test_phase2_network_forward_with_fake_backbone():
    """Bypass the real CRADIOv4 download by stubbing forward_pyramid."""
    import torch
    from libreyolo.models.librevocab1.phase2.nn import LibreVocab1Phase2Network

    net = LibreVocab1Phase2Network(
        size="s",
        hidden_dim=64,
        num_queries=10,
        num_layers=1,
        n_heads=4,
        dim_ff=128,
        device="cpu",
    )

    # Patch backbone.forward_pyramid to return synthetic ViT intermediates.
    embed_dim = net.backbone.embed_dim
    H, W = 64, 64

    def _fake_pyramid(image: torch.Tensor):
        B = image.shape[0]
        return [torch.randn(B, embed_dim, H // 16, W // 16) for _ in range(3)]

    # Patch on the Module instance.
    net.backbone.forward_pyramid = _fake_pyramid  # type: ignore[assignment]

    image = torch.randn(2, 3, H, W)
    text_emb = torch.randn(5, 1152)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    out = net(image, text_emb)
    assert out["pred_logits"].shape == (2, 10, 5)
    assert out["pred_boxes"].shape == (2, 10, 4)


# ---------------------------------------------------------------------------
# BaseModel-style isolation
# ---------------------------------------------------------------------------


def test_phase2_basemodel_class_does_not_route_real_checkpoints():
    from libreyolo.models.librevocab1.phase2 import LibreVocab1Phase2

    sample_dicts = [
        {"backbone.norm.weight": 0, "head.cls_score.weight": 0},
        {"decoder.pre_bbox_head.cls_score.weight": 0},
        {"model.backbone.0.weight": 0},
    ]
    for d in sample_dicts:
        assert LibreVocab1Phase2.can_load(d) is False, f"can_load leaked True on {d}"


def test_phase2_filename_detection():
    from libreyolo.models.librevocab1.phase2 import LibreVocab1Phase2

    assert LibreVocab1Phase2.detect_size_from_filename("LibreVocab1Phase2s.pt") == "s"
    assert LibreVocab1Phase2.detect_size_from_filename("LibreVocab1Phase2h.pt") == "h"
    assert LibreVocab1Phase2.detect_size_from_filename("LibreVocab1s.pt") is None
    assert LibreVocab1Phase2.detect_size_from_filename("random.pt") is None


def test_phase2_does_not_leak_into_registry_on_plain_import():
    import importlib
    import sys

    drops = [m for m in list(sys.modules) if m.startswith("libreyolo")]
    for m in drops:
        del sys.modules[m]

    importlib.import_module("libreyolo")
    from libreyolo.models.base import BaseModel
    names = [c.__name__ for c in BaseModel._registry]
    assert "LibreVocab1Phase2" not in names, (
        f"LibreVocab1Phase2 leaked into registry: {names}"
    )
