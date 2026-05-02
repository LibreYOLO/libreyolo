"""LibreVocab1 architecture-level smoke tests.

Verifies the *structural* integration of the vendored SAM3 stack with a
CRADIOv4-shaped trunk, **without** downloading any real weights. Real
inference parity vs ``mranzinger/sam3-radio`` is a separate test that
requires SAM3 + CRADIOv4 weight downloads (~6.5 GB combined) and lives
behind an env var gate elsewhere.

Marked experimental — the family is not yet registered in the LibreYOLO
factory and the API surface is still in flux.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn as nn


# Skip the whole module if the vocab1 extra isn't installed.
_REQUIRED = ("einops", "timm", "huggingface_hub", "iopath", "ftfy", "regex")
_MISSING = [m for m in _REQUIRED if importlib.util.find_spec(m) is None]
pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        bool(_MISSING),
        reason=f"libreyolo[vocab1] extras missing: {_MISSING!r}",
    ),
]


class _StubCRADIO(nn.Module):
    """Stand-in for `torch.hub.load('NVlabs/RADIO', 'radio_model', ...)`.

    Mirrors only the surface the SAM3 swap touches: callable that returns
    ``{'sam3': (summary, features)}`` where ``features`` is ``[B, N, C]``.
    """

    def __init__(self, embed_dim: int = 1024, patch_size: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.conv = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        feat = self.conv(x)  # (B, C, H/p, W/p)
        feat_seq = feat.flatten(2).transpose(1, 2)  # (B, N, C)
        summary = feat.mean(dim=(2, 3))  # (B, C)
        return {"sam3": (summary, feat_seq)}


def _build_sam3_no_weights():
    """Build the SAM3 image stack on CPU without loading any checkpoint."""
    from libreyolo.models.librevocab1.sam3 import model_builder as mb
    mb._load_checkpoint = lambda model, ckpt_path: None  # bypass download/load
    return mb.build_sam3_image_model(
        bpe_path=None,
        device="cpu",
        checkpoint_path="dummy",
        load_from_HF=False,
        eval_mode=True,
    )


def test_sam3_stack_builds_on_cpu_without_weights():
    """SAM3 stack should construct on CPU with no real weights."""
    sam3_model = _build_sam3_no_weights()
    assert type(sam3_model).__name__ == "Sam3Image"
    n_params = sum(p.numel() for p in sam3_model.parameters())
    # Roughly ~840M for the default ViT-L+ SAM3 image variant.
    assert 700e6 < n_params < 1_000e6, f"unexpected param count: {n_params/1e6:.1f}M"
    assert hasattr(sam3_model, "transformer")
    assert hasattr(sam3_model, "segmentation_head")
    assert hasattr(sam3_model, "dot_prod_scoring")


def test_radio_swap_replaces_trunk_and_preserves_shape():
    """RADIO_Adaptor swap must replace the trunk and produce stride-16 output."""
    from libreyolo.models.librevocab1.sam3.sam3_radio_utils import replace_sam3_encoder
    sam3_model = _build_sam3_no_weights()
    sam3_dim = sam3_model.backbone.vision_backbone.trunk.patch_embed.proj.out_channels
    stub_radio = _StubCRADIO(embed_dim=sam3_dim, patch_size=16)
    sam3_model = replace_sam3_encoder(sam3_model, stub_radio, device="cpu")

    new_trunk = sam3_model.backbone.vision_backbone.trunk
    assert type(new_trunk).__name__ == "RADIO_Adaptor"
    assert new_trunk.channel_list == [sam3_dim]


def test_swapped_trunk_forward_produces_expected_pyramid():
    """Forward pass through the swapped trunk yields stride-16 NCHW features."""
    from libreyolo.models.librevocab1.sam3.sam3_radio_utils import replace_sam3_encoder
    sam3_model = _build_sam3_no_weights()
    sam3_dim = sam3_model.backbone.vision_backbone.trunk.patch_embed.proj.out_channels
    stub_radio = _StubCRADIO(embed_dim=sam3_dim, patch_size=16)
    sam3_model = replace_sam3_encoder(sam3_model, stub_radio, device="cpu")

    x = torch.randn(1, 3, 1152, 1152)
    trunk_out = sam3_model.backbone.vision_backbone.trunk(x)
    assert isinstance(trunk_out, list) and len(trunk_out) == 1
    out = trunk_out[0]
    # 1152 / 16 = 72; channel dim matches SAM3's expected vision embed.
    assert out.shape == (1, sam3_dim, 72, 72), f"got {tuple(out.shape)}"
    assert out.dtype == torch.float32


def test_librevocab1_class_does_not_route_real_checkpoints():
    """Factory safety: LibreVocab1.can_load() must reject every real checkpoint."""
    from libreyolo.models.librevocab1 import LibreVocab1
    sample_dicts = [
        {"backbone.norm.weight": 0, "head.cls_score.weight": 0},   # YOLOX-ish
        {"decoder.pre_bbox_head.cls_score.weight": 0},              # DEIM
        {"model.backbone.0.weight": 0},                             # generic
    ]
    for d in sample_dicts:
        assert LibreVocab1.can_load(d) is False, f"can_load leaked True on {d}"


def test_plain_libreyolo_import_does_not_register_librevocab1():
    """Plain ``import libreyolo`` must not pull librevocab1 into the registry.

    Path A is experimental and must stay out of factory routing until the
    inference parity test passes.
    """
    import importlib
    import sys
    # Drop cached modules so we re-evaluate the registration side effects.
    drops = [m for m in list(sys.modules) if m.startswith("libreyolo")]
    for m in drops:
        del sys.modules[m]

    importlib.import_module("libreyolo")
    from libreyolo.models.base import BaseModel
    names = [c.__name__ for c in BaseModel._registry]
    assert "LibreVocab1" not in names, (
        f"LibreVocab1 leaked into the registry on plain import: {names}"
    )
