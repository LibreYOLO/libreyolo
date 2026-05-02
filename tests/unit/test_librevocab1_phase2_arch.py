"""LibreVocab1 Phase 2 architecture-level smoke tests.

Verifies the *structural* integration of CRADIOv4 backbone wrapper + neck +
encoder + decoder *interfaces* — without loading real CRADIOv4 weights and
without invoking the not-yet-implemented heavy pieces.

The decoder, neck, and hybrid encoder are scaffolded; their forward passes
raise NotImplementedError. This test asserts that fact, so we do not silently
ship a partial Phase 2.
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


def test_phase2_backbone_wrapper_constructs_without_weights():
    """CRADIOv4Backbone constructs without downloading anything."""
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


def test_phase2_network_assembles_without_implementations():
    """LibreVocab1Phase2Network composes without erroring at __init__."""
    from libreyolo.models.librevocab1.phase2.nn import LibreVocab1Phase2Network

    net = LibreVocab1Phase2Network(size="s", device="cpu")
    assert net.backbone.embed_dim == 1152
    assert net.neck.hidden_dim == 256
    assert net.encoder.num_levels == 3
    assert net.decoder.num_queries == 300
    assert net.decoder.num_layers == 6
    assert net.decoder.text_dim == 1152


def test_phase2_components_advertise_unimplemented_clearly():
    """Heavy components must raise NotImplementedError, not silently no-op."""
    import torch
    from libreyolo.models.librevocab1.phase2.nn import (
        SPMv2Neck,
        HybridEncoder,
        OpenVocabDecoder,
    )

    neck = SPMv2Neck(embed_dim=1152, hidden_dim=256)
    with pytest.raises(NotImplementedError, match="DEIMv2 branch merge"):
        neck([], torch.zeros(1, 3, 800, 800))

    enc = HybridEncoder(hidden_dim=256, num_levels=3)
    with pytest.raises(NotImplementedError, match="DEIMv2 branch merge"):
        enc([])

    dec = OpenVocabDecoder(hidden_dim=256, num_queries=300, num_layers=6, text_dim=1152)
    with pytest.raises(NotImplementedError, match="DEIMv2 branch merge"):
        dec([], torch.zeros(1, 1152))


def test_phase2_basemodel_class_does_not_route_real_checkpoints():
    """Factory safety: LibreVocab1Phase2.can_load() rejects every real ckpt."""
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
    """Plain ``import libreyolo`` must not pull phase2 into the registry."""
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
