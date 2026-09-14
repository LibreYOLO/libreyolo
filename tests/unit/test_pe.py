"""Focused unit tests for the LibrePE (Perception Encoder Core) family.

These tests use synthetic tensors and the closed configuration table only. Exact
parity against ``open_clip_torch==3.2.0`` lives in ``weights/parity_pe.py``, and
cleared-cache download tests live in the network tier -- neither belongs here.

No test in this file may fetch the ``facebook/PE-Video`` dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.models.pe.model import DEFAULT_CLIP_FRAMES, LibrePE
from libreyolo.models.pe.nn import PE_CONFIGS, build_pe_model
from libreyolo.utils.video import uniform_frame_indices

pytestmark = [pytest.mark.unit, pytest.mark.pe]

# g14 is a 1.88B-parameter vision tower; constructing it is a slow-tier concern.
LIGHT_SIZES = ("t16", "s16", "b16")
ALL_SIZES = tuple(PE_CONFIGS)


# =============================================================================
# Configuration table
# =============================================================================


def test_expected_sizes_present():
    assert ALL_SIZES == ("t16", "s16", "b16", "l14", "g14")


@pytest.mark.parametrize(
    ("size", "image_size", "embed_dim", "context"),
    [
        ("t16", 384, 512, 32),
        ("s16", 384, 512, 32),
        ("b16", 224, 1024, 32),
        ("l14", 336, 1024, 32),
        ("g14", 448, 1280, 72),
    ],
)
def test_config_matches_published_series(size, image_size, embed_dim, context):
    """Guards the per-size table against a silent edit."""
    cfg = PE_CONFIGS[size]
    assert cfg.image_size == image_size
    assert cfg.projection_dim == embed_dim
    assert cfg.context_length == context


def test_gigantic_is_the_only_class_token_free_size():
    """bigG drops the class token and the rope grid offset; the rest keep both."""
    assert PE_CONFIGS["g14"].class_token is False
    assert PE_CONFIGS["g14"].rope_grid_offset == 0.0
    for size in ("t16", "s16", "b16", "l14"):
        assert PE_CONFIGS[size].class_token is True
        assert PE_CONFIGS[size].rope_grid_offset == 1.0


def test_input_sizes_exposed_per_size():
    assert LibrePE.INPUT_SIZES == {s: c.image_size for s, c in PE_CONFIGS.items()}


def test_family_constants():
    assert LibrePE.FAMILY == "pe"
    assert LibrePE.FILENAME_PREFIX == "LibrePE"
    assert LibrePE.SUPPORTED_TASKS == ("classify", "embed")
    assert LibrePE.DEFAULT_TASK == "classify"
    assert LibrePE.WEIGHT_TASKS == ("classify",)
    assert LibrePE.REQUIRE_TASK_SUFFIX is True
    assert LibrePE.TRAIN_CONFIG is None
    assert LibrePE.VIDEO_EMBED_MODE == "clip"


def test_registered_in_model_group_s():
    from libreyolo.models.registry import MODEL_GROUPS

    assert MODEL_GROUPS["pe"] == "s"


def test_build_rejects_unknown_size():
    with pytest.raises(ValueError, match="Unknown PE size"):
        build_pe_model("xl99")


# =============================================================================
# Architecture / forward contracts
# =============================================================================


@pytest.mark.parametrize("size", LIGHT_SIZES)
def test_forward_shapes(size):
    cfg = PE_CONFIGS[size]
    model = build_pe_model(size).eval()
    res = cfg.image_size
    with torch.no_grad():
        img = model.encode_image(torch.zeros(2, 3, res, res))
        txt = model.encode_text(torch.zeros(3, cfg.context_length, dtype=torch.long))
    assert img.shape == (2, cfg.projection_dim)
    assert txt.shape == (3, cfg.projection_dim)
    assert img.dtype is torch.float32


@pytest.mark.parametrize("size", LIGHT_SIZES)
def test_encode_video_pools_and_normalizes_once(size):
    cfg = PE_CONFIGS[size]
    model = build_pe_model(size).eval()
    res = cfg.image_size
    clips = torch.randn(2, 3, 3, res, res)
    with torch.no_grad():
        out = model.encode_video(clips)
        frame_feats = model.encode_image(clips.reshape(6, 3, res, res)).reshape(2, 3, -1)
        expected = torch.nn.functional.normalize(frame_feats.mean(dim=1), dim=-1)
    assert out.shape == (2, cfg.projection_dim)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        out.norm(dim=-1), torch.ones(2), rtol=0, atol=1e-5
    )


def test_encode_video_rejects_non_5d():
    model = build_pe_model("t16").eval()
    with pytest.raises(ValueError, match="5D"):
        model.encode_video(torch.zeros(2, 3, 384, 384))


def test_rope_buffer_is_not_persisted():
    """RoPE tables are config-derived; they must not be required by a checkpoint."""
    model = build_pe_model("t16")
    assert not any("pos_embed_cat" in k for k in model.state_dict())
    assert not any("attn_mask" in k for k in model.state_dict())


# =============================================================================
# Checkpoint recognition
# =============================================================================


def _reference_state(size: str) -> dict:
    return build_pe_model(size).state_dict()


@pytest.mark.parametrize("size", LIGHT_SIZES)
def test_can_load_accepts_own_state(size):
    assert LibrePE.can_load(_reference_state(size)) is True


@pytest.mark.parametrize("size", LIGHT_SIZES)
def test_detect_size_roundtrip(size):
    assert LibrePE.detect_size(_reference_state(size)) == size


def test_detect_size_returns_none_for_foreign_state():
    assert LibrePE.detect_size({"visual.conv1.weight": torch.zeros(3)}) is None


def test_can_load_rejects_sibling_families():
    """A greedy recognizer would steal CLIP / SigLIP2 / DINOv2 checkpoints."""
    clip_like = {
        "visual.conv1.weight": torch.zeros(1),
        "text_projection": torch.zeros(1),
        "logit_scale": torch.zeros(()),
    }
    siglip_like = {
        "vision_model.embeddings.patch_embedding.weight": torch.zeros(1),
        "text_model.head.weight": torch.zeros(1),
        "logit_bias": torch.zeros(()),
        "logit_scale": torch.zeros(()),
    }
    dinov2_like = {"backbone.blocks.0.attn.qkv.weight": torch.zeros(1)}
    for foreign in (clip_like, siglip_like, dinov2_like):
        assert LibrePE.can_load(foreign) is False


def test_detect_nb_classes_is_open_vocabulary():
    assert LibrePE.detect_nb_classes(_reference_state("t16")) is None


# =============================================================================
# Converter guards
# =============================================================================


def test_converter_rejects_conflicting_size(tmp_path):
    import safetensors.torch as st

    from weights.convert_pe_weights import convert

    src = tmp_path / "src.safetensors"
    st.save_file(_reference_state("t16"), str(src))
    with pytest.raises(ValueError, match="conflicts with the source config"):
        convert(str(src), str(tmp_path / "out.pt"), size="b16")


def test_converter_rejects_foreign_checkpoint(tmp_path):
    import safetensors.torch as st

    from weights.convert_pe_weights import infer_size

    st.save_file({"visual.conv1.weight": torch.zeros(4)}, str(tmp_path / "x.safetensors"))
    with pytest.raises(ValueError, match="does not look like"):
        infer_size({"visual.conv1.weight": torch.zeros(4)})


# =============================================================================
# Deterministic uniform video sampling
# =============================================================================


def test_sampling_includes_both_endpoints():
    idx = uniform_frame_indices(30, 8)
    assert idx[0] == 0 and idx[-1] == 29
    assert idx == sorted(idx)
    assert len(idx) == 8


def test_sampling_exact_length_is_identity():
    assert uniform_frame_indices(8, 8) == list(range(8))


def test_sampling_short_video_repeats_last_frame_only():
    assert uniform_frame_indices(3, 8) == [0, 1, 2, 2, 2, 2, 2, 2]


def test_sampling_single_frame_video():
    assert uniform_frame_indices(1, 8) == [0] * 8


def test_sampling_single_requested_frame():
    assert uniform_frame_indices(30, 1) == [0]


def test_sampling_long_video_is_uniform():
    idx = uniform_frame_indices(1000, 5)
    # step = 999/4 = 249.75, so the interior points round to 250/500/749.
    assert idx == [0, 250, 500, 749, 999]


@pytest.mark.parametrize("bad", [0, -1])
def test_sampling_rejects_non_positive_request(bad):
    with pytest.raises(ValueError, match="num_frames must be positive"):
        uniform_frame_indices(30, bad)


def test_sampling_rejects_empty_video():
    with pytest.raises(ValueError, match="zero frames"):
        uniform_frame_indices(0, 8)


def test_default_clip_frames_is_eight():
    assert DEFAULT_CLIP_FRAMES == 8


# =============================================================================
# No-regression: other families keep frame-by-frame video behavior
# =============================================================================


def test_other_families_default_to_frame_by_frame():
    from libreyolo.models.base.model import BaseModel
    from libreyolo.models.clip.model import LibreCLIP
    from libreyolo.models.dinov2.model import LibreDINOv2
    from libreyolo.models.siglip2.model import LibreSigLIP2

    assert BaseModel.VIDEO_EMBED_MODE == "frames"
    for family in (LibreCLIP, LibreSigLIP2, LibreDINOv2):
        assert family.VIDEO_EMBED_MODE == "frames", family.__name__


def test_clip_mode_stays_opt_in_and_explicit():
    """Clip mode is opt-in, and the opted-in set is small and named.

    PE was the only clip-mode family when it landed; V-JEPA 2 and LeVJEPA then
    joined it on the same shared route rather than adding competing ones. This
    asserts the set is exactly the families that deliberately opted in, so a
    family cannot acquire whole-clip behaviour by accident.
    """
    from libreyolo.models.base.model import BaseModel

    clip_families = {
        cls.__name__
        for cls in BaseModel.__subclasses__()
        if getattr(cls, "VIDEO_EMBED_MODE", "frames") == "clip"
    }
    assert clip_families == {"LibrePE", "LibreVJEPA2", "LibreLeVJEPA"}
