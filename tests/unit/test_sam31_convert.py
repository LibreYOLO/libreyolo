"""Offline coverage for the SAM 3.1 image-path conversion and LibreSAM31 wiring."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from libreyolo.models.sam.model import _ALIASES
from libreyolo.models.sam.sam3 import LibreSAM3, LibreSAM31
from weights import convert_sam31_weights as conv

pytestmark = [pytest.mark.unit, pytest.mark.sam]

_VB = "detector.backbone.vision_backbone."


def _sam31_state_dict():
    """A tiny state dict with the SAM 3.1 key layout (shapes are irrelevant)."""
    t = torch.arange(6, dtype=torch.float32)
    return {
        _VB + "trunk.pos_embed": torch.arange(2 * 3 * 4.0).reshape(1, 3, 8),
        _VB + "trunk.blocks.0.attn.qkv.weight": torch.arange(12.0).reshape(6, 2),
        _VB + "trunk.blocks.0.attn.freqs_cis": torch.zeros(2),
        _VB + "convs.0.conv_1x1.weight": t.clone(),
        _VB + "interactive_convs.0.conv_1x1.weight": t + 1,
        _VB + "propagation_convs.0.conv_1x1.weight": t + 2,
        "detector.backbone.language_backbone.encoder.text_projection": torch.arange(6.0).reshape(2, 3),
        "detector.transformer.decoder.layers.0.cross_attn.in_proj_bias": torch.arange(6.0),
        "detector.geometry_encoder.points_direct_project.weight": t.clone(),
        "tracker.model.interactive_sam_mask_decoder.iou_prediction_head.layers.2.weight": t + 3,
        "tracker.model.interactive_sam_mask_decoder.transformer.layers.0.mlp.lin1.weight": t + 4,
        "tracker.model.interactive_sam_mask_decoder.transformer.layers.0.norm4.bias": t + 5,
        "tracker.model.interactive_sam_mask_decoder.output_upscaling.1.weight": t + 6,
        "tracker.model.interactive_sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix": t + 7,
        **{
            f"tracker.model.interactive_sam_prompt_encoder.point_embeddings.{i}.weight": torch.full((1, 2), float(i))
            for i in range(4)
        },
        "tracker.model.interactivity_no_mem_embed": t + 8,
        "tracker.model.sam_mask_decoder.mask_tokens.weight": t + 9,
        "tracker.model.maskmem_backbone.pix_feat_proj.weight": t + 10,
    }


def test_detects_sam31_layout_only():
    assert conv.is_sam31_state_dict(_sam31_state_dict())
    assert not conv.is_sam31_state_dict({_VB + "sam2_convs.0.conv_1x1.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="Not a SAM 3.1 checkpoint"):
        conv.sam31_to_sam3_layout({"tracker.sam_mask_decoder.x": torch.zeros(1)})


def test_sam31_renames_interactive_path_and_drops_video_keys():
    out = conv.sam31_to_sam3_layout(_sam31_state_dict())
    assert torch.equal(out[_VB + "sam2_convs.0.conv_1x1.weight"], torch.arange(6.0) + 1)
    assert torch.equal(out["tracker.no_mem_embed"], torch.arange(6.0) + 8)
    assert "tracker.sam_mask_decoder.transformer.layers.0.mlp.lin1.weight" in out
    assert not any("propagation_convs" in k for k in out)
    assert not any(k.startswith("tracker.model.") for k in out)


def test_dropped_fourth_fpn_level_is_zero_filled_in_both_necks():
    out = conv.sam31_to_sam3_layout(_sam31_state_dict())
    for neck in ("convs", "sam2_convs"):
        weight = out[f"{_VB}{neck}.3.conv_1x1.weight"]
        assert weight.shape == (256, 1024, 1, 1)
        assert not weight.any()
        assert out[f"{_VB}{neck}.3.conv_3x3.weight"].shape == (256, 256, 3, 3)


def test_transformers_layout_names_and_tensor_transforms():
    src = _sam31_state_dict()
    out = conv.sam3_layout_to_transformers(conv.sam31_to_sam3_layout(src))
    pos = out["detector_model.vision_encoder.backbone.embeddings.position_embeddings"]
    assert torch.equal(pos, src[_VB + "trunk.pos_embed"][:, 1:, :])
    q = out["detector_model.vision_encoder.backbone.layers.0.attention.q_proj.weight"]
    assert torch.equal(q, torch.arange(4.0).reshape(2, 2))
    assert "detector_model.vision_encoder.backbone.layers.0.rotary_emb.rope_embeddings" not in out
    assert torch.equal(
        out["detector_model.text_encoder.text_projection.weight"],
        torch.arange(6.0).reshape(2, 3).T,
    )
    assert torch.equal(
        out["detector_model.detr_decoder.layers.0.vision_cross_attn.v_proj.bias"],
        torch.tensor([4.0, 5.0]),
    )
    assert not any(".geometry_encoder.points_" in k for k in out)
    assert torch.equal(
        out["tracker_neck.fpn_layers.0.proj1.weight"], torch.arange(6.0) + 1
    )
    assert "tracker_neck.fpn_layers.3.proj2.weight" in out
    head = "tracker_model.mask_decoder."
    assert head + "iou_prediction_head.proj_out.weight" in out
    assert head + "transformer.layers.0.mlp.proj_in.weight" in out
    assert head + "transformer.layers.0.layer_norm4.bias" in out
    assert head + "upscale_layer_norm.weight" in out
    assert torch.equal(
        out["tracker_model.prompt_encoder.point_embed.weight"],
        torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )
    shared = out["tracker_model.shared_image_embedding.positional_embedding"]
    prompt_pe = out["tracker_model.prompt_encoder.shared_embedding.positional_embedding"]
    assert torch.equal(shared, prompt_pe)
    assert shared.data_ptr() != prompt_pe.data_ptr()
    assert torch.equal(out["tracker_model.no_memory_embedding"], torch.arange(6.0) + 8)
    assert not any("mask_tokens" in k and "tracker_model.mask_decoder" not in k for k in out)


def test_convert_state_dict_from_file(tmp_path):
    src = tmp_path / "sam3.1_multiplex.pt"
    torch.save({"model": _sam31_state_dict()}, src)
    loaded = conv.convert_sam31_state_dict(src)
    assert "tracker_model.no_memory_embedding" in loaded


def test_write_payload_refuses_unpinned_inputs(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / conv.SAM31_CHECKPOINT).write_bytes(b"not the pinned file")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        conv.write_payload(tmp_path / "out", src)
    assert not (tmp_path / "out").exists()


def test_sam31_aliases_and_class_contract():
    for alias in ("sam3.1", "sam-3.1", "sam31", "sam3.1-large"):
        assert _ALIASES[alias] == (LibreSAM31, "large")
    assert issubclass(LibreSAM31, LibreSAM3)
    assert LibreSAM31.INPUT_SIZES == {"large": 1008}
    assert LibreSAM31.FAMILY == "sam31"
    with pytest.raises(NotImplementedError, match="image inference only"):
        object.__new__(LibreSAM31).track()


def test_sam31_loads_from_the_gated_mirror():
    assert LibreSAM31.HF_REPOS == {"large": conv.TARGET_REPO}
    assert "*.pt" in LibreSAM31.SNAPSHOT_IGNORE_PATTERNS


def test_gated_sam31_download_error_is_actionable(monkeypatch, tmp_path):
    class Forbidden(Exception):
        response = SimpleNamespace(status_code=403)

    def deny(*args, **kwargs):
        raise Forbidden()

    model = object.__new__(LibreSAM31)
    model.size = "large"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=deny)
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace())
    with pytest.raises(RuntimeError) as caught:
        model._ensure_weights()
    assert "https://huggingface.co/LibreYOLO/LibreSAM31large" in str(caught.value)
    assert "HF_TOKEN" in str(caught.value)
