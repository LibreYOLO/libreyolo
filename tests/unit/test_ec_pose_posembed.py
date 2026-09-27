"""EC pose keypoint position-embedding semantics.

Upstream EdgeCrafter's ``DeformableTransformerDecoderLayer.with_pos_embed``
(ecpose/engine/edgecrafter/detrpose_transformer.py) adds the embedding in place
in eval mode and out of place in training. The in-place add changes the layer
input itself, so at inference the embedding also reaches the value, residual
and gate input, and the previous layer's keypoint features the decoder reuses.
LibreYOLO reproduces that without mutating tensors. The references below are
literal in-place formulations of the upstream code path.
"""

from __future__ import annotations

import copy
import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch

from libreyolo.models.ec.decoder import (
    ECPoseTransformer,
    PoseDeformableTransformerDecoderLayer,
)

pytestmark = pytest.mark.unit


def _upstream_inplace_forward(
    layer,
    tgt_pose,
    tgt_pose_query_pos,
    tgt_pose_reference_points,
    attn_mask=None,
    memory=None,
    memory_spatial_shapes=None,
):
    """Upstream eval-mode forward, including its in-place position add."""

    def add_pos(tensor, pos):
        tensor[:, :, -pos.shape[2] :] += pos
        return tensor

    bs, nq, num_kpt, d_model = tgt_pose.shape
    q = k = add_pos(tgt_pose, tgt_pose_query_pos).flatten(0, 1)
    tgt2 = layer.within_attn(q, k, tgt_pose.flatten(0, 1))[0]
    tgt_pose = layer.within_norm(tgt_pose + tgt2.reshape(bs, nq, num_kpt, d_model))
    tgt_pose = tgt_pose.transpose(1, 2).flatten(0, 1)
    tgt2 = layer.across_attn(tgt_pose, tgt_pose, tgt_pose, attn_mask=attn_mask)[0]
    tgt_pose = layer.across_norm(tgt_pose + tgt2)
    tgt_pose = tgt_pose.reshape(bs, num_kpt, nq, d_model).transpose(1, 2)
    cross = layer.cross_attn(
        add_pos(tgt_pose, tgt_pose_query_pos).flatten(1, 2),
        tgt_pose_reference_points,
        memory,
        memory_spatial_shapes,
    ).reshape(bs, nq, num_kpt, d_model)
    tgt_pose = layer.gateway(tgt_pose, cross)
    return layer.forward_ffn(tgt_pose)


def _layer_inputs(d_model=32, n_heads=4, bs=2, nq=5, num_kpt=17):
    torch.manual_seed(0)
    shapes = torch.tensor([[8, 8], [4, 4], [2, 2]])
    # Pose deformable attention takes the value pre-split per level as
    # (bs * n_heads, head_dim, h * w), as ECPoseTransformer passes it.
    memory = tuple(
        torch.randn(bs * n_heads, d_model // n_heads, int(h * w)) for h, w in shapes
    )
    return {
        "tgt_pose": torch.randn(bs, nq, num_kpt + 1, d_model),
        "tgt_pose_query_pos": torch.randn(bs, nq, num_kpt, d_model),
        "tgt_pose_reference_points": torch.rand(bs, nq, 1, num_kpt + 1, 2),
        "memory": memory,
        "memory_spatial_shapes": shapes,
    }


def _cloned(inputs):
    return {
        k: tuple(t.clone() for t in v) if isinstance(v, tuple) else v.clone()
        for k, v in inputs.items()
    }


def _layer(d_model=32):
    torch.manual_seed(1)
    return PoseDeformableTransformerDecoderLayer(
        d_model=d_model, d_ffn=64, n_levels=3, n_heads=4, n_points=2
    )


def test_eval_layer_matches_upstream_inplace_semantics():
    layer = _layer().eval()
    inputs = _layer_inputs()
    with torch.no_grad():
        ours = layer(**_cloned(inputs))
        ref = _upstream_inplace_forward(layer, **_cloned(inputs))
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-5)


def test_eval_layer_does_not_mutate_its_input():
    layer = _layer().eval()
    inputs = _layer_inputs()
    tgt = inputs["tgt_pose"].clone()
    with torch.no_grad():
        layer(**inputs)
    # (the layer never writes to memory or the reference points either)
    torch.testing.assert_close(inputs["tgt_pose"], tgt, rtol=0, atol=0)


def test_training_layer_keeps_query_key_only_embedding():
    """Upstream training adds the embedding out of place, to queries/keys only."""
    layer = _layer().train()
    inputs = _layer_inputs()
    ours = layer(**inputs)

    tgt, pos = inputs["tgt_pose"], inputs["tgt_pose_query_pos"]
    bs, nq, num_kpt, d = tgt.shape
    with_pos = layer.with_pos_embed(tgt, pos)
    q = with_pos.flatten(0, 1)
    t = layer.within_norm(
        tgt + layer.within_attn(q, q, tgt.flatten(0, 1))[0].reshape(bs, nq, num_kpt, d)
    )
    t = t.transpose(1, 2).flatten(0, 1)
    t = layer.across_norm(t + layer.across_attn(t, t, t)[0])
    t = t.reshape(bs, num_kpt, nq, d).transpose(1, 2)
    cross = layer.cross_attn(
        layer.with_pos_embed(t, pos).flatten(1, 2),
        inputs["tgt_pose_reference_points"],
        inputs["memory"],
        inputs["memory_spatial_shapes"],
    ).reshape(bs, nq, num_kpt, d)
    ref = layer.forward_ffn(layer.gateway(t, cross))
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-5)
    ours.sum().backward()


def test_eval_decoder_matches_upstream_input_aliasing():
    """The decoder reproduces the in-place add leaking into the previous layer."""
    torch.manual_seed(0)
    ours = ECPoseTransformer(
        hidden_dim=32,
        nhead=4,
        num_queries=6,
        num_decoder_layers=3,
        dim_feedforward=64,
        dec_n_points=2,
        eval_spatial_size=(64, 64),
    ).eval()
    ref = copy.deepcopy(ours)
    for layer in ref.decoder.layers:
        # The literal in-place forward aliases the previous layer's output
        # itself, so the decoder must not add the leak a second time.
        layer.eval_pos_aliases_input = False
        layer.forward = types.MethodType(_upstream_inplace_forward, layer)
    feats = [torch.randn(1, 32, s, s) for s in (8, 4, 2)]
    with torch.no_grad():
        a = ours([f.clone() for f in feats])
        b = ref([f.clone() for f in feats])
    for key in ("pred_logits", "pred_keypoints"):
        torch.testing.assert_close(a[key], b[key], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Opt-in real-weight parity against the upstream EdgeCrafter ECPose decoder.
#   EC_POSE_UPSTREAM=/path/to/EdgeCrafter/ecpose \
#     pytest tests/unit/test_ec_pose_posembed.py -m 'unit and external_data'
# Downloads LibreEC{size}-pose.pt through the normal weight path.
# ---------------------------------------------------------------------------

_UPSTREAM = os.environ.get("EC_POSE_UPSTREAM")
_DEC_CFG = {
    "s": (192, 512, 3),
    "m": (256, 512, 4),
    "l": (256, 1024, 4),
    "x": (256, 2048, 4),
}


def _import_upstream_decoder():
    if not _UPSTREAM or not Path(_UPSTREAM).exists():
        pytest.skip("EC_POSE_UPSTREAM not set to an EdgeCrafter ecpose checkout")
    sys.path.insert(0, _UPSTREAM)
    stubbed = []
    try:
        for _ in range(40):
            try:
                from engine.edgecrafter.detrpose_transformer import DETRTransformer

                return DETRTransformer
            except ModuleNotFoundError as exc:
                # Training/eval-only upstream deps (tensorboard, calflops,
                # faster_coco_eval) are not needed to build the decoder.
                parts = (exc.name or "").split(".")
                for i in range(1, len(parts) + 1):
                    name = ".".join(parts[:i])
                    if name and name not in sys.modules:
                        stub = mock.MagicMock(name=name)
                        stub.__path__ = []
                        stub.__spec__ = None
                        sys.modules[name] = stub
                        stubbed.append(name)
                for key in [
                    k for k in sys.modules if k == "engine" or k.startswith("engine.")
                ]:
                    del sys.modules[key]
        pytest.skip("could not import the upstream EdgeCrafter decoder")
    finally:
        sys.path.remove(_UPSTREAM)


@pytest.mark.external_data
@pytest.mark.parametrize("size", ["s", "m", "l", "x"])
def test_real_weights_match_upstream_decoder(size, monkeypatch):
    from libreyolo import LibreYOLO

    tensorboard = types.ModuleType("torch.utils.tensorboard")
    tensorboard.SummaryWriter = object
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tensorboard)
    DETRTransformer = _import_upstream_decoder()

    model = LibreYOLO(f"LibreEC{size}-pose.pt", device="cpu")
    core = model.model.eval()
    hidden, ffn, layers = _DEC_CFG[size]
    upstream = DETRTransformer(
        hidden_dim=hidden,
        dim_feedforward=ffn,
        num_decoder_layers=layers,
        nhead=8,
        num_queries=60,
        return_intermediate_dec=True,
        num_feature_levels=3,
        dec_n_points=4,
        learnable_tgt_init=True,
        two_stage_type="standard",
        num_body_points=17,
        feat_strides=[8, 16, 32],
        reg_max=32,
        reg_scale=4,
        eval_spatial_size=[640, 640],
    ).eval()
    state = {
        k[len("decoder.") :]: v
        for k, v in core.state_dict().items()
        if k.startswith("decoder.")
    }
    upstream.load_state_dict(state, strict=True)
    torch.manual_seed(0)
    image = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        feats = core.encoder(core.backbone(image))
        ours = core.decoder([f.clone() for f in feats])
        ref = upstream([f.clone() for f in feats], None)
    torch.testing.assert_close(
        ours["pred_logits"], ref["pred_logits"], rtol=0, atol=1e-4
    )
    torch.testing.assert_close(
        ours["pred_keypoints"], ref["pred_keypoints"], rtol=0, atol=1e-5
    )
