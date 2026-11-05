"""Convert the SAM 3.1 checkpoint into a Transformers SAM 3 image snapshot.

Builds the ``LibreYOLO/LibreSAM31large`` mirror that ``LibreSAM31`` loads.

SAM 3.1 ships one original-format file, ``sam3.1_multiplex.pt``, and no
Transformers weights. Its image path (detector plus interactive SAM head) has
the SAM 3 architecture with retrained weights; only the key names differ, and
the unused fourth (0.5x) FPN level was dropped. The Object Multiplex video
tracker is new and has no Transformers counterpart, so it is not converted.

Conversion runs in two steps:

1. ``sam31_to_sam3_layout`` renames the 3.1 keys into the public SAM 3
   checkpoint layout and zero-fills the dropped fourth FPN level. That level is
   computed but never read on the image path: ``Sam3Model`` discards the last
   FPN level and ``Sam3TrackerModel`` reads only the first three. Randomizing
   it leaves both models' outputs bit-identical.
2. ``sam3_layout_to_transformers`` maps the SAM 3 layout onto the
   ``Sam3VideoModel`` state-dict names that ``Sam3Model`` and
   ``Sam3TrackerModel`` load from. The key table, the QKV split, the
   position-embedding class-token strip, the CLIP projection transpose and the
   point-embedding concatenation are adapted from the Apache-2.0 Hugging Face
   Transformers converters (Copyright 2025 The Meta AI Authors and The
   HuggingFace Inc. team):

   * src/transformers/models/sam3/convert_sam3_to_hf.py at commit
     22278df3198c4219f033f2a4b0931da3e5c21af4
   * src/transformers/models/sam3_video/convert_sam3_video_to_hf.py at commit
     92d02b508707d6e190e358171c2eece10817f7e1

   restricted to the image path. ``--verify-sam3`` applies step 2 to the SAM 3
   ``sam3.pt`` and requires every image-path tensor of the official
   ``facebook/sam3`` ``model.safetensors`` to match exactly.

Learned values are only renamed, split, concatenated or transposed. The output
remains under Meta's SAM License; the mirror carries that license verbatim.

Usage:
    python weights/convert_sam31_weights.py --output /tmp/LibreSAM31large \
        [--sam31-dir DIR_WITH_PINNED_FILES] \
        [--verify-sam3 sam3.pt sam3_model.safetensors]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable

import torch

SAM31_REPO = "facebook/sam3.1"
SAM31_REVISION = "daa63191845a41281374e725f4c9e51c7a824460"
SAM31_CHECKPOINT = "sam3.1_multiplex.pt"
SAM31_CHECKPOINT_SHA256 = (
    "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
)
SAM3_REPO = "facebook/sam3"
SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
SAM3_CHECKPOINT_SHA256 = (
    "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
)
SAM3_SAFETENSORS_SHA256 = (
    "6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a"
)
TRANSFORMERS_REPO = "https://github.com/huggingface/transformers"
TRANSFORMERS_IMAGE_CONVERTER = (
    "22278df3198c4219f033f2a4b0931da3e5c21af4",
    "src/transformers/models/sam3/convert_sam3_to_hf.py",
)
TRANSFORMERS_VIDEO_CONVERTER = (
    "92d02b508707d6e190e358171c2eece10817f7e1",
    "src/transformers/models/sam3_video/convert_sam3_video_to_hf.py",
)
TARGET_REPO = "LibreYOLO/LibreSAM31large"

# Non-weight files copied byte-for-byte from the pinned facebook/sam3.1
# revision. config.json and processor_config.json are the SAM 3 ones.
SAM31_COPIED_FILES = {
    "LICENSE": "bec48f70bd37bf8280a9d1ebf01642d26086f6122ba735baa08fe03c5a6e7448",
    "config.json": "4616385e4b21f2e5e22c875b65679185cbccfa95de42542b9166f7dc3d57160f",
    "processor_config.json": "6420cf2671fa9309ea95bc0144a8b9861666d1c5f43c8db09e410dacda974fce",
    "special_tokens_map.json": "2cdb3b8331a60c92fc1e55a13e9fd61fd2293c5a51275fdcccd62b780052530e",
    "tokenizer.json": "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35",
    "tokenizer_config.json": "39670ad98457fe8f14ca59f6bb74591e9fc850974380a63993e5b8ffc865baa2",
    "vocab.json": "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363",
    "merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
}

_VB = "detector.backbone.vision_backbone."

# Step 1: SAM 3.1 names -> SAM 3 public names (image path only).
_SAM31_RENAMES = (
    (_VB + "interactive_convs.", _VB + "sam2_convs."),
    ("tracker.model.interactive_sam_mask_decoder.", "tracker.sam_mask_decoder."),
    ("tracker.model.interactive_sam_prompt_encoder.", "tracker.sam_prompt_encoder."),
    ("tracker.model.interactivity_no_mem_embed", "tracker.no_mem_embed"),
)

# SAM 3.1 keys that belong to the multiplex video tracker, not the image path.
_SAM31_VIDEO_ONLY = (
    _VB + "propagation_convs.",
    "tracker.model.",
)

# Shapes of the fourth (0.5x) FPN level that SAM 3.1 dropped. It is a 1x1 and
# a 3x3 conv after a weightless max-pool.
_LEVEL3_SHAPES = {
    "conv_1x1.weight": (256, 1024, 1, 1),
    "conv_1x1.bias": (256,),
    "conv_3x3.weight": (256, 256, 3, 3),
    "conv_3x3.bias": (256,),
}


def is_sam31_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    """True for the SAM 3.1 original checkpoint layout."""
    return any(k.startswith(_VB + "interactive_convs.") for k in state_dict) and any(
        k.startswith("tracker.model.interactive_sam_mask_decoder.") for k in state_dict
    )


def sam31_to_sam3_layout(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Rename the SAM 3.1 image-path keys into the public SAM 3 layout."""
    if not is_sam31_state_dict(state_dict):
        raise ValueError("Not a SAM 3.1 checkpoint (interactive head keys missing).")
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in _SAM31_RENAMES:
            if key.startswith(old):
                new_key = new + key[len(old) :]
                break
        if new_key == key and key.startswith(_SAM31_VIDEO_ONLY):
            continue
        out[new_key] = value
    for neck in ("convs", "sam2_convs"):
        for suffix, shape in _LEVEL3_SHAPES.items():
            key = f"{_VB}{neck}.3.{suffix}"
            if key in out:
                raise ValueError(f"Unexpected SAM 3.1 key {key}")
            out[key] = torch.zeros(shape, dtype=torch.float32)
    return out


# Step 2: SAM 3 public names -> Transformers Sam3VideoModel names.
# Adapted from the Apache-2.0 Transformers SAM 3 converters (see module doc).
# fmt: off
_DETECTOR_PATTERNS = (
    (r"backbone\.vision_backbone\.trunk\.", "vision_encoder.backbone."),
    (r"vision_encoder\.backbone\.pos_embed", "vision_encoder.backbone.embeddings.position_embeddings"),
    (r"vision_encoder\.backbone\.patch_embed\.proj\.", "vision_encoder.backbone.embeddings.patch_embeddings.projection."),
    (r"vision_encoder\.backbone\.ln_pre\.", "vision_encoder.backbone.layer_norm."),
    (r"vision_encoder\.backbone\.blocks\.(\d+)\.norm1\.", r"vision_encoder.backbone.layers.\1.layer_norm1."),
    (r"vision_encoder\.backbone\.blocks\.(\d+)\.norm2\.", r"vision_encoder.backbone.layers.\1.layer_norm2."),
    (r"vision_encoder\.backbone\.blocks\.(\d+)\.attn\.qkv\.", r"vision_encoder.backbone.layers.\1.attention.qkv."),
    (r"vision_encoder\.backbone\.blocks\.(\d+)\.attn\.proj\.", r"vision_encoder.backbone.layers.\1.attention.o_proj."),
    (r"vision_encoder\.backbone\.blocks\.(\d+)\.attn\.freqs_cis", r"vision_encoder.backbone.layers.\1.rotary_emb.rope_embeddings"),
    (r"vision_encoder\.backbone\.blocks\.(\d+)\.mlp\.fc1\.", r"vision_encoder.backbone.layers.\1.mlp.fc1."),
    (r"vision_encoder\.backbone\.blocks\.(\d+)\.mlp\.fc2\.", r"vision_encoder.backbone.layers.\1.mlp.fc2."),
    (r"backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2_0\.", r"vision_encoder.neck.fpn_layers.\1.scale_layers.0."),
    (r"backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2_1\.", r"vision_encoder.neck.fpn_layers.\1.scale_layers.2."),
    (r"backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2\.", r"vision_encoder.neck.fpn_layers.\1.scale_layers.0."),
    (r"backbone\.vision_backbone\.convs\.(\d+)\.conv_1x1\.", r"vision_encoder.neck.fpn_layers.\1.proj1."),
    (r"backbone\.vision_backbone\.convs\.(\d+)\.conv_3x3\.", r"vision_encoder.neck.fpn_layers.\1.proj2."),
    (r"backbone\.vision_backbone\.sam2_convs\.(\d+)\.dconv_2x2_0\.", r"vision_encoder.tracker_neck.fpn_layers.\1.scale_layers.0."),
    (r"backbone\.vision_backbone\.sam2_convs\.(\d+)\.dconv_2x2_1\.", r"vision_encoder.tracker_neck.fpn_layers.\1.scale_layers.2."),
    (r"backbone\.vision_backbone\.sam2_convs\.(\d+)\.dconv_2x2\.", r"vision_encoder.tracker_neck.fpn_layers.\1.scale_layers.0."),
    (r"backbone\.vision_backbone\.sam2_convs\.(\d+)\.conv_1x1\.", r"vision_encoder.tracker_neck.fpn_layers.\1.proj1."),
    (r"backbone\.vision_backbone\.sam2_convs\.(\d+)\.conv_3x3\.", r"vision_encoder.tracker_neck.fpn_layers.\1.proj2."),
    (r"backbone\.language_backbone\.encoder\.", "text_encoder."),
    (r"text_encoder\.token_embedding\.", "text_encoder.text_model.embeddings.token_embedding."),
    (r"text_encoder\.positional_embedding", "text_encoder.text_model.embeddings.position_embedding.weight"),
    (r"text_encoder\.ln_final\.", "text_encoder.text_model.final_layer_norm."),
    (r"text_encoder\.text_projection", "text_encoder.text_projection.weight"),
    (r"text_encoder\.transformer\.resblocks\.(\d+)\.attn\.in_proj_", r"text_encoder.text_model.encoder.layers.\1.self_attn.in_proj_"),
    (r"text_encoder\.transformer\.resblocks\.(\d+)\.attn\.out_proj\.", r"text_encoder.text_model.encoder.layers.\1.self_attn.out_proj."),
    (r"text_encoder\.transformer\.resblocks\.(\d+)\.ln_1\.", r"text_encoder.text_model.encoder.layers.\1.layer_norm1."),
    (r"text_encoder\.transformer\.resblocks\.(\d+)\.ln_2\.", r"text_encoder.text_model.encoder.layers.\1.layer_norm2."),
    (r"text_encoder\.transformer\.resblocks\.(\d+)\.mlp\.c_fc\.", r"text_encoder.text_model.encoder.layers.\1.mlp.fc1."),
    (r"text_encoder\.transformer\.resblocks\.(\d+)\.mlp\.c_proj\.", r"text_encoder.text_model.encoder.layers.\1.mlp.fc2."),
    (r"backbone\.language_backbone\.resizer\.", "text_projection."),
    (r"geometry_encoder\.encode\.(\d+)\.cross_attn_image\.out_proj\.", r"geometry_encoder.layers.\1.cross_attn.o_proj."),
    (r"geometry_encoder\.encode\.(\d+)\.cross_attn_image\.", r"geometry_encoder.layers.\1.cross_attn."),
    (r"geometry_encoder\.encode\.(\d+)\.self_attn\.out_proj\.", r"geometry_encoder.layers.\1.self_attn.o_proj."),
    (r"geometry_encoder\.encode\.(\d+)\.self_attn\.", r"geometry_encoder.layers.\1.self_attn."),
    (r"geometry_encoder\.encode\.(\d+)\.linear1\.", r"geometry_encoder.layers.\1.mlp.fc1."),
    (r"geometry_encoder\.encode\.(\d+)\.linear2\.", r"geometry_encoder.layers.\1.mlp.fc2."),
    (r"geometry_encoder\.encode\.(\d+)\.norm1\.", r"geometry_encoder.layers.\1.layer_norm1."),
    (r"geometry_encoder\.encode\.(\d+)\.norm2\.", r"geometry_encoder.layers.\1.layer_norm2."),
    (r"geometry_encoder\.encode\.(\d+)\.norm3\.", r"geometry_encoder.layers.\1.layer_norm3."),
    (r"geometry_encoder\.img_pre_norm\.", "geometry_encoder.vision_layer_norm."),
    (r"geometry_encoder\.norm\.", "geometry_encoder.prompt_layer_norm."),
    (r"geometry_encoder\.encode_norm\.", "geometry_encoder.output_layer_norm."),
    (r"transformer\.encoder\.layers\.(\d+)\.cross_attn_image\.out_proj\.", r"detr_encoder.layers.\1.cross_attn.o_proj."),
    (r"transformer\.encoder\.layers\.(\d+)\.cross_attn_image\.", r"detr_encoder.layers.\1.cross_attn."),
    (r"transformer\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.", r"detr_encoder.layers.\1.self_attn.o_proj."),
    (r"transformer\.encoder\.layers\.(\d+)\.self_attn\.", r"detr_encoder.layers.\1.self_attn."),
    (r"transformer\.encoder\.layers\.(\d+)\.cross_attn\.out_proj\.", r"detr_encoder.layers.\1.cross_attn.o_proj."),
    (r"transformer\.encoder\.layers\.(\d+)\.cross_attn\.", r"detr_encoder.layers.\1.cross_attn."),
    (r"transformer\.encoder\.layers\.(\d+)\.linear1\.", r"detr_encoder.layers.\1.mlp.fc1."),
    (r"transformer\.encoder\.layers\.(\d+)\.linear2\.", r"detr_encoder.layers.\1.mlp.fc2."),
    (r"transformer\.encoder\.layers\.(\d+)\.norm1\.", r"detr_encoder.layers.\1.layer_norm1."),
    (r"transformer\.encoder\.layers\.(\d+)\.norm2\.", r"detr_encoder.layers.\1.layer_norm2."),
    (r"transformer\.encoder\.layers\.(\d+)\.norm3\.", r"detr_encoder.layers.\1.layer_norm3."),
    (r"transformer\.decoder\.query_embed\.", "detr_decoder.query_embed."),
    (r"transformer\.decoder\.reference_points\.", "detr_decoder.reference_points."),
    (r"transformer\.decoder\.instance_query_embed\.", "detr_decoder.instance_query_embed."),
    (r"transformer\.decoder\.instance_reference_points\.", "detr_decoder.instance_reference_points."),
    (r"transformer\.decoder\.presence_token\.", "detr_decoder.presence_token."),
    (r"transformer\.decoder\.presence_token_head\.layers\.0\.", "detr_decoder.presence_head.layer1."),
    (r"transformer\.decoder\.presence_token_head\.layers\.1\.", "detr_decoder.presence_head.layer2."),
    (r"transformer\.decoder\.presence_token_head\.layers\.2\.", "detr_decoder.presence_head.layer3."),
    (r"transformer\.decoder\.presence_token_out_norm\.", "detr_decoder.presence_layer_norm."),
    (r"transformer\.decoder\.norm\.", "detr_decoder.output_layer_norm."),
    (r"transformer\.decoder\.bbox_embed\.layers\.0\.", "detr_decoder.box_head.layer1."),
    (r"transformer\.decoder\.bbox_embed\.layers\.1\.", "detr_decoder.box_head.layer2."),
    (r"transformer\.decoder\.bbox_embed\.layers\.2\.", "detr_decoder.box_head.layer3."),
    (r"transformer\.decoder\.instance_bbox_embed\.layers\.0\.", "detr_decoder.instance_box_head.layer1."),
    (r"transformer\.decoder\.instance_bbox_embed\.layers\.1\.", "detr_decoder.instance_box_head.layer2."),
    (r"transformer\.decoder\.instance_bbox_embed\.layers\.2\.", "detr_decoder.instance_box_head.layer3."),
    (r"transformer\.decoder\.ref_point_head\.layers\.0\.", "detr_decoder.ref_point_head.layer1."),
    (r"transformer\.decoder\.ref_point_head\.layers\.1\.", "detr_decoder.ref_point_head.layer2."),
    (r"transformer\.decoder\.boxRPB_embed_x\.layers\.0\.", "detr_decoder.box_rpb_embed_x.layer1."),
    (r"transformer\.decoder\.boxRPB_embed_x\.layers\.1\.", "detr_decoder.box_rpb_embed_x.layer2."),
    (r"transformer\.decoder\.boxRPB_embed_y\.layers\.0\.", "detr_decoder.box_rpb_embed_y.layer1."),
    (r"transformer\.decoder\.boxRPB_embed_y\.layers\.1\.", "detr_decoder.box_rpb_embed_y.layer2."),
    (r"transformer\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.", r"detr_decoder.layers.\1.self_attn.o_proj."),
    (r"transformer\.decoder\.layers\.(\d+)\.self_attn\.", r"detr_decoder.layers.\1.self_attn."),
    (r"transformer\.decoder\.layers\.(\d+)\.ca_text\.out_proj\.", r"detr_decoder.layers.\1.text_cross_attn.o_proj."),
    (r"transformer\.decoder\.layers\.(\d+)\.ca_text\.", r"detr_decoder.layers.\1.text_cross_attn."),
    (r"transformer\.decoder\.layers\.(\d+)\.cross_attn\.out_proj\.", r"detr_decoder.layers.\1.vision_cross_attn.o_proj."),
    (r"transformer\.decoder\.layers\.(\d+)\.cross_attn\.", r"detr_decoder.layers.\1.vision_cross_attn."),
    (r"transformer\.decoder\.layers\.(\d+)\.linear1\.", r"detr_decoder.layers.\1.mlp.fc1."),
    (r"transformer\.decoder\.layers\.(\d+)\.linear2\.", r"detr_decoder.layers.\1.mlp.fc2."),
    (r"transformer\.decoder\.layers\.(\d+)\.norm1\.", r"detr_decoder.layers.\1.vision_cross_attn_layer_norm."),
    (r"transformer\.decoder\.layers\.(\d+)\.catext_norm\.", r"detr_decoder.layers.\1.text_cross_attn_layer_norm."),
    (r"transformer\.decoder\.layers\.(\d+)\.norm2\.", r"detr_decoder.layers.\1.self_attn_layer_norm."),
    (r"transformer\.decoder\.layers\.(\d+)\.norm3\.", r"detr_decoder.layers.\1.mlp_layer_norm."),
    (r"dot_prod_scoring\.prompt_mlp\.layers\.0\.", "dot_product_scoring.text_mlp.layer1."),
    (r"dot_prod_scoring\.prompt_mlp\.layers\.1\.", "dot_product_scoring.text_mlp.layer2."),
    (r"dot_prod_scoring\.prompt_mlp\.out_norm\.", "dot_product_scoring.text_mlp_out_norm."),
    (r"dot_prod_scoring\.prompt_proj\.", "dot_product_scoring.text_proj."),
    (r"dot_prod_scoring\.hs_proj\.", "dot_product_scoring.query_proj."),
    (r"segmentation_head\.pixel_decoder\.conv_layers\.(\d+)\.", r"mask_decoder.pixel_decoder.conv_layers.\1."),
    (r"segmentation_head\.pixel_decoder\.norms\.(\d+)\.", r"mask_decoder.pixel_decoder.norms.\1."),
    (r"segmentation_head\.mask_predictor\.mask_embed\.layers\.(\d+)\.", r"mask_decoder.mask_embedder.layers.\1."),
    (r"segmentation_head\.mask_embed\.layers\.(\d+)\.", r"mask_decoder.mask_embedder.layers.\1."),
    (r"segmentation_head\.instance_seg_head\.", "mask_decoder.instance_projection."),
    (r"segmentation_head\.semantic_seg_head\.", "mask_decoder.semantic_projection."),
    (r"segmentation_head\.cross_attend_prompt\.out_proj\.", "mask_decoder.prompt_cross_attn.o_proj."),
    (r"segmentation_head\.cross_attend_prompt\.", "mask_decoder.prompt_cross_attn."),
    (r"segmentation_head\.cross_attn_norm\.", "mask_decoder.prompt_cross_attn_norm."),
)

# Interactive SAM head (prompt encoder + mask decoder) of the tracker.
_TRACKER_PATTERNS = (
    (r"^sam_prompt_encoder\.", "prompt_encoder."),
    (r"^sam_mask_decoder\.", "mask_decoder."),
    (r"\.iou_prediction_head\.layers\.0\.", ".iou_prediction_head.proj_in."),
    (r"\.iou_prediction_head\.layers\.1\.", ".iou_prediction_head.layers.0."),
    (r"\.iou_prediction_head\.layers\.2\.", ".iou_prediction_head.proj_out."),
    (r"\.pred_obj_score_head\.layers\.0\.", ".pred_obj_score_head.proj_in."),
    (r"\.pred_obj_score_head\.layers\.1\.", ".pred_obj_score_head.layers.0."),
    (r"\.pred_obj_score_head\.layers\.2\.", ".pred_obj_score_head.proj_out."),
    (r"\.output_hypernetworks_mlps\.(\d+)\.layers\.0\.", r".output_hypernetworks_mlps.\1.proj_in."),
    (r"\.output_hypernetworks_mlps\.(\d+)\.layers\.1\.", r".output_hypernetworks_mlps.\1.layers.0."),
    (r"\.output_hypernetworks_mlps\.(\d+)\.layers\.2\.", r".output_hypernetworks_mlps.\1.proj_out."),
    (r"\.output_upscaling\.0\.", ".upscale_conv1."),
    (r"\.output_upscaling\.1\.", ".upscale_layer_norm."),
    (r"\.output_upscaling\.3\.", ".upscale_conv2."),
    (r"\.mask_downscaling\.0\.", ".mask_embed.conv1."),
    (r"\.mask_downscaling\.1\.", ".mask_embed.layer_norm1."),
    (r"\.mask_downscaling\.3\.", ".mask_embed.conv2."),
    (r"\.mask_downscaling\.4\.", ".mask_embed.layer_norm2."),
    (r"\.mask_downscaling\.6\.", ".mask_embed.conv3."),
    (r"\.point_embeddings\.", ".point_embed."),
    (r"\.pe_layer\.positional_encoding_gaussian_matrix$", ".shared_embedding.positional_embedding"),
    (r"\.mlp\.lin1\.", ".mlp.proj_in."),
    (r"\.mlp\.lin2\.", ".mlp.proj_out."),
    (r"\.norm(\d)\.", r".layer_norm\1."),
    (r"\.norm_final_attn\.", ".layer_norm_final_attn."),
    (r"\.out_proj\.", ".o_proj."),
)
# fmt: on

_TRACKER_HEADS = ("sam_prompt_encoder.", "sam_mask_decoder.")


def _apply(patterns: Iterable[tuple], key: str) -> str:
    for pattern, replacement in patterns:
        key = re.sub(pattern, replacement, key)
    return key


def _split_qkv(state_dict: Dict[str, torch.Tensor]) -> None:
    for key in [k for k in state_dict if ".attention.qkv." in k]:
        q, k, v = torch.chunk(state_dict.pop(key), 3, dim=0)
        state_dict[key.replace(".qkv.", ".q_proj.")] = q
        state_dict[key.replace(".qkv.", ".k_proj.")] = k
        state_dict[key.replace(".qkv.", ".v_proj.")] = v
    for key in [k for k in state_dict if ".in_proj_" in k]:
        q, k, v = torch.chunk(state_dict.pop(key), 3, dim=0)
        kind = "weight" if key.endswith("in_proj_weight") else "bias"
        base = key[: key.rindex("in_proj_")]
        state_dict[f"{base}q_proj.{kind}"] = q
        state_dict[f"{base}k_proj.{kind}"] = k
        state_dict[f"{base}v_proj.{kind}"] = v


def sam3_layout_to_transformers(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Map SAM 3 public checkpoint keys to the Transformers image-path keys.

    Only the image path is produced: the detector (``detector_model.*``), the
    tracker FPN neck (``tracker_neck.*``) and the tracker's interactive SAM
    head, ``no_memory_embedding`` and ``shared_image_embedding``. Video memory
    weights are skipped.
    """
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("detector."):
            new_key = _apply(_DETECTOR_PATTERNS, key[len("detector.") :])
            if new_key.startswith("vision_encoder.tracker_neck."):
                new_key = new_key[len("vision_encoder.") :]
            else:
                new_key = "detector_model." + new_key
            if new_key.endswith(".rope_embeddings"):
                continue  # recomputed by Transformers
            if ".geometry_encoder.points_" in new_key:
                continue  # point-prompt projections absent from Sam3Model
            if new_key == "detector_model.vision_encoder.backbone.embeddings.position_embeddings":
                value = value[:, 1:, :]  # drop the class-token position
            if new_key == "detector_model.text_encoder.text_projection.weight":
                value = value.T
            out[new_key] = value
        elif key.startswith("tracker."):
            sub = key[len("tracker.") :]
            if sub == "no_mem_embed":
                out["tracker_model.no_memory_embedding"] = value
            elif sub.startswith(_TRACKER_HEADS):
                out["tracker_model." + _apply(_TRACKER_PATTERNS, sub)] = value
    _split_qkv(out)
    point_keys = [f"tracker_model.prompt_encoder.point_embed.{i}.weight" for i in range(4)]
    out["tracker_model.prompt_encoder.point_embed.weight"] = torch.cat(
        [out.pop(k) for k in point_keys], dim=0
    )
    out["tracker_model.shared_image_embedding.positional_embedding"] = out[
        "tracker_model.prompt_encoder.shared_embedding.positional_embedding"
    ].clone()
    return {k: v.contiguous() for k, v in out.items()}


def load_original_checkpoint(path: str | Path) -> Dict[str, torch.Tensor]:
    """Safely load an original-format SAM 3 / 3.1 ``.pt`` as a tensor dict."""
    checkpoint = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint structure in {path}")
    return checkpoint


def convert_sam31_state_dict(src: str | Path) -> Dict[str, torch.Tensor]:
    """``sam3.1_multiplex.pt`` -> Transformers image-path state dict."""
    return sam3_layout_to_transformers(sam31_to_sam3_layout(load_original_checkpoint(src)))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: str | Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


IMAGE_PATH_PREFIXES = (
    "detector_model.",
    "tracker_neck.",
    "tracker_model.prompt_encoder.",
    "tracker_model.mask_decoder.",
    "tracker_model.no_memory_embedding",
    "tracker_model.shared_image_embedding.",
)


def verify_sam3_mapping(sam3_pt: str | Path, sam3_safetensors: str | Path) -> int:
    """Check step 2 on SAM 3 against the official Transformers weights.

    Returns the number of image-path tensors that matched exactly.
    """
    from safetensors import safe_open

    require_sha256(sam3_pt, SAM3_CHECKPOINT_SHA256)
    require_sha256(sam3_safetensors, SAM3_SAFETENSORS_SHA256)
    converted = sam3_layout_to_transformers(load_original_checkpoint(sam3_pt))
    with safe_open(str(sam3_safetensors), "pt") as reference:
        expected = {k for k in reference.keys() if k.startswith(IMAGE_PATH_PREFIXES)}
        if set(converted) != expected:
            raise ValueError(
                "Key sets differ: "
                f"extra={sorted(set(converted) - expected)[:5]} "
                f"missing={sorted(expected - set(converted))[:5]}"
            )
        bad = [k for k in expected if not torch.equal(converted[k], reference.get_tensor(k))]
    if bad:
        raise ValueError(f"{len(bad)} tensors differ, e.g. {bad[:5]}")
    return len(expected)


def strict_load(directory: str | Path) -> None:
    """Load the snapshot into both image models with no missing/unexpected keys."""
    from transformers import Sam3Model, Sam3TrackerModel

    for cls in (Sam3Model, Sam3TrackerModel):
        _, info = cls.from_pretrained(str(directory), output_loading_info=True)
        problems = {k: v for k, v in info.items() if k != "error_msgs" and v}
        if problems:
            raise ValueError(f"{cls.__name__} did not load strictly: {problems}")


def _render_readme(model_sha256: str, tensor_count: int) -> str:
    return f"""---
license: other
license_name: sam-license
license_link: LICENSE
extra_gated_prompt: >-
  These weights are redistributed under Meta's SAM License, a copy of which is
  included in this repository as LICENSE. By requesting access you agree to be
  bound by it. Note in particular that the license prohibits use for military
  or warfare purposes, nuclear applications, espionage, and the development or
  use of guns or illegal weapons, and that your use must comply with applicable
  Trade Controls. Access is not available in comprehensively sanctioned
  jurisdictions.
extra_gated_fields:
  Name: text
  Affiliation: text
  Country: country
  I accept the terms of the SAM License: checkbox
  I confirm I am not located in a comprehensively sanctioned jurisdiction: checkbox
library_name: transformers
pipeline_tag: mask-generation
base_model: {SAM31_REPO}
tags:
- libreyolo
- sam3.1
- promptable-segmentation
- image-segmentation
---

# LibreSAM31large

The image path of Meta's [SAM 3.1](https://huggingface.co/{SAM31_REPO})
(detector plus interactive SAM head), converted to the Transformers SAM 3
layout for LibreYOLO's `LibreSAM("sam3.1")`. Meta publishes SAM 3.1 only as
an original-format checkpoint; this snapshot loads with the stock
`Sam3Model` (text prompts) and `Sam3TrackerModel` (point and box prompts).

## License: read this before requesting access

These weights are **not** MIT and are not covered by LibreYOLO's license. They
are a derivative of Meta's SAM 3.1 checkpoint, redistributed under Meta's
**SAM License**, included verbatim as [`LICENSE`](LICENSE). Anyone who
redistributes them further must do so under the same terms and include the
agreement. The license bars military or warfare use, nuclear applications,
espionage, and guns or illegal weapons; use must comply with Trade Controls;
reverse engineering is not allowed; publications must acknowledge SAM.

Meta gates the upstream repository with manual approval. This mirror uses
Hugging Face's automatic gate: users accept the SAM License and submit the
fields above, without manual review.

## Source

- [`{SAM31_REPO}`](https://huggingface.co/{SAM31_REPO}) at revision
  `{SAM31_REVISION}`, file `{SAM31_CHECKPOINT}`
  (SHA-256 `{SAM31_CHECKPOINT_SHA256}`)

## Modifications

- Only the image path is kept. The Object Multiplex video tracker weights are
  not included: Transformers has no implementation for them.
- Keys are renamed to the SAM 3 layout, query/key/value projections are split,
  the class-token position embedding is dropped, the CLIP text projection is
  transposed and the point embeddings are concatenated. The mapping is adapted
  from the Apache-2.0 Transformers SAM 3 converters and reproduces the official
  `{SAM3_REPO}` weights exactly when applied to SAM 3.
- SAM 3.1 dropped the fourth (0.5x) FPN level, which the image path never
  reads. It is filled with zeros so the stock SAM 3 configuration loads.
- No learned value was changed. {tensor_count} tensors.
- `config.json`, `processor_config.json`, the tokenizer files and `LICENSE` are
  copied byte-for-byte from the pinned upstream revision.

`model.safetensors` SHA-256: `{model_sha256}`

## Usage

```python
from libreyolo import LibreSAM

model = LibreSAM("sam3.1")
model.predict("image.jpg", text="person")
model.predict("image.jpg", points=[500, 375], labels=[1])
```
"""


def _render_notice(model_sha256: str, tensor_count: int) -> str:
    return f"""LibreSAM31large provenance notice

Upstream weights
----------------
Repository: https://huggingface.co/{SAM31_REPO}
Revision: {SAM31_REVISION}
File: {SAM31_CHECKPOINT}
SHA-256: {SAM31_CHECKPOINT_SHA256}
License: SAM License (Meta Platforms, Inc.), included verbatim as LICENSE

Conversion implementation
-------------------------
LibreYOLO weights/convert_sam31_weights.py. The SAM 3 to Transformers key
mapping is adapted from {TRANSFORMERS_REPO}:
  {TRANSFORMERS_IMAGE_CONVERTER[1]} at commit {TRANSFORMERS_IMAGE_CONVERTER[0]}
  {TRANSFORMERS_VIDEO_CONVERTER[1]} at commit {TRANSFORMERS_VIDEO_CONVERTER[0]}
Copyright 2025 The Meta AI Authors and The HuggingFace Inc. team
License: Apache License 2.0
No Meta SAM source code was copied or adapted.

LibreYOLO modifications
-----------------------
Image path only (detector and interactive SAM head); the Object Multiplex video
tracker weights are omitted. Keys renamed, QKV projections split, class-token
position embedding dropped, CLIP text projection transposed, point embeddings
concatenated; the dropped fourth FPN level is zero-filled and unused. The
mapping reproduces all image-path tensors of {SAM3_REPO} revision
{SAM3_REVISION} exactly when applied to SAM 3. {tensor_count} tensors.
The non-weight files config.json, processor_config.json, tokenizer files and
LICENSE are copied byte-for-byte from the upstream revision above.

Generated model.safetensors SHA-256: {model_sha256}
"""


def write_payload(output: str | Path, sam31_dir: str | Path) -> Path:
    """Convert and write the complete mirror payload into a new directory."""
    from safetensors.torch import save_file

    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    source = Path(sam31_dir)
    require_sha256(source / SAM31_CHECKPOINT, SAM31_CHECKPOINT_SHA256)
    for name, digest in SAM31_COPIED_FILES.items():
        require_sha256(source / name, digest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(destination.parent)) as tmp:
        staging = Path(tmp) / destination.name
        staging.mkdir()
        state = convert_sam31_state_dict(source / SAM31_CHECKPOINT)
        save_file(state, str(staging / "model.safetensors"), metadata={"format": "pt"})
        for name in SAM31_COPIED_FILES:
            shutil.copyfile(source / name, staging / name)
        model_sha = sha256_file(staging / "model.safetensors")
        (staging / "README.md").write_text(_render_readme(model_sha, len(state)), encoding="utf-8")
        (staging / "NOTICE").write_text(_render_notice(model_sha, len(state)), encoding="utf-8")
        (staging / ".gitattributes").write_text(
            "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
            "tokenizer.json filter=lfs diff=lfs merge=lfs -text\n",
            encoding="utf-8",
        )
        strict_load(staging)
        staging.rename(destination)
    return destination


def download_sam31(cache_dir: str | Path | None = None) -> Path:
    """Fetch the pinned upstream files (gated; needs accepted terms and a token)."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            SAM31_REPO,
            revision=SAM31_REVISION,
            allow_patterns=[SAM31_CHECKPOINT, *SAM31_COPIED_FILES],
            cache_dir=cache_dir,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", required=True, help="new directory for the mirror")
    parser.add_argument("--sam31-dir", help="directory holding the pinned upstream files")
    parser.add_argument(
        "--verify-sam3",
        nargs=2,
        metavar=("SAM3_PT", "SAM3_SAFETENSORS"),
        help="first check the key mapping against the official SAM 3 weights",
    )
    args = parser.parse_args()
    if args.verify_sam3:
        count = verify_sam3_mapping(*args.verify_sam3)
        print(f"SAM 3 mapping check: {count} image-path tensors match exactly")
    source = Path(args.sam31_dir) if args.sam31_dir else download_sam31()
    print(f"Wrote {write_payload(args.output, source)}")


if __name__ == "__main__":
    main()
