"""Grounding DINO Swin backbone parity (transformers-order native Swin)."""

from __future__ import annotations

import torch
from PIL import Image

from libreyolo.models.swin.nn import SwinBackbone, SwinDims

REPO = "LibreYOLO/LibreGroundingDINOt"


def remap(sd):
    new = {}
    for i in range(4):
        for j in range(6):
            p = f"swin.encoder.layers.{i}.blocks.{j}."
            if p + "attention.q_proj.weight" not in sd:
                continue
            b = f"layers.{i}.blocks.{j}."
            new[b + "norm1.weight"] = sd[p + "layernorm_before.weight"]
            new[b + "norm1.bias"] = sd[p + "layernorm_before.bias"]
            new[b + "norm2.weight"] = sd[p + "layernorm_after.weight"]
            new[b + "norm2.bias"] = sd[p + "layernorm_after.bias"]
            new[b + "attn.qkv.weight"] = torch.cat(
                [sd[p + "attention.q_proj.weight"], sd[p + "attention.k_proj.weight"], sd[p + "attention.v_proj.weight"]], 0)
            new[b + "attn.qkv.bias"] = torch.cat(
                [sd[p + "attention.q_proj.bias"], sd[p + "attention.k_proj.bias"], sd[p + "attention.v_proj.bias"]], 0)
            new[b + "attn.proj.weight"] = sd[p + "attention.o_proj.weight"]
            new[b + "attn.proj.bias"] = sd[p + "attention.o_proj.bias"]
            new[b + "attn.relative_position_bias_table"] = sd[p + "attention.relative_position_bias.relative_position_bias_table"]
            new[b + "mlp.fc1.weight"] = sd[p + "mlp.fc1.weight"]
            new[b + "mlp.fc1.bias"] = sd[p + "mlp.fc1.bias"]
            new[b + "mlp.fc2.weight"] = sd[p + "mlp.fc2.weight"]
            new[b + "mlp.fc2.bias"] = sd[p + "mlp.fc2.bias"]
    new["patch_embed.proj.weight"] = sd["swin.embeddings.patch_embeddings.projection.weight"]
    new["patch_embed.proj.bias"] = sd["swin.embeddings.patch_embeddings.projection.bias"]
    new["patch_embed.norm.weight"] = sd["swin.embeddings.norm.weight"]
    new["patch_embed.norm.bias"] = sd["swin.embeddings.norm.bias"]
    for i in range(3):
        new[f"layers.{i+1}.downsample.norm.weight"] = sd[f"swin.encoder.layers.{i}.downsample.norm.weight"]
        new[f"layers.{i+1}.downsample.norm.bias"] = sd[f"swin.encoder.layers.{i}.downsample.norm.bias"]
        new[f"layers.{i+1}.downsample.reduction.weight"] = sd[f"swin.encoder.layers.{i}.downsample.reduction.weight"]
    return new


def main():
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    proc = AutoProcessor.from_pretrained(REPO)
    gd = AutoModelForZeroShotObjectDetection.from_pretrained(REPO, attn_implementation="eager").eval()
    bmod = gd.model.backbone.conv_encoder.model
    # force eager on the swin backbone sub-model (top-level flag may not propagate)
    for m in bmod.modules():
        if hasattr(m, "config"):
            m.config._attn_implementation = "eager"
    sd = {k.split("conv_encoder.model.")[1]: v for k, v in gd.state_dict().items() if "backbone.conv_encoder.model" in k}

    native = SwinBackbone(SwinDims(out_indices=(1, 2, 3), tf_order=True)).eval()
    native.load_state_dict(remap(sd), strict=False)

    img = Image.open("tests/data/parkour.jpg").convert("RGB")
    px = proc(images=img, text="a person.", return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        ref = bmod(px).feature_maps
        feats = native(px)
        ok = True
        for idx, st in enumerate([2, 3, 4]):
            w = sd[f"hidden_states_norms.stage{st}.weight"]
            bb = sd[f"hidden_states_norms.stage{st}.bias"]
            o = torch.nn.functional.layer_norm(feats[idx], (feats[idx].shape[-1],), w, bb).permute(0, 3, 1, 2)
            diff = (ref[idx] - o).abs().max().item()
            print(f"stage{st}: shape={tuple(o.shape)}  max_abs_diff={diff:.3e}  {'OK' if diff < 1e-4 else 'FAIL'}")
            ok = ok and diff < 1e-4
    print("\nGD Swin native parity:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
