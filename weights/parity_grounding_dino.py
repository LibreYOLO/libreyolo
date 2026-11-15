"""Grounding DINO native-port parity vs transformers (logits + pred_boxes)."""

from __future__ import annotations

import torch
from PIL import Image

from libreyolo.models.grounding_dino.nn import GroundingDinoDetectionModel

REPO = "LibreYOLO/LibreGroundingDINOt"


def _swin_remap(sd, dst_prefix):
    """transformers swin.* -> native timm-style backbone keys."""
    out = {}
    for i in range(4):
        for j in range(6):
            p = f"swin.encoder.layers.{i}.blocks.{j}."
            if p + "attention.q_proj.weight" not in sd:
                continue
            b = f"{dst_prefix}layers.{i}.blocks.{j}."
            out[b + "norm1.weight"] = sd[p + "layernorm_before.weight"]
            out[b + "norm1.bias"] = sd[p + "layernorm_before.bias"]
            out[b + "norm2.weight"] = sd[p + "layernorm_after.weight"]
            out[b + "norm2.bias"] = sd[p + "layernorm_after.bias"]
            out[b + "attn.qkv.weight"] = torch.cat(
                [sd[p + "attention.q_proj.weight"], sd[p + "attention.k_proj.weight"], sd[p + "attention.v_proj.weight"]], 0)
            out[b + "attn.qkv.bias"] = torch.cat(
                [sd[p + "attention.q_proj.bias"], sd[p + "attention.k_proj.bias"], sd[p + "attention.v_proj.bias"]], 0)
            out[b + "attn.proj.weight"] = sd[p + "attention.o_proj.weight"]
            out[b + "attn.proj.bias"] = sd[p + "attention.o_proj.bias"]
            out[b + "attn.relative_position_bias_table"] = sd[p + "attention.relative_position_bias.relative_position_bias_table"]
            out[b + "mlp.fc1.weight"] = sd[p + "mlp.fc1.weight"]
            out[b + "mlp.fc1.bias"] = sd[p + "mlp.fc1.bias"]
            out[b + "mlp.fc2.weight"] = sd[p + "mlp.fc2.weight"]
            out[b + "mlp.fc2.bias"] = sd[p + "mlp.fc2.bias"]
    out[dst_prefix + "patch_embed.proj.weight"] = sd["swin.embeddings.patch_embeddings.projection.weight"]
    out[dst_prefix + "patch_embed.proj.bias"] = sd["swin.embeddings.patch_embeddings.projection.bias"]
    out[dst_prefix + "patch_embed.norm.weight"] = sd["swin.embeddings.norm.weight"]
    out[dst_prefix + "patch_embed.norm.bias"] = sd["swin.embeddings.norm.bias"]
    for i in range(3):
        out[f"{dst_prefix}layers.{i+1}.downsample.norm.weight"] = sd[f"swin.encoder.layers.{i}.downsample.norm.weight"]
        out[f"{dst_prefix}layers.{i+1}.downsample.norm.bias"] = sd[f"swin.encoder.layers.{i}.downsample.norm.bias"]
        out[f"{dst_prefix}layers.{i+1}.downsample.reduction.weight"] = sd[f"swin.encoder.layers.{i}.downsample.reduction.weight"]
    return out


def remap(full):
    out = {}
    swin_src = {}
    for k, v in full.items():
        bp = "model.backbone.conv_encoder.model."
        if k.startswith(bp + "swin."):
            swin_src[k[len(bp):]] = v
            continue
        if k.startswith(bp + "hidden_states_norms.stage"):
            st = int(k[len(bp + "hidden_states_norms.stage")])  # 2,3,4
            rest = k.split(f"stage{st}.")[1]
            out[f"backbone_conv.hidden_states_norms.{st-2}.{rest}"] = v
            continue
        if k.startswith("model.decoder.bbox_embed."):
            out[k[len("model."):]] = v  # decoder.bbox_embed.* (tied to bbox_embed)
            continue
        if k.startswith("model.decoder.class_embed") or k.startswith("class_embed"):
            continue  # no params
        if k.startswith("model.backbone.position_embedding"):
            continue
        if k.startswith("model."):
            out[k[len("model."):]] = v
        else:
            out[k] = v  # bbox_embed.*
    out.update(_swin_remap(swin_src, "backbone_conv._backbone."))
    return out


def main():
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    proc = AutoProcessor.from_pretrained(REPO)
    gd = AutoModelForZeroShotObjectDetection.from_pretrained(REPO, attn_implementation="eager").eval()
    for m in gd.modules():
        if hasattr(m, "config"):
            m.config._attn_implementation = "eager"

    native = GroundingDinoDetectionModel(gd.config).eval()
    res = native.load_state_dict(remap(gd.state_dict()), strict=False)
    ign = ("relative_position_index", "num_batches_tracked")
    miss = [k for k in res.missing_keys if not any(s in k for s in ign)]
    unexp = [k for k in res.unexpected_keys if not any(s in k for s in ign)]
    print("missing:", miss[:12], f"(+{len(miss)-12})" if len(miss) > 12 else "")
    print("unexpected:", unexp[:12], f"(+{len(unexp)-12})" if len(unexp) > 12 else "")
    if miss or unexp:
        raise SystemExit("state_dict key mismatch")

    img = Image.open("tests/data/parkour.jpg").convert("RGB")
    inputs = proc(images=img, text="a person. a dog. a skateboard.", return_tensors="pt")
    with torch.no_grad():
        ref = gd(**inputs)
        out = native(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
                     token_type_ids=inputs.get("token_type_ids"), attention_mask=inputs["attention_mask"],
                     pixel_mask=inputs.get("pixel_mask"))

    ok = True
    for key in ("logits", "pred_boxes"):
        r, o = getattr(ref, key), out[key]
        finite = torch.isfinite(r) & torch.isfinite(o)  # logits have -inf padding columns
        diff = (r[finite] - o[finite]).abs().max().item()
        passed = diff < 1e-3
        ok = ok and passed
        print(f"{key:12s} shape={tuple(o.shape)}  max_abs_diff={diff:.3e}  {'OK' if passed else 'FAIL'}")
    print("\nGrounding DINO native parity:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
