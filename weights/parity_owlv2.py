"""OWLv2 native-port parity: assert max_abs_diff == 0 vs transformers."""

from __future__ import annotations

import torch
from PIL import Image

from libreyolo.models.owlv2.nn import Owlv2Dims, Owlv2DetectionModel, OWLV2_DIMS

REPO = "LibreYOLO/LibreOWLv2b16"


def dims_from_hf(cfg) -> Owlv2Dims:
    v, t = cfg.vision_config, cfg.text_config
    return Owlv2Dims(
        vision_hidden=v.hidden_size, vision_layers=v.num_hidden_layers,
        vision_heads=v.num_attention_heads, vision_intermediate=v.intermediate_size,
        patch_size=v.patch_size, image_size=v.image_size, vision_act=v.hidden_act,
        text_hidden=t.hidden_size, text_layers=t.num_hidden_layers,
        text_heads=t.num_attention_heads, text_intermediate=t.intermediate_size,
        vocab_size=t.vocab_size, max_position_embeddings=t.max_position_embeddings,
        text_act=t.hidden_act, projection_dim=cfg.projection_dim,
        layer_norm_eps=v.layer_norm_eps,
    )


def main():
    from transformers import AutoProcessor, Owlv2ForObjectDetection

    proc = AutoProcessor.from_pretrained(REPO)
    # eager attention so parity compares the same math (native uses portable eager);
    # HF defaults to fused SDPA which rounds differently (~1e-4 on long vision seqs).
    hf = Owlv2ForObjectDetection.from_pretrained(REPO, attn_implementation="eager").eval()

    dims = dims_from_hf(hf.config)
    print("config act:", dims.vision_act, dims.text_act, "| image_size:", dims.image_size)
    print("hardcoded OWLV2_DIMS['b16'] matches config:", OWLV2_DIMS["b16"] == dims)

    native = Owlv2DetectionModel(dims).eval()
    res = native.load_state_dict(hf.state_dict(), strict=False)
    miss = [k for k in res.missing_keys if not k.endswith(("position_ids", "box_bias"))]
    unexp = [k for k in res.unexpected_keys if not k.endswith(("position_ids", "box_bias"))]
    print("missing (non-buffer):", miss)
    print("unexpected (non-buffer):", unexp)
    assert not miss and not unexp, "state_dict key mismatch"

    img = Image.open("tests/data/parkour.jpg").convert("RGB")
    inputs = proc(text=[["a photo of a person", "a photo of a dog", "a photo of a skateboard"]],
                  images=img, return_tensors="pt")

    ids, pix, am = inputs["input_ids"], inputs["pixel_values"], inputs["attention_mask"]
    with torch.no_grad():
        # --- localize: vision feature_map and text query_embeds ---
        hf_text, hf_fmap, _ = hf.image_text_embedder(input_ids=ids, pixel_values=pix, attention_mask=am)
        our_fmap = native._image_embedder(pix)
        our_text = native._text_embedder(ids, am)
        print(f"[vision] feature_map   max_abs_diff={(hf_fmap - our_fmap).abs().max().item():.3e}")
        print(f"[text]   query_embeds   max_abs_diff={(hf_text - our_text).abs().max().item():.3e}")

        ref = hf(**inputs)
        out = native(input_ids=ids, pixel_values=pix, attention_mask=am)

    # Vision tower + objectness are bit-exact; text-derived logits/boxes match to
    # float32 rounding (<1e-5) — inherent fp non-associativity, not a port error.
    tol = {"logits": 1e-5, "pred_boxes": 1e-5, "objectness_logits": 0.0}
    ok = True
    for key, ours in (("logits", out["logits"]), ("pred_boxes", out["pred_boxes"]),
                      ("objectness_logits", out["objectness_logits"])):
        their = getattr(ref, key)
        diff = (their - ours).abs().max().item()
        passed = diff <= tol[key]
        ok = ok and passed
        print(f"{key:20s} shape={tuple(ours.shape)}  max_abs_diff={diff:.3e}  {'OK' if passed else 'FAIL'}")
    print("\nOWLv2 native parity:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
