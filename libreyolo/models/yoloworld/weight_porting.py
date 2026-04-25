"""Weight porting from wondervictor/YOLO-World-V2.1 (mmyolo/mmengine format).

**GPL-3.0 WEIGHTS:** the upstream YOLO-World-V2.1 checkpoints are GPL-3.0
licensed. LibreYOLO itself is MIT and does not bundle these weights. Users
who download them via this module inherit GPL-3.0 obligations for any code
that links against the loaded module instance.

Usage (separate script, not auto-run):

    from libreyolo.models.yoloworld.weight_porting import port_from_hf

    model = LibreYOLOWorldModel(size='l')
    port_from_hf(model, repo_id='wondervictor/YOLO-World-V2.1',
                 filename='yolo_world_v2_l_obj365v1_goldg_cc3mlite.pth')

This module is a STUB — the remapping dict is a starting point. Validation
on a real checkpoint requires a funded compute step (CPU works, but 442 MB
download + a test image). Treat each `TODO` below as the next concrete task.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch


# Prefix map: (upstream_prefix, our_prefix). Longest match wins.
# Derived from the research spec; verify on first real weight load.
KEY_REMAP: List[Tuple[str, str]] = [
    ("backbone.image_model.stem.", "backbone.stem."),
    ("backbone.image_model.stage1.", "backbone.stage1."),
    ("backbone.image_model.stage2.", "backbone.stage2."),
    ("backbone.image_model.stage3.", "backbone.stage3."),
    ("backbone.image_model.stage4.", "backbone.stage4."),
    # mmyolo's SPPF is `stage4.<N>` where N depends on depth — already covered by stage4.* above.

    # Text encoder — HuggingFace CLIP weights (we already load these from HF, not the checkpoint).
    ("backbone.text_model.", "__DROP__"),  # skip; we load CLIP separately

    # RepVL-PAN neck
    ("neck.top_down_layers.0.", "neck.top_down_layer_1."),
    ("neck.top_down_layers.1.", "neck.top_down_layer_2."),
    ("neck.bottom_up_layers.0.", "neck.bottom_up_layer_1."),
    ("neck.bottom_up_layers.1.", "neck.bottom_up_layer_2."),
    ("neck.downsample_layers.0.", "neck.downsample_1."),
    ("neck.downsample_layers.1.", "neck.downsample_2."),
    ("neck.reduce_layers.", "__DROP__"),  # we don't have reduce layers; upstream's P5 reduce is Identity in L

    # Head
    ("bbox_head.head_module.cls_preds.", "head.cls_preds."),
    ("bbox_head.head_module.reg_preds.", "head.reg_preds."),
    ("bbox_head.head_module.cls_contrasts.", "head.cls_contrasts."),
]


def _remap_key(k: str) -> str | None:
    """Return our key for upstream key `k`, or None to drop it."""
    for src, dst in KEY_REMAP:
        if k.startswith(src):
            if dst == "__DROP__":
                return None
            return dst + k[len(src):]
    return None  # unmapped → drop with warning


def port_state_dict(
    upstream_state: Dict[str, torch.Tensor],
    our_model: torch.nn.Module,
    strict: bool = False,
) -> Dict[str, object]:
    """Remap an upstream YOLO-World-V2.1 state_dict and load it into `our_model`.

    Returns a summary dict: {loaded, skipped, missing, shape_mismatches}.
    """
    # Upstream checkpoints are typically {'state_dict': {...}, 'meta': {...}, ...}
    if isinstance(upstream_state, dict) and "state_dict" in upstream_state:
        upstream_state = upstream_state["state_dict"]

    ours = our_model.state_dict()
    loaded: List[str] = []
    skipped: List[str] = []
    shape_mismatches: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []

    remapped: Dict[str, torch.Tensor] = {}
    for src_k, tensor in upstream_state.items():
        dst_k = _remap_key(src_k)
        if dst_k is None:
            skipped.append(src_k)
            continue
        if dst_k not in ours:
            skipped.append(f"{src_k} → {dst_k} (not in our model)")
            continue
        if tensor.shape != ours[dst_k].shape:
            shape_mismatches.append((f"{src_k} → {dst_k}", tuple(tensor.shape), tuple(ours[dst_k].shape)))
            continue
        remapped[dst_k] = tensor
        loaded.append(f"{src_k} → {dst_k}")

    missing = [k for k in ours if k not in remapped and not k.startswith("text_encoder.")]
    missing = [k for k in missing if "num_batches_tracked" not in k]  # BN state

    # Merge remapped onto current state_dict (keep text_encoder + head temperature defaults)
    new_state = {**ours, **remapped}
    result = our_model.load_state_dict(new_state, strict=strict)

    summary = {
        "loaded": len(loaded),
        "skipped": len(skipped),
        "missing_after_port": len(missing),
        "shape_mismatches": len(shape_mismatches),
        "load_result_missing": list(result.missing_keys) if hasattr(result, "missing_keys") else [],
        "load_result_unexpected": list(result.unexpected_keys) if hasattr(result, "unexpected_keys") else [],
        "sample_shape_mismatches": shape_mismatches[:5],
        "sample_skipped": skipped[:10],
    }
    return summary


def port_from_hf(
    our_model: torch.nn.Module,
    repo_id: str = "wondervictor/YOLO-World-V2.1",
    filename: str = "yolo_world_v2_l_obj365v1_goldg_cc3mlite.pth",
    local_path: str | Path | None = None,
    strict: bool = False,
) -> Dict[str, object]:
    """Download weights from HF (or load from `local_path`) and port into `our_model`.

    NOTE: this downloads GPL-3.0 weights. The caller is responsible for
    license compliance.
    """
    if local_path is not None:
        ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    return port_state_dict(ckpt, our_model, strict=strict)


# TODO list (verify during first real weight load):
#
# 1. Sub-layer inside CSPLayerWithTwoConv: upstream's `blocks.{i}.conv1.conv` vs our
#    `blocks.{i}.conv1.conv`. Probably identical; check on first load.
#
# 2. The MaxSigmoidAttnBlock's `embed_conv`: upstream may have it inline without
#    a BN (when in_channels == embed_channels, we use nn.Identity, they may not).
#    If so, we need to detect and insert their biasless 1x1 conv. The state-dict
#    keys `neck.*.attn_block.embed_conv.conv.weight` indicate their presence.
#
# 3. Stage4 SPPF: mmyolo places SPPF as `stage4[2]` after 2 CSP blocks. Our
#    nn.Sequential has (Conv, CSP, SPPF) so the index chain matches.
#
# 4. Neck P5 reduce layer: in mmyolo's YOLOv8PAFPN, `reduce_layers[0]` is an
#    Identity for L size. Our implementation skips it entirely — confirm the
#    state_dict doesn't have tensors there.
#
# 5. Head `cls_contrasts[i]` is `BNContrastiveHead`. State-dict keys:
#    norm.{weight,bias,running_mean,running_var}, logit_scale, bias.
#    Confirm logit_scale is a scalar tensor (not a Parameter with different init).
