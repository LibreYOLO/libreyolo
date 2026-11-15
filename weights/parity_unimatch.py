"""Parity check: upstream UniMatch ``inference_flow`` vs LibreYOLO predict.

Runs on a GPU box with an upstream checkout. For every consecutive pair of a
frame directory, the upstream path (verbatim copy of the preprocessing, model
call and postprocessing in upstream ``evaluate_flow.inference_flow``) and the
LibreYOLO path (``LibreYOLO(<converted .pt>)(frames)``, the public sequence
API) both produce a flow field on the original canvas; the script prints the
max and mean absolute difference in pixels and the mean end-point error
between the two fields.

Usage::

    python weights/parity_unimatch.py --upstream /path/to/unimatch \\
        --frames /path/to/frames --size l \\
        --upstream-ckpt gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \\
        --libre-ckpt LibreUniMatchl-flow.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

UPSTREAM_ARGS = {
    "s": dict(num_scales=1, upsample_factor=8, reg_refine=False, padding_factor=16,
              attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1], num_reg_refine=1),
    "l": dict(num_scales=2, upsample_factor=4, reg_refine=True, padding_factor=32,
              attn_splits_list=[2, 8], corr_radius_list=[-1, 4], prop_radius_list=[-1, 1], num_reg_refine=6),
}


@torch.no_grad()
def upstream_flows(upstream_dir, ckpt, size, filenames, device):
    sys.path.insert(0, upstream_dir)
    from unimatch.unimatch import UniMatch  # upstream package

    a = UPSTREAM_ARGS[size]
    model = UniMatch(feature_channels=128, num_scales=a["num_scales"],
                     upsample_factor=a["upsample_factor"], num_head=1,
                     ffn_dim_expansion=4, num_transformer_layers=6,
                     reg_refine=a["reg_refine"], task="flow").to(device)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    model.eval()
    padding_factor = a["padding_factor"]
    out = []
    for test_id in range(len(filenames) - 1):
        # --- verbatim from upstream evaluate_flow.inference_flow ---
        image1 = np.array(Image.open(filenames[test_id])).astype(np.uint8)[..., :3]
        image2 = np.array(Image.open(filenames[test_id + 1])).astype(np.uint8)[..., :3]
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(device)
        transpose_img = False
        if image1.size(-2) > image1.size(-1):
            image1 = torch.transpose(image1, -2, -1)
            image2 = torch.transpose(image2, -2, -1)
            transpose_img = True
        inference_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                          int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]
        ori_size = image1.shape[-2:]
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            image1 = F.interpolate(image1, size=inference_size, mode="bilinear", align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode="bilinear", align_corners=True)
        results_dict = model(image1, image2, attn_type="swin",
                             attn_splits_list=a["attn_splits_list"],
                             corr_radius_list=a["corr_radius_list"],
                             prop_radius_list=a["prop_radius_list"],
                             num_reg_refine=a["num_reg_refine"], task="flow",
                             pred_bidir_flow=False)
        flow_pr = results_dict["flow_preds"][-1]
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode="bilinear", align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]
        if transpose_img:
            flow_pr = torch.transpose(flow_pr, -2, -1)
        out.append(flow_pr[0].permute(1, 2, 0).cpu().numpy())
    return out


@torch.no_grad()
def libre_flows(ckpt, filenames, device):
    from libreyolo import LibreYOLO

    model = LibreYOLO(ckpt, device=device)
    return [np.asarray(r.flow.data, dtype=np.float32) for r in model(filenames)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--size", required=True, choices=["s", "l"])
    p.add_argument("--upstream-ckpt", required=True)
    p.add_argument("--libre-ckpt", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    filenames = sorted(glob(args.frames + "/*.png") + glob(args.frames + "/*.jpg"))
    ref = upstream_flows(args.upstream, args.upstream_ckpt, args.size, filenames, args.device)
    got = libre_flows(args.libre_ckpt, filenames, args.device)
    assert len(ref) == len(got) == len(filenames) - 1, (len(ref), len(got))
    max_abs, sum_abs, n_abs, epe, mags = 0.0, 0.0, 0, [], []
    for r, g in zip(ref, got):
        assert r.shape == g.shape, (r.shape, g.shape)
        d = np.abs(r - g)
        max_abs = max(max_abs, float(d.max()))
        sum_abs += float(d.sum()); n_abs += d.size
        epe.append(float(np.linalg.norm(r - g, axis=-1).mean()))
        mags.append(float(np.linalg.norm(r, axis=-1).mean()))
    report = {
        "size": args.size, "pairs": len(ref), "shape": list(ref[0].shape),
        "max_abs_diff_px": max_abs, "mean_abs_diff_px": sum_abs / n_abs,
        "mean_epe_px": float(np.mean(epe)), "max_pair_epe_px": float(np.max(epe)),
        "mean_upstream_flow_magnitude_px": float(np.mean(mags)),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
