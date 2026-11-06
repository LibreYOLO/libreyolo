"""Compare LibreYOLO RiO-DETRv2 with the pinned upstream on real images.

The upstream checkout must be RicePasteM/RiO-DETRv2 commit
ab8b22df5d800a927774c8c67e77fd29c0fe6967 (the Apache-2.0 release, preserved as
LibreYOLO/RiO-DETRv2 tag ``apache-2.0-ab8b22d``).  Checkpoints are the
Hugging Face RicePasteM/RiO-DETRv2 release at revision
5f578f57929aded1962ebf8c5584a4771461124f; converted files come from
``weights/convert_riodetrv2_weights.py``.  The upstream environment
(``requirements.txt`` at that commit) and LibreYOLO must be importable from the
same interpreter.

For every (dataset, size) and image this checks, at FP32 on CPU:
  1. preprocessed input tensor (upstream ResizeOBB + ConvertPILImage vs
     LibreYOLO ``_preprocess``),
  2. raw ``pred_logits`` / ``pred_boxes``,
  3. the decoded top-300 rotated boxes, scores and labels (upstream
     PostProcessorOBB vs LibreYOLO public ``predict``).
All three must be exactly equal (max_abs_diff == 0).

The released L/X configs say ``use_lab: true`` but the released L/X weights
carry no LearnableAffineBlock tensors; upstream is built with
``use_lab=False`` for those two sizes, the only setting that loads strictly.

Usage::

    python weights/parity_riodetrv2.py --upstream-dir /tmp/rio-apache \\
        --checkpoint-dir /tmp/rio-hf --converted-dir /tmp/rio-conv \\
        --images a.jpg b.jpg --sizes n s m l x --datasets dota diorr
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from _conversion_utils import add_repo_root_to_path
from PIL import Image

add_repo_root_to_path()

PINNED_COMMIT = "ab8b22df5d800a927774c8c67e77fd29c0fe6967"
DATASETS = {
    "dota": ("configs/rio-dota_1_ss/riov2_hgnetv2_{}_dota_1.yml",
             "dota_1_ss/riov2_hgnetv2_{}_dota_1_ss.pth", "LibreRiODETRv2{}-obb.pt"),
    "diorr": ("configs/rio-diorr/riov2_hgnetv2_{}_diorr.yml",
              "diorr/riov2_hgnetv2_{}_diorr.pth", "LibreRiODETRv2{}-obb-diorr.pt"),
}


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return float("inf")
    return 0.0 if a.numel() == 0 else (a.double() - b.double()).abs().max().item()


def run(upstream_dir: Path, ckpt_dir: Path, converted_dir: Path, images, sizes, datasets) -> bool:
    sys.path.insert(0, str(upstream_dir))
    from engine.core import YAMLConfig
    from engine.data.transforms._transforms import ConvertPILImage
    from engine.data.transforms.transforms_obb import ResizeOBB
    from engine.rtv4.postprocessor import PostProcessorOBB

    from libreyolo import LibreYOLO

    ok = True
    for ds in datasets:
        cfg_t, ckpt_t, conv_t = DATASETS[ds]
        for size in sizes:
            cfg = YAMLConfig(str(upstream_dir / cfg_t.format(size)),
                             HGNetv2={"pretrained": False, "use_lab": size in ("n", "s", "m")})
            upstream = cfg.model
            state = torch.load(ckpt_dir / ckpt_t.format(size), map_location="cpu", weights_only=True)["model"]
            upstream.load_state_dict(state, strict=True)
            upstream.eval()
            canvas = tuple(cfg.yaml_cfg["eval_spatial_size"])
            num_classes = int(cfg.yaml_cfg["num_classes"])
            post = PostProcessorOBB(num_classes=num_classes, num_top_queries=300, input_shape=list(canvas))
            resize = ResizeOBB(size=list(canvas), keep_ratio=True, pad_to_size=True, fill=0)
            to_tensor = ConvertPILImage(dtype="float32", scale=True)

            ours = LibreYOLO(str(converted_dir / conv_t.format(size)), device="cpu")
            assert ours.FAMILY == "riodetrv2" and ours.input_size == canvas[0], (ours.FAMILY, ours.input_size)

            for image_path in images:
                pil = Image.open(image_path).convert("RGB")
                w, h = pil.size
                # upstream targets torchvision 0.17's ``_transform`` hook; call it directly.
                x_up = torch.Tensor(to_tensor._transform(resize(pil), {})).unsqueeze(0)
                x_ours, _, _, _ = ours._preprocess(str(image_path))
                with torch.no_grad():
                    out_up = upstream(x_up)
                    out_ours = ours.model(x_ours)
                    dets = post(out_up, torch.tensor([[h, w]]))[0]
                result = ours.predict(str(image_path), conf=0.0, max_det=300)
                result = result[0] if isinstance(result, list) else result
                obb = result.obb.data.cpu()
                keep = dets["scores"] > 0.0
                up_rows = torch.cat([dets["boxes"][keep], dets["scores"][keep, None],
                                     dets["labels"][keep, None].to(dets["boxes"].dtype)], dim=1)
                diffs = {
                    "input": _max_diff(x_up, x_ours),
                    "logits": _max_diff(out_up["pred_logits"], out_ours["pred_logits"]),
                    "boxes": _max_diff(out_up["pred_boxes"], out_ours["pred_boxes"]),
                    "decoded": _max_diff(up_rows, obb.to(up_rows.dtype)),
                }
                top = obb[0].tolist() if len(obb) else []
                passed = all(v == 0.0 for v in diffs.values())
                ok &= passed
                print(f"{ds} {size} {Path(image_path).name} {w}x{h} n_det={len(obb)} "
                      f"n_conf>0.3={int((obb[:, 5] > 0.3).sum())} diffs={diffs} "
                      f"top={[round(v, 3) for v in top]} {'OK' if passed else 'FAIL'}", flush=True)
            del upstream, ours
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--converted-dir", required=True, type=Path)
    parser.add_argument("--images", required=True, nargs="+", type=Path)
    parser.add_argument("--sizes", nargs="+", default=["n", "s", "m", "l", "x"])
    parser.add_argument("--datasets", nargs="+", default=["dota", "diorr"], choices=list(DATASETS))
    args = parser.parse_args()
    head = subprocess.check_output(["git", "-C", str(args.upstream_dir), "rev-parse", "HEAD"], text=True).strip()
    if head != PINNED_COMMIT:
        raise SystemExit(f"Upstream checkout is {head}, expected {PINNED_COMMIT}")
    if not run(args.upstream_dir.resolve(), args.checkpoint_dir, args.converted_dir, args.images,
               args.sizes, args.datasets):
        raise SystemExit("parity FAILED")
    print("parity OK")


if __name__ == "__main__":
    main()
