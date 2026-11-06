"""Parity: LibreGen("lgm") (plain PyTorch) vs upstream LGM + ImageDream, same device, same process.

Stage by stage, each side fed the same inputs: the recentred input (from an
RGBA example, so neither side removes a background), ImageDream's cameras and
LGM's rays, the empty-prompt text tokens, the CLIP image tokens, one forward
of the multi-view UNet, the whole seeded DDIM sampler and its four views, the
LGM network on upstream's views, and end to end (port vs upstream for one
seed, the Gaussians compared after upstream's opacity >= 0.005 pruning).

Measured on an RTX 3090 (torch 2.7.1, xformers 0.0.31, diffusers 0.42.1,
transformers 5.22.2), seed 42, 30 steps, upstream's three examples, fp16: the
recentred image, cameras, rays, text tokens and CLIP image tokens are bitwise
equal; one UNet forward differs by mean 1.6e-4 (xformers vs SDPA in fp16);
the four sampled views by mean 1.2e-4 to 3.6e-4; LGM on the same views by
mean 8e-5; the kept Gaussians number 30234 / 42109 / 39517 upstream against
30229 / 42111 / 39512. On the same box's CPU in fp32 (one example), the same
five stages are bitwise, the UNet forward differs by mean 3.0e-7, the views
by 1.7e-7, LGM by 9e-8, and both keep 31496 Gaussians.

Upstream at 3DTopia/LGM fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c. Its
renderer (diff-gaussian-rasterization, non-commercial) and LPIPS loss are
stubbed: this gate needs neither. On a machine without xformers (CPU, MPS)
upstream's ``xformers.ops.memory_efficient_attention`` is replaced by the
textbook softmax(q k^T / sqrt(d)) v written out below, so the gate still
compares two independent attention implementations; on CUDA with xformers
installed upstream runs its own kernel. CPU runs fp32 on both sides; CUDA
runs upstream's fp16 (``model.half()`` + autocast for LGM, an fp16
ImageDream pipeline), which the port mirrors.

Prereqs (throwaway env, do NOT install into the main .venv):
    pip install -e ".[gen]" kiui==0.2.7 tyro einops
    pip install xformers                       # CUDA only, optional
    git clone https://github.com/3DTopia/LGM && git -C LGM checkout fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c
    hf download ashawkey/LGM model_fp16_fixrot.safetensors --revision bc0097d24ffc5efc2e6a216edd579128f85f53ab --local-dir weights/LGM
    hf download LibreYOLO/LibreTRELLISl birefnet.safetensors --revision 75c784a8fa56ebfb76f46d9b9d830caec25826a7 --local-dir weights
    hf download ashawkey/imagedream-ipmv-diffusers --revision 73a034178e748421506492e91790cc62d6aefef5 --local-dir weights/imagedream

Run:
    python weights/parity_lgm.py --upstream LGM --lgm weights/LGM --imagedream weights/imagedream \\
        --birefnet weights/birefnet.safetensors --device cpu
"""

import argparse
import json
import math
import os
import sys
import time
import types
from unittest import mock

_parser = argparse.ArgumentParser()
_parser.add_argument("--upstream", required=True)
_parser.add_argument(
    "--lgm", required=True, help="folder with model_fp16_fixrot.safetensors"
)
_parser.add_argument(
    "--imagedream", required=True, help="imagedream-ipmv-diffusers folder"
)
_parser.add_argument(
    "--birefnet", required=True, help="birefnet.safetensors (LibreYOLO/LibreTRELLISl)"
)
_parser.add_argument("--device", default="cuda")
_parser.add_argument("--steps", type=int, default=30)
_parser.add_argument("--seed", type=int, default=42)
_parser.add_argument("images", nargs="*")
ARGS = _parser.parse_args()
sys.path.insert(0, ARGS.upstream)

import torch

DEVICE = torch.device(ARGS.device)
# Upstream's DINOv2-style attention uses xformers only when it imports; keep
# it on the plain path off CUDA.
if DEVICE.type != "cuda":
    os.environ["XFORMERS_DISABLED"] = "1"
for _name in ("diff_gaussian_rasterization", "roma", "kiui.lpips", "rembg"):
    sys.modules.setdefault(_name, mock.MagicMock())
try:
    if DEVICE.type != "cuda":
        raise ImportError  # xformers has no CPU or MPS kernel
    import xformers.ops  # noqa: F401

    ATTENTION = "xformers"
except ImportError:

    def _attention(q, k, v, attn_bias=None, op=None):
        weights = (q * q.shape[-1] ** -0.5) @ k.transpose(-2, -1)
        return weights.softmax(dim=-1) @ v

    _ops = types.ModuleType("xformers.ops")
    _ops.memory_efficient_attention = _attention
    _xformers = types.ModuleType("xformers")
    _xformers.ops = _ops
    # diffusers probes importlib.util.find_spec("xformers"), which needs a spec.
    import importlib.machinery

    _xformers.__spec__ = importlib.machinery.ModuleSpec("xformers", None)
    _ops.__spec__ = importlib.machinery.ModuleSpec("xformers.ops", None)
    sys.modules["xformers"], sys.modules["xformers.ops"] = _xformers, _ops
    ATTENTION = "reference softmax attention (no xformers)"

import core.gs
import numpy as np
from PIL import Image
from safetensors.torch import load_file

core.gs.GaussianRenderer = mock.MagicMock()
from core.models import LGM
from core.options import config_defaults
from kiui.op import recenter as upstream_recenter
from mvdream.mv_unet import MultiViewUNetModel, get_camera
from mvdream.pipeline_mvdream import MVDreamPipeline

from libreyolo.models.gen.lgm import pipeline as P
from libreyolo.models.gen.lgm.model import LibreLGM

HALF = DEVICE.type == "cuda"
DTYPE = torch.float16 if HALF else torch.float32
# Stages that must be bitwise equal, and mean-absolute-difference limits for
# the rest (a few times the measured kernel noise, so a wrong cast, layout or
# sampler step fails while attention-kernel noise passes).
BITWISE = ("image", "cameras", "rays", "text_tokens", "image_tokens")
LIMITS = {
    "unet_forward": 2e-3 if HALF else 1e-4,
    "views": 2e-2 if HALF else 2e-3,
    "lgm_gaussians": 2e-3 if HALF else 1e-4,
}


def cmp(a, b):
    a = torch.as_tensor(a).detach().float().cpu()
    b = torch.as_tensor(b).detach().float().cpu()
    if a.shape != b.shape:
        return {"shapes": [list(a.shape), list(b.shape)]}
    d = (a - b).abs()
    return {
        "equal": bool(torch.equal(a, b)),
        "max": float(d.max()),
        "mean": float(d.mean()),
        "scale": float(b.abs().max()),
    }


# -- Upstream --------------------------------------------------------------
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModel,
)

mv = ARGS.imagedream
unet_config = json.load(open(os.path.join(mv, "unet", "config.json")))
up_unet = MultiViewUNetModel(
    **{k: v for k, v in unet_config.items() if not k.startswith("_")}
)
up_unet.load_state_dict(
    load_file(os.path.join(mv, "unet", "diffusion_pytorch_model.safetensors"))
)
up_pipe = MVDreamPipeline(
    vae=AutoencoderKL.from_pretrained(os.path.join(mv, "vae")),
    unet=up_unet,
    tokenizer=CLIPTokenizer.from_pretrained(os.path.join(mv, "tokenizer")),
    text_encoder=CLIPTextModel.from_pretrained(os.path.join(mv, "text_encoder")),
    scheduler=DDIMScheduler.from_pretrained(os.path.join(mv, "scheduler")),
    feature_extractor=CLIPImageProcessor.from_pretrained(
        os.path.join(mv, "feature_extractor")
    ),
    image_encoder=CLIPVisionModel.from_pretrained(os.path.join(mv, "image_encoder")),
)
up_pipe = up_pipe.to(DEVICE, DTYPE)
up_pipe.set_progress_bar_config(disable=True)
opt = config_defaults["big"]
opt.lambda_lpips = 0
up_lgm = LGM(opt)
up_lgm.load_state_dict(
    load_file(os.path.join(ARGS.lgm, "model_fp16_fixrot.safetensors")), strict=False
)
up_lgm = up_lgm.to(DEVICE).eval()
if HALF:
    up_lgm.half()


def upstream_lgm(views):
    """infer.py's second stage."""
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF

    x = torch.from_numpy(views).permute(0, 3, 1, 2).float().to(DEVICE)
    x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    x = TF.normalize(x, P.IMAGENET_MEAN, P.IMAGENET_STD)
    x = torch.cat([x, up_lgm.prepare_default_rays(DEVICE)], dim=1).unsqueeze(0)
    with torch.no_grad():
        if HALF:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return up_lgm.forward_gaussians(x)
        return up_lgm.forward_gaussians(x)


# -- Port ------------------------------------------------------------------
staging = os.path.abspath("weights/_lgm_parity")
os.makedirs(staging, exist_ok=True)
for name, target in (
    (
        "model_fp16_fixrot.safetensors",
        os.path.join(ARGS.lgm, "model_fp16_fixrot.safetensors"),
    ),
    ("imagedream", mv),
    ("birefnet.safetensors", ARGS.birefnet),
):
    link = os.path.join(staging, name)
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(target), link)
port = LibreLGM(checkpoint_dir=staging, device=str(DEVICE))
M = port.pipeline

images = ARGS.images or ["anya_rgba.png", "bird_rgba.png", "catstatue_rgba.png"]
report = []
for name in images:
    row = {"case": name}
    path = (
        name if os.path.isfile(name) else os.path.join(ARGS.upstream, "data_test", name)
    )
    source = Image.open(path).convert("RGBA")
    # Upstream: rembg's cutout, recentred on white. Here the example's own alpha is the cutout.
    carved = np.array(source)
    up_image = (
        upstream_recenter(carved, carved[..., 3] > 0, border_ratio=0.2).astype(
            np.float32
        )
        / 255
    )
    up_image = up_image[..., :3] * up_image[..., 3:4] + (1 - up_image[..., 3:4])
    port_image = P.preprocess_image(source, M.birefnet)
    row["image"] = cmp(port_image, up_image)
    row["cameras"] = cmp(
        P.imagedream_cameras(0.0), get_camera(4, elevation=0, extra_view=True)
    )
    row["rays"] = cmp(P.ray_embeddings(0.0), up_lgm.prepare_default_rays("cpu"))

    with torch.no_grad():
        up_text = up_pipe._encode_prompt("", DEVICE, 1, True, "")
        row["text_tokens"] = cmp(M.text_tokens, up_text[1:])
        up_neg_tokens, up_tokens = up_pipe.encode_image(up_image, DEVICE, 1)
        port_tokens = P.encode_image(M, up_image)
        row["image_tokens"] = cmp(port_tokens, up_tokens)

        # One multi-view UNet forward on the same guided batch.
        g = torch.Generator().manual_seed(1)
        x = torch.randn(10, 4, 32, 32, generator=g).to(DEVICE, DTYPE)
        latents = torch.randn(2, 4, 32, 32, generator=g).to(DEVICE, DTYPE)
        camera = torch.cat([get_camera(4, extra_view=True)] * 2).to(DEVICE, DTYPE)
        context = torch.cat([up_text[1:]] * 10)
        ip = torch.cat([up_neg_tokens] * 5 + [up_tokens] * 5)
        t = torch.tensor([700.0] * 10, device=DEVICE, dtype=DTYPE)
        a = M.unet(x, t, context, camera=camera, num_frames=5, ip=ip, ip_img=latents)
        b = up_pipe.unet(
            x.clone(), t, context, camera=camera, num_frames=5, ip=ip, ip_img=latents
        )
        row["unet_forward"] = cmp(a, b)

    # The whole sampler: upstream seeded the way its demo seeds (global RNG),
    # the port from its device generator.
    torch.manual_seed(ARGS.seed)
    up_views = up_pipe(
        "",
        up_image,
        guidance_scale=5.0,
        num_inference_steps=ARGS.steps,
        elevation=0,
        device=DEVICE,
    )
    up_views = np.stack([up_views[1], up_views[2], up_views[3], up_views[0]], axis=0)
    t0 = time.time()
    port_views = P.sample_views(
        M,
        up_image,
        torch.Generator(device=DEVICE).manual_seed(ARGS.seed),
        steps=ARGS.steps,
        guidance=5.0,
        elevation=0.0,
    )
    seconds_views = time.time() - t0
    row["views"] = cmp(port_views, up_views)

    # LGM from upstream's views.
    up_gaussians = upstream_lgm(up_views)
    port_gaussians = P.views_to_gaussians(M, up_views, 0.0)
    row["lgm_gaussians"] = cmp(port_gaussians, up_gaussians)

    # End to end: each side's whole chain from the same recentred image and seed.
    t0 = time.time()
    out = P.run_stages(
        M, up_image, ARGS.seed, steps=ARGS.steps, guidance=5.0, elevation=0.0
    )
    seconds_port = time.time() - t0
    end_up = up_gaussians[0][up_gaussians[0][:, 3] >= P.MIN_OPACITY]
    end_port = out["lgm"][0][out["lgm"][0][:, 3] >= P.MIN_OPACITY]
    row["end_to_end"] = {
        "gaussians_upstream": int(end_up.shape[0]),
        "gaussians_port": int(end_port.shape[0]),
        "lgm_mean_abs": cmp(out["lgm"], up_gaussians).get("mean"),
        "seconds_views_port": round(seconds_views, 1),
        "seconds_port": round(seconds_port, 1),
    }
    # Frame conversion sanity: half-size, z-up, unit quaternions.
    g = out["gaussians"]
    row["object_frame"] = {
        "max_abs_xyz": float(g["xyz"].abs().max()),
        "max_quat_norm_err": float((g["rotations"].norm(dim=1) - 1).abs().max()),
    }
    failures = [k for k in BITWISE if not row[k].get("equal")]
    failures += [
        k
        for k, limit in LIMITS.items()
        if "mean" not in row[k]
        or not math.isfinite(row[k]["mean"])
        or row[k]["mean"] > limit
    ]
    row["failures"] = failures
    print(json.dumps(row), flush=True)
    report.append(row)

passed = all(not row["failures"] for row in report)
json.dump(
    {
        "torch": torch.__version__,
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        "upstream_attention": ATTENTION,
        "steps": ARGS.steps,
        "seed": ARGS.seed,
        "limits": LIMITS,
        "all_pass": passed,
        "cases": report,
    },
    open("parity_lgm_report.json", "w"),
    indent=1,
)
print("ALL PASS" if passed else "MISMATCH")
sys.exit(0 if passed else 1)
