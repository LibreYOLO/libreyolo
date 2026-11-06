"""Parity: LibreGen("crm") (plain PyTorch) vs upstream CRM, same GPU, same process.

Stage by stage, each side fed the same inputs and the same noise: the
preprocessed image (an RGBA example, so neither side removes a background),
the OpenCLIP text and image tokens, the first diffusion stage (six views,
50 DDIM steps, guidance 5), the second stage from upstream's views (six
CCMs), the reconstruction UNet's triplane from upstream's views and CCMs,
the FlexiCubes grid and raw mesh, the cleaned mesh and the vertex colours.

Upstream needs xformers (its attention asserts it). Its renderer constructs
an nvdiffrast context at import and its cleanup calls kiui/pymeshlab; this
gate stubs nvdiffrast (no rasterization is compared) and runs the cleanup
only when pymeshlab is installed. Upstream reads SD 2.1's scheduler config
from a Hugging Face repo that no longer exists; the gate builds the same
scheduler from its published values.

Prereqs (a CUDA machine; throwaway env, do NOT install into the main .venv):
    pip install -e ".[gen]" xformers omegaconf einops open-clip-torch kiui trimesh pymeshlab
    git clone https://github.com/thu-ml/CRM && git -C CRM checkout 4964e36a593070a3045eb0f300935a771ff5c172
    hf download Zhengyi/CRM --revision 36283fd567b9a3120a166dfe2f2e0965d96261b0 --local-dir weights/LibreCRMl

Run:
    python weights/parity_crm.py --upstream CRM --weights weights/LibreCRMl
"""

import argparse
import json
import sys
import time
from unittest import mock

_parser = argparse.ArgumentParser()
_parser.add_argument("--upstream", required=True)
_parser.add_argument("--weights", required=True)
_parser.add_argument("--seed", type=int, default=0)
_parser.add_argument("--out", default="parity_crm.json")
ARGS = _parser.parse_args()
sys.path.insert(0, ARGS.upstream)
for _name in ("nvdiffrast", "nvdiffrast.torch", "rembg", "gradio"):
    sys.modules.setdefault(_name, mock.MagicMock())

import numpy as np
import torch
from PIL import Image

DEV = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def cmp(a, b):
    a = torch.as_tensor(np.asarray(a) if not torch.is_tensor(a) else a).detach().float().cpu()
    b = torch.as_tensor(np.asarray(b) if not torch.is_tensor(b) else b).detach().float().cpu()
    if a.shape != b.shape:
        return {"shapes": [list(a.shape), list(b.shape)]}
    d = (a - b).abs()
    return {
        "equal": bool(torch.equal(a, b)),
        "max": float(d.max()) if d.numel() else 0.0,
        "mean": float(d.mean()) if d.numel() else 0.0,
        "scale": float(b.abs().max()) if b.numel() else 0.0,
    }


def example():
    """An RGBA object on transparency: a lit sphere in front of a box."""
    size = 512
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rgba = np.zeros((size, size, 4), np.uint8)
    box = (np.abs(xx - 300) < 110) & (np.abs(yy - 330) < 90)
    rgba[box] = (60, 120, 200, 255)
    r = np.sqrt((xx - 220) ** 2 + (yy - 220) ** 2)
    ball = r < 120
    shade = np.clip(1.2 - r / 140, 0.2, 1)[..., None]
    rgba[ball, :3] = (np.array([220, 90, 40]) * shade[ball]).astype(np.uint8)
    rgba[ball, 3] = 255
    return Image.fromarray(rgba, "RGBA")


ROW = {}
t0 = time.time()

# -- Upstream ------------------------------------------------------------------
import diffusers
from omegaconf import OmegaConf

_real_from_pretrained = diffusers.DDIMScheduler.from_pretrained


def _sd21_scheduler(*args, **kwargs):
    # stabilityai/stable-diffusion-2-1-base's scheduler_config.json values.
    return diffusers.DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
        prediction_type="epsilon",
    )


diffusers.DDIMScheduler.from_pretrained = staticmethod(_sd21_scheduler)
import open_clip

_real_create = open_clip.create_model_and_transforms


def _no_pretrained(arch, device=None, pretrained=None, **kw):
    # The CRM checkpoints carry the OpenCLIP weights; skip the LAION download.
    return _real_create(arch, device=device, pretrained=None, **kw)


open_clip.create_model_and_transforms = _no_pretrained
import imagedream.ldm.modules.encoders.modules as uenc

uenc.open_clip.create_model_and_transforms = _no_pretrained
from imagedream.ldm.modules.distributions.distributions import (
    DiagonalGaussianDistribution,
)
from imagedream.ldm.util import instantiate_from_config
from libs.sample import ImageDreamDiffusion, ImageDreamDiffusionStage2
from model import CRM
from run import preprocess_image as up_preprocess

from libreyolo.models.gen.crm import diffusion as P
from libreyolo.models.gen.crm import pipeline as PP
from libreyolo.models.gen.crm import reconstruction as R

NOISE = R.stage_noise(ARGS.seed)
QUEUE = []
_real_sample = DiagonalGaussianDistribution.sample


def _queued_sample(self):
    noise = QUEUE.pop(0).to(self.parameters.device)
    return self.mean + self.std * noise


DiagonalGaussianDistribution.sample = _queued_sample

# Preprocessing (RGBA example: neither side removes a background).
src = example()
up_img = up_preprocess(src, "Alpha as mask", 1.0, (127, 127, 127))
port_img = PP.preprocess_image(src, None)
ROW["preprocess"] = cmp(np.asarray(port_img), np.asarray(up_img))

stage_cfg = {}
for key, cfg_path, stage2 in (
    ("pixel", "configs/nf7_v3_SNR_rd_size_stroke.yaml", False),
    ("ccm", "configs/stage2-v2-snr.yaml", True),
):
    cfg = OmegaConf.load(f"{ARGS.upstream}/{cfg_path}").config
    stage_cfg[key] = cfg


def load_upstream(key, file):
    cfg = stage_cfg[key]
    model_cfg = OmegaConf.load(f"{ARGS.upstream}/{cfg.models.config}").model
    model = instantiate_from_config(model_cfg)
    model.load_state_dict(torch.load(f"{ARGS.weights}/{file}", map_location="cpu"), strict=False)
    model = model.to(DEV).to(torch.float16)
    model.device = DEV
    sampler_cls = ImageDreamDiffusionStage2 if key == "ccm" else ImageDreamDiffusion
    sampler = sampler_cls(model, device=DEV, dtype=torch.float16, **cfg.sampler.params)
    return model, sampler


def load_port(stage2, file):
    module = P.MultiViewDiffusion(stage2=stage2)
    module.load_state_dict(torch.load(f"{ARGS.weights}/{file}", map_location="cpu", weights_only=True))
    return module.eval().requires_grad_(False).half().to(DEV)


def with_latents(model, latents):
    real = model.q_sample

    def q_sample(x_start, t, noise=None):
        return real(x_start, t, noise=latents.to(x_start.device, x_start.dtype))

    model.q_sample = q_sample


# -- Stage 1 -------------------------------------------------------------------
up_model, up_sampler = load_upstream("pixel", "pixel-diffusion.pth")
port_model = load_port(False, "pixel-diffusion.pth")

prompt, negative = P.prompt_tokens(DEV)
ROW["tokens_prompt"] = cmp(prompt, open_clip.tokenize([P.PROMPT]).to(DEV))
ROW["tokens_negative"] = cmp(negative, open_clip.tokenize([P.NEGATIVE_PROMPT]).to(DEV))
with torch.no_grad():
    up_uc = up_model.get_learned_conditioning([P.NEGATIVE_PROMPT])
    port_uc = port_model.clip_model.model.encode_text(negative)
    ROW["text_negative"] = cmp(port_uc, up_uc)
    with torch.autocast("cuda", dtype=torch.float16):
        up_c = up_model.get_learned_conditioning([P.PROMPT])
        port_c = port_model.clip_model.model.encode_text(prompt)
        up_ip = up_model.get_learned_image_conditioning(up_img)
        port_ip = port_model.clip_model.model.visual.penultimate_tokens(
            PP._clip_pixels(up_img, DEV).half()
        )
    ROW["text_prompt"] = cmp(port_c, up_c)
    ROW["image_tokens"] = cmp(port_ip, up_ip)

QUEUE[:] = [NOISE["stage1_posterior"], torch.zeros(1, 4, 32, 32)]
with_latents(up_model, NOISE["stage1_latents"])
up_views = up_sampler.i2i(
    up_model, 256, "3D assets", up_uc, up_sampler.sampler, ip=up_img, step=50, scale=5.0,
    batch_size=7, ddim_eta=0.0, dtype=torch.float16, device=DEV, camera=up_sampler.camera,
    num_frames=7, pixel_control=True, transform=up_sampler.image_transform, offset_noise=True,
)
up_views = np.stack(up_views)
port_views = P.run_stage(
    port_model,
    pixels=PP._pixels(up_img, DEV),
    clip_pixels=PP._clip_pixels(up_img, DEV),
    noise={"posterior": NOISE["stage1_posterior"], "latents": NOISE["stage1_latents"]},
    steps=50,
    guidance=5.0,
    half=True,
)
ROW["stage1_views"] = cmp(port_views, up_views)
ROW["stage1_cameras"] = cmp(P.cameras(P.STAGE1_VIEWS, 6).to(DEV), up_sampler.camera)
del up_model, up_sampler, port_model
torch.cuda.empty_cache()

# -- Stage 2, from upstream's views -----------------------------------------------
up_model, up_sampler = load_upstream("ccm", "ccm-diffusion.pth")
port_model = load_port(True, "ccm-diffusion.pth")
views6 = [Image.fromarray(v) for v in up_views[:6]]
QUEUE[:] = [NOISE["stage2_posterior"], torch.zeros(1, 4, 32, 32)]
with_latents(up_model, NOISE["stage2_latents"])
up_ccms = np.stack(
    up_sampler.i2iStage2(
        up_model, 256, "3D assets", up_sampler.uc, up_sampler.sampler, pixel_images=views6,
        ip=up_img, step=50, scale=5.0, batch_size=6, ddim_eta=0.0, dtype=torch.float16,
        device=DEV, camera=up_sampler.camera, num_frames=6, pixel_control=True,
        transform=up_sampler.image_transform, offset_noise=True,
    )
)
port_ccms = P.run_stage(
    port_model,
    pixels=PP._pixels(up_img, DEV),
    clip_pixels=PP._clip_pixels(up_img, DEV),
    noise={"posterior": NOISE["stage2_posterior"], "latents": NOISE["stage2_latents"]},
    steps=50,
    guidance=5.0,
    stage2_views=torch.stack([PP._pixels(v, DEV)[0] for v in views6]),
    half=True,
)
ROW["stage2_ccms"] = cmp(port_ccms, up_ccms)
del up_model, up_sampler, port_model
torch.cuda.empty_cache()

# -- Reconstruction, from upstream's views and CCMs ---------------------------------
from util.utils import get_tri

specs = json.load(open(f"{ARGS.upstream}/configs/specs_objaverse_total.json"))
up_crm = CRM(specs).to(DEV)
up_crm.load_state_dict(torch.load(f"{ARGS.weights}/CRM.pth", map_location=DEV), strict=False)
up_crm.eval()
port_net = R.CRMNet()
port_net.load_state_dict(torch.load(f"{ARGS.weights}/CRM.pth", map_location="cpu", weights_only=True))
port_net = port_net.eval().to(DEV)
grid = R.Grid(DEV)
ROW["grid_verts"] = cmp(grid.verts, up_crm.renderer.flexicubes.verts)
ROW["grid_indices"] = cmp(grid.indices, up_crm.renderer.flexicubes.indices)
ROW["grid_range"] = [
    float(up_crm.renderer.flexicubes.verts.min()),
    float(up_crm.renderer.flexicubes.verts.max()),
]

np_imgs = np.concatenate(list(up_views[:6]), 1)
np_xyzs = np.concatenate(list(up_ccms), 1)
with torch.no_grad():
    # inference.py's generate3d, step by step.
    color = torch.from_numpy(np_imgs) / 255
    xyz = torch.from_numpy(np_xyzs[:, :, (2, 1, 0)]) / 255
    color, xyz = color.permute(2, 0, 1), xyz.permute(2, 0, 1)

    def get_imgs(c):
        out = [c[:, :, 256 * 5 : 256 * 6]] + [c[:, :, 256 * i : 256 * (i + 1)] for i in range(5)]
        return torch.stack(out, 0)

    up_tri = torch.cat(
        [
            get_tri(get_imgs(color), dim=0, blender=True, scale=1).unsqueeze(0),
            get_tri(get_imgs(xyz), dim=0, blender=True, scale=1, fix=True).unsqueeze(0),
        ],
        dim=1,
    ).to(DEV)
    port_tri = R.triplane_input(up_views[:6], up_ccms).to(DEV)
    ROW["triplane_input"] = cmp(port_tri, up_tri)
    tnew = torch.full([1], 20, dtype=torch.long, device=DEV)
    noise_new = NOISE["triplane"].to(DEV) * 0.5 + 0.5
    up_noisy = up_crm.scheduler.add_noise(up_tri, noise_new, tnew)
    port_noisy = R.add_noise(port_tri, noise_new, 20)
    ROW["triplane_noisy"] = cmp(port_noisy, up_noisy)
    up_feat = up_crm.unet2(up_noisy, tnew)
    port_feat = port_net.unet2(up_noisy, tnew)
    ROW["triplane_features"] = cmp(port_feat, up_feat)
    up_v, up_f = up_crm.decode({"resolution": [1024, 1024]}, up_feat)
    port_v, port_f = R.extract_mesh(port_net, up_feat, grid)
    ROW["raw_vertices"] = cmp(port_v, up_v[0])
    # Same triangles; the face list's row order and each triangle's starting
    # corner may differ, so faces compare as sets of oriented triangles.
    def oriented(faces):
        f = faces.cpu().numpy()
        r = np.argmin(f, 1)
        return set(map(tuple, np.stack([f[np.arange(len(f)), (r + k) % 3] for k in range(3)], 1).tolist()))

    ROW["raw_faces"] = {"equal": oriented(port_f) == oriented(up_f)}
    ROW["raw_counts"] = {"port": [len(port_v), len(port_f)], "upstream": [len(up_v[0]), len(up_f)]}
    pv, pf = R.clean_mesh(port_v.cpu().numpy(), port_f.cpu().numpy())
    ROW["clean_counts_port"] = [len(pv), len(pf)]
    try:
        from kiui.mesh_utils import clean_mesh as up_clean
        from scipy.spatial import cKDTree

        uv_, uf_ = up_clean(
            up_v[0].squeeze().cpu().numpy().astype(np.float32),
            up_f.squeeze().cpu().numpy().astype(np.int32),
            repair=False, remesh=False, remesh_size=0.005,
        )
        ROW["clean_counts_upstream"] = [len(uv_), len(uf_)]
        d1, _ = cKDTree(uv_).query(pv)
        d2, _ = cKDTree(pv).query(uv_)
        ROW["clean_chamfer"] = {"mean": float((d1.mean() + d2.mean()) / 2), "max": float(max(d1.max(), d2.max()))}
        up_dec = up_crm.decoder(up_feat, torch.from_numpy(uv_).to(DEV)[None])
        up_colors = up_crm.rgbMlp(up_dec).squeeze()
        port_colors = R.vertex_colors(port_net, up_feat, torch.from_numpy(uv_).to(DEV))
        ROW["vertex_colors"] = cmp(port_colors, up_colors)
    except ImportError as exc:
        ROW["clean_upstream"] = f"skipped: {exc}"

# -- End to end through the port's own call path ---------------------------------
from libreyolo.models.gen.crm.model import _CRMModules

del up_crm
torch.cuda.empty_cache()
modules = _CRMModules(
    device=DEV,
    half=True,
    pixel=load_port(False, "pixel-diffusion.pth"),
    ccm=load_port(True, "ccm-diffusion.pth"),
    net=port_net,
    grid=grid,
    birefnet=None,
)
torch.cuda.synchronize()
start = time.time()
out = PP.generate_mesh(modules, port_img, ARGS.seed, steps=50, guidance=5.0, texture=True)
torch.cuda.synchronize()
ROW["e2e_seconds"] = time.time() - start
ROW["e2e_views_vs_upstream"] = cmp(out["views"], up_views[:6])
v, f = out["vertices"], out["faces"]
a, b, c = (v[f[:, i]] for i in range(3))
normal = np.cross(b - a, c - a)
ROW["e2e_mesh"] = {
    "vertices": len(v),
    "faces": len(f),
    "bounds": [v.min(0).tolist(), v.max(0).tolist()],
    # Signed volume > 0 when faces wind counter-clockwise seen from outside.
    "signed_volume": float((a * np.cross(b, c)).sum() / 6),
    "texture_covered": float((out["texture"].astype(int).sum(-1) > 3).mean()),
}
Image.fromarray(np.concatenate(list(out["views"]), 1)).save("parity_crm_views.png")
from libreyolo.models.gen.crm.model import mesh_payload
from libreyolo.utils.results import Results

Results(None, (512, 512), meshes=mesh_payload(out, seed=ARGS.seed, params={}), names={}).save(
    "parity_crm_object.obj"
)

# Stages that must be bitwise equal, and limits for the rest (about 3x an
# RTX 3090 run: fp16 kernel noise passes, a wrong layout or cast fails).
BITWISE = (
    "preprocess", "tokens_prompt", "tokens_negative", "stage1_cameras", "grid_verts",
    "grid_indices", "triplane_input", "triplane_noisy",
)
LIMITS = {"text_prompt": 6e-3, "text_negative": 6e-3, "image_tokens": 6e-3,
          "stage1_views": 0.5, "stage2_ccms": 0.5, "triplane_features": 1e-4,
          "raw_vertices": 1e-5, "vertex_colors": 1e-4}
BITWISE += ("raw_faces",)
failures = [k for k in BITWISE if not ROW[k].get("equal")]
failures += [k for k, lim in LIMITS.items() if "mean" not in ROW[k] or ROW[k]["mean"] > lim]
ROW["failures"] = failures
ROW["seconds"] = time.time() - t0
print(json.dumps(ROW, indent=1))
json.dump(ROW, open(ARGS.out, "w"), indent=1)
