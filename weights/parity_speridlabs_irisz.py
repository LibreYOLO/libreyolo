"""Parity: LibreGen("speridlabs-iris-z") vs the Speridlabs Iris-Z SDK, stage by stage.

Each side runs in its own environment (the SDK pins torch 2.9.1 and
transformers 4.57.6; LibreYOLO runs its own stack) and writes a dump; the
compare step reads both:

    # SDK env: pip install -e path/to/iris-z-sdk (commit 6c8925f), hf auth login
    # with an account that accepted the FLUX.2-dev terms (the SDK reads the
    # FLUX.2 VAE from the gated repo; the port does not).
    PYTHONPATH=path/to/iris-z-sdk python weights/parity_speridlabs_irisz.py sdk \
        --weights SNAPSHOT --size 9b --out sdk_9b.pt
    # LibreYOLO env: pip install -e ".[gen]"
    python weights/parity_speridlabs_irisz.py libre \
        --weights SNAPSHOT --size 9b --sdk sdk_9b.pt --out libre_9b.pt
    python weights/parity_speridlabs_irisz.py compare sdk_9b.pt libre_9b.pt

``SNAPSHOT`` is a local copy of ``speridlabs/iris-z`` at the pinned revision
(both sides load the same files). Stages, each compared on CUDA:

1. ``vae_encode``: the FLUX.2 VAE latent of the SDK's prepared image (the
   port reads the bfloat16 klein copy, upstream the float32 dev copy).
2. ``transformer``: one per-grid forward pass on the SDK's saved inputs.
3. ``depth``: image -> depth end to end, one seed and a 3-seed ensemble, at
   the photo's size; plus the port's sampler fed the SDK's latent.
4. ``text``: the prompt embeddings of ["", prompt].
5. ``generate``: prompt -> image end to end, and the port's sampler fed the
   SDK's prompt embedding (isolates the text encoder).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROMPTS = ["a lighthouse at dusk", "a red fox sleeping in fresh snow, photo"]
GEN_SIZE = 512
GEN_SEED = 7
DEPTH_SEEDS = (0, 1, 2)


def _images():
    sample = Path(__file__).resolve().parents[1] / "libreyolo" / "assets" / "guggenheim-bilbao.jpg"
    full = Image.open(sample).convert("RGB")
    return {
        "guggenheim_full": full,  # 1.8 MP -> works at 1024
        "guggenheim_small": full.resize((640, 498), Image.LANCZOS),  # < 0.5 MP -> 512
        "guggenheim_portrait": full.crop((300, 0, 1100, 1194)),
    }


def _folder(size):
    return {"9b": "small", "32b": "big"}[size]


# ---------------------------------------------------------------------------
# SDK side
# ---------------------------------------------------------------------------


def run_sdk(args):
    from iris_z import IrisZPipeline, rgbd

    variant = _folder(args.size)
    dump = {"images": {}, "depth": {}, "ensemble": {}, "text": {}, "generate": {}}
    pipe = IrisZPipeline.from_pretrained(args.weights, variant=variant, load_text_encoder=False)
    for name, image in _images().items():
        rgb = pipe.prepare_image(image, None)
        dump["images"][name] = rgb
        with torch.no_grad(), pipe.autocast():
            latent = pipe.encode_image(rgb)
        dump.setdefault("vae_encode", {})[name] = latent.float().cpu()
        if name == "guggenheim_small":
            # One per-grid forward pass on fixed inputs.
            dtype = next(pipe.model.parameters()).dtype
            gen = torch.Generator(device="cuda").manual_seed(123)
            clean, ids, modality, rgb_length = rgbd.rgbd_sequence(latent.to(dtype), torch.zeros_like(latent, dtype=dtype))
            tokens = clean.clone()
            tokens[:, rgb_length:] = torch.randn(tokens[:, rgb_length:].shape, generator=gen, device="cuda", dtype=dtype)
            modality[:, :rgb_length] = rgbd.GIVEN_RGB
            t = torch.full((1,), 0.7, dtype=dtype, device="cuda")
            inputs = dict(
                x=tokens, x_ids=ids, timesteps=t, ctx=pipe.empty_context[0], ctx_ids=pipe.empty_context[1],
                guidance=None, modality_ids=modality, rgb_timesteps=torch.zeros_like(t), depth_timesteps=t,
                depth_start=rgb_length,
            )
            with torch.no_grad(), pipe.autocast():
                out = pipe.model(**inputs)
            dump["transformer"] = {
                "inputs": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in inputs.items()},
                "output": out.float().cpu(),
            }
        depth = pipe.image_to_depth(image, seeds=(0,))
        dump["depth"][name] = {"log_depth": depth.log_depth, "sky": depth.sky_mask, "internal": depth.log_depth_internal}
        ensemble = pipe.image_to_depth(image, seeds=DEPTH_SEEDS)
        dump["ensemble"][name] = {"log_depth": ensemble.log_depth, "sky": ensemble.sky_mask}
        print("sdk depth", name, depth.log_depth.shape, flush=True)
    del pipe
    torch.cuda.empty_cache()

    pipe = IrisZPipeline.from_pretrained(args.weights, variant=variant, load_text_encoder=True)
    for prompt in PROMPTS:
        encoded = pipe.encode_prompt(prompt).embedding
        dump["text"][prompt] = encoded.float().cpu()
        image = pipe.text_to_image(prompt, width=GEN_SIZE, height=GEN_SIZE, seed=GEN_SEED)
        dump["generate"][prompt] = np.asarray(image)
        print("sdk generate", prompt, flush=True)
    torch.save(dump, args.out)


# ---------------------------------------------------------------------------
# LibreYOLO side
# ---------------------------------------------------------------------------


def run_libre(args):
    from libreyolo import LibreGen
    from libreyolo.models.gen.speridlabs_irisz import pipeline as P

    sdk = torch.load(args.sdk, weights_only=False)
    dump = {"vae_encode": {}, "depth": {}, "ensemble": {}, "depth_from_sdk_latent": {}, "text": {}, "generate": {},
            "generate_from_sdk_text": {}}
    model = LibreGen(args.weights, size=args.size, task="depth", device="cuda")
    model._ensure_loaded()
    nets = model.pipeline
    sky_ceiling = float(nets.config.get("sky_ceiling", 1.0))

    def decode(latent):
        encoded = nets.latents.decode(latent).float().mean(dim=1).clamp(-1, 1)[0].cpu().numpy()
        return P.unsqueeze_from_ceiling(encoded, sky_ceiling)

    for name, image in _images().items():
        rgb = sdk["images"][name]
        with torch.no_grad(), model._autocast():
            pixels = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).cuda()
            dump["vae_encode"][name] = nets.latents.encode(pixels).float().cpu()
            sdk_latent = sdk["vae_encode"][name].cuda()
            members = P.sample_depth(nets.transformer, nets.empty_context, sdk_latent, [0], 4, average_last=2, decode=decode)
        dump["depth_from_sdk_latent"][name] = members[0]
        result = model(image)
        dump["depth"][name] = {"log_depth": np.asarray(result.depth_map.data), "sky": ~np.asarray(result.depth_map.valid_mask)}
        ens = model(image, ensemble=len(DEPTH_SEEDS))
        dump["ensemble"][name] = {"log_depth": np.asarray(ens.depth_map.data), "sky": ~np.asarray(ens.depth_map.valid_mask)}
        print("libre depth", name, flush=True)
    inputs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in sdk["transformer"]["inputs"].items()}
    with torch.no_grad(), model._autocast():
        dump["transformer"] = {"output": nets.transformer(**inputs).float().cpu()}
    del model, nets
    torch.cuda.empty_cache()

    model = LibreGen(args.weights, size=args.size, device="cuda")
    model._ensure_loaded()
    nets = model.pipeline
    for prompt in PROMPTS:
        with torch.no_grad():
            dump["text"][prompt] = nets.text_encoder(["", prompt]).float().cpu()
        result = model(prompt, imgsz=GEN_SIZE, seed=GEN_SEED)
        dump["generate"][prompt] = np.asarray(result.generated.data)
        # The port's sampler and decoder on the SDK's prompt embedding.
        context = P.text_context(sdk["text"][prompt].to(torch.bfloat16).cuda())
        with torch.no_grad(), model._autocast():
            latent = P.sample_rgb(nets.transformer, context, GEN_SIZE, GEN_SIZE, GEN_SEED, model.DEFAULTS["guidance"], 50)
            rgb = nets.latents.decode(latent).float().clamp(-1, 1)
        dump["generate_from_sdk_text"][prompt] = (
            (127.5 * (rgb[0] + 1.0)).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        )
        print("libre generate", prompt, flush=True)
    torch.save(dump, args.out)


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def _diff(a, b):
    a = torch.as_tensor(np.asarray(a, dtype=np.float64) if not torch.is_tensor(a) else a.double())
    b = torch.as_tensor(np.asarray(b, dtype=np.float64) if not torch.is_tensor(b) else b.double())
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)]}
    d = (a - b).abs()
    return {"max": float(d.max()), "mean": float(d.mean()), "equal": bool(torch.equal(a, b))}


def _psnr(a, b):
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)


def _depth_rel(a, b):
    """Disagreement in depth, after mapping log depth in [-1, 1] back to relative depth."""
    a, b = np.exp(np.asarray(a, np.float64)), np.exp(np.asarray(b, np.float64))
    return float(np.mean(np.abs(a - b) / b))


def run_compare(args):
    sdk = torch.load(args.sdk_dump, weights_only=False)
    lib = torch.load(args.libre_dump, weights_only=False)
    print("stage, case: max |diff|, mean |diff| (and extra metrics)")
    for name in sdk["vae_encode"]:
        print("vae_encode", name, _diff(sdk["vae_encode"][name], lib["vae_encode"][name]))
    print("transformer", _diff(sdk["transformer"]["output"], lib["transformer"]["output"]))
    for name in sdk["depth"]:
        s, l = sdk["depth"][name], lib["depth"][name]
        print("depth", name, _diff(s["log_depth"], l["log_depth"]), "rel", round(_depth_rel(l["log_depth"], s["log_depth"]), 6),
              "sky mismatch", int((s["sky"] != l["sky"]).sum()))
        print("depth_from_sdk_latent", name, _diff(s["internal"], lib["depth_from_sdk_latent"][name]))
        s, l = sdk["ensemble"][name], lib["ensemble"][name]
        print("ensemble", name, _diff(s["log_depth"], l["log_depth"]), "rel", round(_depth_rel(l["log_depth"], s["log_depth"]), 6))
    for prompt in sdk["text"]:
        print("text", repr(prompt), _diff(sdk["text"][prompt], lib["text"][prompt]))
        print("generate", repr(prompt), "psnr", round(_psnr(sdk["generate"][prompt], lib["generate"][prompt]), 2))
        print("generate_from_sdk_text", repr(prompt), "psnr",
              round(_psnr(sdk["generate"][prompt], lib["generate_from_sdk_text"][prompt]), 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="side", required=True)
    for side in ("sdk", "libre"):
        p = sub.add_parser(side)
        p.add_argument("--weights", required=True, help="local speridlabs/iris-z snapshot")
        p.add_argument("--size", choices=("9b", "32b"), default="9b")
        p.add_argument("--out", required=True)
        if side == "libre":
            p.add_argument("--sdk", required=True, help="the SDK dump to replay inputs from")
    p = sub.add_parser("compare")
    p.add_argument("sdk_dump")
    p.add_argument("libre_dump")
    args = parser.parse_args()
    {"sdk": run_sdk, "libre": run_libre, "compare": run_compare}[args.side](args)


if __name__ == "__main__":
    sys.exit(main())
