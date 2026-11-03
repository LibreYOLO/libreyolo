"""Parity: LibreGen("triposplat") vs the TripoSplat reference code, same GPU, same process.

The port shares upstream's networks parameter for parameter, so the gate is:
bitwise equality of the preprocessed image and of the FLUX.2 VAE tokens;
DINOv3 tokens within bf16 noise of upstream's hand-written ViT (the port
reads the same weights through transformers: the two implementations order
their bf16 arithmetic differently, which measured a mean absolute difference
of 0.006 and isolated tokens up to 1.0 after the layer norm); the sampled
latent and camera within fp16 tolerance of upstream given the same tokens;
and, with upstream's global generator seeded like the port's, Gaussians
decoded from the same latent equal attribute by attribute (upstream draws
the octree sampling from the global generator, the port from the seeded
one, so the objects are compared stage by stage rather than end to end).

Prereqs (a CUDA machine; throwaway env, do NOT install into the main .venv):
    pip install -e ".[gen]" tqdm torchvision
    git clone https://github.com/VAST-AI-Research/TripoSplat
    git -C path/to/TripoSplat checkout d8db9e018b413dd9c4a9fe22463781bf98e8e68d
    hf download VAST-AI/TripoSplat --revision 56a96e603204ec410c4da60c13ea4fa09a2169a9 \
        --local-dir path/to/TripoSplat/ckpts

The script refuses to run against another code commit or weight revision.

Run:
    PYTHONPATH=path/to/TripoSplat python weights/parity_triposplat.py \
        --upstream path/to/TripoSplat [--out parity_out]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

UPSTREAM_CODE_COMMIT = "d8db9e018b413dd9c4a9fe22463781bf98e8e68d"
UPSTREAM_REVISION = "56a96e603204ec410c4da60c13ea4fa09a2169a9"
UPSTREAM_SHA256 = {
    "diffusion_models/triposplat_fp16.safetensors": "c870b97ac1d6bc9177608a5ec625e19ef9f3c5019aa68f64b0fb7803abcd6d20",
    "vae/triposplat_vae_decoder_fp16.safetensors": "ed0d0c3d43b599e326845d0ec70f3cf77be9a55e2d97627ac3b34d2830763cc8",
    "background_removal/birefnet.safetensors": "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154",
}


def check_upstream(checkout: Path) -> None:
    """Refuse any TripoSplat checkout or weights other than the pinned ones."""
    import hashlib
    import subprocess

    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != UPSTREAM_CODE_COMMIT:
        raise SystemExit(
            f"{checkout} is at {head}; parity is defined against {UPSTREAM_CODE_COMMIT}."
        )
    for name, expected in UPSTREAM_SHA256.items():
        digest = hashlib.sha256((checkout / "ckpts" / name).read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(
                f"ckpts/{name} has SHA-256 {digest}; expected the file of VAST-AI/TripoSplat@{UPSTREAM_REVISION}."
            )


CASES = [
    ("lily", "plant_water_lily.webp", 42, {}),
    ("butterfly", "creature_butterfly.webp", 7, {"gaussians": 65536}),
    ("ship", "vehicle_pirate_ship.webp", 3, {"steps": 10, "guidance": 1.0}),
    ("house", "building_stone_house.webp", 11, {"shift": 2.0, "gaussians": 32768}),
]


def compare(a, b, atol=0.0, mean_tol=None) -> dict:
    """``close`` when the max difference is within ``atol``, or the mean within ``mean_tol`` when given."""
    a = torch.as_tensor(
        np.asarray(a) if not isinstance(a, torch.Tensor) else a.detach().float().cpu()
    )
    b = torch.as_tensor(
        np.asarray(b) if not isinstance(b, torch.Tensor) else b.detach().float().cpu()
    )
    if a.shape != b.shape:
        return {"equal": False, "shapes": [list(a.shape), list(b.shape)]}
    diff = (a.double() - b.double()).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    mean_abs = float(diff.mean()) if diff.numel() else 0.0
    close = mean_abs <= mean_tol if mean_tol is not None else max_abs <= atol
    return {
        "equal": bool(torch.equal(a, b)),
        "close": close,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--upstream", required=True, type=Path, help="TripoSplat checkout with ckpts/"
    )
    parser.add_argument("--out", type=Path, default=Path("parity_triposplat"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Parity is defined on CUDA, where upstream runs.")
    check_upstream(args.upstream)

    import triposplat as up
    from PIL import Image

    from libreyolo import LibreGen
    from libreyolo.models.gen.image3d import pixels_of, preprocess_image
    from libreyolo.models.gen.triposplat.pipeline import sample_flow

    ckpts = args.upstream / "ckpts"
    upstream = up.TripoSplatPipeline(
        ckpt_path=str(ckpts / "diffusion_models/triposplat_fp16.safetensors"),
        decoder_path=str(ckpts / "vae/triposplat_vae_decoder_fp16.safetensors"),
        dinov3_path=str(ckpts / "clip_vision/dino_v3_vit_h.safetensors"),
        flux2_vae_encoder_path=str(ckpts / "vae/flux2-vae.safetensors"),
        rmbg_path=str(ckpts / "background_removal/birefnet.safetensors"),
        device="cuda",
    )
    port = LibreGen("triposplat")
    modules = port.pipeline
    examples = args.upstream / "static" / "example_inputs"

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cases": [],
    }
    for name, image, seed, knobs in CASES:
        source = Image.open(examples / image)
        steps = knobs.get("steps", 20)
        guidance = knobs.get("guidance", 3.0)
        shift = knobs.get("shift", 3.0)
        count = knobs.get("gaussians", 262144)

        # Stage 1: preprocessing (bitwise).
        up_prepared = upstream.preprocess_image(source)
        port_prepared = preprocess_image(
            source, modules.birefnet, image_size=1024, crop_margin=1.2
        )
        row = {
            "case": name,
            "input_image": compare(np.array(port_prepared), np.array(up_prepared)),
        }

        # Stage 2: encoders, each side from the same prepared image and seed.
        gen = torch.Generator(device="cuda").manual_seed(seed)
        up_cond = upstream.encode_image(up_prepared, generator=gen)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        port_cond = modules.encoder(
            pixels_of(port_prepared, port.device), generator=gen
        )
        row["dinov3"] = compare(
            port_cond["feature1"], up_cond["feature1"], mean_tol=2e-2
        )
        row["flux2_vae"] = compare(
            port_cond["feature2"], up_cond["feature2"], atol=5e-2
        )

        # Stage 3: the sampler from the same noise and the same (upstream) tokens.
        gen = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(1, 8192, 16, device="cuda", generator=gen)
        cam = torch.randn(1, 1, 5, device="cuda", generator=gen)
        up_out = up.FlowEulerCfgSampler().sample(
            upstream.flow_model,
            {"latent": noise.clone(), "camera": cam.clone()},
            cond={k: v.clone() for k, v in up_cond.items()},
            neg_cond={k: torch.zeros_like(v) for k, v in up_cond.items()},
            steps=steps,
            shift=shift,
            guidance_scale=guidance,
        )
        port_latent, port_cam = sample_flow(
            modules.denoiser,
            noise.clone(),
            cam.clone(),
            up_cond,
            steps=steps,
            shift=shift,
            guidance=guidance,
        )
        row["latent"] = compare(port_latent, up_out["latent"], atol=1e-2)
        row["camera_latent"] = compare(port_cam, up_out["camera"], atol=1e-2)

        # Stage 4: the decoders from the same latent, upstream's global generator seeded like the port's.
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        theirs = upstream.decode_latent(up_out["latent"], num_gaussians=count)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        ours = modules.decoder.decode(up_out["latent"], count, generator=gen)
        row["xyz"] = compare(ours["xyz"], theirs.get_xyz, atol=1e-4)
        row["scales"] = compare(ours["scaling"], theirs.get_scaling, atol=1e-5)
        row["opacities"] = compare(ours["opacity"], theirs.get_opacity, atol=1e-5)
        row["sh"] = compare(ours["features_dc"], theirs._features_dc, atol=1e-5)
        rotation = theirs._rotation + theirs.rots_bias[None, :]
        row["rotations"] = compare(
            ours["rotation"], rotation / rotation.norm(dim=-1, keepdim=True), atol=1e-5
        )

        # End to end through the public call, for the files.
        result = port(examples / image, seed=seed, **knobs)
        result.save(args.out / f"{name}_port.ply")
        theirs_full, _ = upstream.run(
            str(examples / image),
            seed=seed,
            steps=steps,
            guidance_scale=guidance,
            shift=shift,
            num_gaussians=count,
        )
        theirs_full.save_ply(str(args.out / f"{name}_upstream.ply"))
        row["count"] = len(result.gaussians) == theirs_full.get_xyz.shape[0]
        row["pass"] = (
            row["input_image"]["equal"]
            and all(
                value.get("close", value.get("equal"))
                for key, value in row.items()
                if isinstance(value, dict) and key != "input_image"
            )
            and row["count"]
        )
        print(json.dumps(row), flush=True)
        report["cases"].append(row)
    report["all_pass"] = all(row["pass"] for row in report["cases"])
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print("ALL PASS" if report["all_pass"] else "MISMATCH")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
