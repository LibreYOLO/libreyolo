"""Parity: LibreGen("speridlabs-idol") vs the Speridlabs Idol SDK, same GPU, same process.

The gate is bitwise equality on CUDA for every stage: the preprocessed image,
both encoders' tokens, the sampled latent and camera, every Gaussian
attribute, and the PLY vertex data. ``--compile`` runs both sides on their
compiled paths (upstream ``optimize=True``) and reports the same comparison.

Prereqs (a CUDA machine; throwaway env, do NOT install into the main .venv):
    pip install -e ".[gen]" tensordict plyfile
    git clone https://github.com/speridlabs/idol-sdk   # commit 4702a7f
    hf auth login   # an account that accepted the DINOv3 and FLUX.2-dev terms
                    # (upstream reads its VAE from FLUX.2-dev; the port from klein)

Run:
    ATTN_BACKEND=sdpa PYTHONPATH=path/to/idol-sdk python weights/parity_speridlabs_idol.py \
        --examples path/to/idol-sdk/demo/examples [--compile] [--out parity_out]

``ATTN_BACKEND=sdpa`` keeps upstream on PyTorch SDPA, the only attention the
port uses; with flash-attn installed upstream would otherwise switch to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

UPSTREAM_REVISION = "2731c2291fc65a797c5cb627a5e5526fc1490902"

CASES = [
    ("armchair", "nb-armchair.webp", 7, {}),
    ("zombie", "fig-zombie.webp", 7, {}),
    ("explorer", "fig-explorer.webp", 42, {}),
    ("stand_mixer", "obj-stand-mixer.webp", 42, {}),
    ("castle_1m", "nb-castle-iso.webp", 3, {"gaussians": 1048576}),
    ("robot_32k", "toy-robot-tin.webp", 5, {"gaussians": 32768}),
    ("scooter_knobs", "iris-scooter.webp", 11, {"steps": 20, "guidance": 2.0, "shift": 2.0}),
    ("drill_margin", "obj-power-drill.webp", 9, {"margin": 1.5}),
    ("king_real", "x-jade-chess-king.webp", 13, {"mode": "real"}),
]


def compare(a, b) -> dict:
    a = torch.as_tensor(np.asarray(a) if not isinstance(a, torch.Tensor) else a.detach().cpu())
    b = torch.as_tensor(np.asarray(b) if not isinstance(b, torch.Tensor) else b.detach().cpu())
    if a.shape != b.shape:
        return {"equal": False, "shapes": [list(a.shape), list(b.shape)]}
    diff = (a.double() - b.double()).abs()
    return {"equal": bool(torch.equal(a, b)), "max_abs": float(diff.max()) if diff.numel() else 0.0}


def ply_vertex_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    return raw[raw.index(b"end_header\n") + len(b"end_header\n") :]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", required=True, type=Path, help="idol-sdk/demo/examples")
    parser.add_argument("--compile", action="store_true", help="compare the compiled paths")
    parser.add_argument("--out", type=Path, default=Path("parity_speridlabs_idol"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Parity is defined on CUDA, where upstream runs.")

    import idol.pipeline as upstream_module
    from huggingface_hub import snapshot_download
    from idol.pipeline import WEIGHT_FILES, IdolPipeline

    import libreyolo.models.gen.speridlabs_idol.pipeline as port_module
    from libreyolo import LibreGen

    snapshot = snapshot_download("speridlabs/idol", revision=UPSTREAM_REVISION, allow_patterns=WEIGHT_FILES)
    upstream = IdolPipeline(snapshot, optimize=args.compile)
    port = LibreGen("speridlabs-idol", compile=args.compile)

    # Capture each side's encoder tokens and sampled (latent, camera).
    seen: dict = {}
    up_sample, up_encode = upstream_module.sample_flow, upstream_module.ImageEncoder.__call__
    port_sample, port_encode = port_module.sample_flow, port_module.ImageEncoder.__call__

    def capture(key, value):
        seen[key] = value
        return value

    upstream_module.sample_flow = lambda *a, **k: capture("up_sample", up_sample(*a, **k))
    upstream_module.ImageEncoder.__call__ = lambda self, x: capture("up_cond", up_encode(self, x))
    port_module.sample_flow = lambda *a, **k: capture("port_sample", port_sample(*a, **k))
    port_module.ImageEncoder.__call__ = lambda self, x, generator=None: capture(
        "port_cond", port_encode(self, x, generator=generator)
    )

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"compile": args.compile, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "cases": []}
    for name, image, seed, knobs in CASES:
        source = str(args.examples / image)
        result = port(source, seed=seed, **knobs)
        output = upstream(
            source,
            seed=seed,
            mode=knobs.get("mode", "synthetic"),
            steps=knobs.get("steps", 50),
            cfg_strength=knobs.get("guidance", 3.0),
            rescale_t=knobs.get("shift", 3.0),
            num_gaussians=knobs.get("gaussians", 262144),
            crop_margin=knobs.get("margin"),
        )
        ours, theirs = result.gaussians, output.gaussian
        row = {
            "case": name,
            "input_image": compare(np.array(ours.input_image), np.array(output.preprocessed_image)),
            "dinov3": compare(seen["port_cond"]["feature1"], seen["up_cond"]["feature1"]),
            "flux2_vae": compare(seen["port_cond"]["feature2"], seen["up_cond"]["feature2"]),
            "latent": compare(seen["port_sample"][0], seen["up_sample"]["latent"]),
            "camera_latent": compare(seen["port_sample"][1], seen["up_sample"]["camera"]),
            "xyz": compare(ours.xyz, theirs.xyz),
            "scales": compare(ours.scales, theirs.scaling),
            "rotations": compare(ours.rotations, theirs.rotation),
            "opacities": compare(ours.opacities, theirs.opacity.reshape(-1)),
            "sh": compare(ours.sh, theirs.features_dc),
            "camera": {key: ours.camera[key] for key in output.camera} == output.camera,
        }
        result.save(args.out / f"{name}_port.ply")
        theirs.save_ply(str(args.out / f"{name}_upstream.ply"))
        row["ply_vertex_bytes"] = ply_vertex_bytes(args.out / f"{name}_port.ply") == ply_vertex_bytes(
            args.out / f"{name}_upstream.ply"
        )
        row["bitwise"] = row["camera"] and row["ply_vertex_bytes"] and all(
            value["equal"] for value in row.values() if isinstance(value, dict)
        )
        print(json.dumps(row), flush=True)
        report["cases"].append(row)
    report["all_bitwise"] = all(row["bitwise"] for row in report["cases"])
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print("ALL BITWISE" if report["all_bitwise"] else "MISMATCH")
    return 0 if report["all_bitwise"] else 1


if __name__ == "__main__":
    sys.exit(main())
