"""Convert pinned OpenCLIP-compatible PE Core weights to LibreYOLO format.

The native LibreYOLO PE towers mirror the upstream converted-checkpoint layout
(``visual.trunk.*`` / ``text.*`` / ``logit_scale``) exactly, so conversion is a
strict metadata-wrap with **no key remapping**. Every architectural field is
asserted against the closed configuration table before writing; an unknown or
changed upstream config fails loudly rather than silently producing a wrong
model.

Network fetching is a separate, explicit step (``--download``) from tensor
mapping, so a conversion can always be run against a pinned local snapshot.

Usage::

    # From an explicit local snapshot
    python weights/convert_pe_weights.py open_clip_model.safetensors \\
        weights/LibrePEt16-cls.pt --size t16

    # Fetch the pinned revision first, then convert
    python weights/convert_pe_weights.py --download --size t16 \\
        weights/LibrePEt16-cls.pt

    # All five canonical artifacts
    python weights/convert_pe_weights.py --download --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _conversion_utils import save_checkpoint, wrap_libreyolo_checkpoint  # noqa: E402

from libreyolo.models.pe.nn import PE_CONFIGS, build_pe_model  # noqa: E402

# Pinned weight sources. Both id and revision are recorded into the checkpoint
# metadata and the weight notice.
PINNED_SOURCES = {
    "t16": ("timm/PE-Core-T-16-384", "7fe539ed578ac49a1c2b4f946e4b0747704c825a"),
    "s16": ("timm/PE-Core-S-16-384", "3249a38eb1c432ec19231c6fe1774acb6a4e4efe"),
    "b16": ("timm/PE-Core-B-16", "0038414f37721c5eafc1a5e0da802d291c909de3"),
    "l14": ("timm/PE-Core-L-14-336", "8eff41b3f687e50a323662c2dda5eb3588c6dd35"),
    "g14": ("timm/PE-Core-bigG-14-448", "17aa0c25addfa14198fa2ff73d845a22d433432e"),
}
SOURCE_LICENSE = "Apache-2.0"
SOURCE_WEIGHT_FILE = "open_clip_model.safetensors"


def download_pinned(size: str) -> str:
    """Fetch the pinned revision of one source snapshot. Network step only."""
    from huggingface_hub import hf_hub_download

    repo, revision = PINNED_SOURCES[size]
    return hf_hub_download(repo, SOURCE_WEIGHT_FILE, revision=revision)


def load_source_state(path: str) -> dict:
    if str(path).endswith(".safetensors"):
        import safetensors.torch as st

        return st.load_file(path)
    from _conversion_utils import extract_state_dict, load_checkpoint

    return extract_state_dict(load_checkpoint(path), prefer_ema=True)


def infer_size(state: dict) -> str:
    """Infer the LibreYOLO size from the source tensors alone."""
    patch = state.get("visual.trunk.patch_embed.proj.weight")
    pos = state.get("text.positional_embedding")
    if patch is None or pos is None:
        raise ValueError(
            "Source does not look like an OpenCLIP-compatible PE Core checkpoint "
            "(missing 'visual.trunk.patch_embed.proj.weight' / "
            "'text.positional_embedding')."
        )
    width, patch_size, context = (
        int(patch.shape[0]),
        int(patch.shape[-1]),
        int(pos.shape[0]),
    )
    for size, cfg in PE_CONFIGS.items():
        if (
            cfg.embed_dim == width
            and cfg.patch_size == patch_size
            and cfg.context_length == context
        ):
            return size
    raise ValueError(
        f"No PE size matches the source config (width={width}, "
        f"patch={patch_size}, context={context}). Refusing to guess."
    )


def assert_config(state: dict, size: str) -> None:
    """Validate every architectural invariant before writing a checkpoint."""
    cfg = PE_CONFIGS[size]
    checks = {
        "vision width": (
            int(state["visual.trunk.patch_embed.proj.weight"].shape[0]),
            cfg.embed_dim,
        ),
        "patch size": (
            int(state["visual.trunk.patch_embed.proj.weight"].shape[-1]),
            cfg.patch_size,
        ),
        "context length": (
            int(state["text.positional_embedding"].shape[0]),
            cfg.context_length,
        ),
        "text width": (int(state["text.positional_embedding"].shape[1]), cfg.text_width),
        "vocab size": (
            int(state["text.token_embedding.weight"].shape[0]),
            cfg.vocab_size,
        ),
        "embedding dim": (
            int(state["text.text_projection"].shape[1]),
            cfg.projection_dim,
        ),
        "projection head": (int(state["visual.trunk.head.weight"].shape[0]),
                            cfg.projection_dim),
        "positional tokens": (
            int(state["visual.trunk.pos_embed"].shape[1]),
            cfg.num_patches + cfg.num_prefix_tokens,
        ),
        "attn pool latent": (
            int(state["visual.trunk.attn_pool.latent"].shape[-1]),
            cfg.embed_dim,
        ),
    }
    for label, (found, expected) in checks.items():
        if found != expected:
            raise ValueError(
                f"{label} mismatch for size={size}: source has {found}, "
                f"config expects {expected}."
            )

    depth = 1 + max(
        int(k.split(".")[3])
        for k in state
        if k.startswith("visual.trunk.blocks.")
    )
    if depth != cfg.depth:
        raise ValueError(
            f"vision depth mismatch for size={size}: source has {depth}, "
            f"config expects {cfg.depth}."
        )
    text_layers = 1 + max(
        int(k.split(".")[3])
        for k in state
        if k.startswith("text.transformer.resblocks.")
    )
    if text_layers != cfg.text_layers:
        raise ValueError(
            f"text depth mismatch for size={size}: source has {text_layers}, "
            f"config expects {cfg.text_layers}."
        )

    has_cls = "visual.trunk.cls_token" in state
    if has_cls != cfg.class_token:
        raise ValueError(
            f"class-token mismatch for size={size}: source has cls_token="
            f"{has_cls}, config expects {cfg.class_token}."
        )


def convert(input_path: str, output_path: str, size: str | None = None) -> None:
    state = load_source_state(input_path)
    inferred = infer_size(state)
    if size is not None and size != inferred:
        raise ValueError(
            f"--size {size!r} conflicts with the source config, which is "
            f"{inferred!r}. Refusing to mislabel the checkpoint."
        )
    size = inferred
    cfg = PE_CONFIGS[size]
    assert_config(state, size)

    # Strict dry-load: no missing, unexpected, or silently dropped keys.
    model = build_pe_model(size)
    result = model.load_state_dict(state, strict=True)
    assert not result.missing_keys, f"missing: {result.missing_keys}"
    assert not result.unexpected_keys, f"unexpected: {result.unexpected_keys}"
    print(f"Strict load OK for size={size}: {len(state)} tensors, no key diff.")

    repo, revision = PINNED_SOURCES[size]
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    wrapped = wrap_libreyolo_checkpoint(
        cpu_state,
        model_family="pe",
        size=size,
        # Open-vocabulary: there is no fixed head. 1000 records the ImageNet-1k
        # class set the family defaults to on construction; set_classes()
        # replaces it freely.
        nc=1000,
        task="classify",
        supported_tasks=("classify", "embed"),
        default_task="classify",
    )
    wrapped.update(
        {
            "source_repo": repo,
            "source_revision": revision,
            "source_weight_file": SOURCE_WEIGHT_FILE,
            "source_license": SOURCE_LICENSE,
            "input_size": cfg.image_size,
            "embedding_dim": cfg.projection_dim,
            "text_context_length": cfg.context_length,
            "video_pool": "mean_frame_embeddings",
            "pixel_mean": [0.5, 0.5, 0.5],
            "pixel_std": [0.5, 0.5, 0.5],
        }
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.replace(out)  # atomic
    size_mb = out.stat().st_size / 1e6
    print(f"Wrote {out} (size={size}, {size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", help="source .safetensors / .pt snapshot")
    parser.add_argument("output", nargs="?", help="destination LibreYOLO .pt")
    parser.add_argument("--size", choices=list(PE_CONFIGS))
    parser.add_argument(
        "--download",
        action="store_true",
        help="fetch the pinned revision instead of taking a local input path",
    )
    parser.add_argument(
        "--all", action="store_true", help="convert all five canonical artifacts"
    )
    parser.add_argument("--out-dir", default="weights")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    if args.all:
        for size in PE_CONFIGS:
            dest = Path(args.out_dir) / f"LibrePE{size}-cls.pt"
            convert(download_pinned(size), str(dest), size)
        return

    if args.download:
        if not args.size:
            parser.error("--download requires --size")
        source = download_pinned(args.size)
        dest = args.output or str(
            Path(args.out_dir) / f"LibrePE{args.size}-cls.pt"
        )
    else:
        if not args.input or not args.output:
            parser.error("input and output are required without --download/--all")
        source, dest = args.input, args.output

    convert(source, dest, args.size)


if __name__ == "__main__":
    main()
