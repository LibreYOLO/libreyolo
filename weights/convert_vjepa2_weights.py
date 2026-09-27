"""Convert pinned upstream V-JEPA 2.0 snapshots to LibreYOLO checkpoints.

Encoders (milestone 1)::

    python weights/convert_vjepa2_weights.py --size l256 --task embed \
        --output weights/LibreVJEPA2l256-embed.pt

Attentive probes (milestone 2)::

    python weights/convert_vjepa2_weights.py --size l256 --task classify \
        --variant ssv2 --output weights/LibreVJEPA2l256-cls-ssv2.pt

The snapshot is fetched at its exact pinned revision, so the conversion is
deterministic. Keys are mapped explicitly and loaded strictly: a key is only
dropped when it belongs to a named, asserted set, never by substring match.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

add_repo_root_to_path()

from libreyolo.models.vjepa2.model import (  # noqa: E402
    ENCODER_FRAMES,
    PROBE_CLASSES,
    PROBE_FRAMES,
    LibreVJEPA2,
)
from libreyolo.models.vjepa2.nn import (  # noqa: E402
    VJEPA2_CONFIGS,
    LibreVJEPA2Classifier,
    LibreVJEPA2Encoder,
    VJEPA2Config,
)
from libreyolo.models.vjepa2.preprocess import (  # noqa: E402
    DEFAULT_FRAME_STRIDE,
    PIXEL_MEAN,
    PIXEL_STD,
)

# (repo, revision, weight license) per artifact. Licenses are per artifact and
# are deliberately not collapsed into one family-wide claim: the two g encoders
# are Apache-2.0 while everything else is MIT.
ENCODER_SOURCES: Dict[str, Tuple[str, str, str]] = {
    "l256": ("facebook/vjepa2-vitl-fpc64-256", "b3c1679b7c34d3255ef3547f27c7b226aefab26f", "MIT"),
    "h256": ("facebook/vjepa2-vith-fpc64-256", "b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5", "MIT"),
    "g256": ("facebook/vjepa2-vitg-fpc64-256", "875c192b7b704b87d1e1d99345769632dd5f739a", "Apache-2.0"),
    "g384": ("facebook/vjepa2-vitg-fpc64-384", "12ca91694b230e0d4b5b0078af6f4ae1d51e933d", "Apache-2.0"),
}

PROBE_SOURCES: Dict[Tuple[str, str], Tuple[str, str, str]] = {
    ("l256", "ssv2"): ("facebook/vjepa2-vitl-fpc16-256-ssv2", "4aa02df83918538fc21cfaf576382fa20e489a80", "MIT"),
    ("l256", "diving48"): ("facebook/vjepa2-vitl-fpc32-256-diving48", "71ae2a8b1ff5a297aeeaae9b5e64c7a2e5e6a633", "MIT"),
    ("g384", "ssv2"): ("facebook/vjepa2-vitg-fpc64-384-ssv2", "9f5fd615cb6f79065a28edcf1cc3ef25010dddfa", "MIT"),
    ("g384", "diving48"): ("facebook/vjepa2-vitg-fpc32-384-diving48", "0b48243375319bd8e03e3cd5560d957095429189", "MIT"),
}

# The predictor is the self-supervised head. It is the ONLY group the encoder
# conversion may drop, and it is asserted rather than filtered by substring.
ENCODER_ALLOWED_DROPS = {"predictor"}


def _load_snapshot(repo: str, revision: str) -> Tuple[dict, dict]:
    """Fetch a pinned snapshot and return (state_dict, config dict)."""
    import json

    from huggingface_hub import snapshot_download

    # Fetch only what the conversion reads. These repos ship the weights in
    # BOTH safetensors and pickle form, and pulling both doubles the download
    # and the on-disk cache for no benefit (the h256 repo is ~13 GB with both).
    local = Path(
        snapshot_download(
            repo,
            revision=revision,
            allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"],
        )
    )
    config = json.loads((local / "config.json").read_text(encoding="utf-8"))

    shards = sorted(local.glob("*.safetensors"))
    if shards:
        from safetensors.torch import load_file

        state: dict = {}
        for shard in shards:
            state.update(load_file(str(shard)))
        return state, config

    bins = sorted(local.glob("*.bin"))
    if not bins:
        raise SystemExit(f"no safetensors or .bin weights found in {local}")
    state = {}
    for shard in bins:
        state.update(torch.load(shard, map_location="cpu", weights_only=True))
    return state, config


def _validate_config(size: str, config: dict) -> None:
    """Validate the full pinned config before mapping a single key."""
    table = VJEPA2_CONFIGS[size]
    checks = {
        "hidden_size": table["hidden_size"],
        "num_attention_heads": table["num_attention_heads"],
        "num_hidden_layers": table["num_hidden_layers"],
        "crop_size": table["crop_size"],
        "patch_size": 16,
        "tubelet_size": 2,
    }
    for key, expected in checks.items():
        actual = config.get(key)
        if actual != expected:
            raise SystemExit(
                f"[{size}] pinned config {key}={actual!r} disagrees with the family "
                f"table {expected!r}; re-audit before converting"
            )
    ratio = float(config.get("mlp_ratio", 0))
    if abs(ratio - float(table["mlp_ratio"])) > 1e-9:
        raise SystemExit(
            f"[{size}] pinned config mlp_ratio={ratio} != table {table['mlp_ratio']}"
        )
    # V-JEPA 2.1 and V-JEPA2-AC must never be accepted as 2.0.
    model_type = config.get("model_type")
    if model_type != "vjepa2":
        raise SystemExit(
            f"[{size}] refusing snapshot with model_type={model_type!r}; this "
            "converter only accepts V-JEPA 2.0 encoders. V-JEPA 2.1 and "
            "V-JEPA2-AC are different architectures, not size aliases."
        )
    if "pred_num_mask_tokens" in config and config.get("pred_hidden_size") is None:
        raise SystemExit(f"[{size}] unexpected predictor config layout")


def _base_metadata(size: str, repo: str, revision: str, license_id: str, frames: int) -> dict:
    """Extra checkpoint metadata.

    ``family``/``size``/``task``/``nc``/``names`` are passed to the wrapper as
    explicit arguments, so they are deliberately absent here to avoid
    duplicate-keyword collisions.
    """
    return {
        "source_repo": repo,
        "source_revision": revision,
        "source_license": license_id,
        "input_kind": "video",
        "input_size": VJEPA2_CONFIGS[size]["crop_size"],
        "frames_per_clip": frames,
        "frame_stride": DEFAULT_FRAME_STRIDE,
        "patch_size": 16,
        "tubelet_size": 2,
        "hidden_dim": VJEPA2_CONFIGS[size]["hidden_size"],
        "embedding_pool": "mean_final_tokens_l2",
        "pixel_mean": list(PIXEL_MEAN),
        "pixel_std": list(PIXEL_STD),
    }


def convert_encoder(size: str, output: Path) -> None:
    repo, revision, license_id = ENCODER_SOURCES[size]
    LibreVJEPA2.validate_artifact_name(size, "embed", None)

    state, config = _load_snapshot(repo, revision)
    _validate_config(size, config)

    encoder_state = {
        k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")
    }
    dropped = {k.split(".")[0] for k in state if not k.startswith("encoder.")}
    unexpected = dropped - ENCODER_ALLOWED_DROPS
    if unexpected:
        raise SystemExit(
            f"[{size}] snapshot has unexpected top-level groups {sorted(unexpected)}. "
            "Refusing to drop keys that are not in the asserted set "
            f"{sorted(ENCODER_ALLOWED_DROPS)}."
        )

    model = LibreVJEPA2Encoder(VJEPA2Config.for_size(size, frames_per_clip=ENCODER_FRAMES))
    model.load_state_dict(encoder_state, strict=True)
    params = sum(p.numel() for p in model.parameters())
    print(
        f"[{size}] strict-loaded {len(encoder_state)} tensors ({params / 1e6:.1f}M params); "
        f"dropped {sorted(dropped)}"
    )

    metadata = _base_metadata(size, repo, revision, license_id, ENCODER_FRAMES)
    metadata["variant"] = None

    wrapped = wrap_libreyolo_checkpoint(
        encoder_state,
        model_family="vjepa2",
        size=size,
        # An encoder has no class head. The checkpoint schema requires a
        # positive nc, so this is an explicit placeholder rather than a
        # meaningful class count -- it is deliberately NOT the embedding
        # dimension, which is carried separately as hidden_dim and would
        # otherwise be mistaken for a head width by the rebuild path.
        nc=1,
        names={0: "embedding"},
        task="embed",
        supported_tasks=("embed", "classify"),
        default_task="embed",
        **metadata,
    )
    _write(wrapped, output)


def convert_probe(size: str, variant: str, output: Path) -> None:
    key = (size, variant)
    if key not in PROBE_SOURCES:
        raise SystemExit(
            f"no released probe for size={size!r} variant={variant!r}; "
            f"published: {sorted(PROBE_SOURCES)}"
        )
    LibreVJEPA2.validate_artifact_name(size, "classify", variant)
    repo, revision, license_id = PROBE_SOURCES[key]
    frames = PROBE_FRAMES[key]
    nc = PROBE_CLASSES[variant]

    state, config = _load_snapshot(repo, revision)
    _validate_config(size, config)

    mapped: dict = {}
    for k, v in state.items():
        if k.startswith("vjepa2.encoder."):
            mapped["encoder." + k[len("vjepa2.encoder."):]] = v
        elif k.startswith("pooler.") or k.startswith("classifier."):
            mapped[k] = v
    dropped = {
        k.split(".")[1] if k.startswith("vjepa2.") else k.split(".")[0]
        for k in state
        if not (
            k.startswith("vjepa2.encoder.")
            or k.startswith("pooler.")
            or k.startswith("classifier.")
        )
    }
    unexpected = dropped - ENCODER_ALLOWED_DROPS
    if unexpected:
        raise SystemExit(
            f"[{size}-{variant}] unexpected groups {sorted(unexpected)}; refusing "
            "to drop keys outside the asserted set"
        )

    model = LibreVJEPA2Classifier(
        VJEPA2Config.for_size(size, frames_per_clip=frames), nc=nc
    )
    model.load_state_dict(mapped, strict=True)
    print(
        f"[{size}-{variant}] strict-loaded {len(mapped)} tensors, nc={nc}, "
        f"frames={frames}; dropped {sorted(dropped)}"
    )

    names = config.get("id2label")
    if names:
        names = {int(k): v for k, v in names.items()}
        if len(names) != nc:
            raise SystemExit(
                f"[{size}-{variant}] id2label has {len(names)} entries but nc={nc}"
            )
    else:
        raise SystemExit(
            f"[{size}-{variant}] snapshot supplies no id2label; refusing to invent "
            "generic numeric class names for a published artifact"
        )

    metadata = _base_metadata(size, repo, revision, license_id, frames)
    metadata.update(
        {
            "variant": variant,
            "probe_depth": 3,
            "dataset": variant,
        }
    )

    wrapped = wrap_libreyolo_checkpoint(
        mapped,
        model_family="vjepa2",
        size=size,
        nc=nc,
        names=names,
        task="classify",
        supported_tasks=("embed", "classify"),
        default_task="embed",
        **metadata,
    )
    _write(wrapped, output)


def _write(wrapped: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.replace(output)  # atomic
    size_mb = output.stat().st_size / 1e6
    print(f"Wrote {output} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True, choices=sorted(ENCODER_SOURCES))
    parser.add_argument("--task", required=True, choices=["embed", "classify"])
    parser.add_argument("--variant", choices=sorted(PROBE_CLASSES), default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.task == "embed":
        if args.variant:
            raise SystemExit("--variant is not valid for --task embed")
        convert_encoder(args.size, args.output)
    else:
        if not args.variant:
            raise SystemExit("--task classify requires --variant")
        convert_probe(args.size, args.variant, args.output)


if __name__ == "__main__":
    main()
