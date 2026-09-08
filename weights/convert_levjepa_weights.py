"""Convert the pinned official LeVJEPA Large snapshot to LibreYOLO format.

The converter downloads only ``config.json`` and ``model.safetensors``. It
does not download or import the snapshot's remote Python implementation.

Usage::

    python weights/convert_levjepa_weights.py \
        --output weights/LibreLeVJEPAl-embed.pt
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    repo_root,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

add_repo_root_to_path()

from libreyolo.models.levjepa.nn import (  # noqa: E402
    LEVJEPA_CONFIGS,
    LeVJEPAConfig,
    LeVJEPAModel,
)
from libreyolo.models.levjepa.preprocess import (  # noqa: E402
    PIXEL_MEAN,
    PIXEL_STD,
    TARGET_FPS,
)


SOURCE_REPO = "galilai-group/LeVJEPA-VideoMix-Large"
SOURCE_REVISION = "e831a0347737fcaa660b39c57d41c109de399845"
SOURCE_LICENSE = "CC-BY-NC-4.0"


def _source_tree_version() -> str:
    pyproject = tomllib.loads(
        (repo_root() / "pyproject.toml").read_text(encoding="utf-8")
    )
    return str(pyproject["project"]["version"])


def _load_snapshot() -> tuple[dict, dict]:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    snapshot = Path(
        snapshot_download(
            SOURCE_REPO,
            revision=SOURCE_REVISION,
            allow_patterns=["config.json", "model.safetensors"],
        )
    )
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    return load_file(str(snapshot / "model.safetensors")), config


def _validate_config(config: dict) -> None:
    expected = dict(LEVJEPA_CONFIGS["l"])
    expected["model_type"] = "levjepa"
    for key, value in expected.items():
        if config.get(key) != value:
            raise SystemExit(
                f"pinned config {key}={config.get(key)!r} does not match "
                f"the audited value {value!r}"
            )
    if config.get("token_drop_rate") != 0.0:
        raise SystemExit("released inference snapshot must disable token dropping")


def convert(output: Path) -> None:
    state_dict, config = _load_snapshot()
    _validate_config(config)

    model = LeVJEPAModel(LeVJEPAConfig.for_size("l"))
    model.load_state_dict(state_dict, strict=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != 303_099_904:
        raise SystemExit(f"unexpected parameter count: {parameters}")

    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="levjepa",
        size="l",
        nc=1,
        names={0: "embedding"},
        task="embed",
        libreyolo_version=_source_tree_version(),
        supported_tasks=("embed",),
        default_task="embed",
        source_repo=SOURCE_REPO,
        source_revision=SOURCE_REVISION,
        source_license=SOURCE_LICENSE,
        input_kind="video",
        input_size=224,
        frames_per_clip=16,
        target_fps=TARGET_FPS,
        patch_size=16,
        tubelet_size=1,
        hidden_dim=1024,
        attention_mode="block_causal",
        embedding_pool="cls_final_l2",
        pixel_mean=list(PIXEL_MEAN),
        pixel_std=list(PIXEL_STD),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(wrapped, temporary)
    temporary.replace(output)
    print(
        f"Wrote {output} ({output.stat().st_size / 1e9:.2f} GB, "
        f"{parameters / 1e6:.1f}M parameters)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weights/LibreLeVJEPAl-embed.pt"),
    )
    convert(parser.parse_args().output)
