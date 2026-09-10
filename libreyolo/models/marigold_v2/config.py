"""Released Marigold V2 adapters and their numeric output contracts."""

from dataclasses import dataclass

UPSTREAM_REPO = "huawei-bayerlab/marigold-v2-0"
UPSTREAM_REVISION = "6fd6d1ca246c9d2d99a4d8ac375a4eccc87178ad"
SOURCE_REVISION = "cc6a7031abcd59fd9e1ceff7fdd0d9687d389bc5"


@dataclass(frozen=True)
class Variant:
    task: str
    subfolder: str
    encoding: str | None
    sha256: str


VARIANTS = {
    "log-stage2": Variant(
        "depth",
        "depth/Log-stage2",
        "log_depth",
        "3edec69490ba8e8fdc8d63e130f93cb93f514360d415a22701250b2535056892",
    ),
    "log-stage1": Variant(
        "depth",
        "depth/Log-stage1",
        "log_depth",
        "f81ed1427ded526b393d7953ca8da43645499746240c2c96bd4178dd5c67881b",
    ),
    "log-layered": Variant(
        "depth",
        "depth/Log-layered",
        "log_depth",
        "a2d9d01046efbd370bc176eea76241a646b471677acddb3e603e5aa91fb266a9",
    ),
    "uniform-base": Variant(
        "depth",
        "depth/Uniform-base",
        "depth",
        "aa99ba7a0a6eeb7a0ab6f6784d3d9d3f9034540370f7726a20e3207a482d7fe7",
    ),
    "uniform-layered": Variant(
        "depth",
        "depth/Uniform-layered",
        "depth",
        "d94ab503811d51cca232acdb5636a8b2733675e2a2d992e63efc6e4ccba29be9",
    ),
    "disparity-base": Variant(
        "depth",
        "depth/Disparity-base",
        "inverse_depth",
        "00f2b06aeda7de20a2d3c4c51c353daceb7951dead91c49b84afb69ddc83794b",
    ),
    "disparity-layered": Variant(
        "depth",
        "depth/Disparity-layered",
        "inverse_depth",
        "6961313f5f1dd2a95bba2dbe1349f8565a32a36cafda2d0d90fd2d1eec3079f6",
    ),
    "normal": Variant(
        "normal",
        "normals",
        None,
        "b98dfcafa8cf86cf042d8a8e615b8b1a7d7f17c939f1716954c1b57b1b4af15a",
    ),
    "albedo": Variant(
        "albedo",
        "albedo",
        None,
        "dd955f940becbadb0cf854f51608ed8febec0636891467c05060bc44129d2503",
    ),
}
PROMPTS = {
    "depth": "qwen_edit_2509_qwen_depth_realimg512",
    "normal": "qwen_edit_2509_qwen_normals_dummy512",
    "albedo": "qwen_edit_2509_qwen_albedo_rgb_dummy512",
}


def canonical_filename(variant):
    task = VARIANTS[variant].task
    suffix = "" if variant in {"log-stage2", "normal", "albedo"} else "-" + variant
    return f"LibreMarigoldV2b-{task}{suffix}.pt"


FILENAMES = {canonical_filename(name).lower(): name for name in VARIANTS}


def upstream_url(variant):
    folder = VARIANTS[variant].subfolder
    return f"https://huggingface.co/{UPSTREAM_REPO}/resolve/{UPSTREAM_REVISION}/{folder}/trainables.safetensors"
