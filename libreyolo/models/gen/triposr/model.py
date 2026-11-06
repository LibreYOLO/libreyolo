"""TripoSR: one image in, the object out as a vertex-coloured triangle mesh.

TripoSR (https://github.com/VAST-AI-Research/TripoSR, Tripo AI and Stability
AI, 2024) is a feed-forward image-to-3D model: a DINO ViT-B/16 encodes the
512px input, a transformer decodes a triplane NeRF from it in one pass, and
the mesh is the marching-cubes isosurface of the NeRF's density. There is no
sampling, so the output depends on the image and the knobs only: ``seed`` and
``n`` are accepted like every LibreGen family and give the same mesh.

The port is plain PyTorch and runs in fp32 on CUDA, Apple MPS and CPU, as
upstream does. Two deliberate differences: marching cubes runs through
scikit-image (BSD-3-Clause) instead of torchmcubes, which extracts the same
isosurface with different per-cube triangulation tables; and background
removal runs on LibreYOLO's BiRefNet network instead of rembg's U2Net. The
DINO encoder is written here, reproducing transformers 4.35 (the version
upstream pins), including its position-embedding interpolation.

The mesh is a ``Meshes`` payload (``Results.meshes``) with one row: the
object, in the object frame of ADR 0031 (z up, inside ``[-0.5, 0.5]^3``; the
input view looks along -x, so the object's front faces +x). Upstream's frame
is the same up to a fixed scale: vertices are upstream's divided by
``2 * 0.87``, the side of its query cube. ``body_model`` is ``"object"``,
the body parameters are empty, ``faces`` are wound counter-clockwise seen
from outside and ``extras["vertex_colors"]`` holds RGB in ``[0, 1]``.

Nothing is gated and no login is needed. The network weights are LibreYOLO's
byte-identical mirror ``LibreYOLO/LibreTripoSR`` of ``stabilityai/TripoSR``
(MIT); the BiRefNet weights come from ``LibreYOLO/LibreTRELLISl`` (MIT). Both
at pinned revisions.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Sequence, Tuple

from ....utils.results import Meshes
from ..base import _INSTALL_HINT
from ..base3d import LibreGen3DModel

_CONFIG_FILE = "config.yaml"
_WEIGHTS_FILE = "model.ckpt"
_UPSTREAM_FILES = (_CONFIG_FILE, _WEIGHTS_FILE)
# BiRefNet weights (ZhengPeng7/BiRefNet, MIT), as the TRELLIS mirror ships them.
_BIREFNET_REPO = "LibreYOLO/LibreTRELLISl"
_BIREFNET_REVISION = "75c784a8fa56ebfb76f46d9b9d830caec25826a7"
_BIREFNET_FILE = "birefnet.safetensors"
_BIREFNET_TRAINING_ONLY = ("decoder.conv_ms_spvn_", "decoder.gdt_convs_pred_")
# Upstream's run.py default --foreground-ratio 0.85, as crop side over object size.
_DEFAULT_MARGIN = 1 / 0.85


class LibreTripoSR(LibreGen3DModel):
    """TripoSR behind the LibreGen surface: ``model("chair.png").save("chair.obj")``."""

    FAMILY = "triposr"
    FILENAME_PREFIX = "LibreTripoSR"
    HF_REPOS: ClassVar[Dict[str, str]] = {"": "LibreYOLO/LibreTripoSR"}
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "": "249e9d7749b01a89542aab66c25273cf4ce48646"
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"": 512}
    SNAPSHOT_ALLOW_PATTERNS: ClassVar[Tuple[str, ...]] = _UPSTREAM_FILES
    SAVE_SUFFIXES: ClassVar[Tuple[str, ...]] = (".obj",)
    KNOBS = ("resolution", "threshold", "margin")
    DEFAULTS: ClassVar[Dict[str, Any]] = {
        "resolution": 256,
        "threshold": 25.0,
        "margin": _DEFAULT_MARGIN,
    }
    _LICENSE_NOTICE = ""

    def __init__(self, size=None, *, compile: bool = False, **kwargs):
        if compile:
            raise ValueError(
                "LibreTripoSR has no compiled path; compile=True is a Speridlabs Idol option."
            )
        super().__init__(size, compile=False, **kwargs)

    @property
    def model_name(self) -> str:
        return "triposr"

    @classmethod
    def matches_directory(cls, path: Any) -> bool:
        """True when ``path`` holds upstream's ``config.yaml`` and ``model.ckpt``."""
        try:
            folder = Path(path)
            if not all((folder / name).is_file() for name in _UPSTREAM_FILES):
                return False
            config = (folder / _CONFIG_FILE).read_text(encoding="utf-8")
        except (OSError, TypeError, ValueError):
            return False
        return "tsr.models" in config

    def _check_runtime(self) -> None:
        try:
            import skimage  # noqa: F401
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, snapshot_dir: str) -> None:
        import torch
        import yaml
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        from ...birefnet.nn import LibreBiRefNetModel
        from .nn import CONFIG, TripoSR

        folder = Path(snapshot_dir)
        config = yaml.safe_load((folder / _CONFIG_FILE).read_text(encoding="utf-8"))
        cfg = dict(CONFIG)
        cfg.update(
            cond_image_size=config["cond_image_size"],
            plane_size=config["tokenizer"]["plane_size"],
            radius=config["renderer"]["radius"],
            density_bias=config["renderer"]["density_bias"],
        )
        network = TripoSR(cfg)
        state = torch.load(
            folder / _WEIGHTS_FILE, map_location="cpu", weights_only=True
        )
        network.load_state_dict(state, strict=True)
        network = network.eval().requires_grad_(False).to(self.device)

        birefnet_file = folder / _BIREFNET_FILE
        if not birefnet_file.is_file():
            birefnet_file = Path(
                hf_hub_download(
                    _BIREFNET_REPO, _BIREFNET_FILE, revision=_BIREFNET_REVISION
                )
            )
        birefnet = LibreBiRefNetModel(size="l")
        missing, unexpected = birefnet.load_state_dict(
            load_file(str(birefnet_file)), strict=False
        )
        missing = [k for k in missing if not k.startswith(_BIREFNET_TRAINING_ONLY)]
        if missing or unexpected:
            raise RuntimeError(
                f"birefnet: missing {missing[:5]}, unexpected {unexpected[:5]}"
            )
        birefnet = birefnet.eval().requires_grad_(False).to(_birefnet_device(self.device))
        self._pipeline = _TripoSRModules(network=network, birefnet=birefnet)

    def _move_loaded(self, device) -> None:
        self._pipeline.network.to(device)
        self._pipeline.birefnet.to(_birefnet_device(device))

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def _resolve_knobs(self, knobs: Dict[str, Any]) -> Dict[str, Any]:
        settings = super()._resolve_knobs(knobs)
        resolution = settings["resolution"]
        if (
            isinstance(resolution, bool)
            or not isinstance(resolution, Integral)
            or not 32 <= resolution <= 1024
        ):
            raise ValueError(
                f"resolution must be an integer from 32 to 1024, got {resolution!r}."
            )
        settings["resolution"] = int(resolution)
        for name, low in (("threshold", 0.0), ("margin", 1.0)):
            value = settings[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < low
                or (name == "threshold" and value == low)
            ):
                relation = ">" if name == "threshold" else ">="
                raise ValueError(
                    f"{name} must be a number {relation} {low:g}, got {value!r}."
                )
            settings[name] = float(value)
        return settings

    def _generate_3d(
        self, image, seeds: Sequence[int], knobs: Dict[str, Any]
    ) -> List[Meshes]:
        import torch

        from .nn import extract_mesh
        from .preprocess import preprocess_image, to_tensor

        modules = self._pipeline
        network = modules.network
        with torch.no_grad():
            prepared = preprocess_image(image, modules.birefnet, margin=knobs["margin"])
            pixels = to_tensor(prepared, network.cfg["cond_image_size"]).to(self.device)
            triplane = network(pixels)[0]
            vertices, faces, colours = extract_mesh(
                network, triplane, knobs["resolution"], knobs["threshold"]
            )
        # Upstream's query cube [-0.87, 0.87]^3 -> ADR 0031's [-0.5, 0.5]^3.
        vertices = vertices / (2 * network.radius)
        # Deterministic: every seed gives the same mesh, each its own copy.
        return [
            Meshes(
                torch.zeros(1, 0),
                torch.zeros(1, 0),
                torch.zeros(1, 0),
                torch.zeros(1, 3),
                body_model="object",
                vertices=vertices.unsqueeze(0).clone(),
                faces=faces.clone(),
                extras={"vertex_colors": colours.unsqueeze(0).clone()},
            )
            for _ in seeds
        ]


def _mps_has_deform_conv2d() -> bool:
    """True when the installed torchvision registers an MPS ``deform_conv2d`` kernel."""
    import torch
    import torchvision  # noqa: F401  (registers the torchvision:: operators)

    return torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::deform_conv2d", "MPS")


def _birefnet_device(device):
    """Where BiRefNet runs: ``device``, or the CPU on MPS without a deform_conv2d kernel.

    BiRefNet's decoder calls ``torchvision.ops.deform_conv2d``. Older
    torchvision releases (0.20 among them; the floor is 0.19) have no MPS
    kernel for it, so background removal of an RGB photo would fail there
    before TripoSR runs. Newer releases run it on MPS and keep it there.
    """
    import torch

    device = torch.device(device)
    if device.type == "mps" and not _mps_has_deform_conv2d():
        return torch.device("cpu")
    return device


class _TripoSRModules:
    """The loaded networks; ``model.pipeline`` returns this."""

    def __init__(self, *, network, birefnet):
        self.network = network
        self.birefnet = birefnet
