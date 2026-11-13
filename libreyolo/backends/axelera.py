"""Axelera Metis inference backend for LibreYOLO.

Runs a ``<stem>_axelera_model`` directory written by ``format="axelera"``: the
compiled ``.axm`` executes on the Metis AIPU through Axelera's runtime
(``axelera-rt``). For YOLO-NAS, LibreYOLO decodes the raw head outputs on the
host (``libreyolo.postprocess.yolonas_raw``) before the shared
postprocessing; classifiers return logits that go straight to the shared
classification path.

This module also holds the SDK install and version helpers the exporter
uses, so the backend imports neither torch nor the export package.
Axelera's packages are proprietary software under Axelera's EULA, installed
by the user from Axelera's package index; LibreYOLO never bundles them.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..postprocess.yolonas_raw import (
    YOLONAS_RAW_LAYOUTS,
    check_output_windows,
    decode_raw_heads,
    raw_head_output_names,
)
from ..utils.serialization import warn_on_metadata_schema_version
from .base import BaseBackend, classify_eval_kwargs
from .metadata import ExportMetadataError, parse_export_metadata

logger = logging.getLogger(__name__)

#: SDK release LibreYOLO's Axelera support was validated against.
AXELERA_SDK_VERSION = "1.9.0"
#: Axelera's package index (the SDK is not on PyPI).
AXELERA_INDEX_URL = "https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple"
#: Opt-in switch for installing a missing SDK package automatically.
AUTOINSTALL_ENV = "LIBREYOLO_AUTOINSTALL"
DOCS_TROUBLESHOOTING = "docs/axelera.md#troubleshooting"

_DEVICE_ALIASES = {"auto", "cpu", "axelera", "metis", "aipu"}

#: Layout of classifier exports: one output, the ``[B, nc]`` logits.
CLASSIFY_LOGITS_LAYOUT = "classify_logits_v1"
#: Whole-graph dense exports (super-resolution, depth, semantic, heatmap pose):
#: the outputs are the family's ONNX outputs, decoded by the shared parsers.
GRAPH_OUTPUTS_LAYOUT = "graph_outputs_v1"
#: Super-resolution trunk on Metis; the final pixel shuffle and the nearest-
#: upsampled input skip (Real-ESRGAN's compact network) run on the host.
RESTORE_SHUFFLE_SKIP_LAYOUT = "restore_shuffle_skip_v1"
_PASS_THROUGH_LAYOUTS = (CLASSIFY_LOGITS_LAYOUT, GRAPH_OUTPUTS_LAYOUT)
_HOST_TAIL_LAYOUTS = (RESTORE_SHUFFLE_SKIP_LAYOUT,)


def shuffle_and_skip(body: np.ndarray, image: np.ndarray, scale: int) -> np.ndarray:
    """``pixel_shuffle(body, scale) + nearest_upsample(image, scale)`` in numpy.

    ``body`` is ``[1, C*scale*scale, H, W]`` (PyTorch ``PixelShuffle`` channel
    order) and ``image`` the ``[1, C, H, W]`` network input.
    """
    b, c, h, w = body.shape
    r = int(scale)
    channels = c // (r * r)
    out = body.reshape(b, channels, r, r, h, w).transpose(0, 1, 4, 2, 5, 3)
    out = out.reshape(b, channels, h * r, w * r)
    base = np.repeat(np.repeat(image, r, axis=2), r, axis=3)
    return (out + base[:, :channels]).astype(np.float32, copy=False)


def axelera_install_command(package: str) -> str:
    """The one-line pip command that installs an Axelera SDK package."""
    return (
        f"pip install --extra-index-url {AXELERA_INDEX_URL} "
        f"{package}=={AXELERA_SDK_VERSION}"
    )


def _autoinstall_enabled() -> bool:
    return os.environ.get(AUTOINSTALL_ENV, "").strip().lower() in {"1", "true"}


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_sdk_package(module: str, package: str, purpose: str) -> str | None:
    """Make ``module`` importable or raise an ImportError naming the install."""
    if not _module_available(module):
        command = axelera_install_command(package)
        if not _autoinstall_enabled():
            raise ImportError(
                f"{purpose} requires Axelera's {package} package, which is not "
                f"installed. Install it with:\n  {command}\n"
                f"or set {AUTOINSTALL_ENV}=1 to let LibreYOLO install it. "
                f"{package} is proprietary software under Axelera's EULA and "
                "is installed from Axelera's package index, not PyPI."
            )
        logger.warning(
            "%s=1: installing %s %s from Axelera's package index. It is "
            "proprietary software under Axelera's EULA.",
            AUTOINSTALL_ENV,
            package,
            AXELERA_SDK_VERSION,
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--extra-index-url",
                AXELERA_INDEX_URL,
                f"{package}=={AXELERA_SDK_VERSION}",
            ]
        )
        importlib.invalidate_caches()
        if not _module_available(module):
            raise ImportError(
                f"Installed {package} but {module} still cannot be imported. "
                f"Check the pip output above, then retry:\n  {command}"
            )
    version = _installed_version(package)
    if version is not None and version != AXELERA_SDK_VERSION:
        logger.warning(
            "%s %s is installed; LibreYOLO's Axelera support was validated with "
            "%s. Continuing.",
            package,
            version,
            AXELERA_SDK_VERSION,
        )
    return version


def require_axelera_devkit() -> str | None:
    """Ensure ``axelera.compiler`` is importable; return the devkit version."""
    return _require_sdk_package("axelera.compiler", "axelera-devkit", "Axelera export")


def require_axelera_rt() -> str | None:
    """Ensure ``axelera.runtime`` is importable; return the runtime version."""
    return _require_sdk_package("axelera.runtime", "axelera-rt", "Axelera inference")


def is_axelera_artifact(path: str | Path) -> bool:
    """True for a ``.axm`` file or a directory holding ``.axm`` + ``metadata.yaml``."""
    path = Path(path)
    if path.suffix.lower() == ".axm":
        return True
    return (
        path.is_dir()
        and (path / "metadata.yaml").is_file()
        and any(path.glob("*.axm"))
    )


def _host_l2norm(x: np.ndarray, *, scale, eps: float = 1e-12) -> np.ndarray:
    """``scale[c] * x / ||x||_2`` over the channels of each pixel (SSD's conv4_3 L2Norm, as ``F.normalize``)."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.sqrt((x * x).sum(axis=1, keepdims=True))
    weights = np.asarray(scale, dtype=np.float32).reshape(1, -1, 1, 1)
    return (weights * x / np.maximum(norm, np.float32(eps))).astype(np.float32)


#: Host operations a multi-part (chained) export may run between two ``.axm``
#: parts, by the ``host`` name its metadata gives them.
HOST_OPS: dict[str, Any] = {"l2norm": _host_l2norm}
_HOST_STEP_KEYS = ("host", "input", "output")


def _validate_chain(parts, axm: Path, final_names: list[str]) -> list[dict]:
    """Check a multi-part export's ``parts`` list; return it as plain dicts.

    Each part is a compiled model (``axm``, ``input``, ``outputs``) or a host
    operation (``host``, ``input``, ``output`` plus its parameters). The first
    part is the primary ``.axm`` and takes the images; every input must be an
    earlier output, and the parts together must produce ``final_names``.
    """
    if not isinstance(parts, list) or not parts:
        raise _foreign_artifact(axm, "parts in metadata.yaml must be a non-empty list")
    available = {"images"}
    steps = []
    for index, part in enumerate(parts):
        if not isinstance(part, dict):
            raise _foreign_artifact(axm, f"parts[{index}] is not a mapping")
        step = dict(part)
        source = step.get("input")
        if source not in available:
            raise _foreign_artifact(axm, f"parts[{index}] reads {source!r}, which no earlier part produces")
        if "axm" in step:
            if index == 0 and (step["axm"] != axm.name or source != "images"):
                raise _foreign_artifact(axm, "parts[0] must be the primary .axm reading the images")
            if not (axm.parent / str(step["axm"])).is_file():
                raise FileNotFoundError(f"{axm.parent / str(step['axm'])} (parts[{index}]) is missing.")
            produced = [str(name) for name in step.get("outputs") or ()]
            if not produced:
                raise _foreign_artifact(axm, f"parts[{index}] lists no outputs")
        elif "host" in step:
            if index == 0:
                raise _foreign_artifact(axm, "parts[0] must be a compiled model")
            if step["host"] not in HOST_OPS:
                known = ", ".join(sorted(HOST_OPS))
                raise ValueError(
                    f"{axm.parent / 'metadata.yaml'} asks for host operation {step['host']!r}; "
                    f"this LibreYOLO version knows {known}. Upgrade LibreYOLO or re-export the model."
                )
            produced = [str(step.get("output"))]
        else:
            raise _foreign_artifact(axm, f"parts[{index}] is neither a compiled model nor a host operation")
        clash = available.intersection(produced)
        if clash:
            raise _foreign_artifact(axm, f"parts[{index}] produces {sorted(clash)} a second time")
        available.update(produced)
        steps.append(step)
    missing = [name for name in final_names if name not in available]
    if missing:
        raise _foreign_artifact(axm, f"parts do not produce {missing}")
    return steps


def resolve_axelera_artifact(path: str | Path) -> tuple[Path, Path]:
    """Return ``(axm, metadata.yaml)`` for an export directory or ``.axm`` path."""
    path = Path(path)
    if path.suffix.lower() == ".axm":
        if not path.is_file():
            raise FileNotFoundError(f"Axelera model not found: {path}")
        axm, directory = path, path.parent
    else:
        if not path.is_dir():
            raise FileNotFoundError(f"Axelera model directory not found: {path}")
        candidates = sorted(path.glob("*.axm"))
        if not candidates:
            raise FileNotFoundError(f"No .axm file in {path}.")
        if len(candidates) > 1:
            primary = _chained_primary(path)
            if primary is not None:
                return primary, path / "metadata.yaml"
            names = ", ".join(c.name for c in candidates)
            raise ValueError(
                f"{path} holds more than one .axm ({names}); pass the .axm file "
                "you want to load."
            )
        axm, directory = candidates[0], path
    metadata = directory / "metadata.yaml"
    if not metadata.is_file():
        raise ValueError(
            f"{axm} has no metadata.yaml next to it. LibreYOLO loads Axelera "
            "models it exported itself: run model.export(format='axelera')."
        )
    return axm, metadata


def _chained_primary(directory: Path) -> Path | None:
    """The primary ``.axm`` of a multi-part export directory (its metadata lists ``parts``), else ``None``."""
    import yaml

    metadata = directory / "metadata.yaml"
    if not metadata.is_file():
        return None
    try:
        meta = yaml.safe_load(metadata.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(meta, dict) or not meta.get("parts") or not meta.get("axm"):
        return None
    primary = directory / str(meta["axm"])
    return primary if primary.is_file() else None


def _foreign_artifact(axm: Path, detail: str) -> ValueError:
    return ValueError(
        f"{axm.parent} is not a LibreYOLO Axelera export ({detail}). LibreYOLO "
        "only loads Axelera models it exported itself, because it decodes "
        "their raw outputs on the host: export one with "
        "LibreYOLO(<checkpoint>).export(format='axelera')."
    )


def _no_device_error(exc: BaseException) -> RuntimeError:
    return RuntimeError(
        f"Could not run the model on an Axelera Metis device: {exc}. Run "
        "`axdevice` (installed with axelera-rt) to list devices and their "
        "driver and firmware state; if a device is listed but busy, another "
        f"process holds it. See {DOCS_TROUBLESHOOTING}."
    )


class AxeleraBackend(BaseBackend):
    """Axelera Metis inference backend for LibreYOLO models.

    Args:
        model_path: A ``<stem>_axelera_model`` directory (preferred) or the
            ``.axm`` file inside it.
        nb_classes: Number of classes (default: from metadata).
        device: ``"auto"``, ``"axelera"`` / ``"metis"`` or ``"cpu"`` (host-side
            processing; the network still runs on the AIPU).
        task: Optional task override; must match the exported task.

    Example:
        >>> model = AxeleraBackend("weights/LibreYOLONASs_axelera_model")
        >>> result = model("image.jpg")
    """

    # Compiled graphs have one input shape: restoration pads into that canvas
    # instead of running at the image's own size.
    fixed_input_shape = True

    def __init__(
        self,
        model_path: str | Path,
        nb_classes: int | None = None,
        device: str = "auto",
        task: str | None = None,
    ):
        import yaml

        axm, metadata_path = resolve_axelera_artifact(model_path)
        with open(metadata_path, encoding="utf-8") as handle:
            meta = yaml.safe_load(handle) or {}
        if not isinstance(meta, dict) or meta.get("format") != "axelera":
            raise _foreign_artifact(axm, "metadata.yaml has no format: axelera")
        layout = meta.get("output_layout")
        if (
            layout not in YOLONAS_RAW_LAYOUTS
            and layout not in _PASS_THROUGH_LAYOUTS
            and layout not in _HOST_TAIL_LAYOUTS
        ):
            if isinstance(layout, str) and layout:
                raise ValueError(
                    f"{metadata_path} uses output layout {layout!r}, which this "
                    "LibreYOLO version cannot decode. Upgrade LibreYOLO or "
                    "re-export the model."
                )
            raise _foreign_artifact(axm, "no output_layout in metadata.yaml")
        warn_on_metadata_schema_version(
            meta, artifact=f"Axelera metadata {metadata_path}", logger=logger
        )
        try:
            parsed = parse_export_metadata(
                meta, artifact=str(metadata_path), strict=True
            )
        except ExportMetadataError as exc:
            raise _foreign_artifact(axm, str(exc)) from exc
        if meta.get("axm") and meta["axm"] != axm.name:
            raise ValueError(
                f"{metadata_path} describes {meta['axm']!r}, not {axm.name!r}; "
                "keep the files of one export together."
            )

        requested = (device or "auto").lower()
        if requested.startswith(("cuda", "gpu", "mps")):
            logger.warning(
                "device=%r ignored: Axelera models run on the Metis AIPU.", device
            )
        elif requested not in _DEVICE_ALIASES:
            raise ValueError(
                f"Unsupported device {device!r} for an Axelera model; use 'auto', "
                "'axelera' (or 'metis'), or 'cpu'."
            )

        self._layout = layout
        strides = tuple(int(s) for s in meta.get("strides", (8, 16, 32)))
        if layout == CLASSIFY_LOGITS_LAYOUT:
            layout_names = ["logits"]
        elif layout == GRAPH_OUTPUTS_LAYOUT:
            layout_names = [str(name) for name in meta.get("graph_outputs") or ()]
            if not layout_names:
                raise _foreign_artifact(axm, "no graph_outputs in metadata.yaml")
        elif layout == RESTORE_SHUFFLE_SKIP_LAYOUT:
            layout_names = ["body"]
            if int(meta.get("upscale", 0)) < 1:
                raise _foreign_artifact(axm, "no upscale in metadata.yaml")
        else:
            layout_names = raw_head_output_names(layout, strides)
        # ``output_names`` lists the outputs in the order the compiled model
        # returns them, which need not be the graph's order.
        self._output_names = list(meta.get("output_names") or layout_names)
        if sorted(self._output_names) != sorted(layout_names):
            raise ValueError(
                f"{metadata_path} lists outputs {self._output_names}; layout "
                f"{layout} expects {layout_names}. Re-export the model."
            )
        self._layout_order = [self._output_names.index(name) for name in layout_names]
        try:
            output_windows = (
                {}
                if layout in _PASS_THROUGH_LAYOUTS or layout in _HOST_TAIL_LAYOUTS
                else check_output_windows(layout, meta.get("output_windows"))
            )
        except ValueError as exc:
            raise ValueError(f"{metadata_path}: {exc} Re-export the model.") from exc
        self._upscale = int(meta.get("upscale", 1))
        self._decode_params: dict[str, Any] = {
            "strides": strides,
            "reg_max": int(meta.get("reg_max", 16)),
            "grid_cell_offset": float(meta.get("grid_cell_offset", 0.5)),
            "output_windows": output_windows,
        }
        if meta.get("anchor_sizes") is not None:  # RetinaNet
            self._decode_params["anchor_sizes"] = meta["anchor_sizes"]
            self._decode_params["aspect_ratios"] = meta["aspect_ratios"]
            self._decode_params["class_index_map"] = meta.get("class_index_map")
        if meta.get("anchors") is not None:  # anchor-based YOLO layouts
            self._decode_params["anchors"] = [
                [(float(w), float(h)) for w, h in level] for level in meta["anchors"]
            ]
            if meta.get("scale_x_y") is not None:
                self._decode_params["scale_x_y"] = [float(s) for s in meta["scale_x_y"]]
        if layout == "ssd_raw_heads_v1":  # SSD300: 91 logits incl. background -> the listed classes
            self._decode_params["class_index_map"] = meta.get("class_index_map")
        if meta.get("exp_on_reg") is not None:  # RTMDet
            self._decode_params["exp_on_reg"] = bool(meta["exp_on_reg"])
        if parsed.task == "pose":
            self._decode_params["pose_offset_multiplier"] = float(
                meta.get("pose_offset_multiplier", 1.0)
            )
            self._decode_params["compensate_grid_cell_offset"] = bool(
                meta.get("compensate_grid_cell_offset", True)
            )
        if parsed.imgsz is None:
            raise _foreign_artifact(axm, "no imgsz in metadata.yaml")
        imgsz = parsed.imgsz
        self._fixed_input_hw = (
            (int(imgsz[0]), int(imgsz[1]))
            if isinstance(imgsz, (tuple, list))
            else (int(imgsz), int(imgsz))
        )
        self.axm_path = axm
        self.runtime_version = require_axelera_rt()
        # A multi-part export (metadata ``parts``) runs its compiled parts and
        # host operations in order; any other artifact is one .axm, as before.
        self._chain = None
        self._chain_models: list = []
        if meta.get("parts") is not None:
            self._chain = _validate_chain(meta["parts"], axm, layout_names)
            self._chain_models = [
                self._load(axm.parent / str(step["axm"])) if "axm" in step else None
                for step in self._chain
            ]
            self._model = self._chain_models[0]
        else:
            self._model = self._load(axm)
        logger.info("Axelera model %s loaded; inference runs on the Metis AIPU.", axm.name)

        resolved_nb_classes = parsed.nb_classes if nb_classes is None else int(nb_classes)
        super().__init__(
            model_path=str(axm.parent),
            nb_classes=resolved_nb_classes,
            device="axelera",
            imgsz=imgsz,
            model_family=parsed.model_family,
            names=parsed.names,
            model_size=parsed.model_size,
            task=task if task is not None else parsed.task,
            supported_tasks=(parsed.task,),
            default_task=parsed.task,
            letterbox_pad=parsed.runtime.get("letterbox_pad"),
            channels=parsed.runtime.get("channels"),
            **classify_eval_kwargs(parsed.runtime),
            **parsed.pose,
        )

    @staticmethod
    def _load(axm: Path):
        from axelera import runtime
        from axelera.runtime import op

        device_errors = tuple(
            getattr(runtime, name)
            for name in ("DeviceInUse", "IncompatibleDevice", "ConnectionError")
            if isinstance(getattr(runtime, name, None), type)
        )
        try:
            return op.load(str(axm))
        except device_errors as exc:
            raise _no_device_error(exc) from exc

    def _invoke(self, image: np.ndarray) -> list[np.ndarray]:
        """Run one ``(1, C, H, W)`` float32 image; outputs in layout order."""
        if self._chain is not None:
            return self._run_chain([image], batch=False)[0]
        try:
            outputs = self._model(image)
        except RuntimeError as exc:
            raise _no_device_error(exc) from exc
        return self._to_layout(outputs)

    def _invoke_batch(self, images: list[np.ndarray]) -> list[list[np.ndarray]]:
        """Submit several images at once so the runtime can spread them over cores."""
        if self._chain is not None:
            return self._run_chain(images, batch=True)
        try:
            results = self._model.batch(images)
        except RuntimeError as exc:
            raise _no_device_error(exc) from exc
        return [self._to_layout(outputs) for outputs in results]

    def _run_chain(self, images: list[np.ndarray], *, batch: bool) -> list[list[np.ndarray]]:
        """Run a multi-part export: each compiled part (batched when asked) and host op in order."""
        envs = [{"images": image} for image in images]
        for index, (step, model) in enumerate(zip(self._chain, self._chain_models)):
            if model is None:
                operation = HOST_OPS[step["host"]]
                params = {k: v for k, v in step.items() if k not in _HOST_STEP_KEYS}
                for env in envs:
                    env[step["output"]] = operation(env[step["input"]], **params)
                continue
            inputs = [np.ascontiguousarray(env[step["input"]], dtype=np.float32) for env in envs]
            try:
                if batch and len(inputs) > 1:
                    results = model.batch(inputs)
                else:
                    results = [model(array) for array in inputs]
            except RuntimeError as exc:
                raise _no_device_error(exc) from exc
            names = step["outputs"]
            for env, outputs in zip(envs, results):
                if not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]
                if len(outputs) != len(names):
                    raise ValueError(
                        f"Part {index} ({step['axm']}) returned {len(outputs)} outputs; its "
                        f"metadata lists {len(names)}. Re-export the model."
                    )
                for name, output in zip(names, outputs):
                    array = np.asarray(output, dtype=np.float32)
                    env[name] = array if array.ndim == 4 else array.reshape((1, *array.shape))
        ordered = [[env[name] for name in self._output_names] for env in envs]
        return [[arrays[i] for i in self._layout_order] for arrays in ordered]

    def _to_layout(self, outputs) -> list[np.ndarray]:
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        if len(outputs) != len(self._output_names):
            raise ValueError(
                f"The Axelera model returned {len(outputs)} outputs; its metadata "
                f"lists {len(self._output_names)} ({', '.join(self._output_names)}). "
                "The .axm and metadata.yaml do not belong together, or the model "
                "needs recompiling with this LibreYOLO version."
            )
        arrays = [np.asarray(out, dtype=np.float32) for out in outputs]
        if self._layout == CLASSIFY_LOGITS_LAYOUT:
            arrays = [a.reshape(1, -1) for a in arrays]
        elif self._layout in (GRAPH_OUTPUTS_LAYOUT, RESTORE_SHUFFLE_SKIP_LAYOUT):
            pass  # already the ONNX graph's tensors, batch dimension included
        else:
            arrays = [a if a.ndim == 4 else a.reshape((1, *a.shape)) for a in arrays]
        return [arrays[index] for index in self._layout_order]

    def _supports_batched_inference(self) -> bool:
        # The compiled model takes one image; batched calls go through the
        # runtime's batch submission, which keeps several cores busy.
        if self._chain is not None:
            return all(hasattr(m, "batch") for m in self._chain_models if m is not None)
        return hasattr(self._model, "batch")

    def _run_inference(self, blob: np.ndarray) -> list:
        self._check_fixed_input_size(blob, "Axelera")
        blob = np.ascontiguousarray(blob, dtype=np.float32)
        images = [blob[index : index + 1] for index in range(blob.shape[0])]
        if len(images) > 1 and self._supports_batched_inference():
            raw = self._invoke_batch(images)
        else:
            raw = [self._invoke(image) for image in images]
        if self._layout in _PASS_THROUGH_LAYOUTS:
            decoded = [[np.array(a, copy=True) for a in arrays] for arrays in raw]
        elif self._layout == RESTORE_SHUFFLE_SKIP_LAYOUT:
            decoded = [
                [shuffle_and_skip(np.asarray(arrays[0], np.float32), image, self._upscale)]
                for arrays, image in zip(raw, images)
            ]
        else:
            decoded = [
                decode_raw_heads(self._layout, arrays, **self._decode_params)
                for arrays in raw
            ]
        if len(decoded) == 1:
            return decoded[0]
        return [np.concatenate(parts, axis=0) for parts in zip(*decoded)]


__all__ = [
    "AUTOINSTALL_ENV",
    "HOST_OPS",
    "AXELERA_INDEX_URL",
    "AXELERA_SDK_VERSION",
    "CLASSIFY_LOGITS_LAYOUT",
    "GRAPH_OUTPUTS_LAYOUT",
    "RESTORE_SHUFFLE_SKIP_LAYOUT",
    "AxeleraBackend",
    "axelera_install_command",
    "is_axelera_artifact",
    "require_axelera_devkit",
    "require_axelera_rt",
    "resolve_axelera_artifact",
]
