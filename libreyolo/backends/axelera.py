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
_PASS_THROUGH_LAYOUTS = (CLASSIFY_LOGITS_LAYOUT, GRAPH_OUTPUTS_LAYOUT)


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
        if layout not in YOLONAS_RAW_LAYOUTS and layout not in _PASS_THROUGH_LAYOUTS:
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
                if layout in _PASS_THROUGH_LAYOUTS
                else check_output_windows(layout, meta.get("output_windows"))
            )
        except ValueError as exc:
            raise ValueError(f"{metadata_path}: {exc} Re-export the model.") from exc
        self._decode_params: dict[str, Any] = {
            "strides": strides,
            "reg_max": int(meta.get("reg_max", 16)),
            "grid_cell_offset": float(meta.get("grid_cell_offset", 0.5)),
            "output_windows": output_windows,
        }
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
        try:
            outputs = self._model(image)
        except RuntimeError as exc:
            raise _no_device_error(exc) from exc
        return self._to_layout(outputs)

    def _invoke_batch(self, images: list[np.ndarray]) -> list[list[np.ndarray]]:
        """Submit several images at once so the runtime can spread them over cores."""
        try:
            results = self._model.batch(images)
        except RuntimeError as exc:
            raise _no_device_error(exc) from exc
        return [self._to_layout(outputs) for outputs in results]

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
        elif self._layout == GRAPH_OUTPUTS_LAYOUT:
            pass  # already the ONNX graph's tensors, batch dimension included
        else:
            arrays = [a if a.ndim == 4 else a.reshape((1, *a.shape)) for a in arrays]
        return [arrays[index] for index in self._layout_order]

    def _supports_batched_inference(self) -> bool:
        # The compiled model takes one image; batched calls go through the
        # runtime's batch submission, which keeps several cores busy.
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
    "AXELERA_INDEX_URL",
    "AXELERA_SDK_VERSION",
    "CLASSIFY_LOGITS_LAYOUT",
    "GRAPH_OUTPUTS_LAYOUT",
    "AxeleraBackend",
    "axelera_install_command",
    "is_axelera_artifact",
    "require_axelera_devkit",
    "require_axelera_rt",
    "resolve_axelera_artifact",
]
