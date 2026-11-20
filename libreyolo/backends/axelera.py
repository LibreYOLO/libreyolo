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
import time
from dataclasses import dataclass, field
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


#: Host operations a multi-part (chained) export may run between its compiled
#: parts, by the ``host`` name its metadata gives them. Each takes its input
#: arrays positionally plus the step's parameters as keywords, and returns one
#: array, or a tuple with one array per output.
def _host_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise sum of two parts' outputs (Real-ESRGAN's trunk skip around the RRDB body)."""
    return (np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)).astype(np.float32)


def _host_pack_levels(*levels, boxes, canvas) -> np.ndarray:
    """Place FPN levels on one zero canvas at their ``[y, x, h, w]`` boxes (1-pixel zero gutters between them)."""
    first = np.asarray(levels[0])
    out = np.zeros((first.shape[0], first.shape[1], int(canvas[0]), int(canvas[1])), dtype=np.float32)
    for level, (y, x, h, w) in zip(levels, np.asarray(boxes, dtype=np.int64)):
        out[:, :, y : y + h, x : x + w] = np.asarray(level, dtype=np.float32)
    return out


def _host_gn_relu_levels(x, *, boxes, groups, gamma, beta, eps=1e-5) -> np.ndarray:
    """GroupNorm (per level box, ``groups`` groups) + affine + ReLU on a packed canvas; zero outside the boxes."""
    x = np.asarray(x, dtype=np.float32)
    n, c = x.shape[:2]
    out = np.zeros_like(x)
    gamma = np.asarray(gamma, dtype=np.float32).reshape(1, c, 1, 1)
    beta = np.asarray(beta, dtype=np.float32).reshape(1, c, 1, 1)
    g = int(groups)
    for y, x0, h, w in np.asarray(boxes, dtype=np.int64):
        region = x[:, :, y : y + h, x0 : x0 + w].reshape(n, g, c // g, h, w)
        mean = region.mean(axis=(2, 3, 4), keepdims=True)
        var = region.var(axis=(2, 3, 4), keepdims=True)
        normed = ((region - mean) / np.sqrt(var + np.float32(eps))).reshape(n, c, h, w)
        out[:, :, y : y + h, x0 : x0 + w] = np.maximum(normed * gamma + beta, 0.0)
    return out


def _host_fcos_decode(y, *, boxes, n_cls, anchors, class_indices) -> np.ndarray:
    """FCOS's export tensor from the packed predictor canvas (``cls | reg | ctrness`` channels per level box).

    Per level, flatten in (h, w) order, decode the distances against the fixed-canvas anchors and score
    ``sqrt(sigmoid(cls) * sigmoid(ctrness))`` for the exported classes: ``(1, N, 4 + 1 + classes)`` with the
    level id in column 4, as ``FCOSExportWrapper`` emits it.
    """
    y = np.asarray(y, dtype=np.float32)
    n_cls = int(n_cls)
    cls_rows, reg_rows, ctr_rows, lvl_rows = [], [], [], []
    for level, (yy, x0, h, w) in enumerate(np.asarray(boxes, dtype=np.int64)):
        crop = y[0, :, yy : yy + h, x0 : x0 + w].reshape(y.shape[1], -1).T
        cls_rows.append(crop[:, :n_cls])
        reg_rows.append(np.maximum(crop[:, n_cls : n_cls + 4], 0.0))
        ctr_rows.append(crop[:, n_cls + 4 : n_cls + 5])
        lvl_rows.append(np.full((crop.shape[0], 1), float(level), dtype=np.float32))
    cls, reg, ctr, lvl = (np.concatenate(r, axis=0) for r in (cls_rows, reg_rows, ctr_rows, lvl_rows))
    a = np.asarray(anchors, dtype=np.float32)
    cx, cy = 0.5 * (a[:, 0] + a[:, 2]), 0.5 * (a[:, 1] + a[:, 3])
    aw, ah = a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]
    d = reg * np.stack([aw, ah, aw, ah], axis=-1)
    xyxy = np.stack([cx - d[:, 0], cy - d[:, 1], cx + d[:, 2], cy + d[:, 3]], axis=-1)
    scores = np.sqrt(1.0 / (1.0 + np.exp(-cls)) * (1.0 / (1.0 + np.exp(-ctr))))
    scores = scores[:, np.asarray(class_indices, dtype=np.int64)]
    return np.concatenate([xyxy, lvl, scores], axis=-1)[None].astype(np.float32)


def _bilinear_matrix(n: int, factor: int) -> np.ndarray:
    """``[n * factor, n]`` weights of a bilinear upsample by ``factor`` with ``align_corners=False`` (as
    ``F.interpolate(..., size=n * factor, mode="bilinear")``)."""
    out = np.zeros((n * factor, n), dtype=np.float32)
    src = np.maximum((np.arange(n * factor, dtype=np.float32) + 0.5) / factor - 0.5, 0.0)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    lam = (src - i0).astype(np.float32)
    rows = np.arange(n * factor)
    np.add.at(out, (rows, i0), 1.0 - lam)
    np.add.at(out, (rows, i1), lam)
    return out


def _host_concat(*xs, upsample, mode: str = "nearest") -> np.ndarray:
    """Channel concat of parts' outputs, each first upsampled by its integer factor (``mode`` nearest, as
    ``nn.Upsample``, or bilinear with ``align_corners=False``, as PIDNet's pyramid pooling).

    Pure data movement: RTMDet's PAFPN joins as ``cat([upsample(high), low])`` and ``cat([down, high])``.
    """
    if mode not in ("nearest", "bilinear"):
        raise ValueError(f"concat upsample mode must be 'nearest' or 'bilinear', got {mode!r}")
    arrays = []
    for x, factor in zip(xs, upsample):
        x = np.asarray(x, dtype=np.float32)
        f = int(factor)
        if f > 1 and mode == "bilinear":
            mh, mw = _bilinear_matrix(x.shape[2], f), _bilinear_matrix(x.shape[3], f)
            x = np.einsum("oh,nchw,pw->ncop", mh, x, mw, optimize=True).astype(np.float32)
        elif f > 1:
            x = x.repeat(f, axis=2).repeat(f, axis=3)
        arrays.append(x)
    return np.concatenate(arrays, axis=1)


HOST_OPS: dict[str, Any] = {
    "concat": _host_concat,
    "l2norm": _host_l2norm,
    "add": _host_add,
    "pack_levels": _host_pack_levels,
    "gn_relu_levels": _host_gn_relu_levels,
    "fcos_decode": _host_fcos_decode,
}
_HOST_STEP_KEYS = ("host", "input", "output", "inputs", "outputs")
#: A host-operation parameter written as ``"@npz:<key>"`` is the array stored
#: under ``<key>`` in the export's ``host_arrays`` file (an ``.npz``).
_NPZ_PREFIX = "@npz:"


@dataclass
class ChainStep:
    """One step of a multi-part export, as the backend runs it.

    ``kind`` is ``"axm"`` (a compiled part on the Metis AIPU, one input),
    ``"host"`` (a ``HOST_OPS`` operation in numpy) or ``"onnx"`` (an FP32 ONNX
    graph on the host CPU through ONNX Runtime). ``per_item`` compiled parts
    take a stack of items (e.g. regions of interest) and run them one by one.
    """

    kind: str
    source: str
    inputs: list[str]
    outputs: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    per_item: bool = False
    item_shapes: dict[str, list[int]] = field(default_factory=dict)


def _step_names(step: dict, single: str, plural: str) -> list[str] | None:
    """``[step[single]]`` or ``step[plural]`` as a list of names, else ``None``."""
    if plural in step:
        names = step[plural]
        if not isinstance(names, (list, tuple)) or not names:
            return []
        return [str(name) for name in names]
    if single in step and step[single] is not None:
        return [str(step[single])]
    return None


def _resolve_host_param(value, arrays, axm: Path, index: int):
    if isinstance(value, str) and value.startswith(_NPZ_PREFIX):
        key = value[len(_NPZ_PREFIX):]
        if arrays is None or key not in arrays:
            raise _foreign_artifact(axm, f"parts[{index}] refers to host array {key!r}, which host_arrays does not hold")
        return arrays[key]
    return value


def _validate_chain(parts, axm: Path, final_names: list[str], arrays=None) -> list[ChainStep]:
    """Check a multi-part export's ``parts`` list; return its steps.

    Each part is a compiled model (``axm``, ``input``, ``outputs``), a host
    operation (``host``, ``input``/``inputs``, ``output``/``outputs`` plus its
    parameters) or a host ONNX graph (``onnx``, ``inputs``, ``outputs``). The
    first part is the primary ``.axm`` and takes the images; every input must
    be an earlier output, and the parts together must produce ``final_names``.
    ``arrays`` holds the ``host_arrays`` file's arrays, if the export has one.
    """
    if not isinstance(parts, list) or not parts:
        raise _foreign_artifact(axm, "parts in metadata.yaml must be a non-empty list")
    available = {"images"}
    steps: list[ChainStep] = []
    for index, part in enumerate(parts):
        if not isinstance(part, dict):
            raise _foreign_artifact(axm, f"parts[{index}] is not a mapping")
        if "axm" in part:
            sources = _step_names(part, "input", "inputs")
            if sources is not None and len(sources) > 1:
                raise _foreign_artifact(axm, f"parts[{index}] is a compiled part with {len(sources)} inputs; compiled parts take one")
        elif "host" in part:
            sources = _step_names(part, "input", "inputs")
        elif "onnx" in part:
            sources = _step_names(part, "input", "inputs")
        else:
            raise _foreign_artifact(axm, f"parts[{index}] is neither a compiled model nor a host operation")
        for source in sources or [part.get("input")]:
            if source not in available:
                raise _foreign_artifact(axm, f"parts[{index}] reads {source!r}, which no earlier part produces")
        if "axm" in part:
            if index == 0 and (part["axm"] != axm.name or sources != ["images"]):
                raise _foreign_artifact(axm, "parts[0] must be the primary .axm reading the images")
            if not (axm.parent / str(part["axm"])).is_file():
                raise FileNotFoundError(f"{axm.parent / str(part['axm'])} (parts[{index}]) is missing.")
            produced = [str(name) for name in part.get("outputs") or ()]
            if not produced:
                raise _foreign_artifact(axm, f"parts[{index}] lists no outputs")
            shapes = part.get("item_shapes") or {}
            step = ChainStep(
                "axm", str(part["axm"]), sources, produced,
                per_item=bool(part.get("per_item", False)),
                item_shapes={str(k): [int(d) for d in v] for k, v in dict(shapes).items()},
            )
        elif "host" in part:
            if index == 0:
                raise _foreign_artifact(axm, "parts[0] must be a compiled model")
            if part["host"] not in HOST_OPS:
                known = ", ".join(sorted(HOST_OPS))
                raise ValueError(
                    f"{axm.parent / 'metadata.yaml'} asks for host operation {part['host']!r}; "
                    f"this LibreYOLO version knows {known}. Upgrade LibreYOLO or re-export the model."
                )
            produced = _step_names(part, "output", "outputs") or []
            if not produced:
                raise _foreign_artifact(axm, f"parts[{index}] lists no output")
            params = {
                key: _resolve_host_param(value, arrays, axm, index)
                for key, value in part.items()
                if key not in _HOST_STEP_KEYS
            }
            step = ChainStep("host", str(part["host"]), sources, produced, params=params)
        else:
            if index == 0:
                raise _foreign_artifact(axm, "parts[0] must be a compiled model")
            if not (axm.parent / str(part["onnx"])).is_file():
                raise FileNotFoundError(f"{axm.parent / str(part['onnx'])} (parts[{index}]) is missing.")
            produced = _step_names(part, "output", "outputs") or []
            if not produced or not sources:
                raise _foreign_artifact(axm, f"parts[{index}] needs inputs and outputs")
            step = ChainStep("onnx", str(part["onnx"]), sources, produced)
        clash = available.intersection(produced)
        if clash:
            raise _foreign_artifact(axm, f"parts[{index}] produces {sorted(clash)} a second time")
        available.update(produced)
        steps.append(step)
    missing = [name for name in final_names if name not in available]
    if missing:
        raise _foreign_artifact(axm, f"parts do not produce {missing}")
    return steps


def _load_host_arrays(directory: Path, name, axm: Path) -> dict[str, np.ndarray] | None:
    """The ``host_arrays`` ``.npz`` of a multi-part export, or ``None`` if it has none."""
    if not name:
        return None
    path = directory / str(name)
    if not path.is_file():
        raise FileNotFoundError(f"{path} (host_arrays) is missing.")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _host_onnx_session(path: Path, step: ChainStep, axm: Path):
    """An ONNX Runtime CPU session for a host ONNX step, its inputs and outputs checked."""
    try:
        import onnxruntime
    except ImportError as exc:
        raise ImportError(
            f"{axm.parent} runs part of the model on the host CPU as ONNX ({path.name}), "
            "which needs ONNX Runtime: pip install onnxruntime"
        ) from exc
    options = onnxruntime.SessionOptions()
    # Host steps run between device calls: no spin-waiting threads, and
    # OMP_NUM_THREADS (when set) caps the pool like the other host math.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    threads = os.environ.get("OMP_NUM_THREADS", "").strip()
    if threads.isdigit() and int(threads) > 0:
        options.intra_op_num_threads = int(threads)
    session = onnxruntime.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )
    graph_inputs = sorted(i.name for i in session.get_inputs())
    graph_outputs = {o.name for o in session.get_outputs()}
    if graph_inputs != sorted(step.inputs) or not set(step.outputs) <= graph_outputs:
        raise _foreign_artifact(
            axm,
            f"{path.name} takes {graph_inputs} and gives {sorted(graph_outputs)}; "
            f"its part lists {step.inputs} -> {step.outputs}",
        )
    return session


_HOST_POOLS: dict[int, Any] = {}


def _host_workers() -> int:
    """Threads for per-image host steps: ``OMP_NUM_THREADS`` when set, else up to 4."""
    threads = os.environ.get("OMP_NUM_THREADS", "").strip()
    if threads.isdigit() and int(threads) > 0:
        return int(threads)
    return max(1, min(4, os.cpu_count() or 1))


def _host_map(function, items: list) -> list:
    """``[function(item) for item in items]``, spread over a shared thread pool.

    Each item is processed exactly as in the sequential loop (NumPy releases
    the GIL in its array kernels), so the results are identical; only the
    wall time changes. One item, or one worker, stays on the calling thread.
    """
    workers = min(_host_workers(), len(items))
    if workers <= 1:
        return [function(item) for item in items]
    pool = _HOST_POOLS.get(workers)
    if pool is None:
        from concurrent.futures import ThreadPoolExecutor

        pool = _HOST_POOLS[workers] = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="libreyolo-axelera-host"
        )
    # Each item runs in a copy of the caller's context, so context variables
    # (e.g. agnostic NMS) read the same values as on the calling thread.
    import contextvars

    context = contextvars.copy_context()
    return list(pool.map(lambda item: context.copy().run(function, item), items))


def _as_list(outputs) -> list:
    return list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]


def _device_array(output) -> np.ndarray:
    """A compiled part's output as float32 with its batch dimension."""
    array = np.asarray(output, dtype=np.float32)
    return array if array.ndim == 4 else array.reshape((1, *array.shape))


def _host_array(value) -> np.ndarray:
    array = np.ascontiguousarray(value)
    return array.astype(np.float32) if array.dtype == np.float64 else array


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
    # Lists, directories and video files run 8 images per runtime call by
    # default (``predict(batch=1)`` opts out): a batched submission keeps
    # several AIPU cores busy and overlaps the host pre/post-processing.
    default_predict_batch = 8

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
        self._chain_sessions: dict[int, Any] = {}
        #: Wall time a multi-part export spends on the AIPU parts and on the
        #: host steps, summed over the images run so far.
        self.chain_timing = {"images": 0, "aipu_ms": 0.0, "host_ms": 0.0}
        if meta.get("parts") is not None:
            arrays = _load_host_arrays(axm.parent, meta.get("host_arrays"), axm)
            self._chain = _validate_chain(meta["parts"], axm, layout_names, arrays)
            self._chain_sessions = {
                index: _host_onnx_session(axm.parent / step.source, step, axm)
                for index, step in enumerate(self._chain)
                if step.kind == "onnx"
            }
            self._chain_models = [
                self._load(axm.parent / step.source) if step.kind == "axm" else None
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
        """Run a multi-part export: compiled parts (batched when asked) and host steps in order."""
        envs = [{"images": image} for image in images]
        for index, (step, model) in enumerate(zip(self._chain, self._chain_models)):
            start = time.perf_counter()
            if step.kind == "host":
                operation = HOST_OPS[step.source]
                for env in envs:
                    result = operation(*(env[name] for name in step.inputs), **step.params)
                    self._store(env, step, index, result if len(step.outputs) > 1 else (result,))
            elif step.kind == "onnx":
                session = self._chain_sessions[index]
                for env in envs:
                    feeds = {name: _host_array(env[name]) for name in step.inputs}
                    self._store(env, step, index, session.run(step.outputs, feeds))
            elif step.per_item:
                self._run_per_item(step, model, envs, index)
            else:
                inputs = [np.ascontiguousarray(env[step.inputs[0]], dtype=np.float32) for env in envs]
                for env, outputs in zip(envs, self._call_part(model, inputs, batch)):
                    self._store(env, step, index, [_device_array(o) for o in _as_list(outputs)])
            elapsed = (time.perf_counter() - start) * 1000.0
            self.chain_timing["aipu_ms" if step.kind == "axm" else "host_ms"] += elapsed
        self.chain_timing["images"] += len(images)
        ordered = [[env[name] for name in self._output_names] for env in envs]
        return [[arrays[i] for i in self._layout_order] for arrays in ordered]

    @staticmethod
    def _call_part(model, inputs: list[np.ndarray], batch: bool) -> list:
        try:
            if batch and len(inputs) > 1:
                return list(model.batch(inputs))
            return [model(array) for array in inputs]
        except RuntimeError as exc:
            raise _no_device_error(exc) from exc

    def _run_per_item(self, step: ChainStep, model, envs: list[dict], index: int) -> None:
        """Run a per-item part: every item of every image's stack, submitted together."""
        stacks = [np.ascontiguousarray(env[step.inputs[0]], dtype=np.float32) for env in envs]
        items = [stack[i : i + 1] for stack in stacks for i in range(stack.shape[0])]
        results = self._call_part(model, items, batch=hasattr(model, "batch")) if items else []
        offset = 0
        for env, stack in zip(envs, stacks):
            count = stack.shape[0]
            mine = [[_device_array(o) for o in _as_list(r)] for r in results[offset : offset + count]]
            offset += count
            outputs = []
            for position, name in enumerate(step.outputs):
                if mine:
                    outputs.append(np.concatenate([r[position] for r in mine], axis=0))
                else:
                    shape = step.item_shapes.get(name)
                    outputs.append(np.zeros((0, *shape) if shape else (0,), dtype=np.float32))
            self._store(env, step, index, outputs)

    @staticmethod
    def _store(env: dict, step: ChainStep, index: int, outputs) -> None:
        outputs = list(outputs)
        if len(outputs) != len(step.outputs):
            raise ValueError(
                f"Part {index} ({step.source}) returned {len(outputs)} outputs; its "
                f"metadata lists {len(step.outputs)}. Re-export the model."
            )
        env.update(zip(step.outputs, outputs))

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

    #: Tasks whose per-image result building (logit upsampling, argmax, dense
    #: maps) is heavy enough to gain from host threads; box tasks stay
    #: sequential, where dispatch would cost about what it saves.
    _PARALLEL_RESULT_TASKS = frozenset(
        {"semantic", "depth", "normal", "edge", "matte", "restore"}
    )

    def _map_images(self, function, images: list, stage: str = "preprocess") -> list:
        # Per-image host steps of a batch on the host thread pool, while the
        # device is otherwise idle; each image is processed as before.
        if stage == "results" and self.task not in self._PARALLEL_RESULT_TASKS:
            return [function(image) for image in images]
        return _host_map(function, images)

    def _process_in_batches(self, images, batch: int = 1, **kwargs):
        # Pipeline a multi-image predict: while the device runs one step (an
        # image, or a chunk of ``batch`` images) and its results are built, the
        # next step is preprocessed on a prefetch thread (a chunk spreads over
        # the host pool). Every step is processed exactly as before and in
        # order, so the results are unchanged.
        batched = (
            batch > 1
            and self._supports_batched_inference()
            and not getattr(self, "_batched_inference_failed", False)
        )
        step = batch if batched else 1
        # Multi-part exports run host steps between their compiled parts on the
        # calling thread; a concurrent prefetch competes with them for the
        # host cores (SSD-300 measured 0.85x), so they stay sequential.
        if len(images) <= step or self._chain is not None:
            return super()._process_in_batches(images, batch=batch, **kwargs)
        import contextvars
        from concurrent.futures import ThreadPoolExecutor

        context = contextvars.copy_context()
        imgsz, color_format = kwargs.get("imgsz"), kwargs.get("color_format", "auto")
        start_idx = kwargs.pop("start_idx", 0)
        steps = [images[i : i + step] for i in range(0, len(images), step)]
        if batched:
            prepare = lambda chunk: self._prepare_batch(chunk, imgsz, color_format)  # noqa: E731
        else:
            effective_imgsz = self._resolve_predict_imgsz(imgsz)
            prepare = lambda chunk: self._preprocess(chunk[0], effective_imgsz, color_format)  # noqa: E731
        results = []
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="libreyolo-axelera-prefetch"
        ) as prefetch:
            pending = prefetch.submit(context.copy().run, prepare, steps[0])
            for index, chunk in enumerate(steps):
                preprocessed = pending.result()
                if index + 1 < len(steps):
                    pending = prefetch.submit(context.copy().run, prepare, steps[index + 1])
                offset = start_idx + index * step
                if batched:
                    results.extend(
                        self._predict_batch(
                            chunk, start_idx=offset, preprocessed=preprocessed, **kwargs
                        )
                    )
                else:
                    image = chunk[0]
                    results.append(
                        self._predict_single(
                            image,
                            save_stem=(
                                None if isinstance(image, (str, Path)) else f"image{offset}"
                            ),
                            preprocessed=preprocessed,
                            **kwargs,
                        )
                    )
        return results

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
            decoded = _host_map(
                lambda arrays: decode_raw_heads(self._layout, arrays, **self._decode_params),
                raw,
            )
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
