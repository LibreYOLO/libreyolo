"""Axelera Metis helpers for LibreYOLO export and inference.

LibreYOLO calls Axelera's Voyager SDK as a dependency the user installs:
``axelera-devkit`` (quantize and compile) and ``axelera-rt`` (run ``.axm``
models). Both are proprietary software under Axelera's EULA, published on
Axelera's own package index rather than PyPI; LibreYOLO never bundles or
vendors them.

Importing this module does not import torch or the SDK.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

# The SDK install and version helpers live with the backend, which must stay
# importable without torch; they are re-exported here for export callers.
from ..backends.axelera import (
    AUTOINSTALL_ENV,
    AXELERA_INDEX_URL,
    AXELERA_SDK_VERSION,
    DOCS_TROUBLESHOOTING,
    axelera_install_command,
    require_axelera_devkit,
    require_axelera_rt,
)

logger = logging.getLogger(__name__)

#: Compiler settings LibreYOLO owns: the artifact contract depends on them.
_RESERVED_CONFIG_FIELDS = {
    "output_axm_format": "LibreYOLO needs a .axm artifact (always True).",
    "aipu_cores_used": "The runtime spreads one single-core build across cores (always 1).",
    "output_dir": "LibreYOLO compiles in a temporary build directory.",
    "model_name": "The .axm is named after the checkpoint stem.",
    "model_metadata": "Framework-specific runtime hook; LibreYOLO decodes on the host.",
    "compiler_mode": "Export always quantizes and lowers.",
    "remove_output_dir": "LibreYOLO manages the build directory.",
}

#: Default compiler settings for the raw-head YOLO-NAS graph. Min-max
#: activation ranges: the SDK's histogram default clips the rare large
#: logits (positive class logits, distribution peaks), which measurably cost
#: accuracy on YOLO-NAS; see docs/axelera.md, section 7.
DEFAULT_QUANTIZATION_SCHEME = "per_tensor_min_max"
#: Per-task exceptions. Pose keeps more keypoint accuracy with histogram
#: activation ranges (0.7061 vs 0.6855 keypoint mAP50-95 on YOLO-NAS-pose-s).
DEFAULT_QUANTIZATION_SCHEME_BY_TASK = {"pose": "hybrid_per_tensor_per_channel"}
#: Per-model exceptions, keyed by (family, task); they win over the task
#: default. MobileNetV4 needs histogram ranges on top of its residual-stream
#: equalization: top-1 -1.7 -> +0.2 points on MobileNetV4-m (paired
#: 1000-image ImageNetV2 set, docs/axelera.md).
DEFAULT_QUANTIZATION_SCHEME_BY_MODEL = {
    ("mobilenetv4", "classify"): "hybrid_per_tensor_per_channel",
}


def default_quantization_scheme(family: str, task: str) -> str:
    """The activation-range scheme LibreYOLO compiles ``family``/``task`` with."""
    by_model = DEFAULT_QUANTIZATION_SCHEME_BY_MODEL.get((family, task))
    if by_model is not None:
        return by_model
    return DEFAULT_QUANTIZATION_SCHEME_BY_TASK.get(task, DEFAULT_QUANTIZATION_SCHEME)


#: Graphs with depthwise convolutions: per-channel ranges for the inputs of
#: depthwise convolutions (histogram ranges elsewhere). A depthwise conv never
#: mixes channels, so one per-tensor scale over an input whose channels span
#: very different ranges erases the small ones for good: YOLOX-n's first
#: depthwise input has channel maxima from 0.02 to 106 and scored 0.011 vs
#: FP32 0.261 mAP50-95 with per-tensor ranges (docs/axelera.md, section 7).
DEPTHWISE_QUANTIZATION_SCHEME = "hybrid_per_tensor_per_channel"
DEFAULT_RESOURCES_USED = 0.25


def check_compile_platform() -> None:
    """Axelera's compiler runs on Linux only (WSL2 counts as Linux)."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "Axelera export compiles with Axelera's toolchain, which runs on "
            f"Linux (or WSL2 on Windows); this platform is {sys.platform!r}. "
            "Export on Linux and copy the *_axelera_model directory to the "
            "machine that runs it."
        )


def _config_fields(config_cls) -> tuple[dict[str, str], set[str]]:
    """Map every accepted spelling (field name or alias) to its field name."""
    spellings: dict[str, str] = {}
    for name, field in config_cls.model_fields.items():
        spellings[name] = name
        alias = getattr(field, "alias", None)
        if isinstance(alias, str) and alias:
            spellings[alias] = name
    return spellings, set(config_cls.model_fields)


def normalize_compiler_overrides(
    overrides: Mapping[str, Any] | None, config_cls=None
) -> dict[str, Any]:
    """Validate user ``compiler_config`` keys against the real CompilerConfig."""
    if not overrides:
        return {}
    if not isinstance(overrides, Mapping):
        raise TypeError(
            "compiler_config must be a dict of CompilerConfig fields, got "
            f"{type(overrides).__name__}."
        )
    if config_cls is None:
        from axelera.compiler import CompilerConfig as config_cls
    spellings, fields = _config_fields(config_cls)
    normalized: dict[str, Any] = {}
    unknown = sorted(str(key) for key in overrides if key not in spellings)
    if unknown:
        raise ValueError(
            f"Unknown compiler_config field(s): {', '.join(unknown)}. "
            f"Valid fields: {', '.join(sorted(fields))}."
        )
    for key, value in overrides.items():
        name = spellings[key]
        if name in _RESERVED_CONFIG_FIELDS:
            raise ValueError(
                f"compiler_config cannot set {key!r}: {_RESERVED_CONFIG_FIELDS[name]}"
            )
        if name in normalized:
            raise ValueError(f"compiler_config sets {name!r} more than once.")
        normalized[name] = value
    return normalized


def build_compiler_config(
    model_name: str,
    *,
    build_dir: Path,
    quantization_scheme: str = DEFAULT_QUANTIZATION_SCHEME,
    resources_used: float = DEFAULT_RESOURCES_USED,
    overrides: Mapping[str, Any] | None = None,
):
    """Build the CompilerConfig for one export; returns ``(config, overrides)``."""
    from axelera.compiler import CompilerConfig

    normalized = normalize_compiler_overrides(overrides, CompilerConfig)
    settings: dict[str, Any] = {
        "quantization_scheme": quantization_scheme,
        "resources_used": resources_used,
        "save_error_artifact": True,
    }
    settings.update(normalized)
    settings.update(
        model_name=model_name,
        aipu_cores_used=1,
        output_axm_format=True,
        output_dir=Path(build_dir),
    )
    try:
        config = CompilerConfig(**settings)
    except Exception as exc:  # pydantic.ValidationError, without importing pydantic
        raise ValueError(f"Invalid compiler_config: {exc}") from exc
    return config, normalized


def compile_to_axm(
    model_input, calibration: Iterable, config, work_dir: Path, before_compile=None
):
    """Quantize and compile; return ``(axm_path, quantized_model)``.

    ``work_dir`` must be a fresh directory LibreYOLO owns: the compiler clears
    its ``output_dir`` (``work_dir / "build"``) and writes ``<model_name>.axm``
    next to it, in ``work_dir``. ``before_compile(quantized)`` runs between the
    two steps: compiling lowers the quantized model in place, so anything that
    evaluates it on CPU must happen there.
    """
    from axelera import compiler

    work_dir = Path(work_dir)
    build_dir = work_dir / "build"
    config.output_dir = build_dir
    quantized = compiler.quantize(
        model=model_input, config=config, calibration_dataset=iter(calibration)
    )
    if before_compile is not None:
        before_compile(quantized)
    try:
        compiler.compile(model=quantized, config=config, output_dir=build_dir)
    except Exception as exc:
        artifacts = sorted(str(path) for path in work_dir.rglob("*.zip"))
        detail = f" Error artifact: {', '.join(artifacts)}." if artifacts else ""
        raise RuntimeError(
            f"Axelera compilation failed: {exc}.{detail} "
            f"See {DOCS_TROUBLESHOOTING}."
        ) from exc
    expected = work_dir / f"{config.model_name}.axm"
    if expected.is_file():
        return expected, quantized
    found = sorted(work_dir.rglob("*.axm"))
    if len(found) == 1:
        return found[0], quantized
    raise RuntimeError(
        f"Axelera compilation finished but no single .axm was found under "
        f"{work_dir} (found {len(found)}). See {DOCS_TROUBLESHOOTING}."
    )


def _write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    import yaml

    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, default_flow_style=False, sort_keys=False)


def _final_config_toml(config, build_dir: Path) -> str:
    """The resolved config: the compiler's own copy if it wrote one."""
    for candidate in sorted(Path(build_dir).rglob("compiler_config_final.toml")):
        return candidate.read_text(encoding="utf-8")
    return config.to_toml()


def _device_runner(axm: Path, output_names: list[str], reference_names: list[str]):
    """Callable running ``axm`` on a Metis device, outputs in graph order; or a reason string."""
    from ..backends.axelera import require_axelera_rt

    try:
        require_axelera_rt()
        from axelera.runtime import op

        model = op.load(str(axm))
    except Exception as exc:  # noqa: BLE001 - no runtime, no device or device busy: report, do not fail
        return f"not run: {exc}"
    order = [output_names.index(name) for name in reference_names]

    def run(sample):
        outputs = model(sample)
        outputs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
        return [outputs[i] for i in order]

    return run


def collect_quantized_outputs(quantized, reference_model, samples) -> list[tuple]:
    """``(sample, fp32_outputs, quantized_outputs)`` per image, both in graph order.

    Call before ``compile()``: compiling lowers the quantized model in place.
    """
    import numpy as np
    import torch

    collected = []
    for sample in samples:
        sample = np.ascontiguousarray(sample, dtype=np.float32)
        tensor = torch.from_numpy(sample)
        with torch.no_grad():
            expected = [t.detach().cpu().numpy() for t in reference_model(tensor)]
            produced = quantized(tensor)
        if isinstance(produced, torch.Tensor):
            produced = (produced,)
        produced = [
            np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t)
            for t in produced
        ]
        # The quantized model returns compiled order; compare in graph order.
        collected.append((sample, expected, _match_outputs(expected, produced)))
    return collected


def parity_report(collected, names, device=None) -> dict[str, Any]:
    """Per-output cosine / NRMSE: quantized vs FP32, and the device vs both.

    ``device`` is a callable running the compiled model (outputs in graph
    order), or a string saying why it was not run.
    """
    import numpy as np

    from .rknn import compare_rknn_outputs

    pairs: dict[str, dict[int, list[dict[str, Any]]]] = {
        "quantized_vs_fp32": {},
        "device_vs_quantized": {},
        "device_vs_fp32": {},
    }

    constant: set[int] = set()

    def add(key, reference, candidate):
        # A constant reference (e.g. a windowed class map with every logit
        # below the window) has no direction: one quantization step of noise
        # would read as cosine 0. Such outputs are listed, not scored.
        keep = [i for i, ref in enumerate(reference) if np.ptp(np.asarray(ref)) > 0]
        constant.update(set(range(len(reference))) - set(keep))
        items = compare_rknn_outputs(
            [reference[i] for i in keep], [candidate[i] for i in keep], raise_on_failure=False
        )
        for position, item in enumerate(items):
            pairs[key].setdefault(keep[position], []).append(item)

    for sample, expected, quantized in collected:
        add("quantized_vs_fp32", expected, quantized)
        if callable(device):
            try:
                on_device = [np.asarray(t, dtype=np.float32) for t in device(sample)]
            except Exception as exc:  # noqa: BLE001 - the runtime connects on first call; a busy device must not discard the compiled model
                device = f"not run: {exc}"
                continue
            add("device_vs_quantized", quantized, on_device)
            add("device_vs_fp32", expected, on_device)

    def summarize(per_output):
        rows = []
        for index in sorted(per_output):
            items = per_output[index]
            rows.append(
                {
                    "name": names[index] if index < len(names) else str(index),
                    "min_cosine": min(i["cosine_similarity"] for i in items),
                    "mean_cosine": float(np.mean([i["cosine_similarity"] for i in items])),
                    "max_normalized_rmse": max(i["normalized_rmse"] for i in items),
                }
            )
        return rows

    report: dict[str, Any] = {
        "images": len(collected),
        "quantized_vs_fp32": summarize(pairs["quantized_vs_fp32"]),
    }
    never_scored = sorted(constant - set(pairs["quantized_vs_fp32"]))
    if never_scored:
        report["constant_reference"] = [
            names[i] if i < len(names) else str(i) for i in never_scored
        ]
    if callable(device):
        report["device_vs_quantized"] = summarize(pairs["device_vs_quantized"])
        report["device_vs_fp32"] = summarize(pairs["device_vs_fp32"])
    elif device is not None:
        report["device_vs_quantized"] = str(device)
    return report


def _match_outputs(expected, produced):
    """Reorder ``produced`` to ``expected``'s order by shape, cosine on ties."""
    free = set(range(len(produced)))
    ordered = []
    for reference in expected:
        candidates = [i for i in sorted(free) if produced[i].shape == reference.shape]
        if not candidates:
            raise RuntimeError(
                f"No quantized output matches graph output shape {tuple(reference.shape)}."
            )
        best = max(candidates, key=lambda i, ref=reference: _cosine(ref, produced[i]))
        free.discard(best)
        ordered.append(produced[best])
    return ordered


def device_output_order(quantized, reference_model, sample) -> list[str]:
    """Name the compiled model's outputs in the order it returns them.

    The compiler may reorder graph outputs (for YOLO-NAS it returns each
    level's class tensor before its box tensor). The quantized model returns
    outputs in the compiled order; each is matched to a graph output by shape,
    and by cosine similarity with the FP32 graph when shapes repeat.
    """
    import numpy as np
    import torch

    names = list(reference_model.output_names)
    tensor = torch.from_numpy(np.ascontiguousarray(sample, dtype=np.float32))
    with torch.no_grad():
        reference = [t.detach().cpu().numpy() for t in reference_model(tensor)]
        produced = quantized(tensor)
    if isinstance(produced, torch.Tensor):
        produced = (produced,)
    produced = [
        np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t)
        for t in produced
    ]
    if len(produced) != len(reference):
        raise RuntimeError(
            f"The compiled model has {len(produced)} outputs; the exported graph "
            f"has {len(reference)}. See {DOCS_TROUBLESHOOTING}."
        )
    order: list[str] = []
    free = set(range(len(reference)))
    for output in produced:
        candidates = [i for i in sorted(free) if reference[i].shape == output.shape]
        if not candidates:
            raise RuntimeError(
                f"A compiled output of shape {tuple(output.shape)} matches no "
                f"graph output ({', '.join(names)}). See {DOCS_TROUBLESHOOTING}."
            )
        if len(candidates) > 1:
            candidates.sort(key=lambda i, out=output: _cosine(reference[i], out), reverse=True)
        free.discard(candidates[0])
        order.append(names[candidates[0]])
    return order


def _cosine(a, b) -> float:
    import numpy as np

    a, b = np.ravel(a), np.ravel(b)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b)) / norm


def _remember_first(samples: Iterable, keep: list) -> Iterable:
    for sample in samples:
        if not keep:
            keep.append(sample)
        yield sample


def export_axelera(
    *,
    onnx_path: str,
    output_path: str,
    stem: str,
    metadata: Mapping[str, Any],
    calibration: Iterable,
    calibration_info: Mapping[str, Any],
    compiler_overrides: Mapping[str, Any] | None = None,
    devkit_version: str | None = None,
    quantization_scheme: str = DEFAULT_QUANTIZATION_SCHEME,
    resources_used: float = DEFAULT_RESOURCES_USED,
    verify: bool = False,
    reference_model=None,
    verify_samples: Iterable | None = None,
    verify_source: str = "images",
) -> str:
    """Compile ``onnx_path`` and publish ``<stem>.axm`` + sidecars at ``output_path``."""
    import json

    check_compile_platform()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent)
    )
    try:
        with tempfile.TemporaryDirectory(prefix="libreyolo-axelera-") as work:
            work_dir = Path(work)
            config, overrides = build_compiler_config(
                stem,
                build_dir=work_dir / "build",
                quantization_scheme=quantization_scheme,
                resources_used=resources_used,
                overrides=compiler_overrides,
            )
            first: list = []
            found: dict[str, Any] = {
                "output_names": list(metadata.get("output_names") or []),
                "collected": [],
            }

            def before_compile(quantized):
                if reference_model is None:
                    return
                if first:
                    found["output_names"] = device_output_order(
                        quantized, reference_model, first[0]
                    )
                if verify:
                    found["collected"] = collect_quantized_outputs(
                        quantized, reference_model, verify_samples or []
                    )

            axm, _ = compile_to_axm(
                onnx_path,
                _remember_first(calibration, first),
                config,
                work_dir,
                before_compile=before_compile,
            )
            output_names = found["output_names"]
            shutil.copy2(axm, staging / f"{stem}.axm")
            (staging / "compiler_config_final.toml").write_text(
                _final_config_toml(config, work_dir / "build"), encoding="utf-8"
            )
            if verify and reference_model is not None:
                device = _device_runner(
                    staging / f"{stem}.axm", output_names, list(reference_model.output_names)
                )
                report = parity_report(
                    found["collected"], list(reference_model.output_names), device=device
                )
                report["source"] = verify_source
                (staging / f"{stem}.parity.json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                worst = min(
                    (o["min_cosine"] for o in report["quantized_vs_fp32"]),
                    default=float("nan"),
                )
                logger.info(
                    "Axelera verify on %d %s: lowest per-output cosine %.4f "
                    "(quantized vs FP32); report %s.parity.json.",
                    report["images"],
                    verify_source,
                    worst,
                    stem,
                )
                if isinstance(report.get("device_vs_quantized"), str):
                    logger.warning(
                        "Axelera verify: device comparison %s. The compiled model "
                        "is kept; run val on the device to check it.",
                        report["device_vs_quantized"],
                    )
                if isinstance(report.get("device_vs_quantized"), list):
                    logger.info(
                        "Axelera verify: device vs quantized model lowest cosine %.4f; "
                        "device vs FP32 lowest cosine %.4f.",
                        min((o["min_cosine"] for o in report["device_vs_quantized"]), default=float("nan")),
                        min((o["min_cosine"] for o in report["device_vs_fp32"]), default=float("nan")),
                    )
                if worst < 0.99:
                    logger.warning(
                        "Axelera verify: quantized outputs drift from FP32 (cosine "
                        "%.4f). Check calibration data and run val on the device.",
                        worst,
                    )
        scheme = getattr(config, "quantization_scheme", quantization_scheme)
        meta = dict(metadata)
        meta["axm"] = f"{stem}.axm"
        meta["output_names"] = output_names
        meta["axelera"] = {
            "target": "metis",
            "devkit_version": devkit_version,
            "quantization_scheme": getattr(scheme, "value", scheme),
            "aipu_cores_used": int(config.aipu_cores_used),
            "resources_used": float(config.resources_used),
            "compiler_overrides": {k: _plain(v) for k, v in dict(overrides).items()},
            "calibration": dict(calibration_info),
        }
        _write_yaml(staging / "metadata.yaml", meta)
        publish_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return str(destination)


def _plain(value):
    value = getattr(value, "value", value)
    if isinstance(value, Path):
        return str(value)
    return value


def _current_umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def publish_directory(staging: Path, destination: Path) -> Path:
    """Move a finished artifact directory into place, replacing any old one.

    The previous directory is only removed after the new one is in place; on
    failure it is restored.
    """
    staging, destination = Path(staging), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if destination.exists():
        backup = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.old.", dir=destination.parent)
        )
        backup.rmdir()
        destination.replace(backup)
    try:
        staging.replace(destination)
    except BaseException:
        if backup is not None and backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    # The staging directory came from mkdtemp (mode 0700); give the artifact
    # the permissions any other new directory would get.
    destination.chmod(0o777 & ~_current_umask())
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)
    return destination


__all__ = [
    "AUTOINSTALL_ENV",
    "AXELERA_INDEX_URL",
    "AXELERA_SDK_VERSION",
    "DEFAULT_QUANTIZATION_SCHEME",
    "DEFAULT_QUANTIZATION_SCHEME_BY_MODEL",
    "DEFAULT_QUANTIZATION_SCHEME_BY_TASK",
    "DEPTHWISE_QUANTIZATION_SCHEME",
    "DEFAULT_RESOURCES_USED",
    "axelera_install_command",
    "build_compiler_config",
    "check_compile_platform",
    "collect_quantized_outputs",
    "compile_to_axm",
    "default_quantization_scheme",
    "device_output_order",
    "export_axelera",
    "normalize_compiler_overrides",
    "parity_report",
    "publish_directory",
    "require_axelera_devkit",
    "require_axelera_rt",
]
