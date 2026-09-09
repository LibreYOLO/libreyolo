"""Special commands: version, checks, models, formats, cfg, info, metadata."""

import sys
from pathlib import Path
from typing import Any, Optional

import typer

from ..command_utils import (
    exit_stage_error,
    exit_with_error,
    load_model_or_exit,
    resolve_model_or_exit,
)
from ..errors import CLIError
from ..output import OutputHandler
from ...utils.model_info import build_model_info, format_model_info


# =========================================================================
# Helpers
# =========================================================================
def _get_output(json_output: bool, quiet: bool) -> OutputHandler:
    return OutputHandler(json_mode=json_output, quiet=quiet)


def _resolve_embed_face_detector(out: OutputHandler, face_detector: Optional[str], device: str):
    """Resolve a face detector for the two-stage face-embedding task.

    Accepts an OpenCV face-detector ``.onnx`` (wrapped as a 5-landmark
    ``OpenCVFaceDetector``) or a LibreYOLO detector checkpoint/name (wrapped via
    ``resolve_face_detector``).
    """
    if face_detector is None:
        # The runner falls back to the family's default detector
        # (auto-downloaded on first use).
        return None
    if str(face_detector).lower().endswith(".onnx"):
        from libreyolo.models.facerec import OpenCVFaceDetector

        return OpenCVFaceDetector(face_detector)
    fd_path = resolve_model_or_exit(out, face_detector)
    fd_model = load_model_or_exit(
        out, model=face_detector, model_path=fd_path, device=device
    )
    from libreyolo.models.l2cs.face import resolve_face_detector

    return resolve_face_detector(fd_model)


def compare_cmd(
    model: str = typer.Option(..., help="Face-embedding model (path or name)"),
    source: str = typer.Option(..., help="First image"),
    source2: str = typer.Option(..., help="Second image to compare against"),
    face_detector: Optional[str] = typer.Option(
        None, "--face-detector", help="Face detector (YuNet .onnx or LibreYOLO detector)"
    ),
    threshold: float = typer.Option(
        0.4, help="Cosine-similarity threshold for the same-identity decision"
    ),
    device: str = typer.Option("auto", help="Device: 0, cpu, mps, auto"),
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """Verify whether two face images are the same identity (cosine similarity)."""
    out = _get_output(json_output, quiet)

    for label, src in (("source", source), ("source2", source2)):
        if not Path(src).exists() and not str(src).startswith(("http://", "https://")):
            exit_with_error(out, "source_not_found", f"{label} not found: {src}")

    model_path = resolve_model_or_exit(out, model)
    loaded = load_model_or_exit(
        out, model=model, model_path=model_path, device=device, task="facial-recognition"
    )
    if getattr(loaded, "task", None) != "embed":
        exit_with_error(
            out,
            "config_unsupported",
            "compare requires a face-embedding model (task=facial-recognition).",
        )
    loaded.face_detector = _resolve_embed_face_detector(out, face_detector, device)

    try:
        res = loaded.verify(source, source2, threshold=threshold)
    except Exception as exc:  # no face / runtime error
        exit_stage_error(out, stage="compare", detail=exc, code="config_unsupported")

    similarity = round(float(res["similarity"]), 4)
    data = {
        "similarity": similarity,
        "same_person": bool(res["same_person"]),
        "threshold": threshold,
    }
    if not json_output:
        verdict = "SAME person" if data["same_person"] else "DIFFERENT people"
        data["_human_text"] = (
            f"cosine similarity: {similarity:.4f}  ->  {verdict} "
            f"(threshold {threshold})"
        )
    out.result(data)


def enroll_cmd(
    model: str = typer.Option(..., help="Face-embedding model (path or name)"),
    source: str = typer.Option(
        ..., help="Folder-per-person tree: source/<identity>/*.jpg"
    ),
    gallery: str = typer.Option(
        ..., help="Output gallery file (.npz); extended in place if it exists"
    ),
    face_detector: Optional[str] = typer.Option(
        None, "--face-detector", help="Face detector (YuNet .onnx or LibreYOLO detector)"
    ),
    device: str = typer.Option("auto", help="Device: 0, cpu, mps, auto"),
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """Enroll identities into a face gallery from a folder-per-person tree."""
    from libreyolo.utils.image_loader import ImageLoader

    out = _get_output(json_output, quiet)

    source_dir = Path(source)
    if not source_dir.is_dir():
        exit_with_error(
            out,
            "source_not_found",
            f"source must be a directory laid out as <identity>/<images>: {source}",
        )
    identity_dirs = sorted(d for d in source_dir.iterdir() if d.is_dir())
    if not identity_dirs:
        exit_with_error(
            out,
            "config_unsupported",
            f"No identity subfolders found under {source}. Expected "
            "source/<identity_name>/*.jpg (the folder name becomes the identity).",
        )

    model_path = resolve_model_or_exit(out, model)
    loaded = load_model_or_exit(
        out, model=model, model_path=model_path, device=device, task="facial-recognition"
    )
    if getattr(loaded, "task", None) != "embed":
        exit_with_error(
            out,
            "config_unsupported",
            "enroll requires a face-embedding model (task=facial-recognition).",
        )
    loaded.face_detector = _resolve_embed_face_detector(out, face_detector, device)

    from libreyolo.models.facerec import FaceGallery

    gallery_path = Path(gallery)
    if gallery_path.exists():
        book = FaceGallery.load(gallery_path, embedder=loaded)
    else:
        book = FaceGallery(embedder=loaded)

    enrolled: dict[str, int] = {}
    skipped: list[str] = []
    for identity_dir in identity_dirs:
        images = ImageLoader.collect_images(identity_dir)
        if not images:
            skipped.append(identity_dir.name)
            continue
        count = 0
        for image in images:
            try:
                count += book.enroll(identity_dir.name, image)
            except ValueError as exc:  # e.g. no face found in one reference
                if not quiet:
                    typer.echo(f"skip {image}: {exc}", err=True)
        if count:
            enrolled[identity_dir.name] = count
        else:
            skipped.append(identity_dir.name)

    if not enrolled:
        exit_stage_error(
            out,
            stage="enroll",
            detail="no faces found in any identity folder",
            code="config_unsupported",
        )

    try:
        book.save(gallery_path)
    except Exception as exc:
        exit_stage_error(out, stage="enroll", detail=exc, code="io_error")

    data = {
        "gallery": str(gallery_path),
        "identities": len(book),
        "references": sum(enrolled.values()),
        "enrolled": enrolled,
        "skipped": skipped,
    }
    if not json_output:
        data["_human_text"] = (
            f"enrolled {sum(enrolled.values())} reference faces for "
            f"{len(enrolled)} identities -> {gallery_path}"
            + (f" (skipped: {', '.join(skipped)})" if skipped else "")
        )
    out.result(data)


def _metadata_value_for_cli(value: Any) -> Any:
    """Return a compact JSON-safe representation for raw checkpoint metadata."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
        return {"type": type(value).__name__, "shape": shape, "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return [_metadata_value_for_cli(item) for item in value]
    if isinstance(value, dict):
        if len(value) > 200:
            return {"type": "dict", "keys": len(value)}
        return {str(key): _metadata_value_for_cli(item) for key, item in value.items()}
    return str(value)


# =========================================================================
# version
# =========================================================================


def version_cmd(
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """Print LibreYOLO version and environment info."""
    import torch

    from libreyolo import __version__

    cuda_version = torch.version.cuda or None
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    out = _get_output(json_output, quiet)
    data = {
        "version": __version__,
        "python": python_version,
        "torch": torch.__version__,
        "cuda": cuda_version,
    }
    if not json_output:
        data["_human_text"] = (
            f"libreyolo {__version__}\n"
            f"Python {python_version}, torch {torch.__version__}, "
            f"CUDA {cuda_version or 'not available'}"
        )
    out.result(data)


# =========================================================================
# checks
# =========================================================================


def checks_cmd(
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """System info: GPU, CUDA, Python, installed packages."""
    import torch

    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "memory_mb": props.total_memory // (1024 * 1024),
                }
            )

    packages: dict[str, Optional[str]] = {}
    for pkg in (
        "onnx",
        "onnxruntime",
        "tensorrt",
        "openvino",
        "paddlepaddle",
        "x2paddle",
        "mnn",
        "ncnn",
        "onnx2tf",
        "ai-edge-litert",
        "transformers",
        "scipy",
    ):
        try:
            from importlib.metadata import version

            packages[pkg] = version(pkg)
        except Exception:
            packages[pkg] = None

    out = _get_output(json_output, quiet)
    data = {
        "python": python_version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": str(torch.backends.cudnn.version())
        if torch.backends.cudnn.is_available()
        else None,
        "gpu": gpus,
        "packages": packages,
    }

    if not json_output:
        lines = [
            f"Python:  {python_version}",
            f"Torch:   {torch.__version__}",
            f"CUDA:    {torch.version.cuda or 'not available'}",
            f"cuDNN:   {data['cudnn'] or 'not available'}",
        ]
        if gpus:
            for g in gpus:
                lines.append(f"GPU {g['index']}:   {g['name']} ({g['memory_mb']} MB)")
        else:
            lines.append("GPU:     none detected")
        lines.append("")
        lines.append("Packages:")
        for pkg, ver in packages.items():
            lines.append(f"  {pkg}: {ver or 'not installed'}")
        data["_human_text"] = "\n".join(lines)

    out.result(data)


# =========================================================================
# models
# =========================================================================


def models_cmd(
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """List available model families and sizes."""
    from libreyolo.models.inventory import collect_model_inventory
    from libreyolo.tasks import task_to_suffix

    families = []
    for family, metadata in collect_model_inventory().items():
        cli_names = []
        task_sizes = metadata["task_sizes"] or {
            metadata["default_task"]: metadata["default_imgsz"]
        }
        for task, sizes in task_sizes.items():
            suffix = task_to_suffix(task)
            for size in sizes:
                name = f"{family}-{size}" + (f"-{suffix}" if suffix else "")
                if name not in cli_names:
                    cli_names.append(name)
        if metadata.get("cli_command"):
            cli_names = [metadata["cli_command"]]
        extra = metadata["optional_extra"]
        families.append(
            {
                "name": family,
                "sizes": list(metadata["sizes"]),
                "default_imgsz": metadata["default_imgsz"],
                "task_sizes": metadata["task_sizes"],
                "tasks": metadata["tasks"],
                "default_task": metadata["default_task"],
                "cli_names": cli_names,
                "available": metadata["available"],
                "optional_extra": extra,
                "install_hint": f"pip install libreyolo[{extra}]" if extra else None,
            }
        )

    out = _get_output(json_output, quiet)
    data = {"families": families}

    if not json_output:
        lines = ["Available models:", ""]
        for f in families:
            lines.append(f"  {f['name']}:")
            lines.append(
                f"    Tasks: {', '.join(f['tasks'])} (default: {f['default_task']})"
            )
            lines.append(f"    Sizes: {', '.join(f['sizes'])}")
            lines.append(f"    Names: {', '.join(f['cli_names'])}")
            imgsz_str = ", ".join(f"{s}={v}" for s, v in f["default_imgsz"].items())
            lines.append(f"    Input: {imgsz_str}")
            if not f["available"] and f["install_hint"]:
                lines.append(f"    Unavailable: install with {f['install_hint']}")
            lines.append("")
        data["_human_text"] = "\n".join(lines)

    out.result(data)


# =========================================================================
# formats
# =========================================================================


def formats_cmd(
    family: Optional[str] = typer.Option(
        None, "--family", "--model", help="Show tiers for one model family"
    ),
    task: Optional[str] = typer.Option(None, "--task", help="Canonical model task"),
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """List supported export formats."""
    from libreyolo.export.exporter import BaseExporter
    from libreyolo.export.support import get_support
    from libreyolo.models.inventory import collect_model_inventory
    from libreyolo.tasks import normalize_task

    # Trigger registration of optional exporters
    try:
        from libreyolo.export import tensorrt as _  # noqa: F401
    except ImportError:
        pass
    try:
        from libreyolo.export import openvino as _  # noqa: F401
    except ImportError:
        pass
    try:
        from libreyolo.export import ncnn as _  # noqa: F401
    except ImportError:
        pass

    selected_task = None
    if family is not None:
        inventory = collect_model_inventory()
        family = family.lower()
        if family not in inventory:
            raise typer.BadParameter(f"Unknown model family: {family!r}")
        selected_task = normalize_task(task, default=inventory[family]["default_task"])
        if selected_task not in inventory[family]["tasks"]:
            raise typer.BadParameter(
                f"{family!r} does not support task {selected_task!r}."
            )

    formats = []
    for name, cls in sorted(BaseExporter._registry.items()):
        info: dict = {
            "name": name,
            "extension": cls.suffix,
            "int8": cls.supports_int8,
            "fp16": cls.supports_fp16,
            "requires_onnx": cls.requires_onnx,
        }
        aliases = sorted(
            alias for alias, target in BaseExporter._aliases.items() if target == name
        )
        if aliases:
            info["aliases"] = aliases
        if family is not None and selected_task is not None:
            support = get_support(family, selected_task, name)
            info["tier"] = support.tier
            info["reason"] = support.reason
            info["constraint"] = support.constraint
        formats.append(info)

    out = _get_output(json_output, quiet)
    data = {"formats": formats, "family": family, "task": selected_task}

    if not json_output:
        lines = ["Supported export formats:", ""]
        for f in formats:
            alias = f" (alias: {', '.join(f['aliases'])})" if f.get("aliases") else ""
            lines.append(f"  {f['name']}{alias}")
            lines.append(
                f"    Extension: {f['extension']}, FP16: {f['fp16']}, INT8: {f['int8']}"
            )
            if "tier" in f:
                lines.append(f"    Tier: {f['tier']} ({f['reason']})")
                if f.get("constraint"):
                    lines.append(f"    Constraint: {f['constraint']}")
        data["_human_text"] = "\n".join(lines)

    out.result(data)


# =========================================================================
# cfg
# =========================================================================


def cfg_cmd(
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """Print default configuration."""
    from ..config import get_cfg_defaults

    data = get_cfg_defaults()

    out = _get_output(json_output, quiet)

    if not json_output:
        import yaml

        data["_human_text"] = yaml.dump(
            {k: v for k, v in data.items() if not k.startswith("_")},
            default_flow_style=False,
            sort_keys=False,
        )

    out.result(data)


# =========================================================================
# info
# =========================================================================


def info_cmd(
    model: str = typer.Option(..., help="Model name or path to weights"),
    detailed: bool = typer.Option(False, help="Include per-parameter details"),
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """Show model info: family, size, parameters, classes."""
    out = _get_output(json_output, quiet)

    model_path = resolve_model_or_exit(out, model)
    loaded = load_model_or_exit(out, model=model, model_path=model_path, device="cpu")

    info_fn = getattr(loaded, "info", None)
    if callable(info_fn):
        data = info_fn(detailed=detailed, verbose=False)
    else:
        data = build_model_info(loaded, detailed=detailed)
    data["model"] = model
    family = getattr(loaded, "FAMILY", None)
    task = getattr(loaded, "task", "detect")
    if family:
        from libreyolo.export.support import EXPORT_FORMATS, get_support

        data["export_support"] = {
            fmt: get_support(family, task, fmt).tier for fmt in EXPORT_FORMATS
        }

    if not json_output:
        data["_human_text"] = format_model_info(data)

    out.result(data)


# =========================================================================
# metadata
# =========================================================================


def metadata_cmd(
    path: str = typer.Option(..., help="Path to a .pt checkpoint"),
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """Inspect raw LibreYOLO checkpoint metadata without constructing a model."""
    from libreyolo.utils.serialization import (
        REQUIRED_CHECKPOINT_METADATA_KEYS,
        load_untrusted_torch_file,
        validate_checkpoint_metadata,
    )

    out = _get_output(json_output, quiet)
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        err = CLIError(
            "checkpoint_not_found",
            f"Checkpoint not found: {path}",
        )
        out.error(err)
        raise typer.Exit(err.exit_code)

    loaded = load_untrusted_torch_file(
        checkpoint_path,
        map_location="cpu",
        context="checkpoint metadata",
    )
    errors = validate_checkpoint_metadata(loaded, strict=False)
    metadata = {}
    if isinstance(loaded, dict):
        metadata = {
            key: (
                {"type": "dict", "keys": len(value)}
                if key in {"train_model", "ema", "optimizer"}
                and isinstance(value, dict)
                else _metadata_value_for_cli(value)
            )
            for key, value in loaded.items()
            if key != "model"
        }

    data = {
        "path": str(checkpoint_path),
        "valid": not errors,
        "errors": errors,
        "metadata": metadata,
    }
    if not json_output:
        lines = [
            f"Checkpoint: {checkpoint_path}",
            f"Valid LibreYOLO metadata: {'yes' if not errors else 'no'}",
        ]
        if metadata:
            lines.append("")
            lines.append("Metadata:")
            ordered_keys = [
                key for key in REQUIRED_CHECKPOINT_METADATA_KEYS if key in metadata
            ]
            ordered_keys.extend(key for key in metadata if key not in ordered_keys)
            for key in ordered_keys:
                lines.append(f"  {key}: {metadata[key]}")
        if errors:
            lines.append("")
            lines.append("Errors:")
            lines.extend(f"  - {error}" for error in errors)
        data["_human_text"] = "\n".join(lines)

    out.result(data)
    if errors:
        raise typer.Exit(1)
