"""Special commands: version, checks, models, formats, cfg, info."""

import sys
from typing import Optional

import typer

from ..command_utils import load_model_or_exit, resolve_model_or_exit
from ..output import OutputHandler


# =========================================================================
# Helpers
# =========================================================================
def _get_output(json_output: bool, quiet: bool) -> OutputHandler:
    return OutputHandler(json_mode=json_output, quiet=quiet)


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
    for pkg in ("onnx", "onnxruntime", "tensorrt", "openvino", "ncnn", "rfdetr"):
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
    from libreyolo.models.base.model import BaseModel

    families = []
    for cls in BaseModel._registry:
        family = cls.FAMILY
        sizes = list(cls.INPUT_SIZES.keys())
        cli_names = [f"{family}-{s}" for s in sizes]
        families.append(
            {
                "name": family,
                "sizes": sizes,
                "default_imgsz": cls.INPUT_SIZES,
                "cli_names": cli_names,
            }
        )

    # Check RF-DETR (lazily registered, may not be in _registry yet)
    rfdetr_present = any(f["name"] == "rfdetr" for f in families)
    if not rfdetr_present:
        from libreyolo.models import try_ensure_rfdetr

        rfcls = try_ensure_rfdetr()
        if rfcls is not None:
            sizes = list(rfcls.INPUT_SIZES.keys())
            families.append(
                {
                    "name": rfcls.FAMILY,
                    "sizes": sizes,
                    "default_imgsz": rfcls.INPUT_SIZES,
                    "cli_names": [f"{rfcls.FAMILY}-{s}" for s in sizes],
                }
            )

    out = _get_output(json_output, quiet)
    data = {"families": families}

    if not json_output:
        lines = ["Available models:", ""]
        for f in families:
            lines.append(f"  {f['name']}:")
            lines.append(f"    Sizes: {', '.join(f['sizes'])}")
            lines.append(f"    Names: {', '.join(f['cli_names'])}")
            imgsz_str = ", ".join(f"{s}={v}" for s, v in f["default_imgsz"].items())
            lines.append(f"    Input: {imgsz_str}")
            lines.append("")
        data["_human_text"] = "\n".join(lines)

    out.result(data)


# =========================================================================
# formats
# =========================================================================


def formats_cmd(
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """List supported export formats."""
    from libreyolo.export.exporter import BaseExporter

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

    formats = []
    for name, cls in sorted(BaseExporter._registry.items()):
        info: dict = {
            "name": name,
            "extension": cls.suffix,
            "int8": cls.supports_int8,
            "fp16": cls.supports_fp16,
            "requires_onnx": cls.requires_onnx,
        }
        if name == "tensorrt":
            info["aliases"] = ["engine"]
        formats.append(info)

    out = _get_output(json_output, quiet)
    data = {"formats": formats}

    if not json_output:
        lines = ["Supported export formats:", ""]
        for f in formats:
            alias = f" (alias: {', '.join(f['aliases'])})" if f.get("aliases") else ""
            lines.append(f"  {f['name']}{alias}")
            lines.append(
                f"    Extension: {f['extension']}, FP16: {f['fp16']}, INT8: {f['int8']}"
            )
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
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """Show model info: family, size, parameters, classes."""
    out = _get_output(json_output, quiet)

    model_path = resolve_model_or_exit(out, model)
    loaded = load_model_or_exit(out, model=model, model_path=model_path, device="cpu")

    num_params = sum(p.numel() for p in loaded.model.parameters())
    input_size = loaded.INPUT_SIZES.get(loaded.size, 640)

    class_names = {}
    if hasattr(loaded, "names") and loaded.names:
        class_names = (
            {i: n for i, n in enumerate(loaded.names)}
            if isinstance(loaded.names, list)
            else loaded.names
        )

    data = {
        "model": model,
        "model_family": loaded.FAMILY,
        "size": loaded.size,
        "num_classes": loaded.nb_classes,
        "parameters": num_params,
        "input_size": [input_size, input_size],
        "class_names": class_names,
    }

    if not json_output:
        lines = [
            f"Model:      {model}",
            f"Family:     {loaded.FAMILY}",
            f"Size:       {loaded.size}",
            f"Classes:    {loaded.nb_classes}",
            f"Parameters: {num_params:,}",
            f"Input size: {input_size}x{input_size}",
        ]
        data["_human_text"] = "\n".join(lines)

    out.result(data)
