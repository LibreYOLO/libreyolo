"""MNN export helpers built against the public MNN converter interface.

This is an original integration that invokes the ``mnnconvert`` command shipped
by the optional MNN package. No MNN source code is vendored or adapted here.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

from ._scripts import environment_script

logger = logging.getLogger(__name__)

# MNN 3.6.1 can finish conversion and then terminate during Windows process
# teardown with either STATUS_ACCESS_VIOLATION (0xC0000005) or the fail-fast
# status 0xC0000409. Keep both unsigned and signed subprocess representations.
_WINDOWS_POST_SUCCESS_EXIT_CODES = {
    3221225477,
    -1073741819,
    3221226505,
    -1073740791,
}


def _find_mnnconvert() -> Path:
    """Return the converter installed into the active Python environment."""
    found = environment_script("mnnconvert")
    if found is not None:
        return found.resolve()
    raise ImportError(
        "MNN export requires the mnnconvert tool shipped by the MNN package. "
        'Install with: pip install "libreyolo[mnn]"'
    )


def check_mnn_available() -> Path:
    """Validate the optional MNN runtime and converter installation."""
    from ..backends.mnn import import_mnn

    import_mnn("MNN export")
    try:
        importlib.metadata.version("MNN")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError(
            "MNN export requires the optional MNN package. "
            'Install with: pip install "libreyolo[mnn]"'
        ) from exc
    return _find_mnnconvert()


def _onnx_io_contract(onnx_path: Path) -> tuple[list[str], list[str], list[int]]:
    """Read the single-input, fixed-shape contract passed to MNN."""
    import onnx

    model = onnx.load(str(onnx_path))
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializer_names]
    if len(inputs) != 1:
        raise ValueError(
            f"MNN export expects one image input, got {len(inputs)} in {onnx_path}."
        )

    input_names = [inputs[0].name]
    output_names = [item.name for item in model.graph.output]
    if not output_names:
        raise ValueError(f"MNN export requires at least one ONNX output: {onnx_path}")

    dimensions = inputs[0].type.tensor_type.shape.dim
    input_shape = [
        int(dim.dim_value) if int(dim.dim_value) > 0 else -1 for dim in dimensions
    ]
    if len(input_shape) != 4 or any(value <= 0 for value in input_shape):
        raise ValueError(
            "MNN v1 export requires a fixed NCHW input shape; "
            f"got {input_shape} from {onnx_path}."
        )
    return input_names, output_names, input_shape


def _converter_failure_message(
    result: subprocess.CompletedProcess[str], command: list[str]
) -> str:
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    details = "\n".join(part for part in (stdout, stderr) if part)
    if not details:
        details = "The converter produced no diagnostic output."
    return (
        f"MNN conversion failed with exit code {result.returncode}.\n"
        f"Command: {' '.join(command)}\n{details}"
    )


def _validate_mnn_artifact(
    model_path: Path,
    input_names: list[str],
    output_names: list[str],
) -> None:
    """Load a staged artifact with the public CPU Module API."""
    import MNN

    mnn_api = cast(Any, MNN)
    runtime_manager = mnn_api.nn.create_runtime_manager(
        ({"backend": 0, "precision": 1, "numThread": 1},)
    )
    module = mnn_api.nn.load_module_from_file(
        str(model_path),
        input_names,
        output_names,
        runtime_manager=runtime_manager,
        dynamic=False,
        shape_mutable=False,
    )
    if module is None:
        raise RuntimeError("MNN returned no module for the staged artifact.")


def _can_recover_windows_converter_teardown(
    result: subprocess.CompletedProcess[str], staged_model: Path
) -> bool:
    """Recognize MNN 3.6.1's known post-success Windows teardown exits."""
    diagnostics = f"{result.stdout or ''}\n{result.stderr or ''}"
    return (
        os.name == "nt"
        and result.returncode in _WINDOWS_POST_SUCCESS_EXIT_CODES
        and staged_model.is_file()
        and staged_model.stat().st_size > 0
        and "Converted Success!" in diagnostics
    )


def _commit_artifact_pair(
    staged_model: Path,
    staged_sidecar: Path,
    destination: Path,
    sidecar_path: Path,
) -> None:
    """Commit model and sidecar together, restoring any previous pair on error."""
    staged = {destination: staged_model, sidecar_path: staged_sidecar}
    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    success = False
    try:
        for final_path in staged:
            if final_path.exists():
                with tempfile.NamedTemporaryFile(
                    prefix=f".{final_path.name}.",
                    suffix=".backup",
                    dir=final_path.parent,
                    delete=False,
                ) as backup_handle:
                    backup = Path(backup_handle.name)
                backup.unlink()
                os.replace(final_path, backup)
                backups[final_path] = backup
        for final_path, staged_path in staged.items():
            os.replace(staged_path, final_path)
            committed.append(final_path)
        success = True
    except Exception as commit_error:
        rollback_errors = []
        for final_path in reversed(committed):
            try:
                if final_path.exists():
                    final_path.unlink()
            except OSError as exc:
                rollback_errors.append(exc)
        for final_path, backup in backups.items():
            try:
                if backup.exists():
                    os.replace(backup, final_path)
            except OSError as exc:
                rollback_errors.append(exc)
        if rollback_errors:
            retained = ", ".join(
                str(path) for path in backups.values() if path.exists()
            )
            raise RuntimeError(
                "MNN artifact commit failed and rollback was incomplete. "
                f"Recovery backups were retained at: {retained}"
            ) from commit_error
        raise
    finally:
        if success:
            for backup in backups.values():
                if backup.exists():
                    backup.unlink()


def export_mnn(
    onnx_path: str,
    output_path: str,
    *,
    metadata: dict,
    batch: int,
    verbose: bool = False,
) -> str:
    """Convert a fixed-shape ONNX graph to MNN and write its metadata sidecar."""
    converter = check_mnn_available()
    source = Path(onnx_path)
    if not source.is_file():
        raise FileNotFoundError(f"ONNX intermediate not found: {source}")

    input_names, output_names, input_shape = _onnx_io_contract(source)
    if input_shape[0] != int(batch):
        raise ValueError(
            "MNN batch metadata does not match the ONNX input shape: "
            f"batch={batch}, input_shape={input_shape}."
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = Path(f"{destination}.json")

    sidecar = dict(metadata)
    sidecar.update(
        {
            "format": "mnn",
            "dynamic": False,
            "mnn_version": importlib.metadata.version("MNN"),
            "mnn_backend": "cpu",
            "mnn_input_names": input_names,
            "mnn_output_names": output_names,
            "mnn_input_shape": input_shape,
            "mnn_batch": int(batch),
        }
    )

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.stem}.mnn-", dir=destination.parent
    ) as temp_dir:
        staged_model = Path(temp_dir) / destination.name
        staged_sidecar = Path(temp_dir) / sidecar_path.name
        command = [
            str(converter),
            "-f",
            "ONNX",
            "--modelFile",
            str(source),
            "--MNNModel",
            str(staged_model),
            "--batch",
            str(batch),
            "--bizCode",
            "LibreYOLO",
        ]
        environment = os.environ.copy()
        environment["PATH"] = os.pathsep.join(
            (str(converter.parent), environment.get("PATH", ""))
        )
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        if result.returncode != 0:
            if not _can_recover_windows_converter_teardown(result, staged_model):
                raise RuntimeError(_converter_failure_message(result, command))
            try:
                _validate_mnn_artifact(staged_model, input_names, output_names)
            except Exception as exc:
                raise RuntimeError(
                    f"{_converter_failure_message(result, command)}\n"
                    f"The staged artifact also failed MNN CPU validation: {exc}"
                ) from exc
            logger.warning(
                "mnnconvert produced and validated %s, but exited with Windows "
                "status %s during process teardown; accepting the independently "
                "validated artifact.",
                staged_model.name,
                result.returncode,
            )
        if not staged_model.is_file() or staged_model.stat().st_size == 0:
            raise RuntimeError(
                "MNN conversion reported success but did not produce a non-empty "
                f"artifact at {staged_model}."
            )

        staged_sidecar.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8"
        )
        _commit_artifact_pair(
            staged_model,
            staged_sidecar,
            destination,
            sidecar_path,
        )

    if verbose and result.stdout:
        logger.info("mnnconvert output:\n%s", result.stdout.rstrip())
    logger.info("MNN export complete: %s", destination)
    logger.info("MNN metadata sidecar: %s", sidecar_path)
    return str(destination)


__all__ = ["check_mnn_available", "export_mnn"]
