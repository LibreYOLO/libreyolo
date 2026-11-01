"""ExecuTorch export helpers.

This integration is an original implementation against ExecuTorch's documented
public Python API. No ExecuTorch source code is vendored or adapted here.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def _capture_compatibility(metadata: dict):
    """Temporarily select family-local decompositions for strict capture."""
    module = None
    previous = None
    if metadata.get("model_family") == "rtdetrv4":
        from ..models.dfine import ms_deform

        module = ms_deform
        previous = module._FORCE_MANUAL_GRID_SAMPLE_EXPORT
        module._FORCE_MANUAL_GRID_SAMPLE_EXPORT = True
    try:
        yield
    finally:
        if module is not None:
            module._FORCE_MANUAL_GRID_SAMPLE_EXPORT = previous


def check_executorch_available() -> None:
    """Raise an actionable error when the optional ExecuTorch stack is absent."""
    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (  # noqa: F401
            XnnpackPartitioner,
        )
        from executorch.exir import to_edge_transform_and_lower  # noqa: F401
    except (ImportError, OSError) as exc:
        raise ImportError(
            "ExecuTorch export requires the optional ExecuTorch toolchain. "
            "Install it with: pip install libreyolo[executorch]"
        ) from exc


def _delegate_partition_count(edge_program) -> int:
    """Count delegate calls in the public exported edge program."""
    graph = edge_program.exported_program("forward").graph_module.graph
    return sum(
        "executorch_call_delegate" in str(node.target)
        for node in graph.nodes
        if node.op == "call_function"
    )


def _commit_artifact_pair(
    program_bytes: bytes,
    metadata: dict,
    program_path: Path,
) -> None:
    """Commit a program and JSON sidecar together, restoring prior files on error."""
    sidecar_path = Path(f"{program_path}.json")
    program_path.parent.mkdir(parents=True, exist_ok=True)

    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    success = False
    try:
        for final_path, payload in (
            (program_path, program_bytes),
            (
                sidecar_path,
                json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"),
            ),
        ):
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{final_path.name}.",
                suffix=".tmp",
                dir=final_path.parent,
                delete=False,
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                staged[final_path] = Path(handle.name)

        for final_path in (program_path, sidecar_path):
            if final_path.exists():
                with tempfile.NamedTemporaryFile(
                    prefix=f".{final_path.name}.",
                    suffix=".backup",
                    dir=final_path.parent,
                    delete=False,
                ) as backup_handle:
                    backup_path = Path(backup_handle.name)
                backup_path.unlink()
                os.replace(final_path, backup_path)
                backups[final_path] = backup_path

            os.replace(staged[final_path], final_path)
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
        for final_path, backup_path in backups.items():
            try:
                if backup_path.exists():
                    os.replace(backup_path, final_path)
            except OSError as exc:
                rollback_errors.append(exc)
        if rollback_errors:
            retained = ", ".join(
                str(path) for path in backups.values() if path.exists()
            )
            raise RuntimeError(
                "ExecuTorch artifact commit failed and rollback was incomplete. "
                f"Recovery backups were retained at: {retained}"
            ) from commit_error
        raise
    finally:
        for staged_path in staged.values():
            if staged_path.exists():
                staged_path.unlink()
        if success:
            for backup_path in backups.values():
                if backup_path.exists():
                    backup_path.unlink()



def _decompose_grid_sampler(exported):
    """Lower grid sampling into primitives every ExecuTorch runtime has.

    The Linux ExecuTorch 1.2 runtime registers no ``aten::grid_sampler_2d``
    kernel (the macOS one does) and XNNPACK does not delegate it, so a program
    that keeps the op fails to load there with "Missing operator" (RF-DETR's
    deformable attention). ``torch.export`` emits ``aten.grid_sampler``, which
    ``to_edge`` would turn into ``grid_sampler_2d``; decompose both here,
    leaving every other op for the XNNPACK partitioner.
    """
    from torch._decomp import get_decompositions

    aten = torch.ops.aten
    table = get_decompositions([aten.grid_sampler_2d])
    table[aten.grid_sampler.default] = torch.export.default_decompositions()[
        aten.grid_sampler.default
    ]
    return exported.run_decompositions(table)


def export_executorch(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    *,
    output_path: str,
    metadata: dict,
) -> str:
    """Capture, lower to XNNPACK, and serialize a fixed-shape FP32 program."""
    check_executorch_available()

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    with _capture_compatibility(metadata):
        exported = torch.export.export(model, (example_input,), strict=True)
    exported = _decompose_grid_sampler(exported)
    try:
        edge_program = to_edge_transform_and_lower(
            exported,
            partitioner=[XnnpackPartitioner()],
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ExecuTorch XNNPACK lowering could not start a required host tool. "
            "Ensure the ExecuTorch-bundled 'flatc' executable is on PATH; on "
            "Windows, run from a Visual Studio 2022 Developer PowerShell."
        ) from exc
    partition_count = _delegate_partition_count(edge_program)
    if partition_count == 0:
        raise RuntimeError(
            "ExecuTorch produced no XNNPACK delegate partitions. Refusing to "
            "label a portable-kernel-only program as XNNPACK."
        )

    program = edge_program.to_executorch()
    output = Path(output_path)
    sidecar = dict(metadata)
    sidecar.update(
        {
            "executorch_version": importlib.metadata.version("executorch"),
            "executorch_delegate": "xnnpack",
            "executorch_delegate_partitions": partition_count,
        }
    )
    _commit_artifact_pair(program.buffer, sidecar, output)
    return str(output)
