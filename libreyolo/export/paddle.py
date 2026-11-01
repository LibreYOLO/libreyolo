"""PaddlePaddle export through an intermediate ONNX graph and X2Paddle."""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SUPPORTED_PADDLE_VERSION = "2.6.2"
_SUPPORTED_X2PADDLE_VERSION = "1.6.0"
_MAX_ONNX_VERSION = (1, 17)
_MAX_ONNX_OPSET = 15
_X2PADDLE_PATCH_LOCK = threading.Lock()
_PADDLE_INSTALL_LOCK_TIMEOUT = 10 * 60
_PADDLE_INSTALL_LOCK_POLL_SECONDS = 0.05


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _major_minor(value: str) -> tuple[int, int]:
    parts = value.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError) as exc:
        raise ImportError(f"Could not parse dependency version {value!r}.") from exc


def check_paddle_export_available() -> None:
    """Validate the narrow converter stack covered by Paddle parity tests."""
    if sys.version_info >= (3, 12):
        # Measured 2026-11-01: paddle.jit forces full_graph=True on 3.12+,
        # the static-graph step fails, and X2Paddle aborts the conversion.
        raise ImportError(
            "Paddle export needs Python 3.11 or earlier: paddlepaddle 2.6.2 "
            "cannot build the static graph X2Paddle writes on Python "
            f"{sys.version_info[0]}.{sys.version_info[1]}."
        )
    missing = [
        module
        for module in ("onnx", "paddle", "six", "x2paddle")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise ImportError(
            "Paddle export requires the optional Paddle toolchain "
            f"(missing: {', '.join(missing)}). Install with: "
            "pip install libreyolo[paddle]"
        )

    paddle_version = _package_version("paddlepaddle")
    x2paddle_version = _package_version("x2paddle")
    onnx_version = _package_version("onnx")
    if paddle_version != _SUPPORTED_PADDLE_VERSION:
        raise ImportError(
            "Paddle export is validated with paddlepaddle==2.6.2; got "
            f"{paddle_version or 'an unknown installation'}. Install the "
            "tested stack with: pip install libreyolo[paddle]"
        )
    if x2paddle_version != _SUPPORTED_X2PADDLE_VERSION:
        raise ImportError(
            "Paddle export is validated with x2paddle==1.6.0; got "
            f"{x2paddle_version or 'an unknown installation'}. Install the "
            "tested stack with: pip install libreyolo[paddle]"
        )
    if onnx_version is None or _major_minor(onnx_version) > _MAX_ONNX_VERSION:
        raise ImportError(
            "X2Paddle 1.6.0 requires ONNX <=1.17 for this export path; got "
            f"{onnx_version or 'an unknown installation'}. Install the tested "
            "stack with: pip install libreyolo[paddle]"
        )


def _normalize_onnx_for_x2paddle(onnx_path: str | Path) -> None:
    """Normalize equivalent ONNX forms rejected by X2Paddle 1.6.0.

    ONNX defines omitted MaxPool dilation as one. PyTorch writes the explicit
    all-ones attribute, while X2Paddle 1.6.0 rejects it. Removing only that
    redundant default preserves the graph's specified operation.
    """
    import numpy as np
    import onnx

    path = Path(onnx_path)
    graph = onnx.load(str(path))
    opsets = [entry.version for entry in graph.opset_import if not entry.domain]
    if not opsets:
        raise ValueError("Intermediate ONNX graph does not declare a default opset.")
    if max(opsets) > _MAX_ONNX_OPSET:
        raise NotImplementedError(
            "Paddle export through X2Paddle 1.6.0 supports ONNX opset 15 or "
            f"lower, but the intermediate graph uses opset {max(opsets)}."
        )

    tensor_shapes: dict[str, tuple[int, ...]] = {}
    for value in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output):
        dimensions = value.type.tensor_type.shape.dim
        shape = tuple(int(dimension.dim_value) for dimension in dimensions)
        if shape and all(dimension > 0 for dimension in shape):
            tensor_shapes[value.name] = shape
    initializers = {item.name: item for item in graph.graph.initializer}

    changed = False
    static_reshape_after: dict[int, Any] = {}
    clip_replacements: dict[int, list[Any]] = {}
    einsum_replacements: dict[int, list[Any]] = {}
    for node_index, node in enumerate(graph.graph.node):
        if node.op_type == "MaxPool":
            for index in range(len(node.attribute) - 1, -1, -1):
                attribute = node.attribute[index]
                if (
                    attribute.name == "dilations"
                    and attribute.ints
                    and all(value == 1 for value in attribute.ints)
                ):
                    del node.attribute[index]
                    changed = True

        # Paddle 2.6 rejects an int64 tensor as Clip's min/max input. The ONNX
        # definition is exactly min(max(X, min), max), whose primitive mappers
        # retain int64 support in this converter stack.
        if node.op_type == "Clip" and len(node.input) > 1:
            lower = node.input[1] if len(node.input) > 1 else ""
            upper = node.input[2] if len(node.input) > 2 else ""
            if lower or upper:
                replacements = []
                current = node.input[0]
                if lower:
                    lower_output = (
                        f"{node.output[0]}.x2paddle_lower" if upper else node.output[0]
                    )
                    replacements.append(
                        onnx.helper.make_node(
                            "Max",
                            (current, lower),
                            (lower_output,),
                            name=f"{node.output[0]}.x2paddle_lower",
                        )
                    )
                    current = lower_output
                if upper:
                    replacements.append(
                        onnx.helper.make_node(
                            "Min",
                            (current, upper),
                            (node.output[0],),
                            name=f"{node.output[0]}.x2paddle_upper",
                        )
                    )
                clip_replacements[node_index] = replacements
                changed = True

        # ONNX Gather accepts negative indices, while Paddle 2.6 rejects them.
        # Resolve constant negatives against a known static axis dimension.
        if (
            node.op_type == "Gather"
            and len(node.input) == 2
            and node.input[0] in tensor_shapes
            and node.input[1] in initializers
        ):
            from onnx import numpy_helper

            axis_attribute = next(
                (attribute for attribute in node.attribute if attribute.name == "axis"),
                None,
            )
            axis = int(axis_attribute.i) if axis_attribute is not None else 0
            rank = len(tensor_shapes[node.input[0]])
            axis = axis if axis >= 0 else axis + rank
            if 0 <= axis < rank:
                dimension = tensor_shapes[node.input[0]][axis]
                indices = numpy_helper.to_array(initializers[node.input[1]])
                if np.any(indices < 0) and np.all(indices >= -dimension):
                    resolved = np.where(indices < 0, indices + dimension, indices)
                    indices_name = f"{node.output[0]}.x2paddle_indices"
                    graph.graph.initializer.append(
                        numpy_helper.from_array(resolved, name=indices_name)
                    )
                    node.input[1] = indices_name
                    changed = True

        # X2Paddle 1.6 has no Einsum mapper. Segmentation heads use this one
        # batched mask projection, which is exactly a reshape + batch MatMul.
        if node.op_type == "Einsum" and len(node.input) == 2:
            equation = next(
                (
                    attribute.s
                    for attribute in node.attribute
                    if attribute.name == "equation"
                ),
                b"",
            )
            if equation == b"bchw,bnc->bnhw":
                features_name, queries_name = node.input
            elif equation == b"bqc,bchw->bqhw":
                queries_name, features_name = node.input
            else:
                features_name = queries_name = ""
            features_shape = tensor_shapes.get(features_name)
            queries_shape = tensor_shapes.get(queries_name)
            output_shape = tensor_shapes.get(node.output[0])
            if (
                features_shape is not None
                and queries_shape is not None
                and output_shape is not None
                and len(features_shape) == 4
                and len(queries_shape) == 3
                and len(output_shape) == 4
                and features_shape[0] == queries_shape[0] == output_shape[0]
                and features_shape[1] == queries_shape[2]
                and output_shape[1] == queries_shape[1]
                and output_shape[2:] == features_shape[2:]
            ):
                from onnx import numpy_helper

                flat_shape_name = f"{node.output[0]}.x2paddle_flat_shape"
                output_shape_name = f"{node.output[0]}.x2paddle_output_shape"
                graph.graph.initializer.extend(
                    (
                        numpy_helper.from_array(
                            np.asarray(
                                (features_shape[0], features_shape[1], -1),
                                dtype=np.int64,
                            ),
                            name=flat_shape_name,
                        ),
                        numpy_helper.from_array(
                            np.asarray(output_shape, dtype=np.int64),
                            name=output_shape_name,
                        ),
                    )
                )
                flattened = f"{node.output[0]}.x2paddle_flattened"
                projected = f"{node.output[0]}.x2paddle_projected"
                einsum_replacements[node_index] = [
                    onnx.helper.make_node(
                        "Reshape",
                        (features_name, flat_shape_name),
                        (flattened,),
                        name=f"{node.output[0]}.x2paddle_flatten",
                    ),
                    onnx.helper.make_node(
                        "MatMul",
                        (queries_name, flattened),
                        (projected,),
                        name=f"{node.output[0]}.x2paddle_project",
                    ),
                    onnx.helper.make_node(
                        "Reshape",
                        (projected, output_shape_name),
                        (node.output[0],),
                        name=f"{node.output[0]}.x2paddle_restore",
                    ),
                ]
                changed = True

        # ONNX represents a sizes-driven Resize as [X, "", "", sizes].
        # X2Paddle 1.6.0 drops the empty optional inputs internally, but then
        # still indexes slot 3. Convert only fully static shapes to the
        # equivalent scales-driven form that its mapper accepts.
        if (
            node.op_type == "Resize"
            and len(node.input) == 4
            and not node.input[1]
            and not node.input[2]
            and node.input[3] in initializers
            and node.input[0] in tensor_shapes
        ):
            from onnx import numpy_helper

            source_shape = tensor_shapes[node.input[0]]
            target_shape = tuple(
                int(value)
                for value in numpy_helper.to_array(initializers[node.input[3]])
            )
            if (
                len(source_shape) != len(target_shape)
                or len(source_shape) < 3
                or any(value <= 0 for value in target_shape)
            ):
                continue
            scales = np.asarray(target_shape, dtype=np.float32) / np.asarray(
                source_shape, dtype=np.float32
            )
            # Restrict this compatibility rewrite to exact integer or reciprocal
            # integer ratios so float rounding cannot change the output shape.
            ratios = np.maximum(scales, 1.0 / scales)
            if not np.allclose(ratios, np.rint(ratios), rtol=0.0, atol=1e-6):
                continue

            coordinate_mode = next(
                (
                    attribute
                    for attribute in node.attribute
                    if attribute.name == "coordinate_transformation_mode"
                ),
                None,
            )
            if (
                coordinate_mode is not None
                and coordinate_mode.s == b"half_pixel"
                and any(value == 1 for value in target_shape[2:])
            ):
                continue

            source_name = node.input[0]
            sizes_name = node.input[3]
            output_name = node.output[0]
            scales_name = f"{node.output[0]}.x2paddle_scales"
            graph.graph.initializer.append(
                numpy_helper.from_array(scales, name=scales_name)
            )
            del node.input[:]
            node.input.extend((source_name, "", scales_name))
            # The two ONNX modes differ only when an output dimension is one.
            # X2Paddle maps pytorch_half_pixel to Paddle's half-pixel formula.
            if coordinate_mode is not None and coordinate_mode.s == b"half_pixel":
                coordinate_mode.s = b"pytorch_half_pixel"
            resize_output = f"{output_name}.x2paddle_resize"
            node.output[0] = resize_output
            static_reshape_after[node_index] = onnx.helper.make_node(
                "Reshape",
                (resize_output, sizes_name),
                (output_name,),
                name=f"{output_name}.x2paddle_static_shape",
            )
            changed = True
    if changed:
        if static_reshape_after or clip_replacements or einsum_replacements:
            nodes = list(graph.graph.node)
            del graph.graph.node[:]
            for node_index, node in enumerate(nodes):
                replacements = clip_replacements.get(node_index)
                if replacements is None:
                    replacements = einsum_replacements.get(node_index)
                if replacements is None:
                    graph.graph.node.append(node)
                else:
                    graph.graph.node.extend(replacements)
                reshape = static_reshape_after.get(node_index)
                if reshape is not None:
                    graph.graph.node.append(reshape)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(path))


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(metadata, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _recover_interrupted_install(output_dir: Path, backup_dir: Path) -> None:
    """Restore the previous artifact if an earlier install was interrupted."""
    if not backup_dir.exists() and not backup_dir.is_symlink():
        return
    if output_dir.exists() or output_dir.is_symlink():
        _remove_path(backup_dir)
    else:
        os.replace(backup_dir, output_dir)


def _try_lock_install_file(lock_file) -> None:
    lock_file.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_install_file(lock_file) -> None:
    lock_file.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _paddle_install_lock(output_dir: Path):
    """Serialize recovery and replacement for one output across processes."""
    lock_path = output_dir.with_name(f".{output_dir.name}.install.lock")
    with lock_path.open("a+b") as lock_file:
        if lock_file.seek(0, os.SEEK_END) == 0:
            lock_file.write(b"\0")
            lock_file.flush()

        deadline = time.monotonic() + _PADDLE_INSTALL_LOCK_TIMEOUT
        while True:
            try:
                _try_lock_install_file(lock_file)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting to install Paddle export: {output_dir}"
                    ) from exc
                time.sleep(_PADDLE_INSTALL_LOCK_POLL_SECONDS)

        try:
            yield
        finally:
            _unlock_install_file(lock_file)


@contextmanager
def _x2paddle_conversion_compatibility():
    """Apply narrow compatibility fixes to the pinned X2Paddle converter."""
    with _X2PADDLE_PATCH_LOCK:
        try:
            from x2paddle.op_mapper.onnx2paddle.opset_legacy import OpSet
            from x2paddle.optimizer.optimizer import GraphOptimizer
        except ModuleNotFoundError:
            # Unit tests may provide only the public converter module.
            yield
            return

        original_topk = OpSet.TopK
        original_optimize = GraphOptimizer.optimize

        def topk_with_value_alias(mapper, node):
            original_topk(mapper, node)
            mapper.paddle_graph.add_layer(
                "paddle.assign",
                inputs={"x": f"{node.layer_name}_p0"},
                outputs=[node.layer_name],
            )

        OpSet.TopK = topk_with_value_alias
        # X2Paddle always runs its Paddle-graph fusion passes even when
        # enable_optim=False. Those passes can invalidate constants in large
        # transformer graphs; the unfused mapper graph is already complete.
        GraphOptimizer.optimize = lambda self, graph: graph
        try:
            yield
        finally:
            OpSet.TopK = original_topk
            GraphOptimizer.optimize = original_optimize


@contextmanager
def _paddle_unique_name_guard():
    """Give each conversion an isolated Paddle parameter-name registry."""
    try:
        from paddle.utils.unique_name import guard
    except ModuleNotFoundError:
        # Unit tests may provide only the public converter module.
        yield
        return

    with guard():
        yield


def export_paddle(
    onnx_path: str,
    output_path: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Convert a static FP32 ONNX graph into a Paddle inference directory."""
    check_paddle_export_available()
    _normalize_onnx_for_x2paddle(onnx_path)

    from x2paddle.convert import onnx2paddle

    output_dir = Path(output_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = output_dir.with_name(f".{output_dir.name}.previous")
    with _paddle_install_lock(output_dir):
        _recover_interrupted_install(output_dir, backup_dir)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=str(output_dir.parent)
    ) as temporary:
        temporary_root = Path(temporary)
        conversion_dir = temporary_root / "conversion"
        artifact_dir = temporary_root / "artifact"
        with _x2paddle_conversion_compatibility(), _paddle_unique_name_guard():
            try:
                onnx2paddle(
                    onnx_path,
                    str(conversion_dir),
                    enable_optim=False,
                    disable_feedback=True,
                    enable_onnx_checker=True,
                )
            except SystemExit as exc:
                # X2Paddle calls exit(0) when its dynamic-to-static step
                # fails, which would end the caller's process as a success.
                raise RuntimeError(
                    "X2Paddle aborted the conversion (it printed the cause "
                    f"above and called exit({exc.code})); no Paddle model "
                    "was written."
                ) from exc

        generated = conversion_dir / "inference_model"
        required = (generated / "model.pdmodel", generated / "model.pdiparams")
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "X2Paddle did not produce a runnable Paddle inference model "
                f"(missing: {', '.join(missing)})."
            )

        artifact_dir.mkdir()
        for source in required:
            shutil.copy2(source, artifact_dir / source.name)
        parameters_info = generated / "model.pdiparams.info"
        if parameters_info.is_file():
            shutil.copy2(parameters_info, artifact_dir / parameters_info.name)
        _write_metadata(artifact_dir / "metadata.yaml", metadata or {})

        with _paddle_install_lock(output_dir):
            # Another process may have completed or interrupted an install
            # while this conversion was running.
            _recover_interrupted_install(output_dir, backup_dir)
            if output_dir.exists() or output_dir.is_symlink():
                os.replace(output_dir, backup_dir)
            try:
                os.replace(artifact_dir, output_dir)
            except BaseException:
                if backup_dir.exists() or backup_dir.is_symlink():
                    if output_dir.exists() or output_dir.is_symlink():
                        _remove_path(output_dir)
                    os.replace(backup_dir, output_dir)
                raise
            else:
                _remove_path(backup_dir)

    logger.info("Paddle export complete: %s", output_dir)
    return str(output_dir)


__all__ = [
    "check_paddle_export_available",
    "export_paddle",
]
