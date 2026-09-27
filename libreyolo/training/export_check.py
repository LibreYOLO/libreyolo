"""Epoch-0 export sanity check.

Exports the live model to ONNX and, when the raw torch output is a tensor
(or tuple of tensors) of the same layout as the ONNX graph, asserts numeric
parity. Layout mismatches (export-time NMS vs raw training heads) are
logged and skipped: the check still fails if ``export()`` itself raises,
which is the failure mode this is meant to catch before a long run.

Opt-in. Off by default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import torch

logger = logging.getLogger(__name__)


def _as_tensor_tuple(value: Any) -> tuple[torch.Tensor, ...] | None:
    if torch.is_tensor(value):
        return (value.detach().cpu(),)
    if isinstance(value, (tuple, list)) and value and all(torch.is_tensor(v) for v in value):
        return tuple(v.detach().cpu() for v in value)
    return None


def outputs_comparable(
    torch_out: Sequence[torch.Tensor], onnx_out: Sequence[torch.Tensor]
) -> bool:
    if len(torch_out) != len(onnx_out):
        return False
    return all(
        tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype
        for a, b in zip(torch_out, onnx_out)
    )


def assert_close(
    torch_out: Sequence[torch.Tensor],
    onnx_out: Sequence[torch.Tensor],
    *,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> None:
    if not outputs_comparable(torch_out, onnx_out):
        raise ValueError(
            "export_check: torch and ONNX outputs are not comparable "
            f"(torch shapes={[tuple(t.shape) for t in torch_out]}, "
            f"onnx shapes={[tuple(t.shape) for t in onnx_out]})"
        )
    for i, (left, right) in enumerate(zip(torch_out, onnx_out)):
        if not torch.allclose(left.float(), right.float(), rtol=rtol, atol=atol):
            max_abs = (left.float() - right.float()).abs().max().item()
            raise AssertionError(
                f"export_check: ONNX output {i} differs from torch "
                f"(max abs {max_abs:.4g}, rtol={rtol}, atol={atol})"
            )


def run_export_parity_check(
    wrapper: Any,
    *,
    out_dir: str | Path,
    imgsz: int | tuple[int, int] = 640,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> Path:
    """Export ONNX next to the run and compare raw outputs when layouts match.

    Returns the written ONNX path. Raises if export fails or if comparable
    outputs disagree.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "export_check=True requires onnxruntime. "
            "Install it with `pip install onnxruntime`."
        ) from exc

    if isinstance(imgsz, (list, tuple)):
        height, width = int(imgsz[0]), int(imgsz[1])
    else:
        height = width = int(imgsz)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "export_check.onnx"

    export = getattr(wrapper, "export", None)
    if not callable(export):
        raise RuntimeError("export_check=True requires a wrapper with export()")

    live = getattr(wrapper, "model", None)
    swapped = False
    try:
        from .lora import module_has_lora

        if live is not None and module_has_lora(live):
            # export() folds LoRA in place. Swap a copy so the live
            # adapters and the optimizer stay intact.
            import copy

            wrapper.model = copy.deepcopy(live)
            swapped = True
        written = export(
            format="onnx",
            output_path=str(onnx_path),
            imgsz=(height, width),
        )
    finally:
        if swapped:
            wrapper.model = live
    written_path = Path(written if written else onnx_path)
    if not written_path.exists():
        raise RuntimeError(f"export_check: export() did not write {written_path}")

    device = getattr(wrapper, "device", "cpu")
    dummy = torch.zeros(1, 3, height, width, device=device)
    raw = getattr(wrapper, "model", wrapper)
    was_training = raw.training
    raw.eval()
    try:
        with torch.no_grad():
            torch_raw = raw(dummy)
    finally:
        raw.train(was_training)
    torch_out = _as_tensor_tuple(torch_raw)

    session = __import__("onnxruntime").InferenceSession(
        str(written_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    onnx_np = session.run(None, {input_name: dummy.detach().cpu().numpy()})
    onnx_out = tuple(torch.from_numpy(arr) for arr in onnx_np)

    if torch_out is None or not outputs_comparable(torch_out, onnx_out):
        logger.warning(
            "export_check: ONNX export succeeded (%s) but raw torch and "
            "ONNX output layouts differ; numeric compare skipped.",
            written_path,
        )
        return written_path

    assert_close(torch_out, onnx_out, rtol=rtol, atol=atol)
    logger.info("export_check: ONNX matches torch at %s", written_path)
    return written_path
