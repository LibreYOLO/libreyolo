"""Opaque ONNX runtime container for the official OpenCV Zoo LaMa graph.

The network architecture is intentionally not reimplemented here. The exact
published ONNX bytes live in a persistent uint8 buffer so the canonical
LibreYOLO checkpoint remains one safe, state-dict-compatible ``.pt`` file.
"""

from __future__ import annotations

import hashlib
import re
import threading
from typing import Any

import numpy as np
import torch
import torch.nn as nn


OFFICIAL_ONNX_FILENAME = "inpainting_lama_2025jan.onnx"
OFFICIAL_ONNX_SIZE_BYTES = 92_591_623
OFFICIAL_ONNX_SHA256 = (
    "7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2"
)
MINIMUM_ONNXRUNTIME_VERSION = (1, 18)
ONNX_INPUT_SIZE = 512


def onnx_graph_sha256(graph: torch.Tensor) -> str:
    """Hash one contiguous CPU uint8 graph tensor without serializing it again."""

    if not isinstance(graph, torch.Tensor):
        raise TypeError("LaMa ONNX payload must be a torch.Tensor.")
    array = graph.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def is_official_onnx_graph(graph: Any) -> bool:
    """Return whether a value has the official artifact's tensor envelope."""

    return (
        isinstance(graph, torch.Tensor)
        and graph.dtype == torch.uint8
        and graph.ndim == 1
        and graph.numel() == OFFICIAL_ONNX_SIZE_BYTES
    )


def validate_onnx_graph_tensor(graph: Any) -> str:
    """Validate the exact pinned ONNX bytes and return their SHA-256."""

    if not is_official_onnx_graph(graph):
        shape = tuple(graph.shape) if isinstance(graph, torch.Tensor) else None
        dtype = graph.dtype if isinstance(graph, torch.Tensor) else None
        raise ValueError(
            "LibreLaMa requires the pinned OpenCV Zoo ONNX payload as a flat "
            f"uint8 tensor of {OFFICIAL_ONNX_SIZE_BYTES} bytes; got "
            f"shape={shape}, dtype={dtype}."
        )
    digest = onnx_graph_sha256(graph)
    if digest != OFFICIAL_ONNX_SHA256:
        raise ValueError(
            "LibreLaMa embedded ONNX SHA-256 mismatch: expected "
            f"{OFFICIAL_ONNX_SHA256}, got {digest}."
        )
    return digest


def _major_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"\s*(\d+)\.(\d+)", str(version))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


class OpaqueLaMaONNX(nn.Module):
    """Execute the embedded, immutable LaMa ONNX graph with ONNX Runtime."""

    def __init__(self, graph: torch.Tensor | None = None) -> None:
        super().__init__()
        if graph is None:
            graph = torch.empty(0, dtype=torch.uint8)
        graph = graph.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        self.register_buffer("onnx_graph", graph, persistent=True)
        self._session_lock = threading.Lock()
        self._sessions: dict[str, Any] = {}

    def _apply(self, fn, recurse: bool = True):
        """Apply module transforms while keeping serialized graph bytes on CPU.

        Moving a 92 MB byte buffer to CUDA would waste device memory only to
        copy it back when constructing the runtime session. There are no native
        PyTorch parameters in this opaque module, so retaining this one buffer
        on CPU is the correct device behavior.
        """

        graph = self._buffers.pop("onnx_graph")
        try:
            super()._apply(fn, recurse=recurse)
        finally:
            self._buffers["onnx_graph"] = graph
        return self

    def allocate_graph_buffer(self, num_bytes: int) -> None:
        """Resize the load target before strict state-dict loading."""

        self.onnx_graph = torch.empty(int(num_bytes), dtype=torch.uint8, device="cpu")
        self.clear_session()

    def clear_session(self) -> None:
        with self._session_lock:
            self._sessions.clear()

    def _get_session(self, input_device: torch.device | str):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "LibreLaMa requires ONNX Runtime >=1.18 to execute its embedded "
                "opset-21 graph. Install it with: "
                'pip install "libreyolo[onnx]"'
            ) from exc

        parsed_version = _major_minor(getattr(ort, "__version__", ""))
        if parsed_version is None or parsed_version < MINIMUM_ONNXRUNTIME_VERSION:
            raise RuntimeError(
                "LibreLaMa requires ONNX Runtime >=1.18 for opset 21; found "
                f"{getattr(ort, '__version__', 'unknown')}."
            )

        device = torch.device(input_device)
        available = set(ort.get_available_providers())
        use_cuda = device.type == "cuda" and "CUDAExecutionProvider" in available
        if use_cuda:
            device_index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            provider_key = f"cuda:{device_index}"
            providers: list[str | tuple[str, dict[str, int]]] = [
                ("CUDAExecutionProvider", {"device_id": device_index}),
                "CPUExecutionProvider",
            ]
        else:
            provider_key = "cpu"
            providers = ["CPUExecutionProvider"]

        session = self._sessions.get(provider_key)
        if session is not None:
            return session

        with self._session_lock:
            session = self._sessions.get(provider_key)
            if session is not None:
                return session

            validate_onnx_graph_tensor(self.onnx_graph)
            options = ort.SessionOptions()
            options.log_severity_level = 3
            serialized = self.onnx_graph.numpy().tobytes()
            session = ort.InferenceSession(
                serialized,
                sess_options=options,
                providers=providers,
            )
            input_names = [value.name for value in session.get_inputs()]
            output_names = [value.name for value in session.get_outputs()]
            if input_names != ["image", "mask"] or output_names != ["output"]:
                raise RuntimeError(
                    "Unexpected LaMa ONNX I/O contract: expected inputs "
                    "['image', 'mask'] and output ['output'], got "
                    f"inputs={input_names}, outputs={output_names}."
                )
            self._sessions[provider_key] = session
            return session

    def forward(self, guided_input: torch.Tensor) -> torch.Tensor:
        """Run ``[B, BGR, mask]`` input and return BGR values in ``[0, 255]``."""

        if guided_input.ndim != 4 or guided_input.shape[1:] != (
            4,
            ONNX_INPUT_SIZE,
            ONNX_INPUT_SIZE,
        ):
            raise ValueError(
                "LibreLaMa runtime expects [B, 4, 512, 512] containing BGR "
                f"image channels and one binary mask; got {tuple(guided_input.shape)}."
            )
        device = guided_input.device
        array = guided_input.detach().float().cpu().contiguous().numpy()
        session = self._get_session(device)
        output = session.run(
            ["output"],
            {
                "image": np.ascontiguousarray(array[:, :3]),
                "mask": np.ascontiguousarray(array[:, 3:4]),
            },
        )[0]
        return torch.from_numpy(np.asarray(output)).to(device=device)


__all__ = [
    "MINIMUM_ONNXRUNTIME_VERSION",
    "OFFICIAL_ONNX_FILENAME",
    "OFFICIAL_ONNX_SHA256",
    "OFFICIAL_ONNX_SIZE_BYTES",
    "ONNX_INPUT_SIZE",
    "OpaqueLaMaONNX",
    "is_official_onnx_graph",
    "onnx_graph_sha256",
    "validate_onnx_graph_tensor",
]
