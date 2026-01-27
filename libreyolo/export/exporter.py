"""
Unified model export with ONNX and TorchScript backends.
"""

import importlib.util
import json
import warnings
from pathlib import Path
from typing import Optional

import torch


class Exporter:
    """Export a LibreYOLO model to deployment formats (ONNX, TorchScript).

    Args:
        model: A LibreYOLOBase subclass instance (LIBREYOLOX, LIBREYOLO9, etc.).

    Example::

        from libreyolo import LIBREYOLOX
        from libreyolo.export import Exporter

        model = LIBREYOLOX("weights.pt", size="s")
        exporter = Exporter(model)
        path = exporter("onnx", simplify=True, dynamic=True)

        # Or use the model's export() facade directly:
        path = model.export(format="onnx")
    """

    FORMATS = {
        "onnx": {"suffix": ".onnx", "method": "_export_onnx"},
        "torchscript": {"suffix": ".torchscript", "method": "_export_torchscript"},
    }

    def __init__(self, model):
        self.model = model

    def __call__(
        self,
        format: str = "onnx",
        *,
        output_path: Optional[str] = None,
        imgsz: Optional[int] = None,
        opset: int = 13,
        simplify: bool = True,
        dynamic: bool = True,
        half: bool = False,
        batch: int = 1,
        device: Optional[str] = None,
    ) -> str:
        """Export the model to a deployment format.

        Args:
            format: Target format ("onnx", "torchscript").
            output_path: Output file path (auto-generated if None).
            imgsz: Input resolution (default: model's native size).
            opset: ONNX opset version (default: 13).
            simplify: Run ONNX graph simplification (default: True).
            dynamic: Enable dynamic axes (default: True).
            half: Export in FP16 (default: False).
            batch: Batch size for the static graph (default: 1).
            device: Device to trace on (default: model's current device).

        Returns:
            Path to the exported model file.
        """
        # --- validate format ---
        fmt = format.lower()
        if fmt not in self.FORMATS:
            valid = ", ".join(sorted(self.FORMATS))
            raise ValueError(
                f"Unsupported export format: {format!r}. Must be one of: {valid}"
            )
        fmt_info = self.FORMATS[fmt]

        # --- resolve parameters ---
        if imgsz is None:
            imgsz = self.model._get_input_size()

        if device is None:
            device = self.model.device
        else:
            device = torch.device(device)

        if output_path is None:
            model_name = self.model._get_model_name().lower()
            output_path = f"{model_name}_{self.model.size}{fmt_info['suffix']}"

        # --- prepare model ---
        nn_model = self.model.model
        original_training = nn_model.training
        nn_model.eval()

        original_device = next(nn_model.parameters()).device
        nn_model.to(device)

        # --- build dummy input ---
        dummy = torch.randn(batch, 3, imgsz, imgsz, device=device)

        if half:
            nn_model.half()
            dummy = dummy.half()

        # --- dispatch ---
        method = getattr(self, fmt_info["method"])
        try:
            result = method(
                nn_model,
                dummy,
                output_path=output_path,
                opset=opset,
                simplify=simplify,
                dynamic=dynamic,
                half=half,
            )
        finally:
            # restore model state
            nn_model.to(original_device)
            if half:
                nn_model.float()
            if original_training:
                nn_model.train()

        print(
            f"Export complete: {result}  "
            f"({self.model._get_model_name()} {self.model.size}, "
            f"format={fmt}, imgsz={imgsz})"
        )
        return result

    # ------------------------------------------------------------------
    # ONNX
    # ------------------------------------------------------------------

    def _export_onnx(
        self,
        nn_model,
        dummy,
        *,
        output_path: str,
        opset: int,
        simplify: bool,
        dynamic: bool,
        half: bool,
    ) -> str:
        if importlib.util.find_spec("onnx") is None:
            raise ImportError(
                "ONNX export requires the 'onnx' package. "
                "Install with: uv sync --extra onnx  or  pip install onnx"
            )

        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                "images": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch"},
            }

        torch.onnx.export(
            nn_model,
            dummy,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        if simplify:
            self._simplify_onnx(output_path)

        self._embed_onnx_metadata(output_path, dynamic=dynamic, half=half)

        return output_path

    @staticmethod
    def _simplify_onnx(path: str) -> None:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            model_proto = onnx.load(path)
            simplified, ok = onnx_simplify(model_proto)
            if ok:
                onnx.save(simplified, path)
        except ImportError:
            warnings.warn(
                "onnxsim is not installed — skipping ONNX graph simplification. "
                "Install with: pip install onnxsim",
                stacklevel=3,
            )
        except Exception as exc:
            warnings.warn(
                f"ONNX simplification failed (non-fatal): {exc}",
                stacklevel=3,
            )

    def _embed_onnx_metadata(self, path: str, *, dynamic: bool, half: bool) -> None:
        try:
            import onnx
        except ImportError:
            return

        model_proto = onnx.load(path)

        metadata = {
            "libreyolo_version": self._get_version(),
            "model_family": self.model._get_model_name(),
            "model_size": self.model.size,
            "nb_classes": str(self.model.nb_classes),
            "names": json.dumps(
                {str(k): v for k, v in self.model.names.items()}
            ),
            "imgsz": str(self.model._get_input_size()),
            "dynamic": str(dynamic),
            "half": str(half),
        }

        for key, value in metadata.items():
            entry = model_proto.metadata_props.add()
            entry.key = key
            entry.value = value

        onnx.save(model_proto, path)

    @staticmethod
    def _get_version() -> str:
        try:
            from importlib.metadata import version

            return version("libreyolo")
        except Exception:
            return "0.0.0.dev0"

    # ------------------------------------------------------------------
    # TorchScript
    # ------------------------------------------------------------------

    def _export_torchscript(
        self,
        nn_model,
        dummy,
        *,
        output_path: str,
        **_kwargs,
    ) -> str:
        traced = torch.jit.trace(nn_model, dummy)
        traced.save(output_path)
        return output_path
