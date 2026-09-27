"""Task-aware export graphs for LibrePE.

Three separate wrappers, because PE has three distinct exportable behaviors:

* **image embed** - 4D image input to ``[B, D]`` unit rows, class-independent.
* **frozen-class classify** - 4D image input to ``[B, K]`` logits with the
  current ``set_classes`` text matrix baked into a final linear projection, so
  the graph carries neither the text tower nor a tokenizer.
* **video embed** - fixed-``F`` 5D ``(B, F, C, H, W)`` input to ``[B, D]`` unit
  rows, pooling frames inside the graph.

  This graph is **direct-runtime only**. LibreYOLO's exported-backend
  preprocessing builds 4D image tensors, so there is no shared 5D video-input
  contract to drive it through ``LibreYOLO(<artifact>)`` yet. The artifact
  records ``input_kind="video"`` and the backends refuse to load it with an
  actionable message rather than failing on an opaque input-rank error; feed it
  with ``onnxruntime`` / ``torch.jit`` directly. Native parity for the graph is
  still verified (see ``tests/unit/test_pe_export.py``).

Text tokenization is never exported. The exported classify graph is fixed to the
labels and resolution present at export time.

PE has no ``logit_bias`` (that is a SigLIP concept), so the classify head is
``logit_scale.exp() * L2norm(image) @ text.T``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PEImageEmbedder(nn.Module):
    """Image tower -> L2-normalized ``[B, D]`` rows."""

    def __init__(self, visual: nn.Module) -> None:
        super().__init__()
        self.visual = visual

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.visual(images), dim=-1)


class _FrozenPEClassifier(nn.Module):
    """Image tower + baked text-embedding head -> ``[B, K]`` logits."""

    def __init__(self, visual: nn.Module, weight: torch.Tensor) -> None:
        super().__init__()
        self.visual = visual
        # weight = logit_scale.exp() * text_embeds, shape [K, D].
        self.register_buffer("weight", weight)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = F.normalize(self.visual(images), dim=-1)
        return feats @ self.weight.t()


class _PEVideoEmbedder(nn.Module):
    """Fixed-frame clip tower -> L2-normalized ``[B, D]`` rows.

    Mirrors :meth:`LibrePEModel.encode_video`: flatten ``(B, F)`` into the batch,
    encode every frame independently, average over ``F``, normalize once.
    """

    def __init__(self, visual: nn.Module) -> None:
        super().__init__()
        self.visual = visual

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        b, f = clips.shape[0], clips.shape[1]
        frames = clips.reshape(b * f, clips.shape[2], clips.shape[3], clips.shape[4])
        feats = self.visual(frames).reshape(b, f, -1)
        return F.normalize(feats.mean(dim=1), dim=-1)


def _prepare(model, imgsz: Optional[int]):
    res = int(imgsz or model.input_size)
    device = next(model.model.visual.parameters()).device
    visual = model.model.visual.to("cpu").eval()
    return res, device, visual


def _classifier_weight(model) -> torch.Tensor:
    if model._text_embeds is None:
        raise RuntimeError("No classes set; call set_classes() before export().")
    scale = float(model.model.logit_scale.exp().detach().cpu())
    return scale * model._text_embeds.detach().to("cpu", torch.float32)


def _default_name(model, suffix: str, ext: str) -> str:
    return f"{model.FILENAME_PREFIX}{model.size}-{suffix}.{ext}"


def build_metadata(model, kind: str, frames: int) -> dict:
    """Metadata describing an exported PE graph.

    Written into ONNX ``metadata_props`` and the TorchScript
    ``libreyolo_metadata.json`` extra file, so a reloaded artifact knows its
    family, task, resolution and -- for clip graphs -- its fixed frame count,
    tensor layout and pooling rule, instead of falling back to detection
    defaults.
    """
    cfg = model.model.cfg
    task = "classify" if kind == "classify" else "embed"
    metadata: dict[str, object] = {
        "model_family": model.FAMILY,
        "model_size": model.size,
        "size": model.size,
        "task": task,
        "default_task": model.DEFAULT_TASK,
        "supported_tasks": list(model.SUPPORTED_TASKS),
        "imgsz": cfg.image_size,
        "input_size": cfg.image_size,
        "embedding_dim": cfg.projection_dim,
        "pixel_mean": [0.5, 0.5, 0.5],
        "pixel_std": [0.5, 0.5, 0.5],
        "input_kind": "video" if kind == "video" else "image",
    }
    if kind == "video":
        metadata.update(
            {
                "frames": int(frames),
                "input_layout": "BFCHW",
                "video_pool": "mean_frame_embeddings",
                "video_sampling": (
                    "uniform over the finite source, endpoints included; the "
                    "last frame repeats only when decoding yields fewer frames "
                    "than requested"
                ),
                "dynamic_frames": False,
            }
        )
    if kind == "classify":
        metadata["names"] = [model.names[i] for i in sorted(model.names)]
        metadata["nc"] = len(model.names)
    return metadata


def _stringify(metadata: dict) -> dict:
    """ONNX metadata_props values must be strings."""
    import json

    return {
        key: value if isinstance(value, str) else json.dumps(value)
        for key, value in metadata.items()
    }


def build_export_module(model, kind: str, frames: int = 8):
    """Build the CPU-resident export graph for ``kind``.

    Returns ``(module, dummy_input, input_names, output_names, original_device)``.
    The caller must move ``model.model.visual`` back to ``original_device``.
    """
    res, device, visual = _prepare(model, None)
    if kind == "embed":
        module = _PEImageEmbedder(visual).eval()
        dummy = torch.zeros(1, 3, res, res)
        return module, dummy, ["images"], ["embeddings"], device
    if kind == "classify":
        module = _FrozenPEClassifier(visual, _classifier_weight(model)).eval()
        dummy = torch.zeros(1, 3, res, res)
        return module, dummy, ["images"], ["logits"], device
    if kind == "video":
        module = _PEVideoEmbedder(visual).eval()
        dummy = torch.zeros(1, int(frames), 3, res, res)
        return module, dummy, ["clip"], ["embeddings"], device
    raise ValueError(f"Unknown export kind {kind!r}.")


def export_onnx(
    model,
    kind: str,
    imgsz: Optional[int] = None,
    opset: int = 17,
    output: Optional[str] = None,
    dynamic_batch: bool = True,
    frames: int = 8,
) -> str:
    """Export one PE graph to ONNX.

    Dynamic batch is enabled by default. The frame count of a video graph is
    **static**: a dynamic ``F`` is not advertised because it has not been parity
    tested in the target runtimes.
    """
    if opset < 14:
        raise ValueError("LibrePE ONNX export needs opset >= 14 (ViT attention).")

    module, dummy, input_names, output_names, device = build_export_module(
        model, kind, frames=frames
    )
    suffix = {"embed": "embed", "classify": "cls", "video": "vembed"}[kind]
    output = str(output or _default_name(model, suffix, "onnx"))
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {input_names[0]: {0: "batch"}, output_names[0]: {0: "batch"}}

    try:
        with torch.no_grad():
            torch.onnx.export(
                module,
                dummy,
                output,
                input_names=input_names,
                output_names=output_names,
                opset_version=opset,
                dynamic_axes=dynamic_axes,
                dynamo=False,
            )
    finally:
        model.model.visual.to(device)

    from ...export.onnx import embed_onnx_metadata

    embed_onnx_metadata(output, _stringify(build_metadata(model, kind, frames)))
    return output


def export_torchscript(
    model,
    kind: str,
    imgsz: Optional[int] = None,
    output: Optional[str] = None,
    frames: int = 8,
) -> str:
    """Export one PE graph to TorchScript via tracing."""
    module, dummy, _, _, device = build_export_module(model, kind, frames=frames)
    suffix = {"embed": "embed", "classify": "cls", "video": "vembed"}[kind]
    output = str(output or _default_name(model, suffix, "torchscript"))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    try:
        import json

        with torch.no_grad():
            traced = torch.jit.trace(module, dummy, strict=False)
            extra_files = {
                "libreyolo_metadata.json": json.dumps(
                    build_metadata(model, kind, frames)
                )
            }
            torch.jit.save(traced, output, _extra_files=extra_files)
    finally:
        model.model.visual.to(device)
    return output
