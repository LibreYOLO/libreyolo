"""CoreML (Apple .mlpackage) export implementation.

Strategy: wrap the PyTorch model in a thin per-family preprocessing module so
the *traced* graph always accepts a canonical RGB float tensor in [0, 1].
That lets us configure ``ct.ImageType`` uniformly (RGB, scale=1/255, no bias)
regardless of the family's internal preprocessing convention.

Family conventions, mapped by the wrapper:
  * yolox            : BGR float in [0, 255]  (no normalization)
  * yolo9 / rtdetr   : RGB float in [0, 1]   (identity)
  * rfdetr           : RGB float, ImageNet (mean/std) normalized
"""

from __future__ import annotations

import json
from typing import Any

import torch
import torch.nn as nn

# ImageNet stats used by RF-DETR.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _YoloxPreprocess(nn.Module):
    """Map canonical RGB[0,1] input → BGR[0,255] expected by YOLOX."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Any:
        x = x * 255.0
        x = x[:, [2, 1, 0], :, :]
        return self.model(x)


class _RfdetrPreprocess(nn.Module):
    """Map canonical RGB[0,1] input → ImageNet-normalized RGB expected by RF-DETR."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def forward(self, x: torch.Tensor) -> Any:
        x = (x - self._mean) / self._std
        return self.model(x)


class _RtdetrOutputAdapter(nn.Module):
    """Flatten RT-DETR's dict output into traceable tensor outputs."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        if isinstance(out, dict):
            return out["pred_logits"], out["pred_boxes"]
        return out


def _wrap_for_family(nn_model: nn.Module, model_family: str | None) -> nn.Module:
    family = (model_family or "").lower()
    if family == "yolox":
        return _YoloxPreprocess(nn_model)
    if family == "rfdetr":
        return _RfdetrPreprocess(nn_model)
    if family == "rtdetr":
        return _RtdetrOutputAdapter(nn_model)
    # yolo9 and any others use canonical input directly.
    return nn_model


def _prepare_rtdetr_static_eval(nn_model: nn.Module, height: int, width: int) -> None:
    """Precompute RT-DETR eval tensors for the fixed CoreML export image size."""
    device = next(nn_model.parameters(), torch.empty(0)).device
    eval_spatial_size = (height, width)

    encoder = getattr(nn_model, "encoder", None)
    if encoder is not None and hasattr(encoder, "build_2d_sincos_position_embedding"):
        encoder.eval_spatial_size = eval_spatial_size
        for idx in getattr(encoder, "use_encoder_idx", []):
            stride = encoder.feat_strides[idx]
            pos_embed = encoder.build_2d_sincos_position_embedding(
                width // stride,
                height // stride,
                encoder.hidden_dim,
                encoder.pe_temperature,
            ).to(device)
            setattr(encoder, f"pos_embed{idx}", pos_embed)

    decoder = getattr(nn_model, "decoder", None)
    if decoder is not None and hasattr(decoder, "_generate_anchors"):
        decoder.eval_spatial_size = eval_spatial_size
        anchors, valid_mask = decoder._generate_anchors(device=device)
        if "anchors" in decoder._buffers:
            decoder._buffers["anchors"] = anchors
        else:
            decoder.register_buffer("anchors", anchors, persistent=False)
        if "valid_mask" in decoder._buffers:
            decoder._buffers["valid_mask"] = valid_mask
        else:
            decoder.register_buffer("valid_mask", valid_mask, persistent=False)


class _NMSOutputAdapter(nn.Module):
    """Map detector outputs to CoreML NMS inputs: confidence and cxcywh boxes."""

    def __init__(self, model: nn.Module, model_family: str | None):
        super().__init__()
        self.model = model
        self.model_family = (model_family or "").lower()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)

        if self.model_family == "yolox":
            # YOLOX export output: (B, N, 5 + C), cxcywh + objectness + class scores.
            confidence = out[..., 5:] * out[..., 4:5]
            coordinates = out[..., :4]
        elif self.model_family == "yolo9":
            # YOLO9 export output: (B, 4 + C, N), xyxy + class scores.
            pred = out.transpose(1, 2)
            xyxy = pred[..., :4]
            x1, y1, x2, y2 = xyxy.unbind(dim=-1)
            coordinates = torch.stack(
                ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1),
                dim=-1,
            )
            confidence = pred[..., 4:]
        elif self.model_family == "rtdetr":
            if not isinstance(out, (tuple, list)) or len(out) < 2:
                raise RuntimeError("RT-DETR CoreML NMS export expects two outputs")
            first, second = out[0], out[1]
            if first.shape[-1] == 4:
                coordinates, logits = first, second
            else:
                logits, coordinates = first, second
            confidence = torch.sigmoid(logits)
        else:
            raise NotImplementedError(
                f"nms=True is not supported for model family {self.model_family!r}"
            )

        # CoreML's feature-engineering NMS model expects 2D arrays.
        return confidence[0], coordinates[0]


def _stringify_metadata(metadata: dict) -> dict:
    """Convert metadata values to strings (CoreML user_defined_metadata requires str).

    Dict-typed values (e.g. ``names``) are JSON-encoded so they round-trip cleanly.
    """
    out: dict[str, str] = {}
    for k, v in metadata.items():
        if isinstance(v, dict):
            out[str(k)] = json.dumps(v)
        else:
            out[str(k)] = str(v)
    return out


def _to_compute_unit(compute_units: str):
    """Map a string compute_units value to a coremltools.ComputeUnit enum.

    Accepted: 'all', 'cpu_and_gpu', 'cpu_and_ne', 'cpu_only' (case-insensitive).
    """
    import coremltools as ct

    key = compute_units.lower()
    mapping = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }
    if key not in mapping:
        raise ValueError(
            f"Invalid compute_units {compute_units!r}. "
            f"Must be one of: {sorted(mapping)}"
        )
    return mapping[key]


def export_coreml(
    nn_model,
    dummy,
    *,
    output_path: str,
    precision: str = "fp32",
    compute_units: str = "all",
    nms: bool = False,
    metadata: dict | None = None,
    model_family: str | None = None,
) -> str:
    """Export a PyTorch model to CoreML (.mlpackage / ML Program format).

    The traced graph is built from a wrapped model that accepts canonical
    RGB float input in [0, 1]; the wrapper handles family-specific transforms
    (YOLOX BGR/0-255, RF-DETR ImageNet normalization).

    Args:
        nn_model: The PyTorch nn.Module to export. Must already be in eval/export mode.
        dummy: Reference input tensor — only its (B, C, H, W) shape is used.
        output_path: Destination .mlpackage path (a directory bundle).
        precision: 'fp32' or 'fp16'.
        compute_units: 'all' | 'cpu_and_gpu' | 'cpu_and_ne' | 'cpu_only'.
        nms: If True, embed Apple's NonMaximumSuppression as a CoreML pipeline.
            Not supported for RF-DETR.
        metadata: Dict of metadata to embed under user_defined_metadata.
        model_family: Family string (yolox | yolo9 | rtdetr | rfdetr) — selects
            the preprocess wrapper.

    Returns:
        ``output_path`` on success.
    """
    import coremltools as ct

    family = (model_family or "").lower()
    if nms and family == "rfdetr":
        raise NotImplementedError(
            "nms=True is not supported for RF-DETR; export with nms=False and run "
            "NMS in your application."
        )

    if family == "rtdetr":
        _prepare_rtdetr_static_eval(
            nn_model,
            height=int(dummy.shape[2]),
            width=int(dummy.shape[3]),
        )

    wrapped = _wrap_for_family(nn_model.eval(), model_family).eval()
    if nms:
        wrapped = _NMSOutputAdapter(wrapped, model_family).eval()

    # Always feed the wrapper canonical RGB float in [0, 1], on the model's device.
    canonical_dummy = torch.zeros(
        dummy.shape[0], 3, dummy.shape[2], dummy.shape[3],
        dtype=torch.float32, device=dummy.device,
    )
    if nms and canonical_dummy.shape[0] != 1:
        raise RuntimeError("CoreML embedded NMS export currently requires batch=1")
    traced = torch.jit.trace(wrapped, canonical_dummy)

    image_input = ct.ImageType(
        name="image",
        shape=tuple(canonical_dummy.shape),
        scale=1.0 / 255.0,
        bias=[0.0, 0.0, 0.0],
        color_layout=ct.colorlayout.RGB,
    )
    compute_precision = (
        ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    )

    convert_kwargs = {
        "inputs": [image_input],
        "convert_to": "mlprogram",
        "compute_precision": compute_precision,
        "minimum_deployment_target": ct.target.iOS15,
    }
    if nms:
        convert_kwargs["outputs"] = [
            ct.TensorType(name="confidence"),
            ct.TensorType(name="coordinates"),
        ]

    mlmodel = ct.convert(traced, **convert_kwargs)

    mlmodel.compute_unit = _to_compute_unit(compute_units)

    if nms:
        mlmodel = _wrap_with_nms(mlmodel, model_family=model_family)
        if metadata is None:
            metadata = {}
        metadata = {**metadata, "nms": True}

    if metadata:
        mlmodel.user_defined_metadata.update(_stringify_metadata(metadata))

    mlmodel.save(output_path)
    return output_path


def _wrap_with_nms(mlmodel: Any, *, model_family: str | None) -> Any:
    """Wrap a detector mlmodel in a Pipeline that embeds Apple's NMS layer.

    Output names: 'confidence' (N x nb_classes), 'coordinates' (N x 4 normalized xywh).
    """
    if (model_family or "").lower() == "rfdetr":
        raise NotImplementedError(
            "nms=True is not supported for RF-DETR; export with nms=False and run "
            "NMS in your application."
        )

    import coremltools as ct

    model_spec = mlmodel.get_spec()
    output_by_name = {out.name: out for out in model_spec.description.output}
    if {"confidence", "coordinates"} - output_by_name.keys():
        raise RuntimeError(
            "CoreML NMS wrapping requires converted outputs named "
            "'confidence' and 'coordinates'."
        )

    confidence_shape = _multiarray_shape(output_by_name["confidence"])
    coordinates_shape = _multiarray_shape(output_by_name["coordinates"])
    if len(confidence_shape) != 2 or coordinates_shape != [confidence_shape[0], 4]:
        raise RuntimeError(
            "CoreML NMS wrapping requires confidence shape (N, C) and "
            f"coordinates shape (N, 4); got {confidence_shape} and {coordinates_shape}."
        )

    nms_spec = ct.proto.Model_pb2.Model()
    nms_spec.specificationVersion = 5
    _add_multiarray_feature(
        nms_spec.description.input, "confidence", confidence_shape
    )
    _add_multiarray_feature(
        nms_spec.description.input, "coordinates", coordinates_shape
    )
    _add_multiarray_feature(
        nms_spec.description.output, "confidence", confidence_shape
    )
    _add_multiarray_feature(
        nms_spec.description.output, "coordinates", coordinates_shape
    )

    nms = nms_spec.nonMaximumSuppression
    nms.iouThreshold = 0.45
    nms.confidenceThreshold = 0.25
    nms.confidenceInputFeatureName = "confidence"
    nms.coordinatesInputFeatureName = "coordinates"
    nms.confidenceOutputFeatureName = "confidence"
    nms.coordinatesOutputFeatureName = "coordinates"
    nms.pickTop.perClass = False

    pipeline_spec = ct.proto.Model_pb2.Model()
    pipeline_spec.specificationVersion = max(
        model_spec.specificationVersion,
        nms_spec.specificationVersion,
    )
    pipeline_spec.pipeline
    pipeline_spec.description.input.extend(model_spec.description.input)
    pipeline_spec.description.output.extend(nms_spec.description.output)
    pipeline_spec.pipeline.models.add().CopyFrom(model_spec)
    pipeline_spec.pipeline.models.add().CopyFrom(nms_spec)

    return ct.models.MLModel(pipeline_spec, weights_dir=mlmodel.weights_dir)


def _multiarray_shape(feature: Any) -> list[int]:
    return [int(dim) for dim in feature.type.multiArrayType.shape]


def _add_multiarray_feature(features: Any, name: str, shape: list[int]) -> None:
    import coremltools as ct

    feature = features.add()
    feature.name = name
    multiarray = feature.type.multiArrayType
    multiarray.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32
    multiarray.shape.extend(shape)
