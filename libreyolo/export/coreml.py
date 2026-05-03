"""CoreML (Apple .mlpackage) export implementation.

Direct PyTorch -> coremltools.convert path. Produces ML Program format
(.mlpackage), targeting iOS 15 / macOS 12 minimum.
"""

from __future__ import annotations

import json
from typing import Any


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

    Args:
        nn_model: The PyTorch nn.Module to export. Must already be in eval/export mode.
        dummy: Dummy input tensor of shape (B, 3, H, W) for tracing.
        output_path: Destination .mlpackage path (a directory bundle).
        precision: 'fp32' or 'fp16'. Maps to ct.precision.FLOAT32/FLOAT16.
        compute_units: 'all' | 'cpu_and_gpu' | 'cpu_and_ne' | 'cpu_only'.
        nms: If True, embed Apple's NonMaximumSuppression as a CoreML pipeline.
            Not supported for RF-DETR (raises NotImplementedError).
        metadata: Dict of metadata to embed under user_defined_metadata.
        model_family: Model family string (e.g. 'yolox', 'yolo9', 'rfdetr') used
            to gate features (NMS pipeline shape).

    Returns:
        ``output_path`` on success.
    """
    import coremltools as ct
    import torch

    traced = torch.jit.trace(nn_model, dummy)

    image_input = ct.ImageType(
        name="image",
        shape=tuple(dummy.shape),
        scale=1.0 / 255.0,
        bias=[0.0, 0.0, 0.0],
    )
    compute_precision = (
        ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    )

    mlmodel = ct.convert(
        traced,
        inputs=[image_input],
        convert_to="mlprogram",
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.iOS15,
    )

    mlmodel.compute_unit = _to_compute_unit(compute_units)

    if nms:
        mlmodel = _wrap_with_nms(mlmodel, model_family=model_family)

    if metadata:
        mlmodel.user_defined_metadata.update(_stringify_metadata(metadata))

    mlmodel.save(output_path)
    return output_path


def _wrap_with_nms(mlmodel: Any, *, model_family: str | None) -> Any:
    """Wrap a detector mlmodel in a Pipeline that embeds Apple's NMS layer.

    Output names: 'confidence' (N x nb_classes), 'coordinates' (N x 4 normalized xywh).
    """
    if model_family == "rfdetr":
        raise NotImplementedError(
            "nms=True is not supported for RF-DETR; export with nms=False and run "
            "NMS in your application."
        )

    import coremltools as ct
    from coremltools.models import pipeline as ct_pipeline

    # Build an NMS model spec. Defaults match LibreYOLO's runtime defaults.
    nms_spec = ct.proto.Model_pb2.Model()
    nms_spec.specificationVersion = 5
    nms = nms_spec.nonMaximumSuppression
    nms.iouThreshold = 0.45
    nms.confidenceThreshold = 0.25
    nms.confidenceInputFeatureName = "confidence"
    nms.coordinatesInputFeatureName = "coordinates"
    nms.confidenceOutputFeatureName = "confidence"
    nms.coordinatesOutputFeatureName = "coordinates"
    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"

    nms_model = ct.models.MLModel(nms_spec)

    pipeline = ct_pipeline.Pipeline(
        input_features=[("image", None)],
        output_features=[("confidence", None), ("coordinates", None)],
    )
    pipeline.add_model(mlmodel)
    pipeline.add_model(nms_model)
    # Carry compute_unit forward
    pipeline.spec.compute_unit = mlmodel.compute_unit
    return ct.models.MLModel(pipeline.spec)