"""
OpenVINO export implementation.

Converts ONNX models to OpenVINO IR format with support for
FP32, FP16, and INT8 precision modes.
"""

from pathlib import Path
from typing import Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .calibration import CalibrationDataLoader


def check_openvino_available() -> None:
    """Check if OpenVINO is available and raise helpful error if not."""
    try:
        import openvino as ov
        _ = ov.__version__
    except ImportError:
        raise ImportError(
            "OpenVINO export requires the 'openvino' package.\n\n"
            "Installation:\n"
            "  pip install openvino\n\n"
            "Or install with the libreyolo extras:\n"
            "  pip install libreyolo[openvino]"
        )


def export_openvino(
    onnx_path: str,
    output_path: str,
    *,
    half: bool = True,
    int8: bool = False,
    calibration_data: Optional["CalibrationDataLoader"] = None,
    verbose: bool = False,
    metadata: Optional[dict] = None,
) -> str:
    """
    Export ONNX model to OpenVINO IR format.

    The output is a directory containing model.xml, model.bin, and
    optionally metadata.yaml.

    Args:
        onnx_path: Path to ONNX model file.
        output_path: Output directory for OpenVINO model.
        half: Compress weights to FP16 (default: True).
        int8: Enable INT8 quantization (default: False).
             Requires calibration_data.
        calibration_data: CalibrationDataLoader for INT8 quantization.
                         Required when int8=True.
        verbose: Enable verbose logging (default: False).
        metadata: Optional dict of model metadata to save as metadata.yaml.

    Returns:
        Path to exported OpenVINO model directory.

    Raises:
        ImportError: If OpenVINO is not installed.
        ValueError: If int8=True but calibration_data is not provided.
    """
    check_openvino_available()
    import openvino as ov

    # Validate INT8 requirements
    if int8 and calibration_data is None:
        raise ValueError(
            "INT8 quantization requires calibration data.\n"
            "Provide calibration_data or set int8=False."
        )

    # Convert ONNX to OpenVINO IR
    print(f"Converting ONNX to OpenVINO IR: {onnx_path}")
    ov_model = ov.convert_model(onnx_path)

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.xml"

    # Apply INT8 quantization
    if int8:
        ov_model = _quantize_int8(ov_model, calibration_data, verbose)
        print(f"Saving OpenVINO INT8 model: {model_path}")
        ov.save_model(ov_model, str(model_path))
    elif half:
        # FP16 weight compression via save_model's compress_to_fp16 flag
        print(f"Saving OpenVINO FP16 model: {model_path}")
        ov.save_model(ov_model, str(model_path), compress_to_fp16=True)
    else:
        print(f"Saving OpenVINO FP32 model: {model_path}")
        ov.save_model(ov_model, str(model_path))

    # Save metadata
    if metadata:
        _save_metadata(output_dir, metadata)

    print(f"OpenVINO export complete: {output_dir}")
    return str(output_dir)


def _quantize_int8(
    ov_model,
    calibration_data: "CalibrationDataLoader",
    verbose: bool = False,
):
    """
    Quantize OpenVINO model to INT8 using NNCF.

    Args:
        ov_model: OpenVINO model.
        calibration_data: Calibration data loader.
        verbose: Verbose logging.

    Returns:
        Quantized OpenVINO model.
    """
    try:
        import nncf
    except ImportError:
        raise ImportError(
            "INT8 quantization requires the 'nncf' package.\n"
            "Install with: pip install nncf"
        )

    print("Quantizing model to INT8 with NNCF...")

    def transform_fn(batch):
        return batch

    calibration_dataset = nncf.Dataset(calibration_data, transform_fn)

    quantized_model = nncf.quantize(
        ov_model,
        calibration_dataset,
        preset=nncf.QuantizationPreset.MIXED,
        subset_size=len(calibration_data) * calibration_data.batch,
    )

    print("INT8 quantization complete")
    return quantized_model


def _save_metadata(output_dir: Path, metadata: dict) -> None:
    """Save metadata.yaml for OpenVINO model."""
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    print(f"Saved metadata: {metadata_path}")
