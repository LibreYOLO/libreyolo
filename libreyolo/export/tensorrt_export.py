"""
TensorRT export implementation.

Converts ONNX models to optimized TensorRT engines with support for
FP32, FP16, and INT8 precision modes.
"""

import warnings
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .calibration import CalibrationDataLoader
    from .config import TensorRTExportConfig


def check_tensorrt_available() -> None:
    """Check if TensorRT is available and raise helpful error if not."""
    try:
        import tensorrt as trt
        _ = trt.__version__
    except ImportError:
        raise ImportError(
            "TensorRT export requires the 'tensorrt' package.\n\n"
            "Installation options:\n"
            "  1. pip install tensorrt  (requires NVIDIA GPU + CUDA)\n"
            "  2. pip install nvidia-tensorrt  (alternative package name)\n\n"
            "Requirements:\n"
            "  - NVIDIA GPU with compute capability >= 5.0\n"
            "  - CUDA toolkit installed\n"
            "  - cuDNN installed\n\n"
            "For Jetson devices, TensorRT is pre-installed with JetPack."
        )


def export_tensorrt(
    onnx_path: str,
    output_path: str,
    *,
    half: bool = True,
    int8: bool = False,
    workspace: float = 4.0,
    calibration_data: Optional["CalibrationDataLoader"] = None,
    dynamic: bool = False,
    verbose: bool = False,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 8,
    hardware_compatibility: str = "none",
    device: int = 0,
    config: Optional[Union[str, Path, dict, "TensorRTExportConfig"]] = None,
) -> str:
    """
    Export ONNX model to TensorRT engine.

    The engine is optimized for the GPU it's built on. By default, engines are NOT
    portable between different GPU architectures (e.g., an engine built
    on RTX 4090 won't work on RTX 3080). Use hardware_compatibility to
    enable portability at the cost of some performance.

    Args:
        onnx_path: Path to ONNX model file.
        output_path: Output path for .engine file.
        half: Enable FP16 precision (default: True). Provides ~2x speedup
              on most GPUs with minimal accuracy loss.
        int8: Enable INT8 precision (default: False). Requires calibration_data.
              Provides additional speedup but may impact accuracy.
        workspace: GPU workspace size in GiB for kernel optimization (default: 4.0).
                  Larger values may find faster kernels but use more GPU memory
                  during build. Does not affect inference memory usage.
        calibration_data: CalibrationDataLoader for INT8 quantization.
                         Required when int8=True.
        dynamic: Enable dynamic batch size (default: False). When True, the
                engine supports batch sizes from min_batch to max_batch.
        verbose: Enable verbose TensorRT logging (default: False).
        min_batch: Minimum batch size for dynamic batching (default: 1).
        opt_batch: Optimal batch size for dynamic batching (default: 1).
                  TensorRT optimizes kernels for this batch size.
        max_batch: Maximum batch size for dynamic batching (default: 8).
        hardware_compatibility: Hardware compatibility level (default: "none").
            - "none": Optimize for current GPU only (fastest, not portable)
            - "ampere_plus": Works on Ampere (RTX 30xx, A100) and newer GPUs
            - "same_compute_capability": Works on GPUs with same SM version
        device: GPU device ID for multi-GPU systems (default: 0).
        config: Optional TensorRTExportConfig or path to YAML config file.
               If provided, overrides individual parameters.

    Returns:
        Path to exported .engine file.

    Raises:
        ImportError: If TensorRT is not installed.
        ValueError: If int8=True but calibration_data is not provided.
        RuntimeError: If engine building fails.

    Example::

        # FP16 export (recommended for most use cases)
        export_tensorrt("model.onnx", "model.engine", half=True)

        # INT8 export with calibration
        from libreyolo.export.calibration import get_calibration_dataloader
        calib = get_calibration_dataloader("coco8.yaml", imgsz=640)
        export_tensorrt("model.onnx", "model_int8.engine", int8=True,
                       calibration_data=calib)

        # Export with config file
        export_tensorrt("model.onnx", "model.engine",
                       config="tensorrt_default.yaml")
    """
    # Load config if provided
    if config is not None:
        from .config import load_export_config
        cfg = load_export_config(config)
        # Config values override function parameters
        half = cfg.half
        int8 = cfg.int8
        workspace = cfg.workspace
        verbose = cfg.verbose
        hardware_compatibility = cfg.hardware_compatibility
        device = cfg.device
        if cfg.dynamic.enabled:
            dynamic = True
            min_batch = cfg.dynamic.min_batch
            opt_batch = cfg.dynamic.opt_batch
            max_batch = cfg.dynamic.max_batch
    check_tensorrt_available()
    import tensorrt as trt

    # Set GPU device for multi-GPU systems
    if device != 0:
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            print(f"Using GPU device: {device} ({torch.cuda.get_device_name(device)})")

    # Validate parameters
    if int8 and calibration_data is None:
        raise ValueError(
            "INT8 quantization requires calibration data.\n"
            "Provide calibration_data parameter or use data='coco8.yaml' "
            "in the export() call."
        )

    if half and int8:
        warnings.warn(
            "Both half=True and int8=True specified. Using INT8 precision "
            "(INT8 includes FP16 fallback for unsupported layers)."
        )

    # Verify ONNX file exists
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    # Set up TensorRT logger
    log_level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(log_level)

    # Create builder and network
    builder = trt.Builder(logger)

    # Use explicit batch mode (required for modern TensorRT)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    # Parse ONNX model
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        onnx_data = f.read()

    if not parser.parse(onnx_data):
        error_msgs = []
        for i in range(parser.num_errors):
            error_msgs.append(str(parser.get_error(i)))
        raise RuntimeError(
            f"Failed to parse ONNX model:\n" + "\n".join(error_msgs)
        )

    # Log network info
    print(f"Network: {network.num_inputs} inputs, {network.num_outputs} outputs, "
          f"{network.num_layers} layers")

    # Create builder configuration
    builder_config = builder.create_builder_config()

    # Set workspace memory (convert GiB to bytes)
    workspace_bytes = int(workspace * (1 << 30))
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    print(f"Workspace: {workspace} GiB")

    # Set hardware compatibility level
    if hardware_compatibility != "none":
        try:
            compat_level = {
                "ampere_plus": trt.HardwareCompatibilityLevel.AMPERE_PLUS,
                "same_compute_capability": trt.HardwareCompatibilityLevel.NONE,  # TRT 8.x fallback
            }.get(hardware_compatibility)

            if compat_level is not None:
                builder_config.hardware_compatibility_level = compat_level
                print(f"Hardware compatibility: {hardware_compatibility}")
            else:
                warnings.warn(
                    f"Unknown hardware_compatibility '{hardware_compatibility}'. "
                    f"Using default (none)."
                )
        except AttributeError:
            # Older TensorRT versions may not have this attribute
            warnings.warn(
                f"TensorRT version does not support hardware_compatibility_level. "
                f"Engine will only work on current GPU architecture."
            )

    # Configure precision
    precision_str = "FP32"

    if half or int8:
        if builder.platform_has_fast_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
            precision_str = "FP16"
        else:
            warnings.warn(
                "GPU does not support fast FP16. Falling back to FP32."
            )

    if int8:
        if builder.platform_has_fast_int8:
            builder_config.set_flag(trt.BuilderFlag.INT8)
            precision_str = "INT8"

            # Set up INT8 calibrator (use runtime-created class with proper inheritance)
            CalibratorClass = get_calibrator_class()
            calibrator = CalibratorClass(
                calibration_data,
                cache_file=str(Path(output_path).with_suffix(".cache")),
            )
            builder_config.int8_calibrator = calibrator
            print(f"INT8 calibration: {len(calibration_data)} batches, "
                  f"batch size {calibration_data.batch}")
        else:
            warnings.warn(
                "GPU does not support fast INT8. Falling back to FP16."
            )

    print(f"Precision: {precision_str}")

    # Handle dynamic shapes - only if ONNX was exported with dynamic batch
    # Check if input has dynamic dimension (indicated by -1 or symbolic dim)
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape

    # Check if batch dimension is dynamic (-1 means dynamic in ONNX/TRT)
    has_dynamic_batch = input_shape[0] == -1

    if dynamic and has_dynamic_batch:
        profile = builder.create_optimization_profile()

        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_name = input_tensor.name
            input_shape = input_tensor.shape

            # Shape is (batch, channels, height, width)
            # Batch dimension is -1 for dynamic
            _, c, h, w = input_shape

            min_shape = (min_batch, c, h, w)
            opt_shape = (opt_batch, c, h, w)
            max_shape = (max_batch, c, h, w)

            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            print(f"Dynamic input '{input_name}': "
                  f"min={min_shape}, opt={opt_shape}, max={max_shape}")

        builder_config.add_optimization_profile(profile)
    elif dynamic and not has_dynamic_batch:
        # ONNX has static batch, so we use static shape optimization
        print(f"Note: ONNX has static batch size ({input_shape[0]}), using static optimization")

    # Build the engine
    print("Building TensorRT engine... (this may take several minutes)")

    serialized_engine = builder.build_serialized_network(network, builder_config)

    if serialized_engine is None:
        raise RuntimeError(
            "TensorRT engine build failed. Common causes:\n"
            "  - Unsupported ONNX operations\n"
            "  - Insufficient GPU memory\n"
            "  - CUDA/cuDNN version mismatch\n"
            "Try running with verbose=True for detailed error messages."
        )

    # Save engine
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    # Report engine size
    engine_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Engine saved: {output_path} ({engine_size_mb:.1f} MB)")

    return str(output_path)


class TensorRTCalibrator:
    """
    INT8 entropy calibrator for TensorRT.

    Feeds calibration images to TensorRT to compute optimal quantization
    scales for each layer. Uses entropy calibration (IInt8EntropyCalibrator2)
    which generally provides the best accuracy for detection models.

    The calibration cache is saved to disk so subsequent builds can skip
    the calibration step.
    """

    def __init__(
        self,
        data_loader: "CalibrationDataLoader",
        cache_file: str = "calibration.cache",
    ):
        """
        Initialize the calibrator.

        Args:
            data_loader: CalibrationDataLoader providing calibration batches.
            cache_file: Path to save/load calibration cache.
        """
        import tensorrt as trt

        # Inherit from the correct calibrator class
        self._calibrator_base = trt.IInt8EntropyCalibrator2

        self.data_loader = data_loader
        self.cache_file = Path(cache_file)
        self.batch_iter = None
        self.current_batch = None

        # Allocate device memory for calibration batches
        self._device_input = None
        self._batch_size = data_loader.batch

    def _ensure_cuda_memory(self, batch: np.ndarray) -> int:
        """Allocate CUDA memory if needed and copy batch to device."""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401 - initializes CUDA context
        except ImportError:
            raise ImportError(
                "INT8 calibration requires pycuda for GPU memory management.\n"
                "Install with: pip install pycuda"
            )

        # Ensure batch is contiguous float32
        batch = np.ascontiguousarray(batch, dtype=np.float32)

        # Allocate device memory if needed
        if self._device_input is None:
            self._device_input = cuda.mem_alloc(batch.nbytes)

        # Copy to device
        cuda.memcpy_htod(self._device_input, batch)

        return int(self._device_input)

    def get_batch_size(self) -> int:
        """Return the calibration batch size."""
        return self._batch_size

    def get_batch(self, names) -> list:
        """
        Get the next batch of calibration data.

        Args:
            names: List of input tensor names (provided by TensorRT).

        Returns:
            List of device pointers to input data, or None if no more batches.
        """
        if self.batch_iter is None:
            self.batch_iter = iter(self.data_loader)

        try:
            batch = next(self.batch_iter)
            device_ptr = self._ensure_cuda_memory(batch)
            return [device_ptr]
        except StopIteration:
            return None

    def read_calibration_cache(self) -> bytes:
        """
        Read calibration cache from disk if available.

        Returns:
            Cached calibration data, or None if no cache exists.
        """
        if self.cache_file.exists():
            print(f"Loading calibration cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """
        Write calibration cache to disk.

        Args:
            cache: Calibration data to cache.
        """
        print(f"Saving calibration cache: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# Make TensorRTCalibrator inherit from the TensorRT base class at runtime
# This avoids import errors when TensorRT is not installed
def _create_calibrator_class():
    """Create calibrator class that inherits from TensorRT base."""
    try:
        import tensorrt as trt

        class _TensorRTCalibratorImpl(trt.IInt8EntropyCalibrator2):
            """Runtime-created calibrator with proper inheritance."""

            def __init__(self, data_loader, cache_file="calibration.cache"):
                super().__init__()
                self.data_loader = data_loader
                self.cache_file = Path(cache_file)
                self.batch_iter = None
                self._device_input = None
                self._batch_size = data_loader.batch

            def get_batch_size(self):
                return self._batch_size

            def get_batch(self, names):
                if self.batch_iter is None:
                    self.batch_iter = iter(self.data_loader)

                try:
                    batch = next(self.batch_iter)
                    device_ptr = self._ensure_cuda_memory(batch)
                    return [device_ptr]
                except StopIteration:
                    return None

            def _ensure_cuda_memory(self, batch):
                batch = np.ascontiguousarray(batch, dtype=np.float32)

                # Try cuda-python/cuda-bindings first (newer, better maintained)
                try:
                    from cuda.bindings import runtime as cudart
                    if self._device_input is None:
                        err, self._device_input = cudart.cudaMalloc(batch.nbytes)
                        if err != cudart.cudaError_t.cudaSuccess:
                            raise RuntimeError(f"cudaMalloc failed: {err}")
                    err, = cudart.cudaMemcpy(
                        self._device_input, batch.ctypes.data,
                        batch.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
                    )
                    if err != cudart.cudaError_t.cudaSuccess:
                        raise RuntimeError(f"cudaMemcpy failed: {err}")
                    return int(self._device_input)
                except ImportError:
                    pass

                # Fall back to pycuda
                try:
                    import pycuda.driver as cuda
                    import pycuda.autoinit  # noqa: F401

                    if self._device_input is None:
                        self._device_input = cuda.mem_alloc(batch.nbytes)

                    cuda.memcpy_htod(self._device_input, batch)
                    return int(self._device_input)
                except ImportError:
                    raise ImportError(
                        "INT8 calibration requires cuda-python or pycuda.\n"
                        "Install with: pip install cuda-python\n"
                        "Or: pip install pycuda (requires python3-dev)"
                    )

            def read_calibration_cache(self):
                if self.cache_file.exists():
                    print(f"Loading calibration cache: {self.cache_file}")
                    with open(self.cache_file, "rb") as f:
                        return f.read()
                return None

            def write_calibration_cache(self, cache):
                print(f"Saving calibration cache: {self.cache_file}")
                with open(self.cache_file, "wb") as f:
                    f.write(cache)

        return _TensorRTCalibratorImpl

    except ImportError:
        # Return dummy class if TensorRT not available
        return TensorRTCalibrator


def get_calibrator_class():
    """Get the appropriate calibrator class based on TensorRT availability."""
    return _create_calibrator_class()
