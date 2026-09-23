"""HTTP client backend for NVIDIA Triton Inference Server."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np

from ..tasks import resolve_task
from ..utils.serialization import warn_on_metadata_schema_version
from .base import BaseBackend, ImageSize, classify_eval_kwargs
from .metadata import ExportMetadataError, parse_export_metadata

logger = logging.getLogger(__name__)

TRITON_METADATA_PARAMETER = "libreyolo_metadata"

_ONNX_TO_TRITON_CONFIG_DTYPE = {
    "BOOL": "BOOL",
    "UINT8": "UINT8",
    "UINT16": "UINT16",
    "UINT32": "UINT32",
    "UINT64": "UINT64",
    "INT8": "INT8",
    "INT16": "INT16",
    "INT32": "INT32",
    "INT64": "INT64",
    "FLOAT16": "FP16",
    "FLOAT": "FP32",
    "DOUBLE": "FP64",
    "STRING": "STRING",
}


class TritonBackendError(RuntimeError):
    """Raised when a Triton server or model cannot satisfy the backend contract."""


@dataclass(frozen=True)
class TritonModelURL:
    """Parsed LibreYOLO Triton model URL."""

    scheme: str
    server_url: str
    model_name: str
    model_version: str


def is_triton_model_url(value: Any) -> bool:
    """Return whether *value* uses LibreYOLO's Triton HTTP URL form."""
    if not isinstance(value, str):
        return False
    return urlsplit(value).scheme.lower() in {"http", "https"}


def parse_triton_model_url(url: str) -> TritonModelURL:
    """Parse ``http(s)://host:port/model[/version]``.

    A version is an optional positive decimal path segment. When it is omitted,
    Triton's model-version policy selects the served version.
    """
    if not isinstance(url, str):
        raise TypeError("Triton model URL must be a string.")
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("Triton model URL must use http:// or https://.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Triton v1 URLs do not support embedded credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("Triton model URL must not include a query or fragment.")
    if not parsed.hostname:
        raise ValueError("Triton model URL must include a host.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Triton model URL has an invalid port: {url!r}.") from exc
    if port is None:
        raise ValueError("Triton model URL must include an explicit port.")

    raw_segments = parsed.path.split("/")[1:]
    if len(raw_segments) not in {1, 2} or any(not segment for segment in raw_segments):
        raise ValueError("Triton model URL path must be /model or /model/version.")
    segments = [unquote(segment) for segment in raw_segments]
    if any("/" in segment or "\\" in segment for segment in segments):
        raise ValueError("Triton model URL path segments must not contain slashes.")
    model_name = segments[0]
    if not model_name.strip() or model_name != model_name.strip():
        raise ValueError(
            "Triton model name must be non-empty and contain no edge spaces."
        )

    model_version = segments[1] if len(segments) == 2 else ""
    if model_version and (not model_version.isdecimal() or int(model_version) <= 0):
        raise ValueError("Triton model version must be a positive decimal integer.")

    host = parsed.hostname
    formatted_host = f"[{host}]" if ":" in host else host
    return TritonModelURL(
        scheme=scheme,
        server_url=f"{formatted_host}:{port}",
        model_name=model_name,
        model_version=model_version,
    )


def _normalized_datatype(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TritonBackendError(f"{location} has no datatype.")
    datatype = value.strip().upper()
    if datatype.startswith("TYPE_"):
        datatype = datatype[5:]
    return datatype


def _metadata_json_from_config(config: dict[str, Any], *, artifact: str) -> dict:
    parameters = config.get("parameters")
    if not isinstance(parameters, dict):
        raise ExportMetadataError(f"{artifact} is missing config.pbtxt parameters.")
    parameter = parameters.get(TRITON_METADATA_PARAMETER)
    if isinstance(parameter, dict):
        raw_metadata = parameter.get("string_value")
    else:
        raw_metadata = parameter
    if not isinstance(raw_metadata, str) or not raw_metadata.strip():
        raise ExportMetadataError(
            f"{artifact} config.pbtxt parameter {TRITON_METADATA_PARAMETER!r} "
            "must contain a JSON string_value."
        )
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError as exc:
        raise ExportMetadataError(
            f"{artifact} parameter {TRITON_METADATA_PARAMETER!r} contains invalid JSON."
        ) from exc
    if not isinstance(metadata, dict):
        raise ExportMetadataError(
            f"{artifact} parameter {TRITON_METADATA_PARAMETER!r} must decode to an object."
        )
    return metadata


def create_triton_config(
    onnx_path: str | Path,
    config_path: str | Path,
    *,
    model_name: str | None = None,
    max_batch_size: int = 8,
) -> str:
    """Write a CPU ``config.pbtxt`` preserving LibreYOLO ONNX metadata.

    The ONNX graph remains unchanged. Input and output declarations are emitted
    in graph order, and the complete flat metadata map is encoded as one Triton
    string parameter.
    """
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "Creating a Triton config requires ONNX. Install with: "
            "pip install 'libreyolo[onnx]'"
        ) from exc

    onnx_path = Path(onnx_path)
    config_path = Path(config_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    try:
        max_batch_size = int(max_batch_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_batch_size must be a non-negative integer.") from exc
    if max_batch_size < 0:
        raise ValueError("max_batch_size must be a non-negative integer.")
    resolved_name = model_name or config_path.parent.name
    if (
        not isinstance(resolved_name, str)
        or not resolved_name
        or resolved_name != resolved_name.strip()
        or any(char in resolved_name for char in "\r\n\0")
    ):
        raise ValueError("model_name must be a non-empty single-line string.")

    model = onnx.load(str(onnx_path), load_external_data=False)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as exc:
        raise ValueError(
            f"Could not resolve ONNX tensor shapes for Triton config: {exc}"
        ) from exc
    metadata = {item.key: item.value for item in model.metadata_props}
    parse_export_metadata(
        metadata,
        artifact=f"ONNX metadata for {onnx_path}",
        strict=True,
    )

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    graph_inputs = [
        value for value in model.graph.input if value.name not in initializer_names
    ]
    graph_outputs = list(model.graph.output)
    if len(graph_inputs) != 1:
        raise ValueError(
            f"Triton v1 requires exactly one ONNX graph input, got {len(graph_inputs)}."
        )
    if not graph_outputs:
        raise ValueError("ONNX graph has no outputs.")

    def tensor_config(value, *, drop_batch_axis: bool) -> tuple[str, str, list[int]]:
        tensor_type = value.type.tensor_type
        onnx_dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
        triton_dtype = _ONNX_TO_TRITON_CONFIG_DTYPE.get(onnx_dtype)
        if triton_dtype is None:
            raise ValueError(
                f"ONNX tensor {value.name!r} uses unsupported datatype {onnx_dtype!r}."
            )
        dims = [
            int(dim.dim_value) if int(dim.dim_value) > 0 else -1
            for dim in tensor_type.shape.dim
        ]
        if drop_batch_axis:
            if not dims:
                raise ValueError(f"ONNX tensor {value.name!r} has no batch axis.")
            dims = dims[1:]
        if not dims:
            raise ValueError(f"ONNX tensor {value.name!r} has scalar shape.")
        return value.name, triton_dtype, dims

    batched = max_batch_size > 0
    input_specs = [tensor_config(graph_inputs[0], drop_batch_axis=batched)]
    if batched:
        first_dim = graph_inputs[0].type.tensor_type.shape.dim[0]
        if int(first_dim.dim_value) > 0:
            raise ValueError(
                "max_batch_size > 0 requires an ONNX model with a dynamic batch axis."
            )
        for value in graph_outputs:
            output_dims = value.type.tensor_type.shape.dim
            if not output_dims or int(output_dims[0].dim_value) > 0:
                raise ValueError(
                    "max_batch_size > 0 requires every ONNX output to have a "
                    f"dynamic batch axis; {value.name!r} does not."
                )
    output_specs = [
        tensor_config(value, drop_batch_axis=batched) for value in graph_outputs
    ]

    def tensor_block(kind: str, specs: list[tuple[str, str, list[int]]]) -> str:
        entries = []
        for name, datatype, dims in specs:
            dims_text = ", ".join(str(dim) for dim in dims)
            entries.append(
                "  {\n"
                f"    name: {json.dumps(name)}\n"
                f"    data_type: TYPE_{datatype}\n"
                f"    dims: [ {dims_text} ]\n"
                "  }"
            )
        return f"{kind} [\n" + ",\n".join(entries) + "\n]\n"

    encoded_metadata = json.dumps(
        json.dumps(metadata, separators=(",", ":")),
        ensure_ascii=False,
    )
    config = (
        f"name: {json.dumps(resolved_name)}\n"
        'platform: "onnxruntime_onnx"\n'
        f"max_batch_size: {max_batch_size}\n"
        + tensor_block("input", input_specs)
        + tensor_block("output", output_specs)
        + "instance_group [{ kind: KIND_CPU }]\n"
        + "parameters {\n"
        + f"  key: {json.dumps(TRITON_METADATA_PARAMETER)}\n"
        + f"  value: {{ string_value: {encoded_metadata} }}\n"
        + "}\n"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config, encoding="utf-8")
    return str(config_path)


class TritonBackend(BaseBackend):
    """Run LibreYOLO exported models through Triton's HTTP/HTTPS V2 client."""

    def __init__(
        self,
        model_url: str,
        *,
        task: str | None = None,
        device: str = "auto",
        timeout: float = 30.0,
    ):
        try:
            import tritonclient.http as httpclient
            from tritonclient.utils import triton_to_np_dtype
        except ImportError as exc:
            raise ImportError(
                "Triton inference requires the HTTP client. Install with: "
                "pip install 'tritonclient[http]'"
            ) from exc

        try:
            timeout = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Triton timeout must be a positive number of seconds."
            ) from exc
        if timeout <= 0:
            raise ValueError("Triton timeout must be a positive number of seconds.")

        target = parse_triton_model_url(model_url)
        self.model_url = model_url
        self.server_url = target.server_url
        self.model_name = target.model_name
        self.model_version = target.model_version
        self.timeout = timeout
        self._httpclient = httpclient
        if device != "auto":
            logger.info(
                "Triton controls model placement on the server; ignoring device=%r.",
                device,
            )

        try:
            self.client = httpclient.InferenceServerClient(
                url=target.server_url,
                ssl=target.scheme == "https",
                connection_timeout=timeout,
                network_timeout=timeout,
            )
            if not self.client.is_server_ready():
                raise TritonBackendError(
                    f"Triton server {target.server_url!r} is not ready."
                )
            if not self.client.is_model_ready(
                target.model_name,
                model_version=target.model_version,
            ):
                version_text = target.model_version or "policy-selected"
                raise TritonBackendError(
                    f"Triton model {target.model_name!r} version "
                    f"{version_text!r} is not ready."
                )
            config = self.client.get_model_config(
                target.model_name,
                model_version=target.model_version,
            )
            server_metadata = self.client.get_model_metadata(
                target.model_name,
                model_version=target.model_version,
            )
        except TritonBackendError:
            raise
        except Exception as exc:
            raise TritonBackendError(
                f"Failed to initialize Triton model {model_url!r}: {exc}"
            ) from exc

        artifact = f"Triton model {target.model_name!r}"
        (
            self.input_name,
            input_datatype,
            input_shape,
            self.output_names,
            self._max_batch_size,
        ) = self._read_model_description(config, server_metadata, artifact=artifact)
        raw_runtime_metadata = _metadata_json_from_config(config, artifact=artifact)
        warn_on_metadata_schema_version(
            raw_runtime_metadata,
            artifact=f"{artifact} runtime metadata",
            logger=logger,
        )
        parsed_metadata = parse_export_metadata(
            raw_runtime_metadata,
            artifact=f"{artifact} runtime metadata",
            strict=True,
        )

        numpy_dtype = triton_to_np_dtype(input_datatype)
        if numpy_dtype is None:
            raise TritonBackendError(
                f"{artifact} input {self.input_name!r} uses unsupported datatype "
                f"{input_datatype!r}."
            )
        self.input_datatype = input_datatype
        self.input_dtype = np.dtype(numpy_dtype)
        if not np.issubdtype(self.input_dtype, np.number):
            raise TritonBackendError(
                f"{artifact} image input must use a numeric datatype, got "
                f"{input_datatype!r}."
            )
        self.input_shape = tuple(input_shape)
        self._dynamic_batch_axis = self._max_batch_size > 0
        self._dynamic_spatial_axes = any(
            not isinstance(dim, int) or dim <= 0 for dim in self.input_shape[2:4]
        )
        self.embedded_nms = parsed_metadata.embedded_nms
        self.embedded_nms_raw_output_index = (
            self.output_names.index("raw")
            if self.embedded_nms and "raw" in self.output_names
            else None
        )

        static_imgsz = self._read_static_input_imgsz(self.input_shape)
        imgsz = static_imgsz or parsed_metadata.imgsz
        if imgsz is None:
            raise ExportMetadataError(
                f"{artifact} has dynamic spatial input axes but no imgsz metadata."
            )
        resolved_task = resolve_task(
            explicit_task=task,
            checkpoint_task=parsed_metadata.task,
            default_task=parsed_metadata.default_task,
            supported_tasks=parsed_metadata.supported_tasks,
        )
        runtime = parsed_metadata.runtime
        super().__init__(
            model_path=model_url,
            nb_classes=parsed_metadata.nb_classes,
            device="triton",
            imgsz=imgsz,
            model_family=parsed_metadata.model_family,
            names=parsed_metadata.names,
            model_size=parsed_metadata.model_size,
            task=resolved_task,
            supported_tasks=parsed_metadata.supported_tasks,
            default_task=parsed_metadata.default_task,
            **classify_eval_kwargs(runtime),
            num_bins=runtime.get("num_bins"),
            bin_width_deg=runtime.get("bin_width_deg"),
            offset_deg=runtime.get("offset_deg"),
            **parsed_metadata.pose,
        )

    @staticmethod
    def _read_static_input_imgsz(shape: tuple[Any, ...]) -> ImageSize | None:
        if len(shape) != 4:
            return None
        h, w = shape[2], shape[3]
        if not isinstance(h, int) or not isinstance(w, int) or h <= 0 or w <= 0:
            return None
        return h if h == w else (h, w)

    @staticmethod
    def _read_model_description(
        config: Any,
        metadata: Any,
        *,
        artifact: str,
    ) -> tuple[str, str, list[Any], list[str], int]:
        if not isinstance(config, dict):
            raise TritonBackendError(
                f"{artifact} configuration response is not an object."
            )
        if not isinstance(metadata, dict):
            raise TritonBackendError(f"{artifact} metadata response is not an object.")

        config_inputs = config.get("input")
        metadata_inputs = metadata.get("inputs")
        if not isinstance(config_inputs, list) or len(config_inputs) != 1:
            count = len(config_inputs) if isinstance(config_inputs, list) else 0
            raise TritonBackendError(
                f"{artifact} must declare exactly one input in config.pbtxt, got {count}."
            )
        if not isinstance(metadata_inputs, list) or len(metadata_inputs) != 1:
            count = len(metadata_inputs) if isinstance(metadata_inputs, list) else 0
            raise TritonBackendError(
                f"{artifact} must expose exactly one input in model metadata, got {count}."
            )

        config_input = config_inputs[0]
        metadata_input = metadata_inputs[0]
        if not isinstance(config_input, dict) or not isinstance(metadata_input, dict):
            raise TritonBackendError(f"{artifact} input descriptions must be objects.")
        input_name = metadata_input.get("name")
        if not isinstance(input_name, str) or not input_name:
            raise TritonBackendError(f"{artifact} input metadata has no name.")
        if config_input.get("name") != input_name:
            raise TritonBackendError(
                f"{artifact} input name differs between config and metadata."
            )
        config_input_dtype = _normalized_datatype(
            config_input.get("data_type"),
            location=f"{artifact} config input {input_name!r}",
        )
        input_datatype = _normalized_datatype(
            metadata_input.get("datatype"),
            location=f"{artifact} metadata input {input_name!r}",
        )
        if config_input_dtype != input_datatype:
            raise TritonBackendError(
                f"{artifact} input datatype differs between config and metadata."
            )
        input_shape = metadata_input.get("shape")
        if not isinstance(input_shape, list) or len(input_shape) != 4:
            raise TritonBackendError(
                f"{artifact} image input must have rank 4 NCHW metadata."
            )

        config_outputs = config.get("output")
        metadata_outputs = metadata.get("outputs")
        if not isinstance(config_outputs, list) or not config_outputs:
            raise TritonBackendError(f"{artifact} config.pbtxt declares no outputs.")
        if not isinstance(metadata_outputs, list) or not metadata_outputs:
            raise TritonBackendError(f"{artifact} model metadata exposes no outputs.")
        config_output_types = {}
        for output in config_outputs:
            if not isinstance(output, dict) or not isinstance(output.get("name"), str):
                raise TritonBackendError(f"{artifact} has a malformed config output.")
            name = output["name"]
            if name in config_output_types:
                raise TritonBackendError(
                    f"{artifact} declares duplicate output {name!r}."
                )
            config_output_types[name] = _normalized_datatype(
                output.get("data_type"),
                location=f"{artifact} config output {name!r}",
            )

        output_names = []
        metadata_output_types = {}
        for output in metadata_outputs:
            if not isinstance(output, dict) or not isinstance(output.get("name"), str):
                raise TritonBackendError(f"{artifact} has a malformed metadata output.")
            name = output["name"]
            if name in metadata_output_types:
                raise TritonBackendError(
                    f"{artifact} exposes duplicate output {name!r}."
                )
            output_names.append(name)
            metadata_output_types[name] = _normalized_datatype(
                output.get("datatype"),
                location=f"{artifact} metadata output {name!r}",
            )
        if set(config_output_types) != set(output_names):
            raise TritonBackendError(
                f"{artifact} output names differ between config and metadata."
            )
        for name in output_names:
            if config_output_types[name] != metadata_output_types[name]:
                raise TritonBackendError(
                    f"{artifact} output {name!r} datatype differs between config "
                    "and metadata."
                )

        try:
            max_batch_size = int(config.get("max_batch_size", 0))
        except (TypeError, ValueError) as exc:
            raise TritonBackendError(f"{artifact} has invalid max_batch_size.") from exc
        if max_batch_size < 0:
            raise TritonBackendError(f"{artifact} max_batch_size must be non-negative.")
        return input_name, input_datatype, input_shape, output_names, max_batch_size

    def _supports_batched_inference(self) -> bool:
        return self._max_batch_size > 1 and not self.embedded_nms

    def _process_in_batches(self, images, batch: int = 1, **kwargs):
        if self._max_batch_size > 0:
            batch = min(int(batch), self._max_batch_size)
        return super()._process_in_batches(images, batch=batch, **kwargs)

    def _run_inference(self, blob: np.ndarray) -> list[np.ndarray]:
        array = np.ascontiguousarray(blob, dtype=self.input_dtype)
        if array.ndim != 4:
            raise TritonBackendError(
                f"Triton input must be rank-4 NCHW, got shape {array.shape}."
            )
        if self._max_batch_size == 0 and array.shape[0] != 1:
            raise TritonBackendError(
                "This Triton model does not support batched requests."
            )
        if self._max_batch_size > 0 and array.shape[0] > self._max_batch_size:
            raise TritonBackendError(
                f"Triton model accepts at most {self._max_batch_size} images per request."
            )
        for axis, (actual, expected) in enumerate(zip(array.shape, self.input_shape)):
            if isinstance(expected, int) and expected > 0 and actual != expected:
                raise TritonBackendError(
                    f"Triton input shape mismatch at axis {axis}: expected "
                    f"{expected}, got {actual}."
                )

        infer_input = self._httpclient.InferInput(
            self.input_name,
            list(array.shape),
            self.input_datatype,
        )
        infer_input.set_data_from_numpy(array, binary_data=True)
        requested_outputs = [
            self._httpclient.InferRequestedOutput(name, binary_data=True)
            for name in self.output_names
        ]
        try:
            result = self.client.infer(
                self.model_name,
                inputs=[infer_input],
                model_version=self.model_version,
                outputs=requested_outputs,
                timeout=max(1, int(self.timeout * 1_000_000)),
            )
            arrays = []
            for name in self.output_names:
                output = result.as_numpy(name)
                if output is None:
                    raise TritonBackendError(
                        f"Triton response is missing requested output {name!r}."
                    )
                arrays.append(output)
            return arrays
        except TritonBackendError:
            raise
        except Exception as exc:
            raise TritonBackendError(
                f"Triton inference failed for model {self.model_name!r}: {exc}"
            ) from exc
