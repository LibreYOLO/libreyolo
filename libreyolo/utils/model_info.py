"""Model information helpers shared by wrappers, backends, and the CLI."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


ImageSize = Union[int, Tuple[int, int], List[int]]


def _numel(value: Any) -> int:
    numel = getattr(value, "numel", None)
    if callable(numel):
        return int(numel())
    return 0


def _shape(value: Any) -> list[int]:
    return [int(dim) for dim in getattr(value, "shape", ())]


def _dtype(value: Any) -> Optional[str]:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype).replace("torch.", "")


def _as_hw_list(value: Any) -> Optional[list[int]]:
    try:
        if value is None:
            return None
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                return None
            return [int(value[0]), int(value[1])]
        size = int(value)
        return [size, size]
    except (TypeError, ValueError):
        return None


def _model_family(model: Any) -> Optional[str]:
    for attr in ("FAMILY", "model_family", "family"):
        value = getattr(model, attr, None)
        if value:
            return str(value)
    get_model_name = getattr(model, "_get_model_name", None)
    if callable(get_model_name):
        try:
            value = get_model_name()
        except Exception:
            value = None
        if value:
            return str(value)
    return None


def _model_size(model: Any) -> Optional[str]:
    for attr in ("size", "model_size"):
        value = getattr(model, attr, None)
        if value:
            return str(value)
    return None


def _input_size(model: Any) -> Optional[list[int]]:
    get_input_size = getattr(model, "_get_input_size", None)
    if callable(get_input_size):
        try:
            size = _as_hw_list(get_input_size())
        except Exception:
            size = None
        if size is not None:
            return size

    for attr in ("input_size", "imgsz"):
        size = _as_hw_list(getattr(model, attr, None))
        if size is not None:
            return size

    input_sizes = getattr(model, "INPUT_SIZES", None)
    size_key = getattr(model, "size", None)
    if isinstance(input_sizes, dict) and size_key is not None:
        size = _as_hw_list(input_sizes.get(size_key))
        if size is not None:
            return size
    return None


def _class_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        class_names = {}
        for key, value in names.items():
            try:
                class_names[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return class_names
    if isinstance(names, (list, tuple)):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}


def _parameters(module: Any) -> list[Any]:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return []
    try:
        return list(parameters())
    except Exception:
        return []


def _named_parameters(module: Any) -> list[tuple[str, Any]]:
    named_parameters = getattr(module, "named_parameters", None)
    if not callable(named_parameters):
        return []
    try:
        return list(named_parameters())
    except Exception:
        return []


def _named_modules(module: Any) -> list[tuple[str, Any]]:
    named_modules = getattr(module, "named_modules", None)
    if not callable(named_modules):
        return []
    try:
        return list(named_modules())
    except Exception:
        return []


def _leaf_module_count(module: Any) -> Optional[int]:
    named_modules = _named_modules(module)
    if not named_modules:
        return None
    count = 0
    for _name, child in named_modules:
        children = getattr(child, "children", None)
        if not callable(children):
            continue
        try:
            if not list(children()):
                count += 1
        except Exception:
            continue
    return count


def _parameter_counts(module: Any) -> tuple[Optional[int], Optional[int], Optional[int]]:
    params = _parameters(module)
    if not params:
        return None, None, None
    total = sum(_numel(param) for param in params)
    trainable = sum(_numel(param) for param in params if getattr(param, "requires_grad", False))
    return total, trainable, total - trainable


def _parameter_details(module: Any) -> list[dict[str, Any]]:
    details = []
    for name, param in _named_parameters(module):
        details.append(
            {
                "name": str(name),
                "shape": _shape(param),
                "parameters": _numel(param),
                "trainable": bool(getattr(param, "requires_grad", False)),
                "dtype": _dtype(param),
            }
        )
    return details


def _core_module(model: Any) -> Any:
    return getattr(model, "model", None)


def build_model_info(model: Any, *, detailed: bool = False) -> Dict[str, Any]:
    """Build a JSON-friendly model information dictionary."""
    core = _core_module(model)
    parameters, trainable, non_trainable = _parameter_counts(core)
    names = _class_names(getattr(model, "names", None))

    data: Dict[str, Any] = {
        "model_family": _model_family(model),
        "size": _model_size(model),
        "task": getattr(model, "task", None),
        "input_size": _input_size(model),
        "num_classes": getattr(model, "nb_classes", getattr(model, "nc", None)),
        "class_names": names,
        "parameters": parameters,
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
        "layers": _leaf_module_count(core),
        "device": str(getattr(model, "device", "")) or None,
        "model_path": str(getattr(model, "model_path", "")) or None,
    }
    if detailed:
        data["details"] = _parameter_details(core)
    return data


def _format_count(value: Optional[int]) -> str:
    return f"{value:,}" if isinstance(value, int) else "unavailable"


def _format_input_size(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{int(value[0])}x{int(value[1])}"
    return "unavailable"


def _format_detail_rows(details: Iterable[dict[str, Any]]) -> list[str]:
    rows = []
    for item in details:
        params = item.get("parameters", 0)
        params = params if isinstance(params, int) else 0
        rows.append(
            "  "
            f"{item.get('name')}: "
            f"{params:,} params, "
            f"shape={item.get('shape')}, "
            f"trainable={item.get('trainable')}"
        )
    return rows


def format_model_info(info: Dict[str, Any]) -> str:
    """Format model information for human-readable logs or CLI output."""
    title = info.get("model") or info.get("model_path") or "model"
    num_classes = info.get("num_classes")
    classes_text = num_classes if num_classes is not None else "unknown"
    lines = [
        f"Model:      {title}",
        f"Family:     {info.get('model_family') or 'unknown'}",
        f"Size:       {info.get('size') or 'unknown'}",
        f"Task:       {info.get('task') or 'unknown'}",
        f"Classes:    {classes_text}",
        f"Parameters: {_format_count(info.get('parameters'))}",
        f"Trainable:  {_format_count(info.get('trainable_parameters'))}",
        f"Layers:     {_format_count(info.get('layers'))}",
        f"Input size: {_format_input_size(info.get('input_size'))}",
    ]
    device = info.get("device")
    if device:
        lines.append(f"Device:     {device}")

    details = info.get("details")
    if details:
        lines.extend(["", "Parameters:"])
        lines.extend(_format_detail_rows(details))

    return "\n".join(lines)
