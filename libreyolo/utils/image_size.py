"""Shared image-size normalization helpers."""

from numbers import Integral
from typing import Any


ImageSize = int | tuple[int, int]


def _positive_dimension(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"{name} dimensions must be integers, got {type(value).__name__}: "
            f"{value!r}."
        )
    dimension = int(value)
    if dimension <= 0:
        raise ValueError(f"{name} dimensions must be positive, got {dimension}.")
    return dimension


def normalize_imgsz(
    value: Any,
    *,
    name: str = "imgsz",
    allow_string: bool = False,
    allow_comma: bool = False,
) -> ImageSize:
    """Normalize a square or ``(height, width)`` image size.

    Square pairs collapse to their scalar form. Strings may use ``HxW`` when
    enabled; comma-separated pairs are an optional legacy CLI alias.
    """
    if isinstance(value, str):
        if not allow_string:
            raise TypeError(
                f"{name} must be an int or (height, width) pair, got str: {value!r}."
            )
        text = value.strip()
        if not text:
            raise ValueError(f"{name} must not be empty.")

        lowered = text.lower()
        if "x" in lowered:
            parts = lowered.split("x")
        elif "," in text:
            if not allow_comma:
                raise ValueError(
                    f"Invalid {name} format: {value!r}. Use 640 or 480x640 (HxW)."
                )
            parts = text.split(",")
        else:
            try:
                return _positive_dimension(int(text), name=name)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {name} format: {value!r}. Use 640 or 480x640 (HxW)."
                ) from exc

        if len(parts) != 2:
            raise ValueError(
                f"Invalid {name} format: {value!r}. Use 640 or 480x640 (HxW)."
            )
        try:
            height, width = (int(part.strip()) for part in parts)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {name} format: {value!r}. Use 640 or 480x640 (HxW)."
            ) from exc
        height = _positive_dimension(height, name=name)
        width = _positive_dimension(width, name=name)
    elif isinstance(value, Integral) and not isinstance(value, bool):
        return _positive_dimension(value, name=name)
    elif isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(
                f"{name} must have exactly 2 elements (height, width), "
                f"got {len(value)}: {value!r}."
            )
        height = _positive_dimension(value[0], name=name)
        width = _positive_dimension(value[1], name=name)
    else:
        raise TypeError(
            f"{name} must be an int or (height, width) pair, got "
            f"{type(value).__name__}: {value!r}."
        )

    return height if height == width else (height, width)


def imgsz_to_hw(
    value: Any,
    *,
    name: str = "imgsz",
    allow_string: bool = False,
    allow_comma: bool = False,
) -> tuple[int, int]:
    """Normalize an image size and return an explicit ``(height, width)`` pair."""
    normalized = normalize_imgsz(
        value,
        name=name,
        allow_string=allow_string,
        allow_comma=allow_comma,
    )
    if isinstance(normalized, int):
        return normalized, normalized
    return normalized


def reject_rectangular_imgsz(model: Any, imgsz: Any, call: str) -> None:
    """Raise a clear error when ``model`` cannot run ``call`` at a rectangle.

    Families list the calls that need a square ``imgsz`` in
    ``SQUARE_IMGSZ_CALLS``; without this, their preprocessing crashes deep
    inside on an ``(h, w)`` tuple.
    """
    if call not in getattr(model, "SQUARE_IMGSZ_CALLS", ()):
        return
    if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
        h, w = int(imgsz[0]), int(imgsz[1])
        if h != w:
            family = getattr(model, "family", type(model).__name__)
            raise ValueError(
                f"{family} {call}() does not support rectangular input sizes, "
                f"got ({h}, {w}). Use a square imgsz."
            )
