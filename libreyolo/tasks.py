"""Task metadata and resolution helpers for LibreYOLO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal


TaskType = Literal["detect", "segment", "pose", "classify"]
TASKS = ("detect", "segment", "pose", "classify")

TASK_ALIASES = {
    "detect": "detect",
    "detection": "detect",
    "det": "detect",
    "segment": "segment",
    "segmentation": "segment",
    "seg": "segment",
    "pose": "pose",
    "keypoint": "pose",
    "keypoints": "pose",
    "classify": "classify",
    "classification": "classify",
    "class": "classify",
    "cls": "classify",
}

TASK_TO_SUFFIX = {
    "segment": "seg",
    "pose": "pose",
    "classify": "cls",
}

SUFFIX_TO_TASK = {v: k for k, v in TASK_TO_SUFFIX.items()}


def normalize_task(task: str | None, *, default: str | None = None) -> str | None:
    """Normalize public task aliases to canonical task names."""
    if task is None:
        return default
    normalized = TASK_ALIASES.get(str(task).strip().lower())
    if normalized is None:
        allowed = ", ".join(TASKS)
        raise ValueError(f"Unsupported task: {task!r}. Must be one of: {allowed}.")
    return normalized


def task_to_suffix(task: str | None) -> str | None:
    """Return the filename suffix for a canonical task, or None for detect."""
    task = normalize_task(task)
    return TASK_TO_SUFFIX.get(task)


def suffix_to_task(suffix: str | None) -> str | None:
    """Return the canonical task for a filename suffix."""
    if not suffix:
        return None
    return SUFFIX_TO_TASK.get(suffix.lstrip("-").lower())


def normalize_supported_tasks(
    supported_tasks: Iterable[str] | str | None = None,
) -> tuple[str, ...]:
    """Normalize a supported-task collection from code or exported metadata."""
    if supported_tasks is None:
        return ("detect",)

    if isinstance(supported_tasks, str):
        value = supported_tasks.strip()
        if not value:
            return ("detect",)
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in value.split(",") if part.strip()]
        supported_tasks = parsed if isinstance(parsed, (list, tuple)) else [parsed]

    normalized = []
    for task in supported_tasks:
        canonical = normalize_task(task)
        if canonical is not None and canonical not in normalized:
            normalized.append(canonical)
    return tuple(normalized) or ("detect",)


def task_suffix_pattern(
    supported_tasks: Iterable[str] | str | None = None,
) -> str:
    """Regex fragment matching supported task suffixes."""
    tasks = TASKS if supported_tasks is None else normalize_supported_tasks(supported_tasks)
    suffixes = sorted(
        (TASK_TO_SUFFIX[task] for task in tasks if task in TASK_TO_SUFFIX),
        key=len,
        reverse=True,
    )
    return "|".join(f"-{suffix}" for suffix in suffixes)


def detect_task_suffix(filename: str | Path) -> str | None:
    """Infer a canonical task from a supported filename suffix."""
    stem = Path(filename).stem.lower()
    for suffix, task in SUFFIX_TO_TASK.items():
        if stem.endswith(f"-{suffix}"):
            return task
    return None


def resolve_task(
    *,
    explicit_task: str | None = None,
    checkpoint_task: str | None = None,
    filename_task: str | None = None,
    default_task: str = "detect",
    supported_tasks: Iterable[str] = ("detect",),
) -> str:
    """Resolve task precedence and validate against a model family's support."""
    supported = normalize_supported_tasks(supported_tasks)
    candidates = (explicit_task, checkpoint_task, filename_task, default_task)
    task = next((normalize_task(t) for t in candidates if t is not None), default_task)
    if task not in supported:
        supported_text = ", ".join(str(t) for t in supported)
        raise ValueError(
            f"Task {task!r} is not supported by this model family. "
            f"Supported tasks: {supported_text}."
        )
    return task
