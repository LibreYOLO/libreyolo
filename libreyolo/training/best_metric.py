"""User-selectable metric for best.pt selection and early stopping.

The trainer compares one validation metric per epoch. That comparison
decides both which epoch is saved as ``best.pt`` and when ``patience`` runs
out. ``best_metric`` lets the user choose that metric by a short alias; the
alias is resolved against the model task into the full metric key emitted
by the task's validator.
"""

from __future__ import annotations

# task -> alias -> metric key emitted by that task's validator.
# Every value is higher-is-better; the trainer's strict ``>`` compare relies
# on that. Detection ``f1`` is the maximum F1 over the confidence sweep at
# IoU 0.50 (``metrics/best_conf_f1``), already computed every epoch.
BEST_METRIC_ALIASES: dict[str, dict[str, str]] = {
    "detect": {
        "map50-95": "metrics/mAP50-95",
        "map50": "metrics/mAP50",
        "map75": "metrics/mAP75",
        "f1": "metrics/best_conf_f1",
    },
    "classify": {
        "top1": "metrics/accuracy_top1",
        "top5": "metrics/accuracy_top5",
        "f1": "metrics/f1",
        "precision": "metrics/precision",
        "recall": "metrics/recall",
    },
}

SUPPORTED_BEST_METRIC_TASKS: tuple[str, ...] = tuple(sorted(BEST_METRIC_ALIASES))


def normalize_best_metric(value: object) -> str | None:
    """Return the canonical alias spelling, or ``None`` when unset/empty."""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def resolve_best_metric(alias: str, task: str) -> str:
    """Resolve a ``best_metric`` alias for ``task`` into a full metric key.

    Raises:
        ValueError: if ``task`` has no selectable metrics, or ``alias`` is not
            one of the task's aliases. The message lists the valid values.
    """
    task_name = str(task).strip().lower()
    table = BEST_METRIC_ALIASES.get(task_name)
    if table is None:
        supported = ", ".join(SUPPORTED_BEST_METRIC_TASKS)
        raise ValueError(
            f"best_metric is not supported for task '{task}'. "
            f"Supported tasks: {supported}."
        )
    key = normalize_best_metric(alias)
    if key is None or key not in table:
        valid = ", ".join(sorted(table))
        raise ValueError(
            f"Unknown best_metric '{alias}' for task '{task_name}'. "
            f"Valid values: {valid}."
        )
    return table[key]
