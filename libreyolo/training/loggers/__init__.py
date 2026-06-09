"""Built-in experiment loggers (TensorBoard, MLflow, Weights & Biases).

Loggers are ordinary training callbacks consuming the public hook system
(:mod:`libreyolo.training.callbacks`). Enable them by name::

    model.train(data="data.yaml", loggers="mlflow")

or pass configured instances (mixing with names is fine)::

    from libreyolo.training import MLflowLogger
    model.train(data="data.yaml", loggers=[MLflowLogger(experiment_name="exp"), "tensorboard"])
"""

from __future__ import annotations

from typing import Any, Iterable, List

from .base import BaseLogger as BaseLogger
from .mlflow_logger import MLflowLogger as MLflowLogger
from .tensorboard_logger import TensorBoardLogger as TensorBoardLogger
from .wandb_logger import WandbLogger as WandbLogger

_LOGGER_FACTORIES = {
    "tensorboard": TensorBoardLogger,
    "mlflow": MLflowLogger,
    "wandb": WandbLogger,
}


def resolve_loggers(loggers: Any) -> List[Any]:
    """Resolve the ``loggers=`` train argument into callback instances.

    Accepts ``None``, a logger name (``"tensorboard"``, ``"mlflow"``,
    ``"wandb"``), a callback object, or an iterable mixing both.
    """
    if loggers is None:
        return []
    if isinstance(loggers, str) or not isinstance(loggers, Iterable):
        loggers = [loggers]

    resolved: List[Any] = []
    for item in loggers:
        if isinstance(item, str):
            key = item.strip().lower()
            if key not in _LOGGER_FACTORIES:
                raise ValueError(
                    f"Unknown logger {item!r}. "
                    f"Valid names: {sorted(_LOGGER_FACTORIES)}"
                )
            resolved.append(_LOGGER_FACTORIES[key]())
        else:
            resolved.append(item)
    return resolved
