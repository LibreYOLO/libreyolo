"""Built-in experiment loggers layered on the public training hooks.

Loggers are ordinary training callbacks consuming the public hook system
(:mod:`libreyolo.training.callbacks`). Enable them by name::

    model.train(data="data.yaml", loggers="mlflow")

or pass configured instances (mixing with names is fine)::

    from libreyolo.training import MLflowLogger
    model.train(data="data.yaml", loggers=[MLflowLogger(experiment_name="exp"), "tensorboard"])
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .base import BaseLogger as BaseLogger
from .clearml_logger import ClearMLLogger as ClearMLLogger
from .comet_logger import CometLogger as CometLogger
from .dvclive_logger import DVCLiveLogger as DVCLiveLogger
from .hf_hub_logger import HuggingFaceHubLogger as HuggingFaceHubLogger
from .mlflow_logger import MLflowLogger as MLflowLogger
from .neptune_logger import NeptuneLogger as NeptuneLogger
from .tensorboard_logger import TensorBoardLogger as TensorBoardLogger
from .wandb_logger import WandbLogger as WandbLogger

_LOGGER_FACTORIES = {
    "clearml": ClearMLLogger,
    "comet": CometLogger,
    "dvc": DVCLiveLogger,
    "dvclive": DVCLiveLogger,
    "neptune": NeptuneLogger,
    "tensorboard": TensorBoardLogger,
    "mlflow": MLflowLogger,
    "wandb": WandbLogger,
}

# The Hub logger needs a target repo, so its string form carries it inline:
# loggers="hf:owner/repo" (also accepted: "huggingface:owner/repo").
_HF_LOGGER_PREFIXES = ("hf:", "huggingface:")


def resolve_loggers(loggers: Any) -> list[Any]:
    """Resolve the ``loggers=`` train argument into callback instances.

    Accepts ``None``, a registered logger name, a callback object, or an
    iterable mixing both. ``"dvc"`` is an alias for ``"dvclive"``.
    """
    if loggers is None:
        return []
    if isinstance(loggers, str) or not isinstance(loggers, Iterable):
        loggers = [loggers]

    resolved: list[Any] = []
    for item in loggers:
        if isinstance(item, str):
            key = item.strip().lower()
            if key.startswith(_HF_LOGGER_PREFIXES):
                repo_id = item.strip().partition(":")[2]
                resolved.append(HuggingFaceHubLogger(repo_id))
            elif key in ("hf", "huggingface"):
                raise ValueError(
                    "The Hugging Face Hub logger needs a target repository: "
                    "use loggers='hf:owner/repo', or pass a configured "
                    "HuggingFaceHubLogger instance."
                )
            elif key not in _LOGGER_FACTORIES:
                raise ValueError(
                    f"Unknown logger {item!r}. Valid names: "
                    f"{sorted(_LOGGER_FACTORIES) + ['hf:owner/repo']}"
                )
            else:
                resolved.append(_LOGGER_FACTORIES[key]())
        else:
            resolved.append(item)
    return resolved
