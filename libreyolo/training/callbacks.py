"""Public training callback types."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Protocol


@dataclass(frozen=True)
class TrainEpochEvent:
    """Data emitted after a training epoch has completed."""

    epoch: int
    total_epochs: int
    model_family: str
    model_size: str | None
    task: str
    save_dir: str
    train_loss: float
    train_loss_items: Mapping[str, float]
    lr: Mapping[str, float]
    val_metrics: Mapping[str, float]
    validated: bool
    is_best: bool
    best_metric: float | None
    best_metric_name: str | None
    best_epoch: int | None
    epoch_seconds: float

    def __post_init__(self):
        object.__setattr__(
            self, "train_loss_items", MappingProxyType(dict(self.train_loss_items))
        )
        object.__setattr__(self, "lr", MappingProxyType(dict(self.lr)))
        object.__setattr__(
            self, "val_metrics", MappingProxyType(dict(self.val_metrics))
        )


class TrainCallback(Protocol):
    """Protocol for object-style training callbacks."""

    def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
        """Handle an epoch-complete training event."""


TrainEpochCallable = Callable[[TrainEpochEvent], None]
TrainCallbackLike = TrainCallback | TrainEpochCallable
TrainCallbacks = TrainCallbackLike | Iterable[TrainCallbackLike] | None


class TrainCallbackList:
    """Dispatch training events to callback objects or plain callables."""

    def __init__(self, callbacks: TrainCallbacks = None):
        if callbacks is None:
            self._callbacks: list[TrainCallbackLike] = []
        elif hasattr(callbacks, "on_train_epoch_end") or callable(callbacks):
            self._callbacks = [callbacks]
        else:
            self._callbacks = list(callbacks)

    def __bool__(self) -> bool:
        return bool(self._callbacks)

    def __len__(self) -> int:
        return len(self._callbacks)

    def append(self, callback: TrainCallbackLike) -> None:
        self._callbacks.append(callback)

    def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
        for callback in self._callbacks:
            method = getattr(callback, "on_train_epoch_end", None)
            if method is not None:
                if not callable(method):
                    raise TypeError(
                        "Train callback attribute on_train_epoch_end must be callable"
                    )
                method(event)
            elif callable(callback):
                callback(event)
            else:
                raise TypeError(
                    "Train callback must be callable or define on_train_epoch_end"
                )
