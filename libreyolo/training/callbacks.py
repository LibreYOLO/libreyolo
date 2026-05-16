"""Public training callback types."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Protocol


@dataclass(frozen=True)
class TrainStartEvent:
    """Data emitted after trainer setup, before the first training epoch."""

    start_epoch: int
    total_epochs: int
    model_family: str
    model_size: str | None
    task: str
    save_dir: str


@dataclass(frozen=True)
class TrainEpochEvent:
    """Data emitted after a training epoch has completed.

    ``current_metric`` is the selected validation metric for this epoch.
    ``best_metric`` is the trainer's best-so-far value after this epoch updates
    best-state tracking.
    """

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
    current_metric: float | None
    current_metric_name: str | None
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


@dataclass(frozen=True)
class TrainEndEvent:
    """Data emitted after training completes and results are built."""

    total_epochs: int
    completed_epochs: int
    model_family: str
    model_size: str | None
    task: str
    save_dir: str
    final_loss: float
    best_metric: float | None
    best_epoch: int | None
    total_seconds: float
    results: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))


@dataclass(frozen=True)
class TrainExceptionEvent:
    """Data emitted when training raises before returning results."""

    epoch: int | None
    total_epochs: int
    model_family: str
    model_size: str | None
    task: str
    save_dir: str
    exception: BaseException
    exception_type: str
    exception_message: str
    elapsed_seconds: float


class TrainCallback(Protocol):
    """Protocol for object-style training callbacks."""

    def on_train_start(self, event: TrainStartEvent) -> None:
        """Handle the start of training."""

    def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
        """Handle an epoch-complete training event."""

    def on_train_end(self, event: TrainEndEvent) -> None:
        """Handle successful training completion."""

    def on_train_exception(self, event: TrainExceptionEvent) -> None:
        """Handle a training exception."""


TrainEpochCallable = Callable[[TrainEpochEvent], None]
TrainCallbackLike = TrainCallback | TrainEpochCallable
TrainCallbacks = TrainCallbackLike | Iterable[TrainCallbackLike] | None


class TrainCallbackList:
    """Dispatch training events to callback objects or plain callables."""

    def __init__(self, callbacks: TrainCallbacks = None):
        if callbacks is None:
            self._callbacks: list[TrainCallbackLike] = []
        elif self._is_callback_object(callbacks) or callable(callbacks):
            self._callbacks = [callbacks]
        else:
            self._callbacks = list(callbacks)

    def __bool__(self) -> bool:
        return bool(self._callbacks)

    def __len__(self) -> int:
        return len(self._callbacks)

    def append(self, callback: TrainCallbackLike) -> None:
        self._callbacks.append(callback)

    @staticmethod
    def _is_callback_object(callback) -> bool:
        return any(
            hasattr(callback, name)
            for name in (
                "on_train_start",
                "on_train_epoch_end",
                "on_train_end",
                "on_train_exception",
            )
        )

    def _dispatch(self, method_name: str, event, *, call_plain: bool = False) -> None:
        for callback in self._callbacks:
            method = getattr(callback, method_name, None)
            if method is not None:
                if not callable(method):
                    raise TypeError(
                        f"Train callback attribute {method_name} must be callable"
                    )
                method(event)
            elif call_plain and callable(callback):
                callback(event)
            elif not callable(callback) and not self._is_callback_object(callback):
                raise TypeError(
                    "Train callback must be callable or define a train callback method"
                )

    def on_train_start(self, event: TrainStartEvent) -> None:
        self._dispatch("on_train_start", event)

    def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
        self._dispatch("on_train_epoch_end", event, call_plain=True)

    def on_train_end(self, event: TrainEndEvent) -> None:
        self._dispatch("on_train_end", event)

    def on_train_exception(self, event: TrainExceptionEvent) -> None:
        self._dispatch("on_train_exception", event)
