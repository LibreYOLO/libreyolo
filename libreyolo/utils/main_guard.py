"""Clear exit for ``train()`` / ``val()`` called outside a main guard (#925).

Under the ``spawn`` and ``forkserver`` start methods (macOS, Windows, and Linux
from Python 3.14) every DataLoader worker re-imports the ``__main__`` script
before it runs. A top-level ``train()`` or ``val()`` call then runs again
inside each worker, which fails with a long multiprocessing traceback. The
worker is stopped here instead, with one line that says what to change.
"""

from __future__ import annotations

import functools
import inspect
import multiprocessing
import sys
from typing import Callable

MESSAGE = (
    "LibreYOLO: {action} was called at the top level of your script. DataLoader "
    "workers re-run that script on this platform, so put the call under "
    '`if __name__ == "__main__":`, or pass workers=0.'
)


def exit_if_reimporting_main(action: str, workers=None) -> None:
    """Stop a worker process that is re-running the user's ``__main__`` script.

    ``_inheriting`` is the flag multiprocessing itself sets while a spawned or
    forkserver child imports ``__main__``; it is never set in the main process
    or in a worker that is already running its target. A call that starts no
    DataLoader workers (``workers=0``) does not fail there, so it is let through.
    """
    try:
        if workers is not None and int(workers) <= 0:
            return
    except (TypeError, ValueError):
        pass
    if getattr(multiprocessing.current_process(), "_inheriting", False):
        # Printed before exiting: a forkserver's preload of ``__main__`` drops
        # the SystemExit message.
        print(MESSAGE.format(action=action), file=sys.stderr, flush=True)
        raise SystemExit(1)


def guard_main_reimport(action: str) -> Callable[[Callable], Callable]:
    """Decorate a public method so it calls :func:`exit_if_reimporting_main` first."""

    def decorate(fn: Callable) -> Callable:
        if getattr(fn, "_libreyolo_main_guarded", False):
            return fn

        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if getattr(multiprocessing.current_process(), "_inheriting", False):
                try:
                    bound = signature.bind_partial(*args, **kwargs)
                    bound.apply_defaults()
                    workers = bound.arguments.get("workers")
                except TypeError:
                    workers = None
                exit_if_reimporting_main(action, workers)
            return fn(*args, **kwargs)

        wrapper._libreyolo_main_guarded = True
        return wrapper

    return decorate


__all__ = ["exit_if_reimporting_main", "guard_main_reimport"]
