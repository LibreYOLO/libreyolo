"""``train()`` / ``val()`` called outside a main guard (#925).

Under the ``spawn`` and ``forkserver`` start methods (macOS, Windows, and Linux
from Python 3.14) every DataLoader worker re-imports the ``__main__`` script
before it runs. A top-level ``train()`` or ``val()`` call then runs again
inside each worker, which fails with a long multiprocessing traceback.

The main process reads the script and runs such a call with ``workers=0``,
logging one line that says how to get the workers back. A worker that still
re-runs a call (a guard this check cannot see) is stopped with one line that
says what to change.
"""

from __future__ import annotations

import ast
import functools
import inspect
import logging
import multiprocessing
import sys
from typing import Callable, Optional

logger = logging.getLogger(__name__)

MESSAGE = (
    "LibreYOLO: {action} was called at the top level of your script. DataLoader "
    "workers re-run that script on this platform, so put the call under "
    '`if __name__ == "__main__":`, or pass workers=0.'
)

FALLBACK_MESSAGE = (
    "{action} is called at the top level of your script, and DataLoader workers "
    "re-run that script on this platform, so it runs with workers=0. Put the call "
    'under `if __name__ == "__main__":` to load data in parallel.'
)

_warned: set[str] = set()


def _reimports_main() -> bool:
    """True when a new process re-runs the ``__main__`` script before its target.

    Follows ``multiprocessing.spawn``: ``spawn`` and ``forkserver`` children
    import ``__main__`` again by module name (skipped for a package's
    ``__main__``) or by file path; a REPL or notebook has neither.
    """
    method = (
        multiprocessing.get_start_method(allow_none=True)
        or multiprocessing.get_all_start_methods()[0]
    )
    if method not in ("spawn", "forkserver"):
        return False
    main = sys.modules.get("__main__")
    name = getattr(getattr(main, "__spec__", None), "name", None)
    if name is not None:
        return name != "__main__" and not name.endswith(".__main__")
    return getattr(main, "__file__", None) is not None


def _main_line() -> Optional[int]:
    """The line of the ``__main__`` script's top-level code that led to this call."""
    main_globals = getattr(sys.modules.get("__main__"), "__dict__", None)
    frame = inspect.currentframe()
    try:
        while frame is not None:
            if frame.f_code.co_name == "<module>" and frame.f_globals is main_globals:
                return frame.f_lineno
            frame = frame.f_back
        return None
    finally:
        del frame


def _is_main_test(node: ast.expr) -> bool:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        # ``if __name__ == "__main__" and ...:`` never runs in a re-import either.
        return any(_is_main_test(value) for value in node.values)
    if not (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
    ):
        return False
    sides = (node.left, node.comparators[0])
    return any(isinstance(n, ast.Name) and n.id == "__name__" for n in sides) and any(
        isinstance(n, ast.Constant) and n.value == "__main__" for n in sides
    )


def is_main_guarded(source, line: int) -> bool:
    """True when ``line`` of ``source`` sits in an ``if __name__ == "__main__":`` body."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.If) and node.body and _is_main_test(node.test):
            if node.body[0].lineno <= line <= node.body[-1].end_lineno:
                return True
    return False


def runs_unguarded() -> bool:
    """True when DataLoader workers would re-run the call that is running now.

    That is a main process under ``spawn`` or ``forkserver`` whose ``__main__``
    script reached this call from top-level code outside an
    ``if __name__ == "__main__":`` body. False whenever the script cannot be
    read or the call did not come from its top-level code.
    """
    if getattr(multiprocessing.current_process(), "_inheriting", False):
        return False
    if not _reimports_main():
        return False
    line = _main_line()
    path = getattr(sys.modules.get("__main__"), "__file__", None)
    if line is None or path is None:
        return False
    try:
        with open(path, "rb") as f:
            return not is_main_guarded(f.read(), line)
    except (OSError, SyntaxError, ValueError):
        return False


def warn_workers_dropped(action: str) -> None:
    """Log, once per action, that an unguarded call runs with ``workers=0``."""
    if action not in _warned:
        _warned.add(action)
        logger.warning(FALLBACK_MESSAGE.format(action=action))


def workers_for_main_process(action: str, workers):
    """The worker count a ``train()`` / ``val()`` call can use without failing.

    ``0`` when DataLoader workers would re-run the call (see
    :func:`runs_unguarded`); ``workers`` otherwise.
    """
    try:
        if workers is not None and int(workers) <= 0:
            return workers
    except (TypeError, ValueError):
        return workers
    if not runs_unguarded():
        return workers
    warn_workers_dropped(action)
    return 0


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
        takes_workers = "workers" in signature.parameters

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            inheriting = getattr(multiprocessing.current_process(), "_inheriting", False)
            if inheriting or (takes_workers and _reimports_main()):
                try:
                    bound = signature.bind_partial(*args, **kwargs)
                    bound.apply_defaults()
                except TypeError:
                    bound = None
                workers = bound.arguments.get("workers") if bound else None
                if inheriting:
                    exit_if_reimporting_main(action, workers)
                elif bound is not None:
                    resolved = workers_for_main_process(action, workers)
                    if resolved != workers:
                        bound.arguments["workers"] = resolved
                        return fn(*bound.args, **bound.kwargs)
            return fn(*args, **kwargs)

        wrapper._libreyolo_main_guarded = True
        return wrapper

    return decorate


__all__ = [
    "exit_if_reimporting_main",
    "guard_main_reimport",
    "is_main_guarded",
    "runs_unguarded",
    "warn_workers_dropped",
    "workers_for_main_process",
]
