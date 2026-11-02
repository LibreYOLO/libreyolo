"""Locate command-line tools installed into the running Python environment."""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from pathlib import Path


def environment_script(name: str) -> Path | None:
    """Return the console script ``name`` from this interpreter's environment.

    Converters such as ``mnnconvert`` and ExecuTorch's ``flatc`` install
    beside the environment's ``python``, which is on ``PATH`` only when the
    virtualenv is activated. IDE run buttons, Jupyter kernels and
    ``venv/bin/python script.py`` do not activate it. ``sysconfig`` gives that
    directory directly; resolving ``sys.executable`` would follow the venv's
    symlink to the base interpreter and miss it.
    """
    filename = f"{name}.exe" if os.name == "nt" else name
    for directory in (sysconfig.get_path("scripts"), Path(sys.executable).parent):
        candidate = Path(directory) / filename
        if candidate.is_file():
            return candidate
    found = shutil.which(name)
    return Path(found) if found else None
