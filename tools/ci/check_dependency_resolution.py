"""Check that every extra in pyproject.toml can be resolved (#989).

Two checks, both resolution only (nothing is downloaded beyond metadata,
nothing is installed):

1. ``uv lock --dry-run``: the universal lock a contributor gets from
   ``uv sync``, including the dev group and the [tool.uv] conflicts.
2. ``uv pip compile --no-sources`` of every extra, for each supported Python
   and platform. ``--no-sources`` drops the [tool.uv.sources] index override,
   so this is what ``pip install "libreyolo[<extra>]"`` sees. Resolving the
   local pyproject (not the PyPI name) means a broken extra fails here instead
   of making pip fall back to an older libreyolo release.

Both checks read live PyPI, so a new upstream release can break them without
any change on our side. That is the point: the scheduled run catches it.

Usage (from the repo root, needs uv on PATH):

    python tools/ci/check_dependency_resolution.py
    python tools/ci/check_dependency_resolution.py --extras all vlm --python 3.12
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]

PYTHONS = ("3.10", "3.11", "3.12", "3.13")

# name -> (uv --python-platform, extra environment)
PLATFORMS = {
    "linux-x86_64": ("x86_64-manylinux_2_28", {}),
    # glibc 2.35: Ubuntu 22.04 and JetPack 6. openvino ships no aarch64
    # wheels for older glibc on Python 3.12+.
    "linux-aarch64": ("aarch64-manylinux_2_35", {}),
    "windows-x86_64": ("x86_64-pc-windows-msvc", {}),
    # uv defaults to a macOS 13 target; onnxruntime ships macOS 14+ wheels.
    "macos-arm64": ("aarch64-apple-darwin", {"MACOSX_DEPLOYMENT_TARGET": "14.0"}),
}

# Extras whose macOS wheels need a newer macOS than the default target above.
MACOS_TARGET_OVERRIDES = {
    # Core AI is a macOS 26 framework; coreai-core ships macOS 26 wheels only.
    "coreai": "26.0",
}

# (extra, platform, python) -> reason. Only for combinations an upstream
# package has no build for and we cannot work around. "*" is a wildcard.
KNOWN_UNRESOLVABLE: dict[tuple[str, str, str], str] = {
    ("paddle", "*", "3.13"): (
        "paddlepaddle 2.6.2, the release X2Paddle 1.6.0 works with, "
        "ships no Python 3.13 wheels"
    ),
}


def _known(extra: str, platform: str, python: str) -> str | None:
    for (e, p, v), reason in KNOWN_UNRESOLVABLE.items():
        if e in ("*", extra) and p in ("*", platform) and v in ("*", python):
            return reason
    return None


def _reason(output: str) -> str:
    match = re.search(r"Because .*?(?:\.\s|$)", output, re.DOTALL)
    if match:
        text = match.group(0)
    else:
        lines = output.strip().splitlines()
        text = lines[-1] if lines else "no output"
    return " ".join(text.split())[:300]


def check_lock() -> str | None:
    proc = subprocess.run(
        ["uv", "lock", "--dry-run"],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return None
    return _reason(proc.stdout + proc.stderr)


def compile_extra(extra: str, platform: str, python: str) -> str | None:
    target, extra_env = PLATFORMS[platform]
    if platform.startswith("macos") and extra in MACOS_TARGET_OVERRIDES:
        extra_env = {
            **extra_env,
            "MACOSX_DEPLOYMENT_TARGET": MACOS_TARGET_OVERRIDES[extra],
        }
    proc = subprocess.run(
        [
            "uv",
            "pip",
            "compile",
            "--no-sources",
            "--quiet",
            "--python-version",
            python,
            "--python-platform",
            target,
            "pyproject.toml",
            "--extra",
            extra,
        ],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **extra_env},
    )
    if proc.returncode == 0:
        return None
    return _reason(proc.stdout + proc.stderr)


def main() -> int:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    all_extras = sorted(pyproject["project"]["optional-dependencies"])

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extras", nargs="+", default=all_extras)
    parser.add_argument("--python", nargs="+", default=list(PYTHONS))
    parser.add_argument("--platform", nargs="+", default=list(PLATFORMS))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--skip-lock", action="store_true")
    args = parser.parse_args()

    unknown = sorted(set(args.extras) - set(all_extras))
    if unknown:
        parser.error(f"unknown extras: {', '.join(unknown)}")

    failures: list[str] = []
    stale: list[str] = []

    if not args.skip_lock:
        lock_error = check_lock()
        print(f"uv lock --dry-run: {'ok' if lock_error is None else 'FAIL'}")
        if lock_error is not None:
            failures.append(f"uv lock: {lock_error}")

    combos = [
        (extra, platform, python)
        for extra in args.extras
        for platform in args.platform
        for python in args.python
    ]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda c: compile_extra(*c), combos))

    unresolved = 0
    for (extra, platform, python), error in zip(combos, results):
        label = f"libreyolo[{extra}] / {platform} / Python {python}"
        known = _known(extra, platform, python)
        unresolved += error is not None
        if error is None and known is not None:
            stale.append(f"{label}: resolves now, drop it from KNOWN_UNRESOLVABLE")
        elif error is not None and known is None:
            failures.append(f"{label}: {error}")

    print(f"pip resolution: {len(combos) - unresolved}/{len(combos)} resolve")
    for line in stale:
        print(f"note: {line}")
    for line in failures:
        print(f"FAIL {line}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
