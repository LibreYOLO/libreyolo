"""Check that every extra in pyproject.toml can be installed from wheels (#989).

Three checks, all resolution only (nothing is downloaded beyond metadata,
nothing is installed):

1. ``uv lock --dry-run``: the universal lock a contributor gets from
   ``uv sync``, including the dev group and the [tool.uv] conflicts.
2. ``uv pip compile --no-sources`` of every extra, for each supported Python
   and platform. ``--no-sources`` drops the [tool.uv.sources] index override,
   so this is what ``pip install "libreyolo[<extra>]"`` sees. Resolving the
   local pyproject (not the PyPI name) means a broken extra fails here instead
   of making pip fall back to an older libreyolo release.
3. For each of those resolutions, whether every pinned version has a wheel for
   that platform. pip takes the newest matching version even when it only has
   an sdist and then compiles it, which fails on machines without the right
   toolchain (pycuda needed the CUDA toolkit headers, onnx-simplifier a C++
   build). Packages that are fine to build from source are listed in
   KNOWN_SOURCE_BUILDS with the reason.

All checks read live PyPI, so a new upstream release can break them without
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
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]

PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")

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
# Prefer an environment marker in pyproject.toml that skips the dependency
# there, so the extra still installs the current release.
KNOWN_UNRESOLVABLE: dict[tuple[str, str, str], str] = {}

# (package, platform, python) -> reason. Source builds we accept: either no
# compiler is involved, or an upstream package pins a version without wheels
# and we cannot avoid it. The reason says which. "*" is a wildcard.
_GEVENT = (
    "gevent publishes no wheels for this Python and platform "
    "(tritonclient[http]); builds with a C compiler"
)
_ONNX2TF_PIN = (
    "pinned exactly by onnx2tf (tflite), no aarch64 wheel for that version on "
    "Python 3.12+; TFLite on Linux ARM needs a C/C++ toolchain"
)
_FIFTYONE_DB = (
    "the Linux build of fiftyone-db downloads MongoDB at install; no compiler "
    "(the fiftyone unit-test job installs it)"
)
KNOWN_SOURCE_BUILDS: dict[tuple[str, str, str], str] = {
    ("tensorrt-cu12", "*", "*"): (
        "pure-Python installer that downloads NVIDIA's wheels; no compiler"
    ),
    ("tensorrt-cu12-libs", "*", "*"): (
        "pure-Python installer that downloads NVIDIA's wheels; no compiler"
    ),
    ("docopt", "*", "*"): "pure Python, published as an sdist only (lerobot)",
    ("antlr4-python3-runtime", "*", "*"): "pure Python, published as an sdist only",
    ("glob2", "*", "*"): "pure Python, published as an sdist only (fiftyone)",
    ("fiftyone-db", "linux-x86_64", "*"): _FIFTYONE_DB,
    ("fiftyone-db", "linux-aarch64", "*"): _FIFTYONE_DB,
    ("gevent", "macos-arm64", "3.10"): _GEVENT,
    ("gevent", "macos-arm64", "3.11"): _GEVENT,
    ("gevent", "linux-aarch64", "3.10"): _GEVENT,
    ("onnxoptimizer", "linux-aarch64", "*"): _ONNX2TF_PIN,
    ("psutil", "linux-aarch64", "*"): _ONNX2TF_PIN,
    ("pymongo", "*", "3.14"): (
        "fiftyone pins pymongo<4.10, which has no Python 3.14 wheels; its C "
        "extensions are optional and it installs as pure Python without a "
        "compiler"
    ),
    ("coremltools", "linux-aarch64", "*"): (
        "ExecuTorch 1.2 itself requires coremltools on Linux, which has no "
        "aarch64 wheels"
    ),
}

_NO_WHEEL = re.compile(
    r"Because ([A-Za-z0-9_.\-]+)(?:\{[^}]*\})?(?:[<>=!~][^ ]*)? has no usable wheels"
)


# Source builds accepted only for one extra: (extra, package, platform, python).
KNOWN_EXTRA_SOURCE_BUILDS: dict[tuple[str, str, str, str], str] = {
    ("vla", "numpy", "*", "3.14"): (
        "lerobot (vla) pins numpy<2.3, which has no Python 3.14 wheels; "
        "builds with a C compiler"
    ),
}


def _known(table: dict, *key: str) -> str | None:
    for pattern, reason in table.items():
        if all(p in ("*", k) for p, k in zip(pattern, key)):
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


def _compile(extra: str, platform: str, python: str, *options: str):
    target, extra_env = PLATFORMS[platform]
    if platform.startswith("macos") and extra in MACOS_TARGET_OVERRIDES:
        extra_env = {
            **extra_env,
            "MACOSX_DEPLOYMENT_TARGET": MACOS_TARGET_OVERRIDES[extra],
        }
    return subprocess.run(
        [
            "uv",
            "pip",
            "compile",
            "--no-sources",
            "--quiet",
            "--no-header",
            "--no-annotate",
            "--python-version",
            python,
            "--python-platform",
            target,
            *options,
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


def _source_builds(extra: str, platform: str, python: str, pins: str) -> list[str]:
    """Pinned packages without a wheel for this target, found one at a time."""
    found: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        constraints = Path(tmp) / "pins.txt"
        constraints.write_text(pins)
        for _ in range(20):
            allowed = [arg for pkg in found for arg in ("--no-binary", pkg)]
            proc = _compile(
                extra,
                platform,
                python,
                "--only-binary",
                ":all:",
                *allowed,
                "--constraint",
                str(constraints),
            )
            if proc.returncode == 0:
                return found
            # uv wraps long lines, which can split a marker or the message.
            match = _NO_WHEEL.search(" ".join(proc.stderr.split()))
            if match is None:
                return found + [f"<unparsed: {_reason(proc.stderr)}>"]
            found.append(match.group(1).lower())
    return found


def check_extra(extra: str, platform: str, python: str):
    """Return (resolution error or None, packages that would build from source)."""
    proc = _compile(extra, platform, python)
    if proc.returncode != 0:
        return _reason(proc.stdout + proc.stderr), []
    return None, _source_builds(extra, platform, python, proc.stdout)


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
    notes: list[str] = []

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
        results = list(pool.map(lambda c: check_extra(*c), combos))

    unresolved = 0
    from_source = 0
    for (extra, platform, python), (error, builds) in zip(combos, results):
        label = f"libreyolo[{extra}] / {platform} / Python {python}"
        known = _known(KNOWN_UNRESOLVABLE, extra, platform, python)
        unresolved += error is not None
        if error is None and known is not None:
            notes.append(f"{label}: resolves now, drop it from KNOWN_UNRESOLVABLE")
        elif error is not None and known is None:
            failures.append(f"{label}: {error}")
        unexpected = [
            pkg
            for pkg in builds
            if _known(KNOWN_SOURCE_BUILDS, pkg, platform, python) is None
            and _known(KNOWN_EXTRA_SOURCE_BUILDS, extra, pkg, platform, python) is None
        ]
        from_source += bool(unexpected)
        if unexpected:
            failures.append(
                f"{label}: no wheel, pip would build from source: "
                + ", ".join(unexpected)
            )

    print(f"pip resolution: {len(combos) - unresolved}/{len(combos)} resolve")
    print(
        f"wheels: {len(combos) - unresolved - from_source}/{len(combos) - unresolved}"
        " resolved installs need no unexpected source build"
    )
    for line in notes:
        print(f"note: {line}")
    for line in failures:
        print(f"FAIL {line}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
