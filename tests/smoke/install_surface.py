"""Install/distribution smoke checks for an installed LibreYOLO package."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


INSTALL_SMOKE_SUITE_VERSION = "1.0"
COMMAND_TIMEOUT_SECONDS = 60


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        joined = " ".join(command)
        raise AssertionError(
            f"Command failed with exit code {result.returncode}: {joined}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def _load_json_stdout(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Expected JSON stdout, got:\n{result.stdout}") from exc
    if not isinstance(data, dict):
        raise AssertionError(f"Expected JSON object, got {type(data).__name__}")
    return data


def _check_import_surface(expect_source: str, source_root: Path | None) -> None:
    import libreyolo
    from libreyolo import LibreYOLO, Results, SAMPLE_IMAGE

    if not callable(LibreYOLO):
        raise AssertionError("LibreYOLO import did not resolve to a callable")
    if Results.__name__ != "Results":
        raise AssertionError("Results import did not resolve correctly")

    package_version = importlib.metadata.version("libreyolo")
    if package_version != libreyolo.__version__:
        raise AssertionError(
            "Package metadata version does not match libreyolo.__version__: "
            f"{package_version!r} != {libreyolo.__version__!r}"
        )

    sample_image = Path(SAMPLE_IMAGE)
    if not sample_image.is_file():
        raise AssertionError(f"SAMPLE_IMAGE does not point to a file: {SAMPLE_IMAGE}")
    if sample_image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise AssertionError(f"SAMPLE_IMAGE has unexpected suffix: {sample_image}")

    if source_root is not None and expect_source != "any":
        package_file = Path(libreyolo.__file__).resolve()
        source_root = source_root.resolve()
        is_inside = _is_relative_to(package_file, source_root)
        if expect_source == "inside" and not is_inside:
            raise AssertionError(
                f"Expected editable import from {source_root}, got {package_file}"
            )
        if expect_source == "outside" and is_inside:
            raise AssertionError(
                f"Expected installed import outside {source_root}, got {package_file}"
            )


def _check_cli_surface() -> None:
    executable = shutil.which("libreyolo")
    if executable is None:
        raise AssertionError("Console script 'libreyolo' was not found on PATH")

    help_result = _run([executable, "--help"])
    if "libreyolo" not in help_result.stdout.lower():
        raise AssertionError("`libreyolo --help` did not mention libreyolo")

    version = _load_json_stdout(_run([executable, "version", "--json", "--quiet"]))
    for key in ("version", "python", "torch", "schema_version"):
        if key not in version:
            raise AssertionError(f"`libreyolo version --json` missing key: {key}")

    checks = _load_json_stdout(_run([executable, "checks", "--json", "--quiet"]))
    for key in ("python", "torch", "packages", "schema_version"):
        if key not in checks:
            raise AssertionError(f"`libreyolo checks --json` missing key: {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Repository root used to verify whether imports come from source.",
    )
    parser.add_argument(
        "--expect-source",
        choices=("any", "inside", "outside"),
        default="any",
        help="Expected import location relative to --source-root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"LibreYOLO install-smoke suite v{INSTALL_SMOKE_SUITE_VERSION}")
    _check_import_surface(args.expect_source, args.source_root)
    _check_cli_surface()
    print("Install-smoke surface checks passed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Install-smoke failed: {exc}", file=sys.stderr)
        raise
