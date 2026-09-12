"""Install/distribution smoke checks for an installed LibreYOLO package."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.resources
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
    import cloudpickle
    import libreyolo
    from libreyolo import (
        EdgeMap,
        EdgeValidator,
        FaceGallery,
        Gallery,
        LibreDexiNed,
        LibreTEED,
        LibreYOLO,
        Results,
        SAMPLE_IMAGE,
    )

    if not callable(LibreYOLO):
        raise AssertionError("LibreYOLO import did not resolve to a callable")
    if Results.__name__ != "Results":
        raise AssertionError("Results import did not resolve correctly")
    if not callable(cloudpickle.dumps):
        raise AssertionError("cloudpickle DDP transport dependency is unavailable")

    # This private file is launched by its resolved packaged path during
    # automatic DDP, so it must be present in wheels and sdists, not only
    # checkouts.
    from libreyolo.training._ddp_coordinator import (
        JOB_PROTOCOL,
        JOB_PROTOCOL_VERSION,
    )

    if JOB_PROTOCOL != "libreyolo.ddp.local-job" or JOB_PROTOCOL_VERSION != 2:
        raise AssertionError("Automatic-DDP coordinator protocol is unavailable")
    if FaceGallery is not Gallery:
        raise AssertionError("FaceGallery did not resolve to the Gallery alias")
    if "Gallery" not in libreyolo.__all__ or "FaceGallery" not in libreyolo.__all__:
        raise AssertionError("Gallery exports are missing from libreyolo.__all__")
    for public_type in (EdgeMap, EdgeValidator, LibreTEED, LibreDexiNed):
        if not isinstance(public_type, type):
            raise AssertionError(
                f"Edge public API did not resolve to a class: {public_type!r}"
            )
    for package_name in (
        "libreyolo.models.teed",
        "libreyolo.models.dexined",
        "libreyolo.models.modus",
    ):
        notice = importlib.resources.files(package_name).joinpath("NOTICE")
        if not notice.is_file():
            raise AssertionError(f"Family NOTICE is missing from {package_name}")

    # The LibreVLM tier exposes a factory plus family classes via lazy imports
    # (transformers is not required to import these names).
    from libreyolo import (
        LibreFlorence2,
        LibreInternVL3,
        LibreKosmos2,
        LibreLFM2VL,
        LibreMODUS,
        LibreModus,
        LibreQwen3VL,
        LibreSmolVLM2,
        LibreVLM,
    )

    from libreyolo import (
        LibreGround,
        LibreGroundFlorence2,
        LibreGroundQwen3VL,
        LibreShowUI,
    )

    if not callable(LibreGround):
        raise AssertionError("LibreGround import did not resolve to a callable")
    for family in (
        LibreShowUI,
        LibreGroundFlorence2,
        LibreGroundQwen3VL,
    ):
        if not isinstance(family, type):
            raise AssertionError(
                f"Ground family export did not resolve to a class: {family!r}"
            )

    if not callable(LibreVLM):
        raise AssertionError("LibreVLM import did not resolve to a callable")
    # The LibreVLA tier is importable without the vla extra (lerobot loads lazily).
    from libreyolo import Actions, LibreSmolVLA, LibreVLA, LibreXVLA

    if not callable(LibreVLA):
        raise AssertionError("LibreVLA import did not resolve to a callable")
    if not all(isinstance(cls, type) for cls in (LibreSmolVLA, LibreXVLA, Actions)):
        raise AssertionError("LibreVLA family/payload exports did not resolve to classes")
    for family in (
        LibreQwen3VL,
        LibreLFM2VL,
        LibreSmolVLM2,
        LibreInternVL3,
        LibreFlorence2,
        LibreKosmos2,
        LibreMODUS,
    ):
        if not isinstance(family, type):
            raise AssertionError(
                f"VLM family export did not resolve to a class: {family!r}"
            )
    if LibreModus is not LibreMODUS:
        raise AssertionError("LibreModus compatibility alias did not resolve correctly")

    # LibreLLM is a lazy extra: importing the class must not require openai.
    from libreyolo import LibreLLM

    if not isinstance(LibreLLM, type):
        raise AssertionError(
            f"LibreLLM import did not resolve to a class: {LibreLLM!r}"
        )

    # The open-vocabulary detector tier follows the same lazy-import rule.
    from libreyolo import (
        LibreGroundingDINO,
        LibreOMDetTurbo,
        LibreOpenVocab,
        LibreOWLv2,
    )

    if not callable(LibreOpenVocab):
        raise AssertionError("LibreOpenVocab import did not resolve to a callable")
    for family in (LibreGroundingDINO, LibreOWLv2, LibreOMDetTurbo):
        if not isinstance(family, type):
            raise AssertionError(
                "Open-vocabulary detector export did not resolve to a class: "
                f"{family!r}"
            )

    # Promptable-segmentation exports are lazy and importable without installing
    # the optional model runtime itself.
    from libreyolo import LibreEdgeTAM, LibreSAM, LibreSAM1, LibreSAM2, LibreSAM3

    if not callable(LibreSAM):
        raise AssertionError("LibreSAM import did not resolve to a callable")
    for family in (LibreSAM1, LibreSAM2, LibreEdgeTAM, LibreSAM3):
        if not isinstance(family, type):
            raise AssertionError(
                f"SAM family export did not resolve to a class: {family!r}"
            )

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
