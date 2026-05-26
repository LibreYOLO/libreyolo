"""Run clean install-smoke modes in a temporary virtual environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


VALID_MODES = ("editable", "wheel", "sdist", "pypi")


def _venv_paths(venv_dir: Path) -> tuple[Path, Path]:
    if os.name == "nt":
        scripts_dir = venv_dir / "Scripts"
        return scripts_dir / "python.exe", scripts_dir
    scripts_dir = venv_dir / "bin"
    return scripts_dir / "python", scripts_dir


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _single_artifact(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one artifact matching {dist_dir / pattern}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _install_mode(mode: str, python: Path, source_root: Path, work_dir: Path) -> None:
    _run([str(python), "-m", "pip", "install", "-U", "pip"])

    if mode == "editable":
        _run([str(python), "-m", "pip", "install", "-e", str(source_root)])
        return

    if mode == "pypi":
        _run([str(python), "-m", "pip", "install", "libreyolo"])
        return

    _run([str(python), "-m", "pip", "install", "-U", "build"])
    dist_dir = work_dir / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(python),
            "-m",
            "build",
            f"--{mode}",
            "--outdir",
            str(dist_dir),
            str(source_root),
        ]
    )
    artifact = _single_artifact(dist_dir, "*.whl" if mode == "wheel" else "*.tar.gz")
    _run([str(python), "-m", "pip", "install", str(artifact)])


def run_install_smoke(mode: str, source_root: Path, keep_work_dir: bool) -> None:
    source_root = source_root.resolve()
    surface_script = source_root / "tests" / "smoke" / "install_surface.py"
    if not surface_script.is_file():
        raise FileNotFoundError(surface_script)

    temp_root = Path(tempfile.mkdtemp(prefix=f"libreyolo-install-smoke-{mode}-"))
    try:
        venv_dir = temp_root / "venv"
        _run([sys.executable, "-m", "venv", str(venv_dir)])
        python, scripts_dir = _venv_paths(venv_dir)

        _install_mode(mode, python, source_root, temp_root)
        _run([str(python), "-m", "pip", "check"])

        smoke_env = os.environ.copy()
        smoke_env["PATH"] = str(scripts_dir) + os.pathsep + smoke_env.get("PATH", "")
        expect_source = "inside" if mode == "editable" else "outside"
        _run(
            [
                str(python),
                str(surface_script),
                "--source-root",
                str(source_root),
                "--expect-source",
                expect_source,
            ],
            cwd=temp_root,
            env=smoke_env,
        )
    finally:
        if keep_work_dir:
            print(f"Kept install-smoke work dir: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=VALID_MODES, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.cwd(),
        help="LibreYOLO repository root for editable/build modes.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep the temporary venv/dist directory for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_install_smoke(args.mode, args.source_root, args.keep_work_dir)


if __name__ == "__main__":
    main()
