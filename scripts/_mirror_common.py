"""Shared helpers for the scripts/mirror_*.py staging tools.

Nothing here depends on the machine it runs on: staging goes to a temporary
directory (override with --staging), and the licence texts and .gitattributes
are fetched from their sources at run time rather than read out of a
pre-populated folder.
"""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

# LFS rules, taken from a weight repo that already has them so a new mirror
# stores its weights in LFS instead of as raw blobs.
GITATTRIBUTES_SOURCE = (
    "https://huggingface.co/LibreYOLO/LibreSegformerb5-sem/resolve/main/.gitattributes"
)


def fetch_text(url: str) -> str:
    """Fetch a text asset (a licence, .gitattributes) from its source."""
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def gitattributes() -> str:
    return fetch_text(GITATTRIBUTES_SOURCE)


def parse_args(
    description: str,
    sizes: list[str],
    *,
    create_staging: bool = True,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "sizes",
        nargs="*",
        default=sizes,
        help=f"sizes to stage (default: {' '.join(sizes)})",
    )
    parser.add_argument(
        "--staging",
        type=Path,
        default=None,
        help="where to build the repo directories (default: a temporary directory)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="stage only; do not create or push the Hugging Face repo",
    )
    args = parser.parse_args()
    if args.staging is None and create_staging:
        args.staging = Path(tempfile.mkdtemp(prefix="libreyolo-mirror-"))
    if create_staging:
        args.staging.mkdir(parents=True, exist_ok=True)
    return args


def check_five_file_contract(out: Path) -> list[str]:
    """The contract in skills/libreyolo-upload-hf-model: exactly five files."""
    files = sorted(p.name for p in out.iterdir() if p.is_file())
    if len(files) != 5:
        raise SystemExit(
            f"{out.name}: expected exactly 5 files, got {len(files)}: {files}"
        )
    return files
