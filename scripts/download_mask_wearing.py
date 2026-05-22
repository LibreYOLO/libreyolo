"""Download the LibreYOLO/mask-wearing-608pr dataset from HuggingFace.

Classes: mask (0), no-mask (1) — 2-class face detection task.
  train: 105 images  |  val: 29  |  test: 15

Usage:
    python scripts/download_mask_wearing.py
    python scripts/download_mask_wearing.py --dest ~/datasets/mask-wearing
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


HF_REPO = "LibreYOLO/mask-wearing-608pr"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
HF_API  = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main"

SPLITS = ["train", "valid", "test"]
SUBS   = ["images", "labels"]

DATA_YAML = """\
path: {dest}
train: train/images
val: valid/images
test: test/images
nc: 2
names:
  0: mask
  1: no-mask
"""


def list_files(split: str, sub: str) -> list[str]:
    url = f"{HF_API}/{split}/{sub}?limit=300"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return [f["path"] for f in resp.json()]


def download_file(rel_path: str, dest_root: Path) -> Path:
    out = dest_root / rel_path
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_BASE}/{rel_path}"
    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()
    with open(out, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default="~/datasets/mask-wearing", help="Output directory")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download workers")
    args = parser.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading mask-wearing-608pr → {dest}")

    # Collect all file paths
    all_files: list[str] = []
    for split in SPLITS:
        for sub in SUBS:
            try:
                files = list_files(split, sub)
                all_files.extend(files)
            except Exception as e:
                print(f"  Warning: could not list {split}/{sub}: {e}", file=sys.stderr)

    print(f"  {len(all_files)} files total")

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_file, p, dest): p for p in all_files}
        with tqdm(total=len(futures), unit="file") as pbar:
            for future in as_completed(futures):
                path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    failed.append((path, str(e)))
                finally:
                    pbar.update(1)

    # Write data.yaml
    yaml_path = dest / "data.yaml"
    yaml_path.write_text(DATA_YAML.format(dest=str(dest)))
    print(f"  data.yaml written to {yaml_path}")

    if failed:
        print(f"\n{len(failed)} files failed:")
        for p, e in failed:
            print(f"  {p}: {e}")
        sys.exit(1)

    print(f"\nDone. Dataset at {dest}")
    print(f"Use --data {yaml_path}")


if __name__ == "__main__":
    main()
