"""The VLM fine-tune checkpoint contract.

A VLM checkpoint is a directory, not a ``.pt``: the PEFT adapter tensors, the
processor snapshot, and ``libreyolo_vlm.json``, the contract file that makes
the directory loadable by ``LibreVLM(path)``. Base weights are never copied
into a checkpoint; the contract records which base repo (and pinned revision)
to resolve, so checkpoints stay megabytes and LibreYOLO never redistributes
upstream weights.

``libreyolo_vlm.json`` fields (schema 1):

- ``schema``: contract version, integer.
- ``family`` / ``size``: adapter class resolution keys.
- ``base_repo`` / ``base_revision``: the exact base the adapter was trained on.
- ``names``: ordered training vocabulary; pre-applied on load.
- ``bbox_key`` / ``coord_divisor`` / ``box_format`` / ``prompt``: the output
  convention the fine-tune was trained to emit, recorded for auditability.
- ``task``: always ``detect`` today.
- ``metrics``: final metrics of the producing run.
- ``libreyolo_version``: producer version string.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CONTRACT_FILENAME = "libreyolo_vlm.json"
CONTRACT_SCHEMA = 1

__all__ = [
    "CONTRACT_FILENAME",
    "is_vlm_checkpoint",
    "read_contract",
    "save_vlm_checkpoint",
]


def is_vlm_checkpoint(path) -> bool:
    """True when ``path`` is a directory carrying the VLM checkpoint contract."""
    try:
        p = Path(path)
    except TypeError:
        return False
    return p.is_dir() and (p / CONTRACT_FILENAME).is_file()


def read_contract(path) -> Dict[str, Any]:
    """Read and minimally validate a checkpoint's contract file."""
    contract_path = Path(path) / CONTRACT_FILENAME
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unreadable VLM checkpoint contract {contract_path}: {exc}") from exc
    schema = contract.get("schema")
    if schema != CONTRACT_SCHEMA:
        raise ValueError(
            f"VLM checkpoint {path} uses contract schema {schema!r}; this "
            f"LibreYOLO understands schema {CONTRACT_SCHEMA}. Upgrade libreyolo."
        )
    for key in ("family", "size", "names"):
        if key not in contract:
            raise ValueError(f"VLM checkpoint contract {contract_path} is missing {key!r}.")
    return contract


def save_vlm_checkpoint(
    directory,
    *,
    peft_model,
    processor,
    wrapper,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write one checkpoint directory: adapter + processor + contract."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(directory))
    processor.save_pretrained(str(directory))

    try:
        from libreyolo import __version__ as libreyolo_version
    except ImportError:  # pragma: no cover
        libreyolo_version = "unknown"

    contract = {
        "schema": CONTRACT_SCHEMA,
        "family": wrapper.FAMILY,
        "size": wrapper.size,
        "base_repo": wrapper.HF_REPOS[wrapper.size],
        "base_revision": wrapper.HF_REVISIONS.get(wrapper.size),
        "names": [wrapper.names[i] for i in range(len(wrapper.names))],
        "bbox_key": wrapper.BBOX_KEY,
        "coord_divisor": float(wrapper.COORD_DIVISOR),
        "box_format": wrapper.BOX_FORMAT,
        "prompt": wrapper._detection_prompt(),
        "task": "detect",
        "metrics": dict(metrics or {}),
        "libreyolo_version": libreyolo_version,
    }
    (directory / CONTRACT_FILENAME).write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return directory
