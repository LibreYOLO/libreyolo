"""Checkpoint directory contract for the LibreVLA tier.

A VLA checkpoint is a directory (ADR 0028): the upstream policy files
(``config.json`` plus ``model.safetensors``), the saved pre and post
processor pipelines, and ``libreyolo_vla.json``, the contract file that lets
``LibreVLA(path)`` rebuild the right family without guessing.

``libreyolo_vla.json`` fields (schema 1):

- ``schema``: 1
- ``family``: LibreVLA family id (``smolvla``)
- ``size``: family size code the fine-tune started from
- ``base_repo`` / ``base_revision``: the pinned base the adapter was built on
- ``data``: dataset repo id or path the fine-tune used
- ``fps``: control rate of the dataset, or null
- ``cameras``: dataset camera names in slot order (what ``predict`` expects)
- ``action_names`` / ``state_names``: per-dimension names, or null
- ``chunk_size``: action horizon of the policy
- ``libreyolo_version``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

CONTRACT_FILENAME = "libreyolo_vla.json"
CONTRACT_SCHEMA = 1

__all__ = [
    "CONTRACT_FILENAME",
    "CONTRACT_SCHEMA",
    "is_vla_checkpoint",
    "read_contract",
    "write_contract",
]


def is_vla_checkpoint(path: Any) -> bool:
    """True when ``path`` is a directory carrying the contract file."""
    try:
        p = Path(path)
    except TypeError:
        return False
    return p.is_dir() and (p / CONTRACT_FILENAME).is_file()


def read_contract(path: Any) -> Dict[str, Any]:
    """Load and validate the contract file of a checkpoint directory."""
    contract_path = Path(path) / CONTRACT_FILENAME
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{path} is not a LibreVLA checkpoint: missing {CONTRACT_FILENAME}."
        ) from exc
    schema = contract.get("schema")
    if schema != CONTRACT_SCHEMA:
        raise ValueError(
            f"{contract_path} has schema {schema!r}; this LibreYOLO understands "
            f"schema {CONTRACT_SCHEMA}. Upgrade libreyolo."
        )
    for key in ("family", "size"):
        if not contract.get(key):
            raise ValueError(f"{contract_path} is missing the {key!r} field.")
    return contract


def write_contract(
    directory: Any,
    *,
    family: str,
    size: str,
    base_repo: Optional[str],
    base_revision: Optional[str],
    data: Optional[str],
    fps: Optional[float],
    cameras: Optional[list],
    action_names: Optional[list],
    state_names: Optional[list],
    chunk_size: Optional[int],
) -> Path:
    """Write the contract file into ``directory`` and return its path."""
    from ... import __version__

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": CONTRACT_SCHEMA,
        "family": family,
        "size": size,
        "base_repo": base_repo,
        "base_revision": base_revision,
        "data": data,
        "fps": fps,
        "cameras": list(cameras) if cameras is not None else None,
        "action_names": list(action_names) if action_names is not None else None,
        "state_names": list(state_names) if state_names is not None else None,
        "chunk_size": int(chunk_size) if chunk_size is not None else None,
        "libreyolo_version": __version__,
    }
    path = directory / CONTRACT_FILENAME
    path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path
