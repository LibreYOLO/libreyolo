#!/usr/bin/env python3
"""LibreVocab1 Phase 2 training entry point.

Usage:
    python tools/train_librevocab1_phase2.py \
        --config configs/librevocab1/phase2_so400m_coco_smoketest.yaml

The config is loaded as a dict and passed to ``LibreVocab1Phase2Config``.
Any keys not in the dataclass are warned about and ignored — same pattern
as the rest of LibreYOLO's per-family configs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Allow ``python tools/train_*`` from repo root without ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train LibreVocab1 Phase 2.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a YAML config understood by LibreVocab1Phase2Config.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override the COCO root in the config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs in the config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the device in the config (cpu | cuda | cuda:0 | mps).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the model + loader + trainer; do not run any training step.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    log = logging.getLogger("librevocab1.phase2.train")

    if not args.config.is_file():
        raise FileNotFoundError(f"config not found: {args.config}")
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f) or {}
    if args.data_dir is not None:
        cfg_dict["data_dir"] = args.data_dir
    if args.epochs is not None:
        cfg_dict["epochs"] = args.epochs
    if args.device is not None:
        cfg_dict["device"] = args.device

    from libreyolo.training.config import LibreVocab1Phase2Config
    from libreyolo.models.librevocab1.phase2 import LibreVocab1Phase2
    from libreyolo.models.librevocab1.phase2.trainer import (
        build_phase2_trainer_from_config,
    )

    config = LibreVocab1Phase2Config.from_kwargs(**cfg_dict)
    log.info("Config: %s", config)

    log.info("Building LibreVocab1Phase2(size=%s) ...", config.size)
    model = LibreVocab1Phase2(size=config.size, device=config.device)
    log.info(
        "Model built. Trainable params: %.1fM",
        sum(p.numel() for p in model.model.parameters() if p.requires_grad) / 1e6,
    )

    text_encode_fn = model.model.backbone.encode_text
    trainer = build_phase2_trainer_from_config(
        config=config,
        model=model.model,
        text_encode_fn=text_encode_fn,
    )

    if args.dry_run:
        log.info("Dry-run: trainer built, no steps taken.")
        return 0

    log.info("Starting training: %d epochs, batch=%d, lr=%g", config.epochs, config.batch, config.lr0)
    result = trainer.fit(epochs=config.epochs)
    log.info(
        "Training done. epochs=%d steps=%d last_loss=%.4f",
        result.epochs_completed, result.last_step, result.last_loss,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
