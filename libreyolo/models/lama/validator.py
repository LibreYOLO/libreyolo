"""Mask-aware paired-image validation for LibreLaMa."""

from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ...data.restore_dataset import (
    RestoreDataset,
    img2restore_target_paths,
    resolve_restore_data,
)
from ...validation.restore_validator import RestoreValidator
from .nn import ONNX_INPUT_SIZE


class LaMaValidationDataset(Dataset):
    """Return aligned source, target, and mask paths for one restore split.

    The normal restoration pairing rules resolve sources and targets. An
    explicit ``mask_dir`` YAML key adds a third directory with the same stems:

    ``inputs/val/a.png`` -> ``targets/val/a.png`` + ``masks/val/a.png``.
    """

    def __init__(self, data_config: dict, split: str) -> None:
        mask_dir = data_config.get("mask_dir")
        if not isinstance(mask_dir, str) or not mask_dir.strip():
            raise ValueError(
                "LibreLaMa validation requires a non-empty 'mask_dir' key in "
                "the restore dataset YAML (for example, mask_dir: masks)."
            )

        paired = RestoreDataset(
            data_config,
            split=split,
            imgsz=ONNX_INPUT_SIZE,
            augment=False,
            scale=1,
        )
        self.source_files = paired.img_files
        self.target_files = paired.target_files
        self.mask_files = img2restore_target_paths(
            self.source_files,
            input_dir=paired.input_dir,
            target_dir=mask_dir.strip(),
            stem_suffixes=("",),
        )
        missing = [str(path) for path in self.mask_files if not path.exists()]
        if missing:
            preview = ", ".join(missing[:3])
            raise FileNotFoundError(
                f"{len(missing)} inpainting mask file(s) missing for split "
                f"'{split}' (e.g. {preview}). Expected masks under "
                f"'{mask_dir.strip()}' with stems matching the input images."
            )

    def __len__(self) -> int:
        return len(self.source_files)

    def __getitem__(self, index: int) -> tuple[str, str, str, int]:
        return (
            str(self.source_files[index]),
            str(self.target_files[index]),
            str(self.mask_files[index]),
            index,
        )


def lama_validation_collate(batch):
    """Keep path records as lists; every item is predicted independently."""

    sources, targets, masks, indices = zip(*batch)
    return list(sources), list(targets), list(masks), list(indices)


def _load_rgb_tensor(path: str) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return (
        torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
    )


class LaMaRestoreValidator(RestoreValidator):
    """Evaluate mask-guided inpainting through the public predict API."""

    def _setup_dataloader(self) -> DataLoader:
        if not self.config.data:
            raise ValueError("LibreLaMa validation requires data= (a dataset YAML).")
        if self.config.augment:
            raise ValueError(
                "LibreLaMa validation does not support augmented inference; "
                "use augment=False."
            )
        size = self.config.imgsz
        size_hw = (
            tuple(int(value) for value in size)
            if isinstance(size, (tuple, list))
            else (int(size), int(size))
        )
        if size_hw != (ONNX_INPUT_SIZE, ONNX_INPUT_SIZE):
            raise ValueError(
                "LibreLaMa validation uses the fixed 512x512 ONNX graph; "
                f"imgsz={size!r} is not supported."
            )

        data_config = resolve_restore_data(
            self.config.data,
            allow_scripts=getattr(self.config, "allow_download_scripts", False),
        )
        dataset = LaMaValidationDataset(
            data_config,
            split=self.config.split or "val",
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=lama_validation_collate,
        )

    def _warmup_model(self, n_warmup: int = 3) -> None:
        # A raw warmup tensor cannot carry the original-canvas mask context.
        # The first public predict call initializes the lazy runtime session.
        del n_warmup

    def _run_validation(self) -> None:
        if self.config.augment:
            raise ValueError(
                "LibreLaMa validation does not support augmented inference; "
                "use augment=False."
            )
        self.model.model.eval()
        progress = tqdm(
            self.dataloader,
            desc="Validating",
            total=len(self.dataloader),
            disable=not self.config.verbose,
        )
        total_start = time.time()

        with torch.no_grad():
            for sources, targets, masks, indices in progress:
                for source, target_path, mask, index in zip(
                    sources, targets, masks, indices
                ):
                    preprocess_start = time.time()
                    target = _load_rgb_tensor(target_path)
                    self.speed["preprocess"] += time.time() - preprocess_start

                    inference_start = time.time()
                    result = self.model.predict(
                        source,
                        mask=mask,
                        imgsz=ONNX_INPUT_SIZE,
                        device=str(self.device),
                    )
                    self.speed["inference"] += time.time() - inference_start

                    postprocess_start = time.time()
                    if isinstance(result, list):
                        if len(result) != 1:
                            raise RuntimeError(
                                "LibreLaMa validation expected one result per "
                                f"source, got {len(result)} for {source}."
                            )
                        result = result[0]
                    restored = getattr(result, "restored", None)
                    if restored is None:
                        raise RuntimeError(
                            "LibreLaMa predict returned no restored image for "
                            f"{source}."
                        )
                    prediction = (
                        torch.from_numpy(np.array(restored.array, copy=True))
                        .permute(2, 0, 1)
                        .float()
                        .div_(255.0)
                    )
                    if prediction.shape != target.shape:
                        raise ValueError(
                            "LibreLaMa validation pair shape mismatch after "
                            f"prediction: source {source} produced "
                            f"{tuple(prediction.shape[-2:])}, but target "
                            f"{target_path} is {tuple(target.shape[-2:])}."
                        )
                    info = {
                        "orig_shape": tuple(int(v) for v in target.shape[-2:]),
                        "target_shape": tuple(int(v) for v in target.shape[-2:]),
                        "img_path": source,
                        "target_path": target_path,
                        "mask_path": mask,
                    }
                    self._update_metrics(
                        prediction.unsqueeze(0),
                        target.unsqueeze(0),
                        [info],
                        [index],
                    )
                    self.speed["postprocess"] += time.time() - postprocess_start
                    self.seen += 1

        self.speed["total"] = time.time() - total_start


__all__ = [
    "LaMaRestoreValidator",
    "LaMaValidationDataset",
    "lama_validation_collate",
]
