"""Paired RGB images and linear-RGB albedo arrays on matching canvases."""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import get_img_files, load_data_config


def resolve_albedo_data(data, allow_scripts=False):
    return load_data_config(str(data), allow_scripts=allow_scripts)


class AlbedoDataset(Dataset):
    """Same-stem ``images/<split>`` RGB and ``albedo/<split>`` float32 NPY.

    Targets are HWC linear RGB in [0,1]; image and target shapes must match.
    A fixed validation/training canvas stretches RGB and albedo together.
    """

    def __init__(self, data_config, split, imgsz, augment=False):
        value = data_config.get(split)
        if not value:
            raise ValueError(f"Albedo dataset has no {split!r} split.")
        self.img_files = [
            Path(p)
            for p in (data_config.get(f"{split}_img_files") or get_img_files(value))
        ]
        if not self.img_files:
            raise FileNotFoundError(f"No albedo input images in split {split!r}.")
        self.imgsz = int(imgsz)
        if self.imgsz <= 0:
            raise ValueError("Albedo dataset imgsz must be positive.")
        self.augment = bool(augment)
        input_dir = str(data_config.get("input_dir", "images"))
        target_dir = str(data_config.get("albedo_dir", "albedo"))
        if any(
            len(Path(part).parts) != 1 or part in {".", ".."}
            for part in (input_dir, target_dir)
        ):
            raise ValueError("input_dir and albedo_dir must be single directory names.")
        self.target_files = []
        for image in self.img_files:
            if input_dir not in image.parts:
                raise ValueError(f"Input image is not under {input_dir!r}: {image}")
            index = len(image.parts) - 1 - image.parts[::-1].index(input_dir)
            target = Path(
                *image.parts[:index], target_dir, *image.parts[index + 1 :]
            ).with_suffix(".npy")
            if not target.is_file():
                raise FileNotFoundError(f"Albedo target not found: {target}")
            self.target_files.append(target)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        image_path, target_path = self.img_files[index], self.target_files[index]
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        target = np.load(target_path, allow_pickle=False)
        if target.shape != rgb.shape:
            raise ValueError(
                f"Albedo target shape differs from its image: {target_path}"
            )
        if not np.issubdtype(target.dtype, np.floating):
            raise ValueError(
                "Albedo NPY targets must contain floating-point linear RGB."
            )
        if not np.isfinite(target).all() or ((target < 0) | (target > 1)).any():
            raise ValueError("Albedo targets must be finite linear RGB in [0,1].")
        original = rgb.shape[:2]
        canvas = (self.imgsz, self.imgsz)
        rgb = cv2.resize(rgb, canvas, interpolation=cv2.INTER_LANCZOS4)
        target = cv2.resize(
            target.astype(np.float32), canvas, interpolation=cv2.INTER_LINEAR
        )
        if self.augment and torch.rand(()) < 0.5:
            rgb, target = rgb[:, ::-1], target[:, ::-1]
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255
        )
        target_tensor = torch.from_numpy(np.ascontiguousarray(target)).permute(2, 0, 1)
        return (
            image_tensor,
            target_tensor,
            {
                "orig_shape": original,
                "img_path": str(image_path),
                "target_path": str(target_path),
            },
            index,
        )


def albedo_collate_fn(batch):
    return (
        torch.stack([x[0] for x in batch]),
        torch.stack([x[1] for x in batch]),
        [x[2] for x in batch],
        [x[3] for x in batch],
    )
