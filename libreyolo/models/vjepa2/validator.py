"""V-JEPA 2 classification validation on its own video clips.

``LibreVJEPA2.train`` reads a video dataset YAML (``names`` plus ``train`` /
``val`` manifests). Validation on that YAML decodes the ``val`` manifest with
the same deterministic clip sampling the trainer uses, so epoch validation and
``val()`` score the probe on what it was trained on. An ImageFolder root keeps
the still-image path of :class:`ClassifyValidator` (each image through the
model's frame preprocessing).
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from ...validation.classify_validator import ClassifyValidator


def _is_video_dataset(data) -> bool:
    return isinstance(data, (str, Path)) and Path(data).suffix.lower() in (
        ".yaml",
        ".yml",
    )


def _collate_for_validator(batch):
    """Clips and labels in the validator's ``(images, targets, info, ids)`` shape."""
    from .dataset import collate_clips

    clips, labels = collate_clips(batch)
    return clips, labels, [{} for _ in range(len(labels))], list(range(len(labels)))


class VJEPA2ClipValidator(ClassifyValidator):
    """Top-1/top-5 of a V-JEPA 2 classifier on the video ``val`` manifest."""

    def _setup_dataloader(self) -> DataLoader:
        if not _is_video_dataset(self.config.data):
            return super()._setup_dataloader()
        from .dataset import VideoClipDataset, load_video_dataset

        if getattr(self.config, "visualize", False):
            # visualize draws the validated image files; a clip has none, so
            # accepting the flag here would write nothing.
            raise ValueError(
                "visualize=True is not supported for V-JEPA 2 video validation; "
                "validate an ImageFolder of frames to draw them."
            )
        data = load_video_dataset(self.config.data)
        if "val" not in data:
            raise ValueError(
                f"{self.config.data} has no 'val' manifest to validate on."
            )
        self._check_label_space(data)
        dataset = VideoClipDataset(
            data["val"],
            int(self.model.clip_frames),
            int(getattr(self.model, "frame_stride", 2)),
            int(self.model.crop_size),
            train=False,
        )
        self._num_classes = int(data["nc"])
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=_collate_for_validator,
        )

    def _check_label_space(self, data: dict) -> None:
        """Refuse labels that would be scored against the wrong logit columns.

        Same rule as the ImageFolder path: when the model carries real class
        names, the manifest's ordered names must match them; generic
        ``class_i`` names defer to the dataset. A manifest with more classes
        than the head would index past it.
        """
        nc = int(data["nc"])
        head = int(getattr(self.model, "nb_classes", nc))
        if nc > head:
            raise ValueError(
                f"{self.config.data} has {nc} classes but the model head has "
                f"{head}; its labels cannot be scored against this model."
            )
        model_classes = self._model_class_names()
        dataset_classes = [str(data["names"][i]) for i in range(nc)]
        if model_classes is not None and model_classes != dataset_classes:
            raise ValueError(
                "Video val classes must match the model class names in order "
                f"(model: {model_classes[:5]}..., dataset: {dataset_classes[:5]}...)."
            )

    def _warmup_model(self, n_warmup: int = 3) -> None:
        # A clip forward runs the encoder over every frame; warming up before
        # each epoch's validation would cost more than it saves.
        if not _is_video_dataset(self.config.data):
            super()._warmup_model(n_warmup)
