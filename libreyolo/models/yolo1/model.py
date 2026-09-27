"""LibreYOLO1 - YOLOv1 detector (the original 2016 "You Only Look Once"), cfg-driven.

Source provenance: architecture and pretrained weights derive from
pjreddie/darknet, which is public domain (the "YOLO LICENSE": *"Darknet is
public domain. Do whatever you want with it."*). See
``libreyolo/models/darknet/`` for the shared engine and provenance detail.

Sizes:
    ``t`` tiny-yolo v1 (448)   8 conv + 1 fully-connected head
    ``b`` yolo v1      (448)   24 conv + a locally-connected + fully-connected head

Notes: YOLOv1 predates anchor boxes and COCO. It uses a fixed 448x448 input
(the fully-connected head forbids dynamic shapes), a single ``[detection]`` head
that decodes a dense ``S*S*(C + B*(coords+1))`` vector (S=7, B=2, C=20 VOC
classes -> 7x7x30), and it is trained on Pascal VOC (2007+2012), NOT COCO. The
``[connected]`` (fully-connected) weight layout is the load-bearing port detail
(see ``darknet/blocks.py::DarknetConnected`` / ``darknet/reader.py``). This
family is inference-only in LibreYOLO.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from PIL import Image

from ...utils.image_loader import ImageInput
from ..darknet.family import DarknetFamily
from ..darknet.preprocess import preprocess_image_stretch as _v1_preprocess_image
from ..darknet.preprocess import preprocess_numpy_stretch as _v1_preprocess_numpy

# Pascal VOC 20-class order, taken verbatim from the Darknet ``data/voc.names``
# data file (NOT from memory). The index order is load-bearing: it is the order
# the pretrained YOLOv1 head emits class scores in.
VOC_NAMES: tuple[str, ...] = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


class LibreYOLO1(DarknetFamily):
    FAMILY = "yolo1"
    # The fully-connected head decodes a single fixed-448 image at a time; the
    # decoder rejects batch > 1. Force the inference runner to loop per image
    # for list inputs instead of attempting a real batch.
    SUPPORTS_BATCHED_PREDICT = False
    FILENAME_PREFIX = "LibreYOLO1"
    # Forward is pure tensor work with no host sync, verified to capture and
    # replay bit-identically (tests/unit/test_cuda_graph_families.py).
    SUPPORTS_CUDA_GRAPH = True
    INPUT_SIZES = {"t": 448, "b": 448}

    UNPUBLISHED_WEIGHTS = {
        "LibreYOLO1t": "The tiny YOLOv1 weights are lost upstream; use LibreYOLO1b.pt.",
    }

    CFG_BY_SIZE = {"t": "yolov1-tiny", "b": "yolov1"}
    # Conv-layer counts are byte-exact vs the real .weights: tiny has 8, full 24.
    CONV_COUNT_TO_SIZE = {8: "t", 24: "b"}
    DEFAULT_SIZE = "b"
    DEFAULT_NB_CLASSES = 20  # Pascal VOC, not COCO
    DEFAULT_NAMES = {index: name for index, name in enumerate(VOC_NAMES)}

    # ``[detection]`` head geometry (fixed for both sizes; from the bundled cfg).
    SIDE = 7            # S: the S x S grid
    BOXES_PER_CELL = 2  # B: predicted boxes per cell
    COORDS = 4          # box coordinates per box

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        """Read the class count from the fully-connected head output width.

        The v2/v3/v4 heuristic (a conv with bias and no BN) does not apply:
        every YOLOv1 conv is batch-normalized and the head is a ``[connected]``
        layer. Its output width is ``S*S*(C + B*(coords+1))``, so
        ``C = out // (S*S) - B*(coords+1)``.
        """
        suffix = ".linear.weight"
        prefix = f"{cls.FAMILY}.layers."
        for key, value in weights_dict.items():
            if key.startswith(prefix) and key.endswith(suffix):
                out = int(value.shape[0])
                nc = out // (cls.SIDE * cls.SIDE) - cls.BOXES_PER_CELL * (cls.COORDS + 1)
                return nc if nc > 0 else None
        return None

    # --- YOLOv1-specific preprocessing ------------------------------------
    # The FC head needs a plain square stretch, not the letterbox the shared
    # DarknetFamily (v2/v3/v4) uses. The matching stretch-inverse in
    # post-processing is selected automatically by the "detection" head kind.
    @staticmethod
    def _get_preprocess_numpy():
        return _v1_preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        size = input_size if input_size is not None else self.input_size
        return _v1_preprocess_image(image, input_size=size, color_format=color_format)
