"""PP-DocLayoutV3 preprocessing and class names.

``preprocess_numpy`` lives in ``libreyolo.preprocess.ppdoclayoutv3`` (torch-free,
shared with the exported backends) and is re-exported here.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch

from ...preprocess.ppdoclayoutv3 import preprocess_numpy
from ...utils.image_loader import ImageInput, ImageLoader

__all__ = ["DOC_LAYOUT_CLASSES", "preprocess_image", "preprocess_numpy"]

# Label order of the released checkpoint (``inference.yml`` ``label_list``).
DOC_LAYOUT_CLASSES = (
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
)


def preprocess_image(
    image: ImageInput,
    input_size: int = 800,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
    """Load and preprocess one image for native inference."""
    loaded = ImageLoader.load(image, color_format=color_format)
    width, height = loaded.size
    chw, ratio = preprocess_numpy(np.asarray(loaded.convert("RGB")), input_size)
    return torch.from_numpy(chw).unsqueeze(0), loaded, (width, height), ratio
