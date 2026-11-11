"""Render the README task gallery by running LibreYOLO on the Guggenheim sample.

    python scripts/readme/make_examples.py

Writes 640x498 JPGs to ``docs/readme/``. Runs on CPU or MPS; weights download
on first use. The OCR input is a parcel label drawn here with Pillow's built-in
font, so every pixel in the gallery comes from this repository.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from libreyolo import LibreYOLO

OUT_DIR = ROOT / "docs" / "readme"
SAMPLE = ROOT / "libreyolo" / "assets" / "guggenheim-bilbao.jpg"
TILE = (640, 498)  # same aspect as the 1536x1194 sample

# output name -> (checkpoint, input, predict kwargs)
TASKS = {
    "detect": ("LibreYOLO9s.pt", "sample", {"conf": 0.5}),
    "segment": ("LibreRFDETRn-seg.pt", "sample", {"conf": 0.6}),
    "pose": ("LibreGTRs-pose.pt", "sample", {"conf": 0.5}),
    "depth": ("LibreDepthAnythingV2s-depth.pt", "sample", {}),
    "matte": ("LibreBEN2b-matte.pt", "sample", {}),
    "ocr": ("LibrePPOCRt-ocr.pt", "label", {}),
}


def draw_label(path: Path) -> None:
    """A parcel label, drawn at 1280x852, scaled and padded to the sample's 1536x1194."""
    img = Image.new("RGB", (1280, 852), (236, 232, 224))
    d = ImageDraw.Draw(img)

    def font(size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.load_default(size=size)

    ink, grey = (20, 20, 20), (90, 90, 90)
    d.rounded_rectangle(
        (170, 70, 1110, 790),
        radius=18,
        fill=(252, 252, 250),
        outline=(40, 40, 40),
        width=4,
    )
    d.rectangle((170, 70, 1110, 190), fill=(30, 30, 30))
    d.text((210, 105), "PRIORITY PARCEL", font=font(56), fill=(255, 255, 255))
    d.text((215, 250), "SHIP TO", font=font(32), fill=grey)
    d.text((215, 330), "LibreYOLO Project", font=font(50), fill=ink)
    d.text((215, 430), "48009 Bilbao, Spain", font=font(46), fill=ink)
    d.line((215, 505, 1070, 505), fill=(60, 60, 60), width=3)
    d.text((215, 565), "2.4 kg", font=font(48), fill=ink)
    d.text((600, 565), "A-20431", font=font(48), fill=ink)
    d.text((215, 685), "Handle with care", font=font(42), fill=(170, 30, 30))
    d.text((760, 685), "No. 4471", font=font(42), fill=ink)
    canvas = Image.new("RGB", (1536, 1194), (236, 232, 224))
    canvas.paste(img.resize((1536, 1022), Image.LANCZOS), (0, 86))
    canvas.rotate(-3, resample=Image.BICUBIC, fillcolor=(236, 232, 224)).save(
        path, quality=95
    )


def to_pil(plotted) -> Image.Image:
    if isinstance(plotted, Image.Image):
        return plotted.convert("RGB")
    return Image.fromarray(
        np.ascontiguousarray(np.asarray(plotted)[..., ::-1])
    )  # BGR -> RGB


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        label = Path(tmp) / "label.jpg"
        draw_label(label)
        sources = {"sample": str(SAMPLE), "label": str(label)}
        for name, (weights, source, kwargs) in TASKS.items():
            model = LibreYOLO(weights)
            result = model(sources[source], **kwargs)
            tile = to_pil(result.plot()).resize(TILE, Image.LANCZOS)
            path = OUT_DIR / f"task-{name}.jpg"
            tile.save(path, quality=82, optimize=True, progressive=True)
            print(f"{path.relative_to(ROOT)}  {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
