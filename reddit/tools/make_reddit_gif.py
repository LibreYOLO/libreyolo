from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageSequence

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "specs.json"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}


@dataclass(frozen=True)
class GifPreset:
    name: str
    width: int
    height: int
    fps: int
    duration_seconds: float
    max_mb: float
    fit: str


def load_specs(path: Path = SPEC_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_preset(name: str | None = None) -> GifPreset:
    specs = load_specs()
    preset_name = name or specs["default_preset"]
    try:
        raw = specs["presets"][preset_name]
    except KeyError as exc:
        choices = ", ".join(sorted(specs["presets"]))
        raise SystemExit(f"Unknown preset '{preset_name}'. Choose one of: {choices}") from exc
    return GifPreset(
        name=preset_name,
        width=int(raw["width"]),
        height=int(raw["height"]),
        fps=int(raw["fps"]),
        duration_seconds=float(raw["duration_seconds"]),
        max_mb=float(raw["max_mb"]),
        fit=str(raw["fit"]),
    )


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
    names += ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if not text:
        return (0, 0)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return (right - left, bottom - top)


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    *,
    max_lines: int | None = None,
) -> list[str]:
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
        if max_lines and len(lines) >= max_lines:
            break
    if not max_lines or len(lines) < max_lines:
        lines.append(current)

    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
    if max_lines and len(lines) == max_lines:
        while lines[-1] and text_size(draw, lines[-1] + "...", font)[0] > max_width:
            lines[-1] = lines[-1][:-1].rstrip()
        if lines[-1] and not lines[-1].endswith("..."):
            lines[-1] += "..."
    return lines


def resize_cover(
    image: Image.Image,
    width: int,
    height: int,
    *,
    zoom: float = 1.0,
    pan_x: float = 0.0,
    pan_y: float = 0.0,
) -> Image.Image:
    source = image.convert("RGB")
    scale = max(width / source.width, height / source.height) * zoom
    resized = source.resize(
        (max(width, round(source.width * scale)), max(height, round(source.height * scale))),
        Image.Resampling.LANCZOS,
    )
    overflow_x = max(0, resized.width - width)
    overflow_y = max(0, resized.height - height)
    left = int((overflow_x / 2) + (pan_x * overflow_x / 2))
    top = int((overflow_y / 2) + (pan_y * overflow_y / 2))
    left = min(max(left, 0), overflow_x)
    top = min(max(top, 0), overflow_y)
    return resized.crop((left, top, left + width, top + height))


def resize_contain(image: Image.Image, width: int, height: int) -> Image.Image:
    source = image.convert("RGB")
    background = resize_cover(source, width, height).filter(ImageFilter.GaussianBlur(24))
    background = ImageOps.autocontrast(background.point(lambda value: int(value * 0.55)))
    contained = ImageOps.contain(source, (width, height), Image.Resampling.LANCZOS)
    x = (width - contained.width) // 2
    y = (height - contained.height) // 2
    background.paste(contained, (x, y))
    return background


def fit_image(image: Image.Image, width: int, height: int, fit: str, *, phase: float = 0.0) -> Image.Image:
    if fit == "contain":
        return resize_contain(image, width, height)
    pan = math.sin(phase * math.tau) * 0.45
    zoom = 1.02 + (0.035 * phase)
    return resize_cover(image, width, height, zoom=zoom, pan_x=pan, pan_y=0.0)


def draw_caption(
    frame: Image.Image,
    title: str | None,
    subtitle: str | None,
    *,
    progress: float,
) -> Image.Image:
    if not title and not subtitle:
        return frame

    image = frame.convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    margin = max(28, width // 28)
    panel_height = max(height // 4, 230)
    panel_y = height - panel_height

    for offset in range(panel_height):
        alpha = int(220 * (offset / panel_height))
        draw.line((0, panel_y + offset, width, panel_y + offset), fill=(7, 8, 10, alpha))

    title_font = load_font(max(34, width // 20), bold=True)
    subtitle_font = load_font(max(22, width // 34), bold=False)
    max_text_width = width - (margin * 2)
    y = panel_y + margin

    if title:
        for line in wrap_text(draw, title, title_font, max_text_width, max_lines=3):
            draw.text((margin, y), line, fill=(255, 255, 255, 255), font=title_font)
            y += text_size(draw, line, title_font)[1] + 12
    if subtitle:
        y += 6
        for line in wrap_text(draw, subtitle, subtitle_font, max_text_width, max_lines=2):
            draw.text((margin, y), line, fill=(220, 224, 230, 245), font=subtitle_font)
            y += text_size(draw, line, subtitle_font)[1] + 8

    bar_width = width - (margin * 2)
    bar_y = height - margin
    draw.rounded_rectangle(
        (margin, bar_y, margin + bar_width, bar_y + 7),
        radius=4,
        fill=(255, 255, 255, 70),
    )
    draw.rounded_rectangle(
        (margin, bar_y, margin + int(bar_width * progress), bar_y + 7),
        radius=4,
        fill=(255, 69, 0, 230),
    )
    return image.convert("RGB")


def draw_text_with_shadow(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    *,
    fill: tuple[int, int, int, int],
    shadow: tuple[int, int, int, int] = (0, 0, 0, 170),
) -> None:
    x, y = xy
    draw.text((x + 2, y + 2), text, fill=shadow, font=font)
    draw.text((x, y), text, fill=fill, font=font)


def draw_clean_left_text(image: Image.Image, draw: ImageDraw.ImageDraw, left_text: str) -> Image.Image:
    width, height = image.size
    margin = max(32, width // 24)
    top = max(40, height // 12)
    panel_width = min(int(width * 0.44), 650)
    title_font = load_font(max(40, width // 24), bold=True)
    body_font = load_font(max(24, width // 54), bold=True)
    small_font = load_font(max(20, width // 68), bold=False)

    raw_lines = [line.strip() for line in left_text.splitlines() if line.strip()]
    if not raw_lines:
        return image
    title = raw_lines[0]
    items = raw_lines[1:]

    item_height = text_size(draw, "Ag", body_font)[1] + 10
    column_count = 2 if len(items) > 9 else 1
    rows = math.ceil(len(items) / column_count) if items else 0
    title_height = text_size(draw, title, title_font)[1]

    draw_text_with_shadow(draw, (margin, top), title, title_font, fill=(255, 91, 35, 255))
    y = top + title_height + 28
    if not items:
        return image

    column_width = panel_width // column_count
    for index, item in enumerate(items):
        column = index // rows if rows else 0
        row = index % rows if rows else index
        x = margin + (column * column_width)
        item_y = y + (row * item_height)
        label = item.upper().replace("_", "-")
        draw_text_with_shadow(draw, (x, item_y), label, body_font, fill=(245, 248, 252, 255))

    footer = "native + export-ready"
    footer_y = y + (rows * item_height) + 8
    draw_text_with_shadow(draw, (margin, footer_y), footer, small_font, fill=(222, 229, 238, 235))
    return image


def draw_left_text(frame: Image.Image, left_text: str | None, *, style: str = "panel") -> Image.Image:
    if not left_text:
        return frame

    image = frame.convert("RGBA")
    draw = ImageDraw.Draw(image)
    if style == "clean":
        return draw_clean_left_text(image, draw, left_text).convert("RGB")

    width, height = image.size
    panel_width = max(width // 3, int(width * 0.38))
    margin = max(28, width // 28)
    title_font = load_font(max(42, width // 17), bold=True)
    body_font = load_font(max(24, width // 34), bold=True)

    for x in range(panel_width):
        alpha = int(210 * (1 - (x / panel_width)))
        draw.line((x, 0, x, height), fill=(6, 8, 12, alpha))

    lines: list[str] = []
    for raw_line in left_text.splitlines():
        if not raw_line.strip():
            lines.append("")
            continue
        font = title_font if not lines else body_font
        lines.extend(wrap_text(draw, raw_line, font, panel_width - (margin * 2), max_lines=3))

    y = max(margin, height // 2 - ((len(lines) * max(36, width // 25)) // 2))
    for index, line in enumerate(lines):
        if not line:
            y += 18
            continue
        font = title_font if index == 0 else body_font
        fill = (255, 255, 255, 255) if index == 0 else (235, 238, 244, 255)
        if index == 0 and any(char.isdigit() for char in line):
            fill = (255, 91, 35, 255)
        draw.text((margin, y), line, fill=fill, font=font)
        y += text_size(draw, line, font)[1] + 12

    return image.convert("RGB")


def text_card_frames(
    width: int,
    height: int,
    fps: int,
    duration_seconds: float,
    title: str,
    subtitle: str | None,
) -> list[Image.Image]:
    frame_count = max(2, round(fps * duration_seconds))
    title_font = load_font(max(42, width // 16), bold=True)
    subtitle_font = load_font(max(24, width // 32), bold=False)
    frames: list[Image.Image] = []
    for index in range(frame_count):
        phase = index / max(1, frame_count - 1)
        image = Image.new("RGB", (width, height), (17, 20, 24))
        draw = ImageDraw.Draw(image)
        for y in range(height):
            mix = y / max(1, height - 1)
            red = int(18 + (34 * mix))
            green = int(24 + (21 * mix))
            blue = int(31 + (18 * mix))
            draw.line((0, y, width, y), fill=(red, green, blue))

        band_x = int(-width * 0.3 + phase * width * 0.18)
        draw.polygon(
            [
                (band_x, 0),
                (band_x + width // 2, 0),
                (band_x + width, height),
                (band_x + width // 2, height),
            ],
            fill=(255, 69, 0),
        )
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 38))
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(image)

        margin = max(42, width // 16)
        max_text_width = width - (margin * 2)
        lines = wrap_text(draw, title, title_font, max_text_width, max_lines=4)
        subtitle_lines = wrap_text(draw, subtitle or "", subtitle_font, max_text_width, max_lines=2)
        text_block_height = sum(text_size(draw, line, title_font)[1] + 14 for line in lines)
        text_block_height += sum(text_size(draw, line, subtitle_font)[1] + 10 for line in subtitle_lines)
        y = max(margin, (height - text_block_height) // 2)

        for line in lines:
            draw.text((margin, y), line, fill=(255, 255, 255), font=title_font)
            y += text_size(draw, line, title_font)[1] + 14
        y += 10
        for line in subtitle_lines:
            draw.text((margin, y), line, fill=(231, 235, 240), font=subtitle_font)
            y += text_size(draw, line, subtitle_font)[1] + 10

        bar_width = width - (margin * 2)
        bar_y = height - margin
        draw.rounded_rectangle((margin, bar_y, margin + bar_width, bar_y + 6), radius=3, fill=(255, 255, 255))
        draw.rounded_rectangle(
            (margin, bar_y, margin + max(8, int(bar_width * phase)), bar_y + 6),
            radius=3,
            fill=(255, 69, 0),
        )
        frames.append(draw_caption(image, None, None, progress=phase))
    return frames


def first_frame(path: Path) -> Image.Image:
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        try:
            import cv2
        except ImportError as exc:
            raise SystemExit("Video input requires opencv-python. It is already a LibreYOLO dependency.") from exc
        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise ValueError(f"Could not read a frame from video: {path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb).convert("RGB")

    image = Image.open(path)
    image.seek(0)
    return ImageOps.exif_transpose(image).convert("RGB")


def parse_crop_box(value: str | None) -> tuple[float, float, float, float]:
    if not value:
        return (0.36, 0.04, 0.62, 0.9)
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise SystemExit("--float-crop-box must be four comma-separated values: x,y,w,h")
    x, y, crop_width, crop_height = parts
    if min(parts) < 0 or x + crop_width > 1 or y + crop_height > 1:
        raise SystemExit("--float-crop-box values must be normalized to the 0..1 source image range")
    return (x, y, crop_width, crop_height)


def crop_by_box(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    x, y, crop_width, crop_height = box
    left = round(image.width * x)
    top = round(image.height * y)
    right = round(image.width * (x + crop_width))
    bottom = round(image.height * (y + crop_height))
    return image.crop((left, top, right, bottom))


def paste_rounded(base: Image.Image, layer: Image.Image, xy: tuple[int, int], radius: int) -> None:
    mask = Image.new("L", layer.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((0, 0, layer.width, layer.height), radius=radius, fill=255)
    base.paste(layer, xy, mask)


def alpha_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    if image.mode != "RGBA":
        raise ValueError("Subject layer must have an alpha channel")
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("Subject layer alpha channel is empty")
    return bbox


def inpaint_background(background: Image.Image, subject_layer: Image.Image, radius: int = 7) -> Image.Image:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Float-layer inpainting requires opencv-python and numpy.") from exc

    bg = background.convert("RGB")
    layer = subject_layer.convert("RGBA")
    if layer.size != bg.size:
        layer = layer.resize(bg.size, Image.Resampling.LANCZOS)

    mask = np.array(layer.getchannel("A"))
    mask = np.where(mask > 8, 255, 0).astype("uint8")
    kernel = np.ones((19, 19), dtype="uint8")
    mask = cv2.dilate(mask, kernel, iterations=1)
    rgb = np.array(bg)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))


def add_subject_shadow(subject: Image.Image, opacity: int = 95) -> Image.Image:
    alpha = subject.getchannel("A")
    shadow = Image.new("RGBA", subject.size, (0, 0, 0, 0))
    shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(14)).point(lambda value: int(value * opacity / 255))
    shadow.putalpha(shadow_alpha)
    return shadow


def floating_layer_frames(
    background_path: Path,
    subject_layer_path: Path,
    width: int,
    height: int,
    frame_count: int,
    fit: str,
    *,
    left_text: str | None,
    left_text_style: str,
    float_pixels: int,
    subject_scale: float,
    subject_offset_x: int,
    subject_offset_y: int,
    inpaint_radius: int,
) -> list[Image.Image]:
    background = first_frame(background_path)
    subject_full = Image.open(subject_layer_path).convert("RGBA")
    if subject_full.size != background.size:
        subject_full = subject_full.resize(background.size, Image.Resampling.LANCZOS)

    clean_background = inpaint_background(background, subject_full, radius=inpaint_radius).convert("RGBA")
    bbox = alpha_bbox(subject_full)
    subject = subject_full.crop(bbox)
    subject_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    if subject_scale <= 0:
        raise SystemExit("--subject-scale must be greater than 0")
    if subject_scale != 1:
        subject = subject.resize(
            (max(2, round(subject.width * subject_scale)), max(2, round(subject.height * subject_scale))),
            Image.Resampling.LANCZOS,
        )

    source_float_pixels = int(float_pixels * (background.height / height))
    frames: list[Image.Image] = []
    for index in range(frame_count):
        phase = index / max(1, frame_count - 1)
        float_y = int(math.sin(phase * math.tau) * source_float_pixels)
        layer_x = round(subject_center[0] - (subject.width / 2) + subject_offset_x)
        layer_y = round(subject_center[1] - (subject.height / 2) + subject_offset_y + float_y)

        composite = clean_background.copy()
        shadow = add_subject_shadow(subject)
        composite.alpha_composite(shadow, (layer_x + 10, layer_y + 12))
        composite.alpha_composite(subject, (layer_x, layer_y))

        frame = fit_image(composite.convert("RGB"), width, height, fit, phase=0)
        frame = draw_left_text(frame, left_text, style=left_text_style)
        frames.append(frame)
    return frames


def floating_crop_frames(
    path: Path,
    width: int,
    height: int,
    frame_count: int,
    *,
    left_text: str | None,
    left_text_style: str,
    float_pixels: int,
    crop_box: tuple[float, float, float, float],
) -> list[Image.Image]:
    source = first_frame(path)
    crop = crop_by_box(source, crop_box)
    max_crop_width = int(width * 0.58)
    max_crop_height = int(height * 0.86)
    crop = ImageOps.contain(crop, (max_crop_width, max_crop_height), Image.Resampling.LANCZOS)

    background = resize_cover(source, width, height, zoom=1.04).filter(ImageFilter.GaussianBlur(16))
    background = ImageOps.autocontrast(background.point(lambda value: int(value * 0.62)))
    frames: list[Image.Image] = []
    base_x = width - crop.width - max(34, width // 30)
    base_y = (height - crop.height) // 2

    for index in range(frame_count):
        phase = index / max(1, frame_count - 1)
        float_y = int(math.sin(phase * math.tau) * float_pixels)
        frame = background.copy().convert("RGBA")

        shadow = Image.new("RGBA", (crop.width + 54, crop.height + 54), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rounded_rectangle(
            (24, 24, crop.width + 34, crop.height + 34),
            radius=max(18, width // 42),
            fill=(0, 0, 0, 145),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(18))
        frame.alpha_composite(shadow, (base_x - 24, base_y + float_y - 10))

        paste_rounded(frame, crop.convert("RGBA"), (base_x, base_y + float_y), max(18, width // 42))
        frame = draw_left_text(frame.convert("RGB"), left_text, style=left_text_style).convert("RGBA")
        frames.append(frame.convert("RGB"))
    return frames


def image_frames(
    path: Path,
    width: int,
    height: int,
    fps: int,
    frame_count: int,
    fit: str,
) -> list[Image.Image]:
    source = Image.open(path)
    source = ImageOps.exif_transpose(source)
    if getattr(source, "is_animated", False):
        raw_frames = [ImageOps.exif_transpose(frame.copy()).convert("RGB") for frame in ImageSequence.Iterator(source)]
        if not raw_frames:
            raise ValueError(f"No frames found in {path}")
        frames = []
        for index in range(frame_count):
            raw = raw_frames[int(index * len(raw_frames) / frame_count) % len(raw_frames)]
            phase = index / max(1, frame_count - 1)
            frames.append(fit_image(raw, width, height, fit, phase=phase))
        return frames

    frames = []
    still = source.convert("RGB")
    for index in range(frame_count):
        phase = index / max(1, frame_count - 1)
        frames.append(fit_image(still, width, height, fit, phase=phase))
    return frames


def video_frames(path: Path, width: int, height: int, fps: int, frame_count: int, fit: str) -> list[Image.Image]:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Video input requires opencv-python. It is already a LibreYOLO dependency.") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or fps
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        raise ValueError(f"Could not read frame count from video: {path}")

    frames: list[Image.Image] = []
    for index in range(frame_count):
        source_index = min(total_frames - 1, round(index * source_fps / fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, frame = capture.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        phase = index / max(1, frame_count - 1)
        frames.append(fit_image(image, width, height, fit, phase=phase))
    capture.release()

    if not frames:
        raise ValueError(f"No frames extracted from video: {path}")
    return frames


def build_frames(
    inputs: Sequence[Path],
    width: int,
    height: int,
    fps: int,
    duration_seconds: float,
    fit: str,
    *,
    title: str | None,
    subtitle: str | None,
    caption: bool = True,
    left_text: str | None = None,
    left_text_style: str = "panel",
    motion: str = "ken-burns",
    float_pixels: int = 16,
    float_crop_box: str | None = None,
    subject_layer: Path | None = None,
    subject_scale: float = 1.12,
    subject_offset_x: int = 0,
    subject_offset_y: int = 0,
    inpaint_radius: int = 7,
) -> list[Image.Image]:
    frame_count = max(2, round(fps * duration_seconds))
    if not inputs:
        if not title:
            raise SystemExit("Provide --input media or --title for a generated text GIF.")
        frames = text_card_frames(width, height, fps, duration_seconds, title, subtitle)
        return [draw_left_text(frame, left_text, style=left_text_style) for frame in frames]

    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise SystemExit(f"Missing input file(s): {', '.join(missing)}")

    frames: list[Image.Image] = []
    per_input = max(2, math.ceil(frame_count / len(inputs)))
    for path in inputs:
        if motion == "float-layer":
            if subject_layer is None:
                raise SystemExit("--motion float-layer requires --subject-layer")
            if not subject_layer.exists():
                raise SystemExit(f"Missing subject layer file: {subject_layer}")
            frames.extend(
                floating_layer_frames(
                    path,
                    subject_layer,
                    width,
                    height,
                    per_input,
                    fit,
                    left_text=left_text,
                    left_text_style=left_text_style,
                    float_pixels=float_pixels,
                    subject_scale=subject_scale,
                    subject_offset_x=subject_offset_x,
                    subject_offset_y=subject_offset_y,
                    inpaint_radius=inpaint_radius,
                )
            )
        elif motion == "float-crop":
            frames.extend(
                floating_crop_frames(
                    path,
                    width,
                    height,
                    per_input,
                    left_text=left_text,
                    left_text_style=left_text_style,
                    float_pixels=float_pixels,
                    crop_box=parse_crop_box(float_crop_box),
                )
            )
        elif path.suffix.lower() in VIDEO_EXTENSIONS:
            frames.extend(video_frames(path, width, height, fps, per_input, fit))
        else:
            frames.extend(image_frames(path, width, height, fps, per_input, fit))

    frames = frames[:frame_count]
    while len(frames) < frame_count:
        frames.append(frames[-1].copy())

    if caption and (title or subtitle):
        frames = [
            draw_caption(frame, title, subtitle, progress=index / max(1, len(frames) - 1))
            for index, frame in enumerate(frames)
        ]
    if left_text and motion not in {"float-crop", "float-layer"}:
        frames = [draw_left_text(frame, left_text, style=left_text_style) for frame in frames]
    return frames


def quantize(frames: Iterable[Image.Image], colors: int) -> list[Image.Image]:
    return [
        frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors)
        for frame in frames
    ]


def save_gif_candidate(
    frames: Sequence[Image.Image],
    path: Path,
    *,
    fps: float,
    colors: int,
) -> None:
    duration_ms = max(20, round(1000 / fps))
    paletted = quantize(frames, colors)
    paletted[0].save(
        path,
        save_all=True,
        append_images=paletted[1:],
        optimize=True,
        loop=0,
        duration=duration_ms,
        disposal=2,
    )


def scaled_frames(frames: Sequence[Image.Image], scale: float) -> list[Image.Image]:
    if scale == 1:
        return [frame.copy() for frame in frames]
    width = max(2, round(frames[0].width * scale))
    height = max(2, round(frames[0].height * scale))
    return [frame.resize((width, height), Image.Resampling.LANCZOS) for frame in frames]


def fps_candidates(fps: int) -> list[int]:
    values = [fps, 12, 10, 8, 6, 5, 4]
    return sorted({value for value in values if 1 <= value <= fps}, reverse=True)


def save_optimized_gif(
    frames: Sequence[Image.Image],
    output: Path,
    *,
    fps: int,
    max_mb: float,
    initial_colors: int = 128,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_mb * 1024 * 1024)
    colors_values = [initial_colors, 96, 64, 48, 32]
    scales = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    best: tuple[int, Path, dict[str, Any]] | None = None

    with tempfile.TemporaryDirectory(prefix="reddit-gif-") as tmp:
        tmp_dir = Path(tmp)
        attempt = 0
        for target_fps in fps_candidates(fps):
            step = max(1, round(fps / target_fps))
            reduced = [frame.copy() for frame in frames[::step]]
            effective_fps = fps / step
            for scale in scales:
                resized = scaled_frames(reduced, scale)
                for colors in colors_values:
                    attempt += 1
                    candidate = tmp_dir / f"candidate-{attempt}.gif"
                    save_gif_candidate(resized, candidate, fps=effective_fps, colors=colors)
                    size = candidate.stat().st_size
                    summary = {
                        "width": resized[0].width,
                        "height": resized[0].height,
                        "frames": len(resized),
                        "fps": round(effective_fps, 2),
                        "colors": colors,
                        "bytes": size,
                        "max_bytes": max_bytes,
                        "within_limit": size <= max_bytes,
                    }
                    if best is None or size < best[0]:
                        best = (size, candidate, summary)
                    if size <= max_bytes:
                        shutil.copyfile(candidate, output)
                        return summary | {"output": str(output)}

        if best is None:
            raise RuntimeError("No GIF candidate was created")
        shutil.copyfile(best[1], output)
        return best[2] | {
            "output": str(output),
            "warning": "Could not reach requested max_mb; wrote smallest candidate.",
        }


def make_gif(
    *,
    inputs: Sequence[Path],
    output: Path,
    preset_name: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    duration_seconds: float | None = None,
    max_mb: float | None = None,
    fit: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: bool = True,
    colors: int = 128,
    left_text: str | None = None,
    left_text_style: str = "panel",
    motion: str = "ken-burns",
    float_pixels: int = 16,
    float_crop_box: str | None = None,
    subject_layer: Path | None = None,
    subject_scale: float = 1.12,
    subject_offset_x: int = 0,
    subject_offset_y: int = 0,
    inpaint_radius: int = 7,
) -> dict[str, Any]:
    preset = get_preset(preset_name)
    final_width = width or preset.width
    final_height = height or preset.height
    final_fps = fps or preset.fps
    final_duration = duration_seconds or preset.duration_seconds
    final_max_mb = max_mb or preset.max_mb
    final_fit = fit or preset.fit
    frames = build_frames(
        inputs,
        final_width,
        final_height,
        final_fps,
        final_duration,
        final_fit,
        title=title,
        subtitle=subtitle,
        caption=caption,
        left_text=left_text,
        left_text_style=left_text_style,
        motion=motion,
        float_pixels=float_pixels,
        float_crop_box=float_crop_box,
        subject_layer=subject_layer,
        subject_scale=subject_scale,
        subject_offset_x=subject_offset_x,
        subject_offset_y=subject_offset_y,
        inpaint_radius=inpaint_radius,
    )
    summary = save_optimized_gif(frames, output, fps=final_fps, max_mb=final_max_mb, initial_colors=colors)
    return {
        "preset": preset.name,
        "target_width": final_width,
        "target_height": final_height,
        "target_fps": final_fps,
        "target_duration_seconds": final_duration,
        "target_max_mb": final_max_mb,
        "fit": final_fit,
        "motion": motion,
        "left_text": left_text,
        "left_text_style": left_text_style,
        "subject_layer": str(subject_layer) if subject_layer else None,
        "subject_scale": subject_scale,
        "subject_offset_x": subject_offset_x,
        "subject_offset_y": subject_offset_y,
    } | summary


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an optimized Reddit-ready GIF.")
    parser.add_argument("--input", action="append", default=[], help="Input image, GIF, or video. Repeat for slides.")
    parser.add_argument("--output", required=True, help="Output GIF path.")
    parser.add_argument("--preset", default=None, help="Preset from reddit/specs.json.")
    parser.add_argument("--width", type=int, default=None, help="Override output width.")
    parser.add_argument("--height", type=int, default=None, help="Override output height.")
    parser.add_argument("--fps", type=int, default=None, help="Override frames per second.")
    parser.add_argument("--duration", type=float, default=None, help="Override duration in seconds.")
    parser.add_argument("--max-mb", type=float, default=None, help="Override target max size in MB.")
    parser.add_argument("--fit", choices=["cover", "contain"], default=None, help="Media fit mode.")
    parser.add_argument("--title", default=None, help="Hook text shown on generated GIF or caption overlay.")
    parser.add_argument("--subtitle", default=None, help="Secondary caption text.")
    parser.add_argument("--no-caption", action="store_true", help="Do not overlay title/subtitle on media inputs.")
    parser.add_argument("--left-text", default=None, help="Text burned into the left side of each frame. Use newlines for multiple lines.")
    parser.add_argument("--left-text-style", choices=["panel", "clean"], default="panel", help="Left text rendering style.")
    parser.add_argument("--motion", choices=["ken-burns", "float-crop", "float-layer"], default="ken-burns", help="Animation style.")
    parser.add_argument("--float-pixels", type=int, default=16, help="Vertical float amount for --motion float-crop.")
    parser.add_argument(
        "--float-crop-box",
        default=None,
        help="Normalized source crop for --motion float-crop as x,y,w,h. Default favors the right side.",
    )
    parser.add_argument("--subject-layer", default=None, help="Transparent PNG subject layer for --motion float-layer.")
    parser.add_argument("--subject-scale", type=float, default=1.12, help="Scale factor for --motion float-layer subject.")
    parser.add_argument("--subject-offset-x", type=int, default=0, help="Source-pixel horizontal offset for --motion float-layer.")
    parser.add_argument("--subject-offset-y", type=int, default=0, help="Source-pixel vertical offset for --motion float-layer.")
    parser.add_argument("--inpaint-radius", type=int, default=7, help="OpenCV inpaint radius for --motion float-layer.")
    parser.add_argument("--colors", type=int, default=128, help="Initial GIF palette color count.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    summary = make_gif(
        inputs=[Path(value) for value in args.input],
        output=Path(args.output),
        preset_name=args.preset,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration_seconds=args.duration,
        max_mb=args.max_mb,
        fit=args.fit,
        title=args.title,
        subtitle=args.subtitle,
        caption=not args.no_caption,
        colors=args.colors,
        left_text=args.left_text,
        left_text_style=args.left_text_style,
        motion=args.motion,
        float_pixels=args.float_pixels,
        float_crop_box=args.float_crop_box,
        subject_layer=Path(args.subject_layer) if args.subject_layer else None,
        subject_scale=args.subject_scale,
        subject_offset_x=args.subject_offset_x,
        subject_offset_y=args.subject_offset_y,
        inpaint_radius=args.inpaint_radius,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
