from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def run_tool(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def test_make_reddit_gif_text_card(tmp_path: Path) -> None:
    output = tmp_path / "post.gif"
    run_tool(
        "reddit/tools/make_reddit_gif.py",
        "--title",
        "LibreYOLO detections in a four second loop",
        "--subtitle",
        "Mobile preview",
        "--width",
        "160",
        "--height",
        "200",
        "--duration",
        "1",
        "--fps",
        "4",
        "--max-mb",
        "1",
        "--output",
        str(output),
    )

    with Image.open(output) as image:
        assert image.size == (160, 200)
        assert getattr(image, "is_animated", False)


def test_reddit_emulator_writes_html_and_png(tmp_path: Path) -> None:
    media = tmp_path / "media.png"
    Image.new("RGB", (80, 100), (255, 69, 0)).save(media)
    html = tmp_path / "preview.html"
    png = tmp_path / "preview.png"

    run_tool(
        "reddit/tools/reddit_emulator.py",
        "--media",
        str(media),
        "--title",
        "A compact Reddit preview",
        "--body",
        "Short supporting context.",
        "--output-html",
        str(html),
        "--output-png",
        str(png),
    )

    assert html.exists()
    with Image.open(png) as image:
        assert image.size == (390, 844)


def test_make_reddit_gif_float_crop_with_left_text(tmp_path: Path) -> None:
    media = tmp_path / "media.png"
    Image.new("RGB", (240, 160), (42, 80, 120)).save(media)
    output = tmp_path / "float.gif"

    run_tool(
        "reddit/tools/make_reddit_gif.py",
        "--input",
        str(media),
        "--title",
        "LibreYOLO supports 16 model families",
        "--left-text",
        "16 models supported",
        "--motion",
        "float-crop",
        "--width",
        "160",
        "--height",
        "200",
        "--duration",
        "1",
        "--fps",
        "4",
        "--max-mb",
        "1",
        "--no-caption",
        "--output",
        str(output),
    )

    with Image.open(output) as image:
        assert image.size == (160, 200)
        assert getattr(image, "is_animated", False)


def test_make_reddit_gif_float_layer(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    subject = tmp_path / "subject.png"
    output = tmp_path / "float-layer.gif"
    Image.new("RGB", (180, 140), (80, 150, 210)).save(background)
    subject_image = Image.new("RGBA", (180, 140), (0, 0, 0, 0))
    for x in range(80, 120):
        for y in range(45, 95):
            subject_image.putpixel((x, y), (230, 230, 240, 255))
    subject_image.save(subject)

    run_tool(
        "reddit/tools/make_reddit_gif.py",
        "--input",
        str(background),
        "--subject-layer",
        str(subject),
        "--motion",
        "float-layer",
        "--subject-scale",
        "1.15",
        "--subject-offset-x",
        "-8",
        "--left-text",
        "16 models supported",
        "--width",
        "160",
        "--height",
        "200",
        "--duration",
        "1",
        "--fps",
        "4",
        "--max-mb",
        "1",
        "--no-caption",
        "--output",
        str(output),
    )

    with Image.open(output) as image:
        assert image.size == (160, 200)
        assert getattr(image, "is_animated", False)
