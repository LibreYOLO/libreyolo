"""Generate the repo-safe matte8 fixture images (deterministic, self-owned).

These procedural images are trivially redistributable (no third-party image
rights) and their sharp edges, gradients, and thin protrusions exercise the
resize/normalize + soft-edge behaviour a background-removal model must handle.
One real photo (the test-only ``tests/data/parkour.jpg``, a person) is copied in for a
natural-image check. Run once to (re)generate ``images/``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
OUT = HERE / "images"
REPO_ROOT = HERE.parents[2]


def _gradient(h: int, w: int, c0, c1) -> np.ndarray:
    t = np.linspace(0, 1, w)[None, :, None]
    return (np.asarray(c0) * (1 - t) + np.asarray(c1) * t).repeat(h, 0).astype(np.uint8)


def gen() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    # 1. product: solid circle on a horizontal gradient
    bg = _gradient(320, 400, (30, 60, 120), (200, 220, 240))
    im = Image.fromarray(bg)
    d = ImageDraw.Draw(im)
    d.ellipse([120, 90, 280, 250], fill=(220, 60, 40))
    im.save(OUT / "product_circle.png")

    # 2. box on checkerboard
    board = np.indices((300, 380)).sum(0) // 24 % 2
    bg = np.where(board[..., None] == 0, 235, 170).repeat(3, 2).astype(np.uint8)
    im = Image.fromarray(bg)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([90, 70, 300, 230], radius=24, fill=(60, 160, 90))
    im.save(OUT / "box_checker.png")

    # 3. fine-hair: white blob with radial thin spikes on dark
    im = Image.new("RGB", (360, 360), (18, 18, 26))
    d = ImageDraw.Draw(im)
    cx, cy = 180, 180
    d.ellipse([120, 120, 240, 240], fill=(235, 232, 220))
    for k in range(180):
        ang = k * np.pi / 90.0
        r0, r1 = 60, 60 + 70 * (0.5 + 0.5 * np.sin(k * 1.7))
        d.line(
            [cx + r0 * np.cos(ang), cy + r0 * np.sin(ang), cx + r1 * np.cos(ang), cy + r1 * np.sin(ang)],
            fill=(228, 224, 210), width=1,
        )
    im.save(OUT / "fine_hair_star.png")

    # 4. two ellipses (animal-like silhouettes) on a vertical gradient
    bg = _gradient(340, 420, (120, 150, 90), (40, 70, 40)).transpose(1, 0, 2)[:340, :420]
    im = Image.fromarray(np.ascontiguousarray(bg))
    d = ImageDraw.Draw(im)
    d.ellipse([60, 120, 200, 260], fill=(50, 40, 35))
    d.ellipse([230, 90, 360, 240], fill=(70, 55, 45))
    im.save(OUT / "two_shapes.png")

    # 5. gradient subject on noise (kept small: noise compresses poorly)
    noise = rng.integers(90, 165, size=(200, 200, 3)).astype(np.uint8)
    im = Image.fromarray(noise)
    d = ImageDraw.Draw(im)
    d.polygon([(100, 25), (170, 135), (100, 175), (30, 135)], fill=(210, 120, 200))
    im.save(OUT / "diamond_noise.png")

    # 6. concentric rings
    im = Image.new("RGB", (300, 300), (245, 245, 245))
    d = ImageDraw.Draw(im)
    for i, col in enumerate([(30, 30, 30), (245, 245, 245), (200, 40, 40), (245, 245, 245), (40, 80, 200)]):
        r = 130 - i * 22
        d.ellipse([150 - r, 150 - r, 150 + r, 150 + r], fill=col)
    im.save(OUT / "rings.png")

    # 7. furry radial-strand blob (soft, anti-aliased edges) on light. Drawn at
    # 3x and downsampled so the fine strands are anti-aliased like real hair/fur
    # rather than 1px hard-aliased (which only stresses resize, not the model).
    ss = 3
    big = Image.new("RGB", (340 * ss, 340 * ss), (240, 238, 235))
    d = ImageDraw.Draw(big)
    cx, cy = 170 * ss, 170 * ss
    d.ellipse([100 * ss, 100 * ss, 240 * ss, 240 * ss], fill=(120, 90, 60))
    for k in range(720):
        ang = k * np.pi / 360.0
        r1 = (70 + 55 * (0.5 + 0.5 * np.sin(k * 0.9)) * (0.6 + 0.4 * rng.random())) * ss
        d.line([cx, cy, cx + r1 * np.cos(ang), cy + r1 * np.sin(ang)], fill=(110, 82, 54), width=2 * ss)
    big.resize((340, 340), Image.LANCZOS).save(OUT / "furry_blob.png")

    # 8. portrait-like silhouette (head + shoulders) on a soft gradient
    bg = _gradient(360, 300, (210, 205, 225), (150, 160, 200))
    im = Image.fromarray(np.ascontiguousarray(bg))
    d = ImageDraw.Draw(im)
    d.ellipse([110, 60, 190, 150], fill=(70, 55, 50))          # head
    d.polygon([(60, 360), (90, 200), (210, 200), (240, 360)], fill=(70, 55, 50))  # shoulders
    im.save(OUT / "portrait_silhouette.png")

    print("wrote", len(list(OUT.glob("*.png"))), "images to", OUT)


if __name__ == "__main__":
    gen()
