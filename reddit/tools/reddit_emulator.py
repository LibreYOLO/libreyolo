from __future__ import annotations

import argparse
import html
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}


@dataclass
class PostPreview:
    title: str
    subreddit: str = "r/libreyolo"
    author: str = "u/libreyolo"
    body: str = ""
    media: Path | None = None
    votes: str = "128"
    comments: str = "24"
    flair: str = "Showcase"
    time_label: str = "just now"


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
    max_lines: int,
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
        if len(lines) >= max_lines:
            break
    if len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines:
        while lines[-1] and text_size(draw, lines[-1] + "...", font)[0] > max_width:
            lines[-1] = lines[-1][:-1].rstrip()
        if lines[-1] and not lines[-1].endswith("..."):
            lines[-1] += "..."
    return lines


def first_media_frame(path: Path) -> Image.Image:
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        try:
            import cv2
        except ImportError as exc:
            raise SystemExit("Video preview requires opencv-python.") from exc
        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise ValueError(f"Could not read a frame from {path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb).convert("RGB")

    image = Image.open(path)
    image.seek(0)
    return ImageOps.exif_transpose(image).convert("RGB")


def contain(image: Image.Image, box: tuple[int, int], background: tuple[int, int, int]) -> Image.Image:
    canvas = Image.new("RGB", box, background)
    fitted = ImageOps.contain(image, box, Image.Resampling.LANCZOS)
    canvas.paste(fitted, ((box[0] - fitted.width) // 2, (box[1] - fitted.height) // 2))
    return canvas


def theme_colors(theme: str) -> dict[str, tuple[int, int, int]]:
    if theme == "dark":
        return {
            "bg": (12, 14, 18),
            "card": (27, 31, 36),
            "line": (52, 58, 66),
            "text": (238, 241, 245),
            "muted": (162, 170, 181),
            "soft": (38, 43, 50),
            "accent": (255, 69, 0),
        }
    return {
        "bg": (246, 247, 248),
        "card": (255, 255, 255),
        "line": (222, 226, 230),
        "text": (28, 31, 35),
        "muted": (101, 109, 118),
        "soft": (238, 241, 244),
        "accent": (255, 69, 0),
    }


def draw_post_card(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    post: PostPreview,
    *,
    x: int,
    y: int,
    width: int,
    theme: str,
) -> int:
    colors = theme_colors(theme)
    title_font = load_font(24 if width < 500 else 28, bold=True)
    meta_font = load_font(13 if width < 500 else 15)
    body_font = load_font(15 if width < 500 else 17)
    stat_font = load_font(14 if width < 500 else 16, bold=True)
    padding = 14 if width < 500 else 18
    content_width = width - (padding * 2)
    cursor = y + padding

    title_lines = wrap_text(draw, post.title, title_font, content_width, 4)
    body_lines = wrap_text(draw, post.body, body_font, content_width, 4) if post.body else []
    media_height = 0
    media_image: Image.Image | None = None
    media_meta = ""
    if post.media:
        media_frame = first_media_frame(post.media)
        media_height = min(round(content_width * 1.25), 520 if width < 500 else 560)
        media_image = contain(media_frame, (content_width, media_height), colors["soft"])
        media_meta = f"{media_frame.width}x{media_frame.height} source"

    card_height = padding * 2 + 22
    card_height += sum(text_size(draw, line, title_font)[1] + 7 for line in title_lines) + 10
    if body_lines:
        card_height += sum(text_size(draw, line, body_font)[1] + 5 for line in body_lines) + 10
    if media_image:
        card_height += media_height + 14
    card_height += 42

    draw.rounded_rectangle((x, y, x + width, y + card_height), radius=10, fill=colors["card"], outline=colors["line"])
    draw.ellipse((x + padding, cursor, x + padding + 18, cursor + 18), fill=colors["accent"])
    meta = f"{post.subreddit} - Posted by {post.author} - {post.time_label}"
    draw.text((x + padding + 26, cursor + 1), meta, fill=colors["muted"], font=meta_font)
    if post.flair:
        flair_width = text_size(draw, post.flair, meta_font)[0] + 16
        flair_x = x + width - padding - flair_width
        draw.rounded_rectangle((flair_x, cursor - 2, flair_x + flair_width, cursor + 21), radius=11, fill=colors["soft"])
        draw.text((flair_x + 8, cursor + 1), post.flair, fill=colors["muted"], font=meta_font)
    cursor += 32

    for line in title_lines:
        draw.text((x + padding, cursor), line, fill=colors["text"], font=title_font)
        cursor += text_size(draw, line, title_font)[1] + 7
    cursor += 7

    for line in body_lines:
        draw.text((x + padding, cursor), line, fill=colors["text"], font=body_font)
        cursor += text_size(draw, line, body_font)[1] + 5
    if body_lines:
        cursor += 8

    if media_image:
        canvas.paste(media_image, (x + padding, cursor))
        if post.media and post.media.suffix.lower() == ".gif":
            badge = "GIF"
            badge_width = text_size(draw, badge, stat_font)[0] + 16
            draw.rounded_rectangle(
                (x + padding + 10, cursor + 10, x + padding + 10 + badge_width, cursor + 38),
                radius=8,
                fill=(0, 0, 0),
            )
            draw.text((x + padding + 18, cursor + 15), badge, fill=(255, 255, 255), font=stat_font)
        cursor += media_height + 6
        draw.text((x + padding, cursor), media_meta, fill=colors["muted"], font=meta_font)
        cursor += 18

    stat_y = y + card_height - padding - 28
    stats = f"^ {post.votes}    Comments {post.comments}    Share    Save"
    draw.text((x + padding, stat_y), stats, fill=colors["muted"], font=stat_font)
    return y + card_height


def render_png(post: PostPreview, output: Path, *, device: str = "mobile", theme: str = "light") -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = theme_colors(theme)
    if device == "desktop":
        size = (1280, 900)
        card_width = 760
        card_x = 340
        card_y = 92
    else:
        size = (390, 844)
        card_width = 370
        card_x = 10
        card_y = 74

    canvas = Image.new("RGB", size, colors["bg"])
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(22, bold=True)
    nav_font = load_font(14)
    draw.rectangle((0, 0, size[0], 56), fill=colors["card"])
    draw.line((0, 55, size[0], 55), fill=colors["line"])
    draw.text((18, 16), "reddit", fill=colors["accent"], font=header_font)

    if device == "desktop":
        draw.rounded_rectangle((38, 92, 278, 460), radius=10, fill=colors["card"], outline=colors["line"])
        for index, label in enumerate(["Home", "Popular", "All", post.subreddit]):
            draw.text((64, 126 + index * 42), label, fill=colors["text"], font=nav_font)
        draw.rounded_rectangle((card_x, 26, card_x + card_width, 44), radius=9, fill=colors["soft"])
        draw.text((card_x + 16, 29), "Search Reddit", fill=colors["muted"], font=nav_font)
    else:
        draw.rounded_rectangle((112, 18, 372, 38), radius=10, fill=colors["soft"])
        draw.text((126, 20), "Search Reddit", fill=colors["muted"], font=nav_font)

    bottom = draw_post_card(draw, canvas, post, x=card_x, y=card_y, width=card_width, theme=theme)
    if bottom > size[1] - 20:
        draw.rectangle((0, size[1] - 34, size[0], size[1]), fill=colors["bg"])
        draw.text((18, size[1] - 28), "Preview clipped: shorten title/body or use desktop preview.", fill=colors["accent"], font=nav_font)

    canvas.save(output)
    return {"output": str(output), "width": size[0], "height": size[1], "device": device, "theme": theme}


def asset_src(media: Path | None, html_path: Path) -> str:
    if media is None:
        return ""
    try:
        return os.path.relpath(media.resolve(), html_path.parent.resolve()).replace("\\", "/")
    except ValueError:
        return media.resolve().as_uri()


def write_html(post: PostPreview, output: Path, *, theme: str = "light") -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = theme_colors(theme)
    media = ""
    if post.media:
        src = html.escape(asset_src(post.media, output))
        suffix = post.media.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            media = f'<video class="media" src="{src}" controls autoplay loop muted></video>'
        else:
            media = f'<img class="media" src="{src}" alt="">'

    bg = f"rgb{colors['bg']}"
    card = f"rgb{colors['card']}"
    text = f"rgb{colors['text']}"
    muted = f"rgb{colors['muted']}"
    line = f"rgb{colors['line']}"
    soft = f"rgb{colors['soft']}"
    accent = f"rgb{colors['accent']}"
    body_html = f"<p>{html.escape(post.body)}</p>" if post.body else ""
    flair_html = f'<span class="flair">{html.escape(post.flair)}</span>' if post.flair else ""
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reddit Post Preview</title>
  <style>
    :root {{
      --bg: {bg};
      --card: {card};
      --text: {text};
      --muted: {muted};
      --line: {line};
      --soft: {soft};
      --accent: {accent};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
    }}
    header {{
      height: 56px;
      display: flex;
      align-items: center;
      gap: 28px;
      padding: 0 20px;
      background: var(--card);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
    }}
    .brand {{ color: var(--accent); font-weight: 800; font-size: 22px; }}
    .search {{
      flex: 1;
      max-width: 620px;
      height: 34px;
      border-radius: 18px;
      background: var(--soft);
      color: var(--muted);
      display: flex;
      align-items: center;
      padding: 0 16px;
      font-size: 14px;
    }}
    main {{
      width: min(760px, calc(100vw - 20px));
      margin: 28px auto;
    }}
    article {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
    }}
    .post-body {{ padding: 16px; }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .dot {{
      width: 18px;
      height: 18px;
      border-radius: 999px;
      background: var(--accent);
      display: inline-block;
    }}
    .flair {{
      padding: 3px 9px;
      border-radius: 999px;
      background: var(--soft);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    h1 {{
      margin: 12px 0 10px;
      font-size: clamp(22px, 3vw, 30px);
      line-height: 1.15;
      letter-spacing: 0;
    }}
    p {{
      margin: 0 0 14px;
      font-size: 16px;
      line-height: 1.45;
    }}
    .media {{
      width: 100%;
      max-height: min(74vh, 720px);
      object-fit: contain;
      background: var(--soft);
      display: block;
    }}
    .actions {{
      display: flex;
      gap: 22px;
      padding: 12px 16px 16px;
      color: var(--muted);
      font-weight: 700;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <header>
    <div class="brand">reddit</div>
    <div class="search">Search Reddit</div>
  </header>
  <main>
    <article>
      <div class="post-body">
        <div class="meta"><span class="dot"></span>{html.escape(post.subreddit)} - Posted by {html.escape(post.author)} - {html.escape(post.time_label)} {flair_html}</div>
        <h1>{html.escape(post.title)}</h1>
        {body_html}
      </div>
      {media}
      <div class="actions"><span>^ {html.escape(post.votes)}</span><span>Comments {html.escape(post.comments)}</span><span>Share</span><span>Save</span></div>
    </article>
  </main>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")
    return {"output": str(output), "theme": theme}


def load_post_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_post(args: argparse.Namespace) -> PostPreview:
    data: dict[str, Any] = {}
    if args.json:
        data = load_post_json(Path(args.json))
    media_value = args.media if args.media is not None else data.get("media")
    media = Path(media_value) if media_value else None
    return PostPreview(
        title=args.title or data.get("title") or "Untitled Reddit post",
        subreddit=args.subreddit or data.get("subreddit") or "r/libreyolo",
        author=args.author or data.get("author") or "u/libreyolo",
        body=args.body if args.body is not None else data.get("body", ""),
        media=media,
        votes=args.votes or str(data.get("votes", "128")),
        comments=args.comments or str(data.get("comments", "24")),
        flair=args.flair if args.flair is not None else data.get("flair", "Showcase"),
        time_label=args.time_label or data.get("time_label", "just now"),
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Reddit-style local post preview.")
    parser.add_argument("--json", default=None, help="Post JSON file created by build_reddit_post_pack.py.")
    parser.add_argument("--media", default=None, help="Media path to preview.")
    parser.add_argument("--title", default=None, help="Post title.")
    parser.add_argument("--body", default=None, help="Optional post body.")
    parser.add_argument("--subreddit", default=None, help="Subreddit label, for example r/libreyolo.")
    parser.add_argument("--author", default=None, help="Author label, for example u/libreyolo.")
    parser.add_argument("--votes", default=None, help="Displayed vote count.")
    parser.add_argument("--comments", default=None, help="Displayed comment count.")
    parser.add_argument("--flair", default=None, help="Displayed flair text.")
    parser.add_argument("--time-label", default=None, help="Displayed relative post time.")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--device", choices=["mobile", "desktop"], default="mobile")
    parser.add_argument("--output-html", default=None, help="Write animated browser preview HTML.")
    parser.add_argument("--output-png", default=None, help="Write static PNG screenshot.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.output_html and not args.output_png:
        raise SystemExit("Provide --output-html, --output-png, or both.")
    post = build_post(args)
    results = []
    if args.output_html:
        results.append(write_html(post, Path(args.output_html), theme=args.theme))
    if args.output_png:
        results.append(render_png(post, Path(args.output_png), device=args.device, theme=args.theme))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
