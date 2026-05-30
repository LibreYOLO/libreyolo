from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_reddit_gif import make_gif
from reddit_emulator import PostPreview, render_png, write_html


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:64] or "reddit-post"


def write_checklist(path: Path, data: dict[str, Any]) -> None:
    lines = [
        "# Posting Checklist",
        "",
        "- Open `preview.html` and check the animated GIF loop.",
        "- Open `preview-mobile.png` and confirm the first frame is readable.",
        "- Open `preview-desktop.png` and confirm the media is not awkwardly cropped.",
        "- Confirm the subreddit allows image/GIF posts and that this post follows its rules.",
        "- Keep the title under 150 characters when possible.",
        "- Do the final Reddit submission manually.",
        "",
        f"Title length: {len(data['title'])} characters",
        f"Media: `{data['media']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a complete local Reddit GIF post pack.")
    parser.add_argument("--input", action="append", default=[], help="Input image, GIF, or video. Repeat for slides.")
    parser.add_argument("--title", required=True, help="Reddit post title and GIF hook.")
    parser.add_argument("--subtitle", default=None, help="Secondary text overlay for the GIF.")
    parser.add_argument("--left-text", default=None, help="Text burned into the left side of the GIF.")
    parser.add_argument("--left-text-style", choices=["panel", "clean"], default="panel", help="Left text rendering style.")
    parser.add_argument("--body", default="", help="Optional Reddit post body.")
    parser.add_argument("--subreddit", default="r/libreyolo", help="Subreddit label.")
    parser.add_argument("--author", default="u/libreyolo", help="Author label for preview.")
    parser.add_argument("--flair", default="Showcase", help="Flair label for preview.")
    parser.add_argument("--preset", default=None, help="GIF preset from reddit/specs.json.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to reddit/outputs/<slug>.")
    parser.add_argument("--width", type=int, default=None, help="Override GIF width.")
    parser.add_argument("--height", type=int, default=None, help="Override GIF height.")
    parser.add_argument("--max-mb", type=float, default=None, help="Override GIF size target.")
    parser.add_argument("--duration", type=float, default=None, help="Override GIF duration.")
    parser.add_argument("--fps", type=int, default=None, help="Override GIF fps.")
    parser.add_argument("--fit", choices=["cover", "contain"], default=None, help="Media fit mode.")
    parser.add_argument("--motion", choices=["ken-burns", "float-crop", "float-layer"], default="ken-burns", help="GIF animation style.")
    parser.add_argument("--float-pixels", type=int, default=16, help="Vertical float amount for float-crop.")
    parser.add_argument("--float-crop-box", default=None, help="Normalized crop box for float-crop as x,y,w,h.")
    parser.add_argument("--subject-layer", default=None, help="Transparent PNG subject layer for float-layer.")
    parser.add_argument("--subject-scale", type=float, default=1.12, help="Scale factor for float-layer subject.")
    parser.add_argument("--subject-offset-x", type=int, default=0, help="Source-pixel horizontal offset for float-layer.")
    parser.add_argument("--subject-offset-y", type=int, default=0, help="Source-pixel vertical offset for float-layer.")
    parser.add_argument("--inpaint-radius", type=int, default=7, help="Inpaint radius for float-layer background fill.")
    parser.add_argument("--no-caption", action="store_true", help="Do not overlay title/subtitle at the bottom of media GIFs.")
    parser.add_argument("--theme", choices=["light", "dark"], default="light", help="Preview theme.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else root / "outputs" / slugify(args.title)
    out_dir.mkdir(parents=True, exist_ok=True)

    gif_path = out_dir / "post.gif"
    gif_summary = make_gif(
        inputs=[Path(value) for value in args.input],
        output=gif_path,
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

    post_data = {
        "title": args.title,
        "body": args.body,
        "subreddit": args.subreddit,
        "author": args.author,
        "flair": args.flair,
        "media": str(gif_path),
        "votes": "128",
        "comments": "24",
        "time_label": "just now",
        "gif_summary": gif_summary,
    }
    post_json = out_dir / "post.json"
    post_json.write_text(json.dumps(post_data, indent=2), encoding="utf-8")

    post = PostPreview(
        title=args.title,
        subreddit=args.subreddit,
        author=args.author,
        body=args.body,
        media=gif_path,
        flair=args.flair,
    )
    html_path = out_dir / "preview.html"
    mobile_png = out_dir / "preview-mobile.png"
    desktop_png = out_dir / "preview-desktop.png"
    write_html(post, html_path, theme=args.theme)
    render_png(post, mobile_png, device="mobile", theme=args.theme)
    render_png(post, desktop_png, device="desktop", theme=args.theme)
    write_checklist(out_dir / "posting-checklist.md", post_data)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "post_json": str(post_json),
                "gif": str(gif_path),
                "preview_html": str(html_path),
                "preview_mobile_png": str(mobile_png),
                "preview_desktop_png": str(desktop_png),
                "gif_summary": gif_summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
