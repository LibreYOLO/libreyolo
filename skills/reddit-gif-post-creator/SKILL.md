---
name: reddit-gif-post-creator
description: Create optimized Reddit GIF post assets and local post packs. Use when Codex needs to turn an idea, image set, GIF, or video clip into a Reddit-ready animated GIF with a title hook, size optimization, preview screenshots, and a manual posting checklist.
---

# Create Reddit GIF Post Assets

Use the local toolkit under `reddit/tools/`. Do not automate Reddit submission, voting, commenting, or account activity; produce assets for a human to post manually.

## Workflow

1. Read `reddit/specs.json` before choosing a canvas.
2. Prefer `mobile-feed-gif` for organic posts unless the user names a stricter subreddit or asks for square/desktop output.
3. Use `build_reddit_post_pack.py` for a complete pack. Use `make_reddit_gif.py` only when the user asks just for a GIF.
4. Inspect the generated GIF summary. If the tool had to shrink below the target canvas, tell the user.
5. Render or review the mobile and desktop previews before final delivery.

## Commands

Complete pack:

```powershell
.\.venv\Scripts\python.exe reddit\tools\build_reddit_post_pack.py `
  --input path\to\media.mp4 `
  --title "Short concrete hook" `
  --subtitle "Optional GIF label" `
  --subreddit "r/example"
```

Text-only GIF pack:

```powershell
.\.venv\Scripts\python.exe reddit\tools\build_reddit_post_pack.py `
  --title "Short concrete hook" `
  --subtitle "A visual reason to click"
```

Strict small GIF:

```powershell
.\.venv\Scripts\python.exe reddit\tools\make_reddit_gif.py `
  --input path\to\clip.mp4 `
  --preset strict-gif `
  --output reddit\outputs\post.gif
```

Floating cropped subject with left text:

```powershell
.\.venv\Scripts\python.exe reddit\tools\build_reddit_post_pack.py `
  --input libreyolo\assets\parkour_result.jpg `
  --title "LibreYOLO now supports 16 model families" `
  --left-text "16 models supported" `
  --motion float-crop `
  --float-pixels 14 `
  --no-caption
```

Use `--float-crop-box x,y,w,h` when the default crop does not center the subject. Values are normalized to the source image.

Transparent subject layer with inpainted background:

```powershell
.\.venv\Scripts\python.exe reddit\tools\build_reddit_post_pack.py `
  --input marketing\parkour.jpg `
  --subject-layer marketing\original-no-background.png `
  --title "LibreYOLO now supports 16 model families" `
  --left-text "16 models supported" `
  --motion float-layer `
  --subject-scale 1.15 `
  --subject-offset-x -120 `
  --float-pixels 18 `
  --no-caption
```

Use `float-layer` when there is a transparent subject cutout. The tool fills the old subject area using the alpha mask and animates only the subject layer.

## Quality Checks

- First frame must make sense as a static thumbnail.
- Title should usually stay under 150 characters.
- On-GIF text should be a label or hook, not paragraph copy.
- Prefer 4-6 second loops unless the user asks otherwise.
- Confirm the target subreddit permits GIF/image posts.
