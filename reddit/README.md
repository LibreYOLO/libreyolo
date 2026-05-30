# Reddit GIF Post Toolkit

Local tools for building Reddit-ready GIF post packs without posting anything automatically.

## Build a Complete Pack

```powershell
.\.venv\Scripts\python.exe reddit\tools\build_reddit_post_pack.py `
  --title "LibreYOLO result in 4 seconds" `
  --subtitle "Fast object detection loop" `
  --subreddit "r/computervision"
```

This creates `reddit/outputs/<slug>/` with:

- `post.gif`
- `post.json`
- `preview.html`
- `preview-mobile.png`
- `preview-desktop.png`
- `posting-checklist.md`

## Create Only a GIF

```powershell
.\.venv\Scripts\python.exe reddit\tools\make_reddit_gif.py `
  --input path\to\clip.mp4 `
  --title "The result after one pass" `
  --subtitle "Watch the detections lock in" `
  --output reddit\outputs\post.gif
```

Useful presets live in `reddit/specs.json`:

- `mobile-feed-gif`: default, 1080x1350, 4:5.
- `strict-gif`: same canvas with a 3 MB target.
- `square-gif`: 1080x1080.
- `desktop-landscape-gif`: 1440x1080.

## Floating Crop Example

```powershell
.\.venv\Scripts\python.exe reddit\tools\build_reddit_post_pack.py `
  --input libreyolo\assets\parkour_result.jpg `
  --title "LibreYOLO now supports 16 model families" `
  --left-text "16 models supported" `
  --motion float-crop `
  --float-pixels 14 `
  --no-caption
```

Use `--float-crop-box x,y,w,h` to tune which part of the source image floats.
The values are normalized to the source image, so `0,0,1,1` is the full image.

## Transparent Subject Layer Example

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

`float-layer` uses the transparent PNG alpha mask to fill the old subject area in the background, then animates only the subject layer.

## Preview or Screenshot a Post

```powershell
.\.venv\Scripts\python.exe reddit\tools\reddit_emulator.py `
  --media reddit\outputs\post.gif `
  --title "LibreYOLO result in 4 seconds" `
  --output-html reddit\outputs\preview.html `
  --output-png reddit\outputs\preview-mobile.png
```

The HTML preview animates GIFs. The PNG screenshot uses the first frame so the thumbnail/readability check is deterministic.
