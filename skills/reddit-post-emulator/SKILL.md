---
name: reddit-post-emulator
description: Preview and screenshot Reddit-style posts locally. Use when Codex needs to inspect how a Reddit post, GIF, image, or video will look in mobile and desktop feeds before a human posts it.
---

# Emulate Reddit Posts Locally

Use `reddit/tools/reddit_emulator.py` for browser HTML previews and deterministic PNG screenshots. This is a local visual emulator, not a Reddit API client.

## Workflow

1. Build or locate the media asset.
2. Generate `preview.html` for animated inspection when the media is a GIF or video.
3. Generate both mobile and desktop PNG screenshots.
4. Check that the title wraps cleanly, the first frame is readable, and the media is not cropped in a misleading way.
5. If screenshots look poor, revise the GIF with `reddit/tools/make_reddit_gif.py` and rerender.

## Commands

Preview one post:

```powershell
.\.venv\Scripts\python.exe reddit\tools\reddit_emulator.py `
  --media reddit\outputs\post.gif `
  --title "Short concrete hook" `
  --body "Optional context" `
  --subreddit "r/example" `
  --output-html reddit\outputs\preview.html `
  --output-png reddit\outputs\preview-mobile.png
```

Desktop screenshot:

```powershell
.\.venv\Scripts\python.exe reddit\tools\reddit_screenshot.py `
  --json reddit\outputs\my-post\post.json `
  --device desktop `
  --output-png reddit\outputs\my-post\preview-desktop.png
```

Dark theme screenshot:

```powershell
.\.venv\Scripts\python.exe reddit\tools\reddit_screenshot.py `
  --json reddit\outputs\my-post\post.json `
  --theme dark `
  --output-png reddit\outputs\my-post\preview-mobile-dark.png
```

## Review Criteria

- The hook should be readable at mobile width.
- The post should still make sense if only the first GIF frame is shown.
- Any body text should support the title, not repeat it.
- Generated assets are local drafts; the human handles final posting.
