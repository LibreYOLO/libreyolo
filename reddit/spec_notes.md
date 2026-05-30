# Reddit Creative Notes

These defaults are tuned for local post creation, not guaranteed upload approval.
Subreddit rules can be stricter than platform limits.

## Defaults

- Use `mobile-feed-gif` first: `1080x1350`, `4:5`, 12 fps, target below 20 MB.
- Use `strict-gif` when a subreddit or campaign wants a small file: same canvas, 10 fps, target below 3 MB.
- Use `square-gif` when the community feed crops tall media or the visual depends on centered composition.
- Use `desktop-landscape-gif` when the target audience is desktop-heavy.

## Sourcing

- Reddit Help says GIF comments are selected from GIPHY, and some communities disable GIFs or NSFW GIF comments.
  Source: https://support.reddithelp.com/hc/en-us/articles/7765031267988-How-do-I-add-GIFs-in-comments
- Reddit Help says video comments can be up to 3 minutes, must be under 1 GB, and recommends less than 300 MB.
  Source: https://support.reddithelp.com/hc/en-us/articles/48109333836692-How-do-I-add-video-in-comments
- Reddit ad/spec aggregators list `4:5`, `1:1`, `4:3`, and `16:9` as supported Reddit creative ratios. The local default chooses `4:5` because it gives mobile feed posts more visual area.
  Reference: https://help.metadata.io/portal/articles/get-started-what-are-the-specifications-for-ad-creative

## Practical Rules

- Make the first frame readable without motion; it is the thumbnail in many contexts.
- Keep headline text short enough to scan in one breath.
- Treat text inside the GIF as a label, not a paragraph.
- Prefer 4-6 seconds for GIF loops. Long GIFs balloon in size and often lose clarity after optimization.
- Test mobile and desktop previews before posting.
