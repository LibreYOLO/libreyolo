"""Build hook for the PyPI page; all other metadata is in pyproject.toml.

PyPI renders the package description from README.md, but it does not draw
Mermaid and it does not resolve paths relative to the repository (#1057).
``long_description()`` returns README.md with each Mermaid block replaced by a
link to the same section on GitHub, and with relative image and link targets
made absolute against the ``release`` branch. README.md itself is unchanged.
"""

import re
from pathlib import Path

REPO = "https://github.com/LibreYOLO/libreyolo"
RAW = "https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/"
BLOB = f"{REPO}/blob/release/"

# CommonMark: up to 3 spaces of indent; a backtick fence's info has no backticks
_FENCE = re.compile(r"^ {0,3}(?:(`{3,})([^`]*)|(~{3,})(.*))$")
_HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$")
_MD_TARGET = re.compile(r"(\]\()([^)\s]+)")
_HTML_ATTR = re.compile(r"(\b(?:src|href)=\")([^\"]+)(\")", re.IGNORECASE)
_ABSOLUTE = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//|#)", re.IGNORECASE)
_IMAGE = re.compile(r"\.(?:png|jpe?g|gif|svg|webp)$", re.IGNORECASE)


def _absolute(target: str) -> str:
    """A repository-relative target as a URL: images from raw, the rest from blob."""
    if _ABSOLUTE.match(target):
        return target
    path = target.removeprefix("./").lstrip("/")
    return (RAW if _IMAGE.search(path.split("#")[0].split("?")[0]) else BLOB) + path


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def long_description(readme: str) -> str:
    out: list[str] = []
    fence = ""  # the opening run of an open code fence, or "" outside one
    mermaid = False
    anchor = ""
    for line in readme.splitlines():
        match = _FENCE.match(line.rstrip())
        run, info = (
            (match.group(1) or match.group(3), match.group(2) or match.group(4) or "")
            if match
            else ("", "")
        )
        if fence:
            if match and run.startswith(fence) and not info.strip():
                fence = ""
                if mermaid:
                    mermaid = False
                    target = f"{REPO}#{anchor}" if anchor else REPO
                    out.append(
                        f"*This diagram is drawn on [GitHub]({target}); PyPI cannot render Mermaid.*"
                    )
                    continue
            if not mermaid:
                out.append(line)
            continue
        if match:
            fence = run
            mermaid = (info.split() or [""])[0].lower() == "mermaid"
            if not mermaid:
                out.append(line)
            continue
        heading = _HEADING.match(line)
        if heading:
            anchor = _slug(heading.group(1))
        line = _MD_TARGET.sub(lambda m: m.group(1) + _absolute(m.group(2)), line)
        line = _HTML_ATTR.sub(
            lambda m: m.group(1) + _absolute(m.group(2)) + m.group(3), line
        )
        out.append(line)
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    from setuptools import setup

    setup(
        long_description=long_description(
            Path(__file__).with_name("README.md").read_text(encoding="utf-8")
        ),
        long_description_content_type="text/markdown",
    )
