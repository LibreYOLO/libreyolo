"""The PyPI page built from README.md by setup.py (#1057)."""

import importlib.util
import re
import runpy
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location("libreyolo_setup", ROOT / "setup.py")
setup_py = importlib.util.module_from_spec(_spec)
# __name__ is not "__main__" here, so setup() does not run
_spec.loader.exec_module(setup_py)

RAW = "https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/"
BLOB = "https://github.com/LibreYOLO/libreyolo/blob/release/"


def test_readme_is_built_by_setup_py():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    assert "readme" not in project
    assert "readme" in project["dynamic"]


def test_mermaid_block_becomes_a_link_to_its_section():
    readme = "## Complete list of models\n\nText.\n\n```mermaid\nflowchart TB\n  A --> B\n```\n\nAfter.\n"
    out = setup_py.long_description(readme)
    assert "mermaid" not in out.split("PyPI cannot render")[0].lower()
    assert "flowchart" not in out
    assert "(https://github.com/LibreYOLO/libreyolo#complete-list-of-models)" in out
    assert out.endswith("After.\n")


def test_relative_targets_become_absolute():
    readme = (
        '<img src="docs/readme/banner.png" alt="b">\n'
        "[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)\n"
        "![shot](./libreyolo/assets/x.jpg) and [guide](CONTRIBUTING.md#setup)\n"
        '<a href="SPONSORS.md">s</a> [top](#install) [docs](https://www.libreyolo.com/docs)\n'
    )
    out = setup_py.long_description(readme)
    assert f'src="{RAW}docs/readme/banner.png"' in out
    assert f"]({BLOB}LICENSE)" in out
    assert "(https://img.shields.io/badge/license-MIT-green)" in out
    assert f"]({RAW}libreyolo/assets/x.jpg)" in out
    assert f"]({BLOB}CONTRIBUTING.md#setup)" in out
    assert f'href="{BLOB}SPONSORS.md"' in out
    assert "](#install)" in out
    assert "](https://www.libreyolo.com/docs)" in out


def test_code_blocks_are_left_alone():
    readme = '```python\nresults = model("image.jpg")\nprint(boxes[0](x))\n```\n'
    assert setup_py.long_description(readme) == readme


def test_setup_py_passes_the_built_description(monkeypatch):
    setuptools = pytest.importorskip("setuptools")
    seen = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kw: seen.update(kw))
    runpy.run_path(str(ROOT / "setup.py"), run_name="__main__")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert seen["long_description"] == setup_py.long_description(readme)
    assert seen["long_description_content_type"] == "text/markdown"


def test_only_real_fences_open_a_block():
    readme = (
        "Example:\n\n    ```mermaid\n\n```foo``` inline\n\n## Install\n\n"
        "[guide](CONTRIBUTING.md)\n"
    )
    out = setup_py.long_description(readme)
    assert "## Install" in out
    assert f"]({BLOB}CONTRIBUTING.md)" in out


def test_mermaid_fence_with_attributes_is_replaced():
    readme = "## Models\n\n~~~mermaid title=x\nflowchart TB\n~~~\n"
    out = setup_py.long_description(readme)
    assert "flowchart" not in out
    assert "#models)" in out


def test_the_real_readme_has_nothing_pypi_cannot_show():
    out = setup_py.long_description((ROOT / "README.md").read_text(encoding="utf-8"))
    assert "```mermaid" not in out
    targets = re.findall(r"\]\(([^)\s]+)", out) + re.findall(
        r'(?:src|href)="([^"]+)"', out
    )
    relative = [
        t
        for t in targets
        if not re.match(r"^(?:[a-z][a-z0-9+.-]*:|//|#)", t, re.IGNORECASE)
    ]
    assert relative == []
