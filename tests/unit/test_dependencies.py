"""Tests for declared dependency floors."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import pytest

pytestmark = pytest.mark.unit


def test_rfdetr_extra_uses_native_dependencies():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["optional-dependencies"]["rfdetr"]
    assert "transformers>=5.1.0" in deps
    assert "scipy>=1.7.0" not in deps
    assert all(not dep.startswith("rfdetr") for dep in deps)


def test_all_extra_includes_triton_serving_dependencies():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    all_deps = pyproject["project"]["optional-dependencies"]["all"]
    assert "libreyolo[triton]" in all_deps


def test_core_dependencies_include_import_chain_requirements():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    assert "Pillow>=9.1.0" in deps
    assert "scipy>=1.7.0" in deps
    assert "torchvision>=0.19.0" in deps


def test_torch_floor_supports_amp_grad_scaler():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    assert "torch>=2.4.0" in deps


def test_paddle_extra_pins_the_measured_converter_stack():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["optional-dependencies"]["paddle"]
    assert "libreyolo[onnx]" in deps
    assert "onnx>=1.14.0,<1.18" in deps
    assert any(dep.startswith("paddlepaddle==2.6.2") for dep in deps)
    assert any(dep.startswith("x2paddle==1.6.0") for dep in deps)
    assert any(dep.startswith("six>=1.16.0") for dep in deps)


def test_openvocab_extra_covers_clip_tokenizer_runtime():
    """OV-DEIM always embeds prompts with the vendored CLIP BPE tokenizer.

    That tokenizer imports ftfy and regex at predict time, so a clean
    ``pip install libreyolo[openvocab]`` must ship them or the first
    LibreOVDEIM prediction raises ImportError (v1.4.0 release blocker).
    """
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["optional-dependencies"]["openvocab"]
    names = {dep.split(">=")[0].split("==")[0].strip() for dep in deps}
    assert "ftfy" in names
    assert "regex" in names


def _extra(name):
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    return pyproject["project"]["optional-dependencies"][name]


def test_extras_avoid_packages_that_compile_on_install():
    # #989: pip compiled these from source and failed without a toolchain.
    assert not any(dep.startswith("pycuda") for dep in _extra("tensorrt"))
    assert not any(dep.startswith("onnx-simplifier") for dep in _extra("tflite"))
    (coremltools,) = [d for d in _extra("coreml") if d.startswith("coremltools")]
    assert "sys_platform == 'darwin'" in coremltools


def test_executorch_extra_pins_the_torch_its_runtime_was_built_for():
    # ExecuTorch 1.2's runtime fails to load next to a newer torch (#989).
    deps = _extra("executorch")
    assert "executorch>=1.2,<1.3" in deps
    assert "torch>=2.11,<2.12" in deps
