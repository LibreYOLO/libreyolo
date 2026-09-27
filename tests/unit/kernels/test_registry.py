"""CPU tests for the library-wide kernel registry and its quant shim."""

from __future__ import annotations

import subprocess
import sys

import pytest

from libreyolo import kernels

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_registry_env(monkeypatch):
    monkeypatch.delenv("LIBREYOLO_KERNELS", raising=False)
    monkeypatch.delenv("LIBREYOLO_QUANT_KERNELS", raising=False)
    kernels.clear_cache()
    yield
    kernels.clear_cache()


def test_quant_shim_forwards_to_registry():
    from libreyolo.quant import kernels as shim

    assert shim.resolve is kernels.resolve
    assert shim.register is kernels.register
    assert shim.active is kernels.active
    assert shim.clear_cache is kernels.clear_cache
    assert shim.REFERENCE_OPS == kernels.REFERENCE_OPS


def test_reference_ops_resolve_on_cpu():
    for op in kernels.REFERENCE_OPS + kernels.UNPACK_OPS:
        assert kernels.resolve(op) is not None, op


def test_env_force_reference(monkeypatch):
    reference = [
        e["impl"]
        for e in kernels._REGISTRY["fake_quant_fp8"]
        if e["name"] == "reference"
    ][0]
    monkeypatch.setenv("LIBREYOLO_KERNELS", "off")
    kernels.clear_cache()
    assert kernels.resolve("fake_quant_fp8") is reference

    monkeypatch.delenv("LIBREYOLO_KERNELS")
    monkeypatch.setenv("LIBREYOLO_QUANT_KERNELS", "reference")
    kernels.clear_cache()
    assert kernels.resolve("fake_quant_fp8") is reference


def test_active_lists_attention_slot(monkeypatch):
    # Both accelerated providers pinned off: the slot must report unavailable.
    monkeypatch.setenv("LIBREYOLO_HUB_KERNELS", "0")
    monkeypatch.setenv("LIBREYOLO_TRITON_MSDA", "0")
    kernels.clear_cache()
    assert kernels.active().get("ms_deform_attn") == "unavailable"


def test_registry_importable_before_quant():
    """Importing libreyolo.kernels first must survive the quant cycle."""
    code = (
        "import libreyolo.kernels as k; "
        "import libreyolo.quant; "
        "assert k.resolve('fake_quant_fp8') is not None; "
        "assert libreyolo.quant.kernels.resolve is k.resolve"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300)
