"""Kernel registry: pluggable accelerated implementations for library ops.

Architecture (vLLM-shaped, scaled down to a library that rents its runtime):

- Op slots are organized by purpose, not backend:
  ``kernels/quant/simulate/`` holds fake-quantization kernels (any device,
  STE backward, run during QAT *and* simulated PTQ/val inference),
  ``kernels/quant/execute/`` holds finalized-only real-precision paths
  (no backward, real hardware), and ``kernels/attention/`` holds attention
  ops shared across model families.
- The reference implementations in ``quant/fake_quant.py`` are always
  registered and always available. They are the numerical oracle: any
  accelerated implementation must match them (parity tests gate in-tree
  kernels, and ``quant/packing.py`` never has variants because it is the
  checkpoint contract).
- Triton kernels are plain Python (JIT compiled at runtime, no build step,
  no wheels) and are imported lazily on first resolution.
- Compiled kernels ship out-of-tree in the optional ``libreyolo-kernels``
  package. If installed, it is imported here and self-registers via
  :func:`register`; the core package never grows a build dependency.
- Hugging Face Hub kernels load through the optional ``kernels`` package
  (the ``libreyolo[hub-kernels]`` extra); installing it is the opt-in, and
  ``LIBREYOLO_HUB_KERNELS=0`` disables it. Absence of the package is the
  normal fallback case.
- Selection is per-op: implementations are tried newest-first and the first
  one whose predicate passes wins, falling back to the reference. The
  ``LIBREYOLO_KERNELS`` environment variable overrides selection:
  ``off``/``reference`` forces the reference implementations; any other
  value selects only implementations registered under that name.
  ``LIBREYOLO_QUANT_KERNELS`` is honored as a legacy alias.

Registered op slots (callables with the reference signatures):
``fake_quant_int8_per_channel``, ``fake_quant_int8_affine``,
``fake_quant_fp8``, ``fake_quant_int_grouped``, ``fake_quant_nvfp4_weight``,
``fake_quant_nvfp4_dynamic``, ``fake_quant_mxfp4_weight``,
``fake_quant_mxfp4_dynamic``, plus the unpack slots ``unpack_int_grouped``
and ``unpack_nvfp4``. GEMM and attention slots (``fp8_gemm``,
``ms_deform_attn``, ``nvfp4_gemm``, ...) have no reference implementation:
callers must check :func:`resolve` returns non-None before use and keep
their portable path as the fallback.
"""

import importlib
import importlib.util
import logging
import os
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from ..quant import fake_quant as _reference

logger = logging.getLogger(__name__)

REFERENCE_OPS = (
    "fake_quant_int8_per_channel",
    "fake_quant_int8_affine",
    "fake_quant_fp8",
    "fake_quant_int_grouped",
    "fake_quant_nvfp4_weight",
    "fake_quant_nvfp4_dynamic",
    "fake_quant_mxfp4_weight",
    "fake_quant_mxfp4_dynamic",
)

_REGISTRY: Dict[str, List[dict]] = {}
_RESOLVED: Dict[str, Callable] = {}


def register(
    op: str,
    impl: Callable,
    *,
    name: str,
    predicate: Optional[Callable[[], bool]] = None,
):
    """Register an implementation for an op slot (newest wins when eligible)."""
    _REGISTRY.setdefault(op, []).insert(
        0, {"name": name, "impl": impl, "predicate": predicate}
    )
    _RESOLVED.pop(op, None)


def unregister(op: str, name: str):
    _REGISTRY[op] = [e for e in _REGISTRY.get(op, []) if e["name"] != name]
    _RESOLVED.pop(op, None)


def clear_cache():
    _RESOLVED.clear()


def _entry_allowed(entry: dict, forced: str) -> bool:
    if forced in ("off", "reference") and entry["name"] != "reference":
        return False
    if forced and forced not in ("off", "reference") and entry["name"] not in (
        forced,
        "reference",
    ):
        return False
    predicate = entry["predicate"]
    try:
        return predicate is None or predicate()
    except Exception as exc:  # a broken predicate must never break inference
        logger.warning(
            "Kernel predicate failed for %s: %s", entry["name"], exc
        )
        return False


def iter_impls(op: str) -> Iterator[Tuple[str, Callable]]:
    """Yield eligible ``(name, impl)`` pairs newest-first.

    Unlike :func:`resolve`, this does not cache and does not stop at the
    first eligible provider. Slots that can return None for a given input
    (attention, GEMM) walk the rest so a rejected newer impl does not hide
    an older one.
    """
    _ensure_intree_loaded()
    forced = _forced()
    for entry in _REGISTRY.get(op, ()):
        if _entry_allowed(entry, forced):
            yield entry["name"], entry["impl"]


def _forced() -> str:
    forced = os.environ.get("LIBREYOLO_KERNELS", "").strip().lower()
    if forced:
        return forced
    # Legacy alias from when the registry lived under libreyolo/quant/.
    return os.environ.get("LIBREYOLO_QUANT_KERNELS", "").strip().lower()


_INTREE_ATTEMPTED = False

# In-tree providers imported lazily on first resolution. Triton modules and
# the attention provider self-register on import; each group fails
# independently so a missing optional dependency never hides the others.
_LAZY_PROVIDERS = (
    # Triton fake-quant kernels (need triton).
    ".quant.simulate",
    # Triton finalized-path kernels.
    ".quant.execute.fp8_fusion",
    ".quant.execute.unpack_int_grouped",
    ".quant.execute.unpack_nvfp4",
    # Hub-backed attention kernels (need the `kernels` package, opt-in).
    ".attention.ms_deform_attn",
)


def _ensure_intree_loaded():
    """Lazily import the in-tree accelerated providers on first resolution.

    Lazy so `import libreyolo` never pays the triton import cost, and
    skipped entirely when the env forces the reference implementations.
    Absence of triton is the normal fallback case.
    """
    global _INTREE_ATTEMPTED
    if _INTREE_ATTEMPTED or _forced() in ("off", "reference"):
        return
    _INTREE_ATTEMPTED = True
    for module_name in _LAZY_PROVIDERS:
        try:
            importlib.import_module(module_name, __name__)
        except Exception as exc:
            logger.debug("In-tree kernel provider %s unavailable: %s", module_name, exc)


def resolve(op: str) -> Optional[Callable]:
    """Return the active implementation for an op, or None if none eligible."""
    _ensure_intree_loaded()
    forced = _forced()
    cache_key = f"{op}|{forced}"
    if cache_key in _RESOLVED:
        return _RESOLVED[cache_key]

    impl = None
    for _name, candidate in iter_impls(op):
        impl = candidate
        break
    _RESOLVED[cache_key] = impl
    return impl


def active() -> Dict[str, str]:
    """Map of op slot to the name of the implementation currently selected."""
    # Importing the optional in-tree kernels registers new slots. Do that
    # before iterating so a first-ever active() call cannot mutate the dict.
    _ensure_intree_loaded()
    out = {}
    for op, entries in list(_REGISTRY.items()):
        impl = resolve(op)
        out[op] = next(
            (e["name"] for e in entries if e["impl"] is impl), "unavailable"
        )
    return out


def _make_proxy(op: str) -> Callable:
    def proxy(*args, **kwargs):
        impl = resolve(op)
        if impl is None:
            raise RuntimeError(f"No implementation available for op '{op}'.")
        return impl(*args, **kwargs)

    proxy.__name__ = op
    return proxy


# Reference implementations: registered first so they sit last in the list
# (always the fallback) and are selectable via the "reference" name.
for _op in REFERENCE_OPS:
    register(_op, getattr(_reference, _op), name="reference")
    globals()[_op] = _make_proxy(_op)

# Unpack slots (finalized-checkpoint dequantization): the packing module is
# the contract and provides the reference implementations; accelerated
# variants register on top exactly like the fake-quant slots.
from ..quant import packing as _packing  # noqa: E402

UNPACK_OPS = ("unpack_int_grouped", "unpack_nvfp4")
register("unpack_int_grouped", _packing.unpack_int_grouped_weight, name="reference")
register("unpack_nvfp4", _packing.unpack_nvfp4_weight, name="reference")
for _op in UNPACK_OPS:
    globals()[_op] = _make_proxy(_op)


# In-tree GEMM kernels built on stock torch (no triton, no build step).
# fp8_gemm: finalized fp8 QuantLinear on the fp8 tensor cores via
# torch._scaled_mm (Ada/Hopper/Blackwell); resolves to None elsewhere.
from .quant.execute import scaled_mm_fp8  # noqa: E402,F401  (self-registers)

# Optional out-of-tree compiled kernels (e.g. the CUTLASS NVFP4 GEMM).
# The package self-registers on import; absence is the normal case.
if importlib.util.find_spec("libreyolo_kernels") is not None:
    try:
        import libreyolo_kernels  # noqa: F401
    except Exception as exc:
        logger.warning("libreyolo_kernels present but failed to import: %s", exc)
