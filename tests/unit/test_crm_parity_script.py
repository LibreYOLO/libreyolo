"""weights/parity_crm.py must always write its report, whatever optional step it skips."""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_parity_gate_measures_every_gated_stage_unconditionally():
    """Every BITWISE / LIMITS key of weights/parity_crm.py is set outside a try block.

    An entry set only on an optional path left the report to die with a
    KeyError after the expensive runs.
    """
    script = Path(__file__).resolve().parents[2] / "weights" / "parity_crm.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    gated = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("BITWISE", "LIMITS"):
                gated |= {c.value for c in ast.walk(node.value) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
        if isinstance(node, ast.AugAssign) and getattr(node.target, "id", None) == "BITWISE":
            gated |= {c.value for c in ast.walk(node.value) if isinstance(c, ast.Constant)}
    optional = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for inner in node.body:
                for sub in ast.walk(inner):
                    if isinstance(sub, ast.Subscript) and getattr(sub.value, "id", None) == "ROW":
                        if isinstance(sub.ctx, ast.Store) and isinstance(sub.slice, ast.Constant):
                            optional.add(sub.slice.value)
    unconditional = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and getattr(node.value, "id", None) == "ROW":
            if isinstance(node.ctx, ast.Store) and isinstance(node.slice, ast.Constant):
                unconditional.add(node.slice.value)
    assert gated and gated <= unconditional
    assert not gated & optional, f"gated only on an optional path: {sorted(gated & optional)}"
