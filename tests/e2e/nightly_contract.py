"""Versioned contract for the nightly e2e test suite."""

NIGHTLY_E2E_SUITE_VERSION = "3.1"
NIGHTLY_E2E_SUITE_CONTRACT = (
    "general=curated smallest native inference cases for public detector families, "
    "each pulled from a public auto-download route (LibreYOLO HF, or Deci's CDN "
    "for YOLO-NAS); gaze (L2CS/Gaze360) is non-redistributable and runs only in "
    "the non-gated per-family suite, not the gated nightly; "
    "flagship=YOLO9/RF-DETR validation, input equivalence, video, tracking, CLI, "
    "and one RF1 training/reload size per flagship family; training-time CUDA graphs, "
    "inference CUDA graphs, export backends, ExecuTorch, and extended task "
    "training remain opt-in"
)
DEFAULT_NIGHTLY_E2E_MARKERS = frozenset({"general_nightly", "flagship_nightly"})
NIGHTLY_E2E_MARKERS = frozenset({*DEFAULT_NIGHTLY_E2E_MARKERS, "training_nightly"})
NIGHTLY_E2E_ADVANCED_MARKERS = frozenset(
    {
        "cuda_graph",
        "executorch",
        "export_backend",
        "extended_training",
        "training_nightly",
    }
)
NIGHTLY_E2E_SUITE_CHANGE_POLICY = (
    "Bump minor for meaningful coverage additions or threshold/runtime changes; "
    "bump major when a green run makes a materially different promise."
)


def nightly_summary_line() -> str:
    """Return a compact one-line suite identity for logs."""
    return f"LibreYOLO nightly e2e suite v{NIGHTLY_E2E_SUITE_VERSION}: {NIGHTLY_E2E_SUITE_CONTRACT}"


def nightly_markdown_summary() -> str:
    """Return a GitHub-step-summary friendly suite identity."""
    return "\n".join(
        [
            f"### LibreYOLO nightly e2e suite v{NIGHTLY_E2E_SUITE_VERSION}",
            "",
            NIGHTLY_E2E_SUITE_CONTRACT,
            "",
            NIGHTLY_E2E_SUITE_CHANGE_POLICY,
        ]
    )


def nightly_advanced_marker_conflicts(marker_names) -> tuple[str, ...]:
    """Return advanced markers that overlap with a default-nightly marker."""
    names = set(marker_names)
    if not names.intersection(DEFAULT_NIGHTLY_E2E_MARKERS):
        return ()
    return tuple(sorted(names.intersection(NIGHTLY_E2E_ADVANCED_MARKERS)))
