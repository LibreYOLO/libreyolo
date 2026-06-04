"""Versioned contract for the nightly e2e test suite."""

NIGHTLY_E2E_SUITE_VERSION = "2.1"
NIGHTLY_E2E_SUITE_CONTRACT = (
    "general=smallest native inference case for every public detector family, "
    "each pulled from a public auto-download route (LibreYOLO HF, or Deci's CDN "
    "for YOLO-NAS); gaze (L2CS/Gaze360) is non-redistributable and runs only in "
    "the non-gated per-family suite, not the gated nightly; "
    "flagship=YOLO9/RF-DETR validation, video, tracking, CLI, and one RF1 "
    "training/reload size per flagship family; export backends remain outside "
    "the default nightly"
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
