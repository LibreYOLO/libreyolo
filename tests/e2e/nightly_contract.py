"""Versioned contract for the nightly e2e test suite."""

NIGHTLY_E2E_SUITE_VERSION = "2.0"
NIGHTLY_E2E_SUITE_CONTRACT = (
    "general=native inference stability across the YOLO9 and RF-DETR flagship "
    "size range, using auto-downloadable LibreYOLO HF weights only; "
    "flagship=heavier native YOLO9/RF-DETR validation, training, video, "
    "tracking, and CLI checks; the wider model zoo and non-redistributable "
    "weights (e.g. YOLO-NAS, L2CS) run in non-gated per-family suites; "
    "export backends remain outside the default nightly"
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
