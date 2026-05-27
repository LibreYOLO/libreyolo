"""Versioned contract for the nightly e2e test suite."""

NIGHTLY_E2E_SUITE_VERSION = "2.0"
NIGHTLY_E2E_SUITE_CONTRACT = (
    "general=YOLO9/RF-DETR native load+inference for every detection size; "
    "flagship=simple YOLO9/RF-DETR load smoke for every detection size, "
    "smallest-size RF1 training, and smallest-size ONNX export+inference"
)
NIGHTLY_E2E_SUITE_CHANGE_POLICY = (
    "Bump minor for meaningful coverage additions or threshold/runtime changes; "
    "bump major when a green run makes a materially different promise."
)


def nightly_summary_line() -> str:
    return (
        f"LibreYOLO nightly e2e suite v{NIGHTLY_E2E_SUITE_VERSION}: "
        f"{NIGHTLY_E2E_SUITE_CONTRACT}"
    )


def nightly_markdown_summary() -> str:
    return (
        f"### LibreYOLO nightly e2e suite v{NIGHTLY_E2E_SUITE_VERSION}\n\n"
        f"{NIGHTLY_E2E_SUITE_CONTRACT}\n\n"
        f"{NIGHTLY_E2E_SUITE_CHANGE_POLICY}\n\n"
        "Job summary generated at run-time.\n"
    )
