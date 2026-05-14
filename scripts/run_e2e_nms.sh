#!/usr/bin/env bash
# run_e2e_nms.sh — E2E smoke tests for the NMS speed-up branch (#202).
#
# Runs in two stages:
#   1. Unit tests  (fast, always)
#   2. RF1 training + val on the two affected families: YOLOX and YOLO9
#
# Usage:
#   ./scripts/run_e2e_nms.sh              # both stages
#   ./scripts/run_e2e_nms.sh --unit-only  # skip e2e (no weights needed)
#   ./scripts/run_e2e_nms.sh --e2e-only   # skip unit tests
#   ./scripts/run_e2e_nms.sh --size n     # only yolox-n and yolo9-n (smallest/fastest)
#
# Requirements:
#   - Project venv activated, or .venv/bin/pytest on PATH
#   - RF1 e2e: weights for the selected size(s) must be reachable
#     (pytest will auto-skip individual cases when weights are missing)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
RUN_UNIT=1
RUN_E2E=1
SIZE_FILTER=""   # e.g. "-n" → test IDs containing "-n"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --unit-only)  RUN_E2E=0;  shift ;;
        --e2e-only)   RUN_UNIT=0; shift ;;
        --size)       SIZE_FILTER="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve pytest
# ---------------------------------------------------------------------------
if command -v pytest &>/dev/null; then
    PYTEST=pytest
elif [[ -x ".venv/bin/pytest" ]]; then
    PYTEST=.venv/bin/pytest
else
    echo "ERROR: pytest not found. Activate the project venv or install pytest." >&2
    exit 1
fi

UNIT_STATUS="skipped"
E2E_STATUS="skipped"

run_stage() {
    local label="$1"; shift
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $label"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if "$PYTEST" "$@"; then
        echo "pass"
    else
        echo "fail"
    fi
}

# ---------------------------------------------------------------------------
# Stage 1 — unit tests
# ---------------------------------------------------------------------------
if [[ $RUN_UNIT -eq 1 ]]; then
    UNIT_STATUS="$(run_stage "Unit tests" tests/unit/ -q --tb=short)"
fi

# ---------------------------------------------------------------------------
# Stage 2 — RF1 e2e: train + val on YOLOX and YOLO9
# ---------------------------------------------------------------------------
if [[ $RUN_E2E -eq 1 ]]; then
    # Parametrized test IDs look like [yolox-n], [yolo9-t], etc.
    # When --size is given, match "yolox-{size} or yolo9-{size}" directly
    # against the bracket suffix (avoids "training" containing common letters).
    if [[ -n "$SIZE_FILTER" ]]; then
        K_EXPR="yolox-${SIZE_FILTER} or yolo9-${SIZE_FILTER}"
    else
        K_EXPR="yolox or yolo9"
    fi

    E2E_STATUS="$(run_stage "RF1 e2e — YOLOX + YOLO9 (k='${K_EXPR}')" \
        tests/e2e/test_rf1_training.py \
        -v -m e2e -k "$K_EXPR" \
        --tb=short)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
_icon() { [[ "$1" == "pass" ]] && echo "✓" || ([[ "$1" == "fail" ]] && echo "✗" || echo "—"); }

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  $(_icon "$UNIT_STATUS") Unit tests   $UNIT_STATUS"
echo "  $(_icon "$E2E_STATUS") E2E (RF1)    $E2E_STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[[ "$UNIT_STATUS" != "fail" && "$E2E_STATUS" != "fail" ]]
