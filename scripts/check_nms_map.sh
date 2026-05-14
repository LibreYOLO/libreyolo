#!/usr/bin/env bash
# check_nms_map.sh — Verify mAP is bit-equivalent before/after the NMS speed-up.
#
# Saves a baseline on the dev branch, then compares on the current branch.
#
# Usage:
#   ./scripts/check_nms_map.sh                        # default: yolox-n yolo9-t
#   ./scripts/check_nms_map.sh --models yolox-n       # specific models
#   ./scripts/check_nms_map.sh --device cuda          # use GPU
#   ./scripts/check_nms_map.sh --baseline-branch main # compare against main instead of dev
#   ./scripts/check_nms_map.sh --skip-baseline        # skip baseline step (reuse existing)
#
# Requirements:
#   - Git repo with a clean working tree (stash or commit any changes first)
#   - Weight files reachable in project root or weights/ subdir
#   - Project venv activated
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BASELINE_BRANCH="dev"
BASELINE_FILE="scripts/baseline.json"
NEW_FILE="scripts/new.json"
MODELS=()          # empty → use Python script default (yolox-n yolo9-t)
DEVICE="auto"
SKIP_BASELINE=0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)          shift; while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done ;;
        --device)          DEVICE="$2"; shift 2 ;;
        --baseline-branch) BASELINE_BRANCH="$2"; shift 2 ;;
        --baseline-file)   BASELINE_FILE="$2"; shift 2 ;;
        --new-file)        NEW_FILE="$2"; shift 2 ;;
        --skip-baseline)   SKIP_BASELINE=1; shift ;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
done

MODELS_ARGS=()
[[ ${#MODELS[@]} -gt 0 ]] && MODELS_ARGS=(--models "${MODELS[@]}")

FEATURE_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
sep() { echo; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
header() { sep; echo "  $*"; sep; }

guard_clean_tree() {
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "ERROR: working tree has uncommitted changes. Stash or commit first." >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Stage 1 — baseline on dev
# ---------------------------------------------------------------------------
if [[ $SKIP_BASELINE -eq 1 ]]; then
    header "Stage 1 — baseline (skipped, reusing $BASELINE_FILE)"
    if [[ ! -f "$BASELINE_FILE" ]]; then
        echo "ERROR: --skip-baseline given but $BASELINE_FILE does not exist." >&2
        exit 1
    fi
else
    header "Stage 1 — baseline on '$BASELINE_BRANCH'"
    guard_clean_tree
    echo
    echo "  >>> Switching to branch: $BASELINE_BRANCH"
    git checkout "$BASELINE_BRANCH"
    echo "  >>> Now on branch: $(git rev-parse --abbrev-ref HEAD)"
    echo
    BASELINE_EXIT=0
    python scripts/spot_check_val_map.py \
        --save-baseline "$BASELINE_FILE" \
        --device "$DEVICE" \
        "${MODELS_ARGS[@]+"${MODELS_ARGS[@]}"}" || BASELINE_EXIT=$?
    echo
    echo "  >>> Switching back to branch: $FEATURE_BRANCH"
    git checkout "$FEATURE_BRANCH"
    echo "  >>> Now on branch: $(git rev-parse --abbrev-ref HEAD)"
    echo
    if [[ $BASELINE_EXIT -ne 0 ]]; then
        echo "  WARNING: baseline had failures (exit $BASELINE_EXIT) — some models may be missing from baseline."
    else
        echo "  Baseline saved to $BASELINE_FILE"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 2 — compare on feature branch
# ---------------------------------------------------------------------------
header "Stage 2 — comparison on '$FEATURE_BRANCH'"
python scripts/spot_check_val_map.py \
    --baseline "$BASELINE_FILE" \
    --save "$NEW_FILE" \
    --device "$DEVICE" \
    "${MODELS_ARGS[@]+"${MODELS_ARGS[@]}"}"
COMPARE_EXIT=$?

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
sep
if [[ $COMPARE_EXIT -eq 0 ]]; then
    echo "  RESULT: PASS — mAP is bit-equivalent on '$FEATURE_BRANCH'."
else
    echo "  RESULT: FAIL — mAP diverged. Check the diff above."
fi
sep
exit $COMPARE_EXIT
