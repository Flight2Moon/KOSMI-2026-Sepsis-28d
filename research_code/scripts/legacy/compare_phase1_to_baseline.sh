#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASELINE_RUN_ID="${1:-20260713_162345}"
CANDIDATE_RUN_ID="${2:-}"

if [[ -z "$CANDIDATE_RUN_ID" ]]; then
  echo "Usage: bash compare_phase1_to_baseline.sh BASELINE_RUN_ID CANDIDATE_RUN_ID"
  exit 2
fi

cd "$PROJECT_DIR"
"$PYTHON_BIN" -u tools/compare_runs.py \
  --baseline-run "$BASELINE_RUN_ID" \
  --candidate-run "$CANDIDATE_RUN_ID"
