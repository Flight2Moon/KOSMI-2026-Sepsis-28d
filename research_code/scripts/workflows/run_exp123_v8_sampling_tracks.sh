#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CONFIG=""
OPTIMIZED_RUN_ID=""
T3_RUN_ID=""
T4_RUN_ID=""
INCLUDE_UNDERSAMPLING=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="${2:?}"; shift 2;;
    --optimized-run-id) OPTIMIZED_RUN_ID="${2:?}"; shift 2;;
    --t3-run-id) T3_RUN_ID="${2:?}"; shift 2;;
    --t4-run-id) T4_RUN_ID="${2:?}"; shift 2;;
    --include-undersampling) INCLUDE_UNDERSAMPLING=1; shift;;
    -h|--help)
      echo "Run V8 sampled tracks after a completed T2 run."
      echo "Required: --config --optimized-run-id --t3-run-id"
      echo "Optional: --include-undersampling --t4-run-id"
      exit 0;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
[[ -n "$CONFIG" ]] || { echo "--config is required" >&2; exit 2; }
[[ -n "$OPTIMIZED_RUN_ID" ]] || { echo "--optimized-run-id is required" >&2; exit 2; }
[[ -n "$T3_RUN_ID" ]] || { echo "--t3-run-id is required" >&2; exit 2; }
python scripts/workflows/run_exp123_v8_smote.py \
  --config "$CONFIG" \
  --run-id "$T3_RUN_ID" \
  --optimized-run-id "$OPTIMIZED_RUN_ID"
if [[ "$INCLUDE_UNDERSAMPLING" -eq 1 ]]; then
  [[ -n "$T4_RUN_ID" ]] || { echo "--t4-run-id is required with --include-undersampling" >&2; exit 2; }
  python scripts/workflows/run_exp123_v8_undersampling.py \
    --config "$CONFIG" \
    --run-id "$T4_RUN_ID" \
    --optimized-run-id "$OPTIMIZED_RUN_ID" \
    --confirm-optional-track
else
  echo "[INFO] T4 Undersampling status: not_requested"
fi
