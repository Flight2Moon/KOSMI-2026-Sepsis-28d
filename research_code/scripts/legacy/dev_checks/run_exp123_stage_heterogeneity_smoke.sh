#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="exp123_stage_heterogeneity_smoke"
OVERWRITE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

ARGS=(--output "$PROJECT_ROOT/runs/$RUN_ID")
if [[ "$OVERWRITE" -eq 1 ]]; then
  ARGS+=(--overwrite)
fi

PYTHONPATH="$PROJECT_ROOT/src" python \
  "$PROJECT_ROOT/tests/run_exp123_stage_heterogeneity_smoke.py" \
  "${ARGS[@]}"
