#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="exp123_oracle_pipeline_smoke"
OVERWRITE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE="--overwrite"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

PYTHONPATH="$PROJECT_ROOT/src" python "$PROJECT_ROOT/tests/run_exp123_oracle_pipeline_smoke.py" \
  --output "$PROJECT_ROOT/runs/$RUN_ID" \
  $OVERWRITE
