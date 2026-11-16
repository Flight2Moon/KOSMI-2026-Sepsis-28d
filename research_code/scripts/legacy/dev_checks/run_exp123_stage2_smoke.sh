#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="exp123_stage2_smoke"
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

python "$PROJECT_DIR/tests/run_exp123_stage2_smoke.py" \
  --output "$PROJECT_DIR/runs/$RUN_ID" \
  $OVERWRITE
