#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${1:-latest}"

if [[ $# -gt 0 ]]; then
  shift
fi

cd "$PROJECT_DIR"

"$PYTHON_BIN" -u tools/verify_baseline.py \
  --run-id "$RUN_ID" \
  "$@"
