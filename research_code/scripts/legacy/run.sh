#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="$PROJECT_DIR/logs"
LOG_PATH="$LOG_DIR/run_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

set +e
"$PYTHON_BIN" -u run_all_experiments.py "$@" 2>&1 | tee "$LOG_PATH"
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo "Console log: $LOG_PATH"
exit "$EXIT_CODE"
