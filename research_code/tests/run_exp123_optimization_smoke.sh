#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-${PROJECT_ROOT}/runs/exp123_optimization_smoke/common/optimization_smoke}"

python "${PROJECT_ROOT}/tests/run_exp123_optimization_smoke.py" \
  --output "${OUTPUT}" \
  --overwrite
