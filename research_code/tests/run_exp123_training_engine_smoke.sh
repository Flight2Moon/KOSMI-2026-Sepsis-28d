#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-${PROJECT_ROOT}/runs/exp123_training_engine_smoke}"

PYTHONPATH="${PROJECT_ROOT}/src" python \
  "${PROJECT_ROOT}/tests/run_exp123_training_engine_smoke.py" \
  --output "${OUTPUT_DIR}" \
  --overwrite
