#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR" \
  python -m pytest -q \
  tests/test_outcomes.py \
  tests/test_v72_outcome_pipeline.py

bash tests/run_exp123_outcome_smoke.sh
bash tests/run_exp123_split_smoke.sh

echo "[OK] EXP123 V7.2 outcome, cohort, external-hold, split, and retry fixes passed."
