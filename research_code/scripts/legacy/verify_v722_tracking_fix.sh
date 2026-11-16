#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  python -m pytest -q \
  tests/test_v722_tracking_concurrency.py \
  tests/test_progress_hardware.py \
  tests/test_v7_observatory.py

python -m compileall -q src run_exp123.py run_exp123_job.py

echo "[OK] V7.2.2 multiprocess tracking verification passed."
