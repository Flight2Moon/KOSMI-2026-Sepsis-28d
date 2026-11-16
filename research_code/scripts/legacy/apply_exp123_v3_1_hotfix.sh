#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f "run_exp123_job.py" || ! -d "src/los_baseline" ]]; then
  echo "[ERROR] Run this script from the patched EXP123 project root." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -d "exp123_release_freeze" ]]; then
  BACKUP="exp123_release_freeze.pre_v3_1_${STAMP}"
  mv "exp123_release_freeze" "$BACKUP"
  echo "[INFO] Preserved the stale release freeze at: $BACKUP"
fi

find . -type d -name '__pycache__' -prune -exec rm -rf {} +
rm -rf .pytest_cache

python -m py_compile \
  run_exp123_job.py \
  src/los_baseline/failure_diagnostics.py

PYTHONPATH=src python -m pytest -q \
  tests/test_run_exp123_job_contract.py \
  tests/test_resource_scheduler_v3.py \
  tests/test_resource_protocol_graph.py

cat <<'MSG'
[OK] EXP123 V3.1 interface hotfix is installed.

Important:
- Existing runs were not deleted.
- Do not resume a run preflighted with the previous source release.
- Create a new release freeze and a new Run ID.

Recommended next command:
  bash run_exp123_cuda_dense.sh --refreeze
MSG
