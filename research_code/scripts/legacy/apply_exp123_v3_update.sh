#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

rm -rf exp123_release_freeze .pytest_cache
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete

chmod +x \
  run_exp123_safe.sh \
  run_exp123_cuda_dense.sh \
  run_exp123_cpu_7800x3d.sh \
  diagnose_exp123_run.py \
  validate_exp123_setup.py

python -m json.tool configs/experiment_resource_cpu.json >/dev/null
python -m json.tool configs/experiment_cuda_max_gpu_throughput_7800x3d_48gb.json >/dev/null

echo "[OK] EXP123 V3 update applied."
echo "[REQUIRED] Run tests, hardware smoke, and create a new release freeze."
echo "[PRESERVED] Existing runs were not deleted."
