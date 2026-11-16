#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -d exp123_release_freeze ]]; then
  mv exp123_release_freeze "exp123_release_freeze.pre_v4_${TIMESTAMP}"
  echo "[INFO] Previous release freeze was preserved as exp123_release_freeze.pre_v4_${TIMESTAMP}"
fi

find . -type d -name __pycache__ -prune -exec rm -rf {} +
rm -rf .pytest_cache tests/_exp123_outcome_smoke

chmod +x \
  prepare_exp123_dataset.py \
  setup_macos_mps.sh \
  run_exp123_mps_m1max.sh \
  check_exp123_mps_m1max.sh \
  launch_exp123_dashboard_mps.sh \
  run_exp123_cuda_dense.sh \
  run_exp123_safe.sh

python -m compileall -q src *.py tests

for config in configs/*.json; do
  python -m json.tool "$config" >/dev/null
done

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q

echo "[OK] EXP123 V4 update validation passed."
echo "[IMPORTANT] Build a new release freeze and use a new Run ID."
