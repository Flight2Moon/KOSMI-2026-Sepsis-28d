#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -d exp123_release_freeze ]]; then
  mv exp123_release_freeze "exp123_release_freeze.pre_v5_${TIMESTAMP}"
  echo "[OK] Existing release freeze was preserved."
fi

find . -type d -name __pycache__ -prune -exec rm -rf {} +
rm -rf .pytest_cache

chmod +x \
  run_exp123_staged_protocol.py \
  build_exp123_targeted_config.py \
  run_exp123_v5_series.sh \
  run_exp123_v5_cuda.sh \
  run_exp123_v5_mps.sh \
  apply_exp123_v5_update.sh

python -m compileall -q src \
  run_exp123_job.py \
  run_exp123_staged_protocol.py \
  build_exp123_targeted_config.py

for script in \
  run_exp123_v5_series.sh \
  run_exp123_v5_cuda.sh \
  run_exp123_v5_mps.sh \
  apply_exp123_v5_update.sh; do
  bash -n "$script"
done

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  python -m pytest -q \
    tests/test_staged_protocol_v5.py \
    tests/test_stable_binary_threshold_v5.py \
    tests/test_stage9_candidate_fallback_v5.py

cat <<'MESSAGE'
[OK] EXP123 V5 update was applied.
[INFO] Existing runs were preserved.
[INFO] Source and configs changed, so create a new release freeze and new Run ID.
[INFO] Recommended CUDA command:
       bash run_exp123_v5_cuda.sh --phase baseline --refreeze
[INFO] Recommended MPS command:
       bash run_exp123_v5_mps.sh --phase baseline --refreeze
MESSAGE
