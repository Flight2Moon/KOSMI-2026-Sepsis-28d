#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -m py_compile \
  prepare_exp123_v8_confirmatory.py \
  run_exp123_v8_confirmatory.py \
  compare_exp123_v8_confirmatory.py \
  src/los_baseline/confirmatory_selection_v8.py \
  src/los_baseline/confirmatory_v8.py \
  src/los_baseline/site_recalibration_v8.py \
  src/los_baseline/confirmatory_reporting_v8.py
pytest -q tests/test_v8_phase5_confirmatory.py
python - <<'PY'
from pathlib import Path
from los_baseline.config import load_config
for name in ('experiment_cuda_v8.json','experiment_cuda_v8_los14.json','experiment_cuda_v8_los28.json'):
    cfg=load_config(Path('configs')/name)
    assert cfg['confirmatory_v8']['threshold_reselection_allowed'] is False
    assert cfg['confirmatory_v8']['raw_external_validation_is_primary'] is True
    assert cfg['confirmatory_v8']['site_recalibration']['repeats'] == 20
print('[OK] V8 Phase 5 config contract')
PY
echo "[OK] EXP123 V8 Phase 5 verification passed."
