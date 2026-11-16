#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="src:.${PYTHONPATH:+:$PYTHONPATH}"
python -m compileall -q src \
  run_exp123_v8_reporting.py compare_exp123_v8_matrix.py \
  run_exp123_v8_smote.py run_exp123_v8_undersampling.py
bash -n run_exp123_v8_horizon.sh run_exp123_v8_matrix.sh run_exp123_v8_sampling_tracks.sh
python - <<'PY'
from pathlib import Path
from los_baseline.config import load_config
for path in sorted(Path('configs').glob('experiment_cuda_v8*.json')):
    cfg=load_config(path.resolve())
    assert cfg['analysis_plan']['task_selection_metrics']['exp2_combined']=='macro_auprc'
    assert cfg['clinical_evaluation']['confirmatory_threshold_reselection_allowed'] is False
    assert cfg['experiment_matrix']['strategies']['T3_smote']['reporting_status']=='phase3_implemented'
    assert cfg['experiment_matrix']['strategies']['T4_undersampling']['reporting_status']=='phase3_implemented'
print('[OK] V8 Phase3 config contracts')
PY
pytest -q \
  tests/test_v8_phase1_matrix.py \
  tests/test_v8_phase1_fixed_cv_metadata.py \
  tests/test_v8_phase2_sampling.py \
  tests/test_v8_phase3_reporting.py \
  tests/test_staged_protocol_v5.py \
  tests/test_evaluation_binary.py \
  tests/test_evaluation_multiclass.py
python run_exp123_v8_reporting.py --help >/dev/null
python compare_exp123_v8_matrix.py --help >/dev/null
bash run_exp123_v8_horizon.sh --help >/dev/null
bash run_exp123_v8_matrix.sh --plan-only >/dev/null
echo '[OK] EXP123 V8 Phase 3 verification completed.'
