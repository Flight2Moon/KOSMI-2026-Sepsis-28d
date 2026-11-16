#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="src:.${PYTHONPATH:+:$PYTHONPATH}"
python -m compileall -q src run_exp123_v8_smote.py run_exp123_v8_undersampling.py
bash -n run_exp123_v8_matrix.sh run_exp123_v8_sampling_tracks.sh
python - <<'PY'
from pathlib import Path
from los_baseline.config import load_config
for path in sorted(Path('configs').glob('experiment_cuda_v8*.json')):
    config = load_config(path.resolve())
    assert config['experiment_matrix']['strategies']['T3_smote']['implementation_status'] == 'phase2_implemented'
    assert config['experiment_matrix']['strategies']['T4_undersampling']['optional'] is True
print('[OK] V8 config contracts')
PY
pytest -q \
  tests/test_v8_phase1_matrix.py \
  tests/test_v8_phase1_fixed_cv_metadata.py \
  tests/test_v8_phase2_sampling.py \
  tests/test_exp123_data.py \
  tests/test_exp123_splits.py \
  tests/test_v73_hpo_protocol.py \
  tests/test_v7_cohort_imbalance_hpo.py
python run_exp123_v8_smote.py --help >/dev/null
python run_exp123_v8_undersampling.py --help >/dev/null
echo '[OK] EXP123 V8 Phase 2 verification completed.'
