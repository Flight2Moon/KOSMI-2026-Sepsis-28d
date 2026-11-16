#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

echo "[CHECK] Python compile"
python -m compileall -q src scripts/workflows scripts/internal scripts/ops scripts/monitor

echo "[CHECK] Active JSON configs"
python - <<'PY'
import json
from pathlib import Path
for path in sorted(Path('configs').glob('*.json')):
    json.loads(path.read_text(encoding='utf-8'))
assert not Path('configs/experiment_cuda_v8_los14.json').exists()
assert Path('legacy_reference/v8_multi_horizon/experiment_cuda_v8_los14.json').is_file()
print('[OK] configs JSON + LOS28-only active config contract')
PY

echo "[CHECK] Shell syntax"
while IFS= read -r file; do bash -n "$file"; done < <(find . -type f -name '*.sh' \
  -not -path './exp123_release_freeze/*' -not -path './runs/*' | sort)

echo "[CHECK] Phase 6 unit tests"
python -m pytest -q tests/test_v8_phase6_los28_high_throughput.py

echo "[CHECK] Phase 1-5 key contract tests"
python -m pytest -q \
  tests/test_v8_phase1_matrix.py \
  tests/test_v8_phase2_sampling.py \
  tests/test_v8_phase3_reporting.py \
  tests/test_v8_phase4a_ablation.py \
  tests/test_v8_phase4b_svm_runtime.py \
  tests/test_v8_phase5_confirmatory.py

echo "[CHECK] LOS28 plan-only command"
./exp123 plan --no-autotune >/tmp/exp123_phase6_verify_plan.log
python - <<'PY'
from pathlib import Path
import json
plans=sorted(Path('series').glob('exp123_v8_matrix_plan_*/matrix_plan.json'), key=lambda p:p.stat().st_mtime)
assert plans
p=json.loads(plans[-1].read_text())
assert p['requested_horizons']==[28]
assert len(p['runs'])==4
assert sum(bool(r['enabled']) for r in p['runs'])==3
print('[OK] LOS28 T1/T2/T3 + optional T4 plan')
PY

echo "[CHECK] Root declutter"
python - <<'PY'
from pathlib import Path
root=Path('.')
files=sorted(p.name for p in root.iterdir() if p.is_file())
assert 'exp123' in files
assert 'run_exp123_v8_matrix.sh' not in files
assert len(files) <= 20, files
print(f'[OK] root file count={len(files)}')
PY

echo "[OK] EXP123 V8 Phase 6 verification passed."
