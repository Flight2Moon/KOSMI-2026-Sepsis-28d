#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest -q tests/test_v8_phase4a_ablation.py tests/test_v8_phase3_reporting.py tests/test_v8_phase1_matrix.py
python -m compileall -q src build_exp123_ablation_config.py run_exp123_v8_ablation.py compare_exp123_v8_ablation.py
python -m json.tool configs/feature_sets.json >/dev/null
bash -n run_exp123_v8_ablation.sh
printf '[OK] V8 Phase 4-A ablation verification passed.\n'
