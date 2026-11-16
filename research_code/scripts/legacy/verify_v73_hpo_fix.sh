#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q \
  tests/test_v73_hpo_protocol.py \
  tests/test_v73_amended_resume.py \
  tests/test_run_exp123_job_contract.py

python -m compileall -q \
  src/los_baseline \
  run_exp123.py \
  run_exp123_job.py \
  run_exp123_amended_staged_protocol.py \
  audit_exp123_v73_repair_inputs.py

python - <<'PY'
from los_baseline.hpo.optuna_engine import HPO_PROTOCOL_VERSION
assert HPO_PROTOCOL_VERSION == "exp123_hpo_v7_3"
print("[OK] HPO protocol:", HPO_PROTOCOL_VERSION)
PY

echo "[OK] EXP123 V7.3 HPO and amended-resume verification passed."
