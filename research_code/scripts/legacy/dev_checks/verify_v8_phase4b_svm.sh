#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
python -m compileall -q src benchmark_exp123_v8_svm.py
python - <<'PY'
from pathlib import Path
from los_baseline.config import load_config
for name in ("experiment_cuda_v8.json", "experiment_cuda_v8_los14.json", "experiment_cuda_v8_los28.json"):
    config = load_config(Path("configs") / name)
    assert config["svm_runtime_optimization"]["enabled"] is True
    assert config["svm_runtime_optimization"]["search_backends"] == ["exact_rbf", "nystroem_linear"]
    assert config["models"]["svm"]["parameters"]["probability"] is False
    assert config["resources"]["scheduler"]["max_concurrent_by_model"]["svm"] == 2
print("[OK] Phase 4-B V8 configs")
PY
pytest -q tests/test_v8_phase4b_svm_runtime.py tests/test_v73_hpo_protocol.py tests/test_v8_phase2_sampling.py
python benchmark_exp123_v8_svm.py --help >/dev/null
printf '[OK] V8 Phase 4-B SVM verification completed.\n'
