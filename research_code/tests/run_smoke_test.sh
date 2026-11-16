#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
rm -rf tests/generated
mkdir -p tests/generated
PYTHONPATH=src python tests/create_synthetic_data.py --output-dir tests/generated --rows 240
python run_all_experiments.py --config tests/smoke_config.json --run-id smoke_test

test -f tests/generated/runs/smoke_test/comparison_internal_test.csv
test -f tests/generated/runs/smoke_test/comparison_external_eicu.csv
test -f tests/generated/runs/smoke_test/data_audit/mimic_missingness.csv

python - <<'PY'
import json
from pathlib import Path
import pandas as pd

run = Path("tests/generated/runs/smoke_test")
summary = pd.read_csv(run / "comparison_all_splits.csv")
required = {
    "sample_count",
    "positive_count",
    "negative_count",
    "positive_rate",
    "accuracy_at_0_5",
    "specificity_at_0_5",
    "split_inference_seconds",
    "split_evaluation_seconds",
}
missing = required - set(summary.columns)
assert not missing, missing
assert "auroc_at_0_5" not in summary.columns
assert len(summary) == 60, len(summary)

manifest = json.loads((run / "run_manifest.json").read_text())
assert manifest["audit"]["eicu_harmonizer"]["status"] == "disabled"

PY

python tools/compare_runs.py \
  --baseline-run tests/generated/runs/smoke_test \
  --candidate-run tests/generated/runs/smoke_test \
  --output-dir tests/generated/regression_self_check

echo "Phase 1 smoke test completed successfully."
