#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${PROJECT_ROOT}/tests/_exp123_outcome_smoke"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

PYTHONPATH="${PROJECT_ROOT}/src" python "${PROJECT_ROOT}/tests/create_synthetic_data.py" \
  --output-dir "$WORK_DIR/data" \
  --rows 120

python - "$PROJECT_ROOT" "$WORK_DIR" <<'PY'
import json
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
work_dir = Path(sys.argv[2])
config = json.loads((project_root / "configs" / "experiment.json").read_text())
config["paths"]["mimic_csv"] = str(work_dir / "data" / "mimic.csv")
config["paths"]["eicu_csv"] = str(work_dir / "data" / "eicu.csv.gz")
config["paths"]["runs_dir"] = str(work_dir / "runs")
config_path = work_dir / "smoke_config.json"
config_path.write_text(json.dumps(config, indent=2) + "\n")
PY

PYTHONPATH="${PROJECT_ROOT}/src" python - "$PROJECT_ROOT" "$WORK_DIR/smoke_config.json" <<'PY'
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
config_path = Path(sys.argv[2])
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from los_baseline.config import load_config
from run_exp123 import run_outcome_step

config = load_config(config_path)
_, issue_count = run_outcome_step(
    config=config,
    run_id="exp123_outcome_smoke",
    dataset_selection="both",
    overwrite=False,
)
assert issue_count == 0
PY

OUTPUT="$WORK_DIR/runs/exp123_outcome_smoke/common/outcomes"
test -f "$OUTPUT/mimic_outcome_audit.json"
test -f "$OUTPUT/eicu_outcome_audit.json"
test -f "$OUTPUT/mimic_outcome_assignments.csv.gz"
test -f "$OUTPUT/eicu_outcome_assignments.csv.gz"
test -f "$OUTPUT/outcome_step_manifest.json"

python - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
manifest = json.loads((output / "outcome_step_manifest.json").read_text())
assert manifest["status"] == "ready", manifest
assert manifest["model_training_enabled"] is True
for dataset in ("mimic", "eicu"):
    audit = manifest["dataset_audits"][dataset]
    assert audit["status"] == "ready", audit
    assert sum(audit["class_counts"].values()) == audit["retained_row_count"]
print("[OK] EXP123 outcome smoke test passed.")
PY
