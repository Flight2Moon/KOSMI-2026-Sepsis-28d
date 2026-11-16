#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${1:-/tmp/exp123_split_smoke}"
RUN_ID="exp123_split_smoke"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

python - "$PROJECT_DIR/configs/experiment.json" "$WORK_DIR" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

config_path = Path(sys.argv[1])
work_dir = Path(sys.argv[2])
config = json.loads(config_path.read_text())

rng = np.random.default_rng(42)

def make_dataset(path: Path, n_per_class: int, id_offset: int) -> None:
    rows = []
    source_id = id_offset
    for class_code in [0, 1, 2]:
        for index in range(n_per_class):
            if class_code == 0:
                los = float(rng.uniform(1.0, 27.0))
                death = 0
            elif class_code == 1:
                los = float(rng.uniform(1.0, 28.0))
                death = 1
            else:
                los = float(rng.uniform(28.0, 50.0))
                death = int(index % 5 == 0)
            admission = pd.Timestamp("2020-01-01") + pd.to_timedelta(
                source_id, unit="h"
            )
            rows.append(
                {
                    "subject_id": source_id,
                    "hadm_id": 100000 + source_id,
                    "stay_id": 200000 + source_id,
                    "admittime": admission,
                    "dischtime": admission + pd.to_timedelta(los, unit="D"),
                    "LOS": los,
                    "hospital_expire_flag": death,
                }
            )
            source_id += 1
    pd.DataFrame(rows).to_csv(path, index=False)

mimic_path = work_dir / "mimic.csv"
eicu_path = work_dir / "eicu.csv"
make_dataset(mimic_path, 60, 1)
make_dataset(eicu_path, 30, 10000)

config["paths"]["mimic_csv"] = str(mimic_path)
config["paths"]["eicu_csv"] = str(eicu_path)
config["paths"]["runs_dir"] = str(work_dir / "runs")
config_path_out = work_dir / "config.json"
config_path_out.write_text(json.dumps(config, indent=2))
PY

cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR/src" python - "$PROJECT_DIR" "$WORK_DIR/config.json" "$RUN_ID" <<'PY'
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
config_path = Path(sys.argv[2])
run_id = sys.argv[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from los_baseline.config import load_config
from los_baseline.exp123_splits import run_common_split_step
from run_exp123 import run_outcome_step

config = load_config(config_path)
_, issue_count = run_outcome_step(
    config=config,
    run_id=run_id,
    dataset_selection="both",
    overwrite=False,
)
assert issue_count == 0
run_common_split_step(config=config, run_id=run_id, overwrite=False)
PY

python - "$WORK_DIR/runs/$RUN_ID/common/splits" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

split_dir = Path(sys.argv[1])
manifest = json.loads((split_dir / "split_manifest.json").read_text())
integrity = json.loads((split_dir / "split_integrity.json").read_text())
assignment = pd.read_csv(split_dir / "split_assignments.csv.gz")

assert manifest["status"] == "ready"
assert len(set(manifest["task_split_references"].values())) == 1
assert integrity["no_patient_overlap"] is True
assert integrity["all_classes_present_per_split"] is True
assert integrity["stage1_both_classes_per_split"] is True
assert integrity["stage2_both_classes_per_split"] is True
assert len(assignment) == 180
print("[OK] EXP123 outcome-to-common-split smoke test passed.")
print(f"[OK] split_hash={manifest['split_hash']}")
PY
