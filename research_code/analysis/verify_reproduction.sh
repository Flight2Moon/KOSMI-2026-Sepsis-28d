#!/usr/bin/env bash
# Verify that the packaged analysis reproduces the shipped results exactly.
#
# Runs the paired bootstrap on all five evaluation cohorts, rebuilds the
# figures, and diffs both against the reference copies under
# analysis/reference/. Exits non-zero on any mismatch.
#
# Usage:
#   bash analysis/verify_reproduction.sh
#   PYTHON_BIN=python3.12 bash analysis/verify_reproduction.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS="analysis/results"
REFERENCE="analysis/reference"

echo "[INFO] Project root: $ROOT"
echo "[INFO] Python:       $($PYTHON_BIN --version 2>&1)"

# --- 1. dataset integrity -------------------------------------------------- #
echo
echo "[STEP 1/4] Verifying dataset checksums against configs/dataset_registry_28d.json"
"$PYTHON_BIN" - <<'PYCODE'
import hashlib
import json
import sys
from pathlib import Path

root = Path.cwd()
registry = json.loads(
    (root / "configs/dataset_registry_28d.json").read_text(encoding="utf-8")
)
seen: dict[str, str] = {}
failures = 0
for name, spec in registry["datasets"].items():
    path = root / "data" / spec["file"]
    if not path.is_file():
        print(f"[ERROR] Missing dataset file: {path}")
        failures += 1
        continue
    if spec["file"] not in seen:
        seen[spec["file"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    actual = seen[spec["file"]]
    expected = spec["expected_sha256"]
    status = "OK" if actual == expected else "MISMATCH"
    if status == "MISMATCH":
        failures += 1
    print(f"  {status:8s} {name:14s} {spec['file']}")
sys.exit(1 if failures else 0)
PYCODE

# --- 2. paired bootstrap --------------------------------------------------- #
echo
echo "[STEP 2/4] Running the paired bootstrap on all five cohorts"
rm -f "$RESULTS"/boot_*.csv
for SITE in mimic_temporal eicu_alt eicu_primary sicdb nwicu; do
  echo "  - $SITE"
  "$PYTHON_BIN" analysis/paired_bootstrap_flat_vs_cascade.py "$SITE" >/dev/null
done
"$PYTHON_BIN" analysis/paired_bootstrap_flat_vs_cascade.py --merge

# --- 3. figures ------------------------------------------------------------ #
echo
echo "[STEP 3/4] Rebuilding figures"
"$PYTHON_BIN" analysis/make_panel_a.py >/dev/null
"$PYTHON_BIN" analysis/make_panel_a.py --width-mm 160 --aspect 0.60 \
  --out analysis/figures/PanelA_160mm_tall >/dev/null
"$PYTHON_BIN" analysis/make_panel_a.py --width-mm 100 --aspect 0.98 \
  --out analysis/figures/PanelA_square >/dev/null
echo "  figures written to analysis/figures/"

# --- 4. diff against reference --------------------------------------------- #
echo
echo "[STEP 4/4] Comparing against analysis/reference/"
"$PYTHON_BIN" - <<'PYCODE'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path.cwd()
reference = root / "analysis/reference/flat_vs_cascade_paired_bootstrap.csv"
produced = root / "analysis/results/flat_vs_cascade_paired_bootstrap.csv"

old = pd.read_csv(reference).sort_values(["site", "model"]).reset_index(drop=True)
new = pd.read_csv(produced).sort_values(["site", "model"]).reset_index(drop=True)

if len(old) != len(new):
    print(f"[ERROR] Row count differs: reference {len(old)} vs produced {len(new)}")
    sys.exit(1)

numeric = [
    column for column in old.columns
    if column not in ("site", "model") and pd.api.types.is_numeric_dtype(old[column])
]
difference = (new[numeric] - old[numeric]).abs().to_numpy()
largest = float(difference.max())

print(f"  rows compared:    {len(old)}")
print(f"  columns compared: {len(numeric)}")
print(f"  max abs diff:     {largest:.3e}")

if largest == 0.0:
    print("  RESULT: bit-identical reproduction")
elif largest < 5e-5:
    print("  RESULT: identical to 4 decimal places (acceptable)")
else:
    print("[ERROR] Reproduction mismatch exceeds tolerance.")
    sys.exit(1)
PYCODE

echo
echo "[OK] Reproduction verified."
