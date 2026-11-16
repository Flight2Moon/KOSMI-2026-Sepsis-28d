#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash run_study28_full_pipeline.sh --dataset-root /path/to/final_dataset [options]

Required:
  --dataset-root PATH   Directory containing:
                        feature_list.csv
                        mimic_final.csv
                        eicu_final.csv
                        sicdb_final.csv
                        nwicu_final.csv

Options:
  --run-root PATH       Root output directory (default: runs/study28_full)
  --python PATH         Python executable (default: python)
  --resource-profile P  maximum_throughput (default) or balanced
  --replan-resources    Re-detect hardware instead of reusing the frozen run plan
  -h, --help            Show this help message

Pipeline:
  0. Frozen dataset audit
  1. Alpha 2: 5x3 nested CV, MIMIC temporal evaluation, full-MIMIC model freeze
  2. Alpha 3: mandatory raw external validation
             eICU-alt, eICU-primary, SICdb, NWICU
  3. Alpha 4: reverse validation, local recalibration, DCA,
             subgroup analysis, eICU hospital heterogeneity
  4. Alpha 5: SHAP, pre-ICU LOS sensitivity, feature-availability,
             missingness robustness, HFD28
  5. Alpha 6: manuscript tables, figures, evidence index,
             statistical-methods evidence, finalization checklist

Resume behavior:
  Re-run the same command with the same --run-root. Existing verified
  checkpoints are reused by the underlying Alpha runners where supported.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
DATASET_ROOT=""
RUN_ROOT="runs/study28_full"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESOURCE_PROFILE="maximum_throughput"
REPLAN_RESOURCES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="${2:-}"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --resource-profile)
      RESOURCE_PROFILE="${2:-}"
      shift 2
      ;;
    --replan-resources)
      REPLAN_RESOURCES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DATASET_ROOT" ]]; then
  echo "[ERROR] --dataset-root is required." >&2
  usage >&2
  exit 2
fi

if [[ "$RESOURCE_PROFILE" != "maximum_throughput" && "$RESOURCE_PROFILE" != "balanced" ]]; then
  echo "[ERROR] Invalid --resource-profile: $RESOURCE_PROFILE" >&2
  exit 2
fi

DATASET_ROOT="$(cd "$DATASET_ROOT" 2>/dev/null && pwd)" || {
  echo "[ERROR] Dataset root does not exist: $DATASET_ROOT" >&2
  exit 2
}

if [[ "$RUN_ROOT" != /* ]]; then
  RUN_ROOT="$PROJECT_ROOT/$RUN_ROOT"
fi
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"

required_files=(
  feature_list.csv
  mimic_final.csv
  eicu_final.csv
  sicdb_final.csv
  nwicu_final.csv
)

missing=0
for name in "${required_files[@]}"; do
  if [[ ! -f "$DATASET_ROOT/$name" ]]; then
    echo "[ERROR] Missing required dataset file: $DATASET_ROOT/$name" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

AUDIT_DIR="$RUN_ROOT/alpha1_audit"
ALPHA2_DIR="$RUN_ROOT/alpha2"
ALPHA3_DIR="$RUN_ROOT/alpha3"
ALPHA4_DIR="$RUN_ROOT/alpha4"
ALPHA5_DIR="$RUN_ROOT/alpha5"
ALPHA6_DIR="$RUN_ROOT/alpha6"
LOG_FILE="$RUN_ROOT/full_pipeline.log"
RESOURCE_PLAN="$RUN_ROOT/hardware_execution_plan.json"
RESOURCE_ENV="$RUN_ROOT/hardware_execution_env.sh"
RESOURCE_USAGE="$RUN_ROOT/hardware_usage.csv"

exec > >(tee -a "$LOG_FILE") 2>&1

cd "$PROJECT_ROOT"
if [[ "$REPLAN_RESOURCES" -eq 0 && -f "$RESOURCE_PLAN" && -f "$RESOURCE_ENV" ]]; then
  echo "[RESOURCE] Reusing frozen hardware execution plan: $RESOURCE_PLAN"
else
  echo "[RESOURCE] Detecting hardware and freezing execution plan ($RESOURCE_PROFILE)"
  "$PYTHON_BIN" tools/plan_study28_resources.py \
    --output-json "$RESOURCE_PLAN" \
    --output-env "$RESOURCE_ENV" \
    --profile "$RESOURCE_PROFILE"
fi
# shellcheck disable=SC1090
source "$RESOURCE_ENV"

"$PYTHON_BIN" tools/monitor_study28_resources.py --output "$RESOURCE_USAGE" --interval 2.0 &
RESOURCE_MONITOR_PID=$!

on_error() {
  local code=$?
  if [[ -n "${RESOURCE_MONITOR_PID:-}" ]]; then kill "$RESOURCE_MONITOR_PID" 2>/dev/null || true; wait "$RESOURCE_MONITOR_PID" 2>/dev/null || true; fi
  echo
  echo "[FAILED] Study28 full pipeline stopped with exit code $code."
  echo "[FAILED] Inspect log: $LOG_FILE"
  echo "[FAILED] Re-run the same command with the same --run-root after fixing the issue; verified checkpoints will be reused where supported."
  exit "$code"
}
trap on_error ERR

echo "============================================================"
echo "Study28 V9 full research pipeline"
echo "Project root : $PROJECT_ROOT"
echo "Dataset root : $DATASET_ROOT"
echo "Run root     : $RUN_ROOT"
echo "Python       : $PYTHON_BIN"
echo "Resource     : $RESOURCE_PROFILE"
echo "Resource plan: $RESOURCE_PLAN"
echo "============================================================"

echo
echo "[0/5] Frozen dataset audit"
"$PYTHON_BIN" tools/audit_study28.py \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$AUDIT_DIR"

echo
echo "[1/5] Alpha 2 - development, nested CV, temporal evaluation, model freeze"
"$PYTHON_BIN" tools/run_study28_alpha2.py \
  --dataset-root "$DATASET_ROOT" \
  --project-root "$PROJECT_ROOT" \
  --output-dir "$ALPHA2_DIR"

echo
echo "[2/5] Alpha 3 - mandatory raw external validation"
"$PYTHON_BIN" tools/run_study28_alpha3.py \
  --dataset-root "$DATASET_ROOT" \
  --alpha2-output-dir "$ALPHA2_DIR" \
  --project-root "$PROJECT_ROOT" \
  --output-dir "$ALPHA3_DIR"

echo
echo "[3/5] Alpha 4 - sensitivity and clinical utility"
"$PYTHON_BIN" tools/run_study28_alpha4.py \
  --dataset-root "$DATASET_ROOT" \
  --alpha2-output-dir "$ALPHA2_DIR" \
  --alpha3-output-dir "$ALPHA3_DIR" \
  --project-root "$PROJECT_ROOT" \
  --output-dir "$ALPHA4_DIR"

echo
echo "[4/5] Alpha 5 - robustness, interpretability, HFD28"
"$PYTHON_BIN" tools/run_study28_alpha5.py \
  --dataset-root "$DATASET_ROOT" \
  --alpha2-output-dir "$ALPHA2_DIR" \
  --alpha3-output-dir "$ALPHA3_DIR" \
  --alpha4-output-dir "$ALPHA4_DIR" \
  --project-root "$PROJECT_ROOT" \
  --output-dir "$ALPHA5_DIR"

echo
echo "[5/5] Alpha 6 - manuscript evidence and final reporting"
"$PYTHON_BIN" tools/run_study28_alpha6.py \
  --alpha2-output-dir "$ALPHA2_DIR" \
  --alpha3-output-dir "$ALPHA3_DIR" \
  --alpha4-output-dir "$ALPHA4_DIR" \
  --alpha5-output-dir "$ALPHA5_DIR" \
  --project-root "$PROJECT_ROOT" \
  --output-dir "$ALPHA6_DIR"

CHECKLIST="$ALPHA6_DIR/finalization_checklist.json"
if [[ ! -f "$CHECKLIST" ]]; then
  echo "[ERROR] Finalization checklist was not created: $CHECKLIST" >&2
  exit 2
fi

"$PYTHON_BIN" - "$CHECKLIST" <<'PY_CHECK'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
payload = json.loads(p.read_text(encoding="utf-8"))
if payload.get("passed") is not True:
    raise SystemExit(f"Finalization checklist did not pass: {p}")
required = {"eicu_alt", "eicu_primary", "sicdb", "nwicu"}
observed = set(payload.get("required_external_sites", []))
if observed != required:
    raise SystemExit(
        f"Required external-site contract mismatch: expected={sorted(required)}, observed={sorted(observed)}"
    )
privacy = payload.get("privacy_guard", {})
if privacy.get("passed") is not True:
    raise SystemExit(f"Privacy guard did not pass: {privacy}")
print("[VERIFY] Alpha 6 finalization checklist: PASS")
print("[VERIFY] Mandatory external cohorts: eICU-alt, eICU-primary, SICdb, NWICU")
print("[VERIFY] Privacy guard: PASS")
PY_CHECK

if [[ -n "${RESOURCE_MONITOR_PID:-}" ]]; then
  kill "$RESOURCE_MONITOR_PID" 2>/dev/null || true
  wait "$RESOURCE_MONITOR_PID" 2>/dev/null || true
fi

"$PYTHON_BIN" - "$RESOURCE_USAGE" <<'PY_USAGE'
import csv, sys
from pathlib import Path
p=Path(sys.argv[1])
rows=list(csv.DictReader(p.open())) if p.is_file() else []
cpu=[float(r['cpu_percent']) for r in rows if r.get('cpu_percent')]
ram=[float(r['ram_percent']) for r in rows if r.get('ram_percent')]
if cpu:
    print(f"[RESOURCE] CPU utilization mean={sum(cpu)/len(cpu):.1f}% peak={max(cpu):.1f}%")
if ram:
    print(f"[RESOURCE] RAM utilization peak={max(ram):.1f}%")
print(f"[RESOURCE] Usage log: {p}")
PY_USAGE

echo
echo "============================================================"
echo "[DONE] Study28 full pipeline completed successfully."
echo "[DONE] Raw external validation : $ALPHA3_DIR"
echo "[DONE] Sensitivity/utility      : $ALPHA4_DIR"
echo "[DONE] Robustness/SHAP/HFD28    : $ALPHA5_DIR"
echo "[DONE] Final manuscript package : $ALPHA6_DIR"
echo "[DONE] Final checklist          : $ALPHA6_DIR/finalization_checklist.json"
echo "[DONE] Full log                 : $LOG_FILE"
echo "[DONE] Hardware execution plan  : $RESOURCE_PLAN"
echo "[DONE] Hardware utilization log : $RESOURCE_USAGE"
echo "============================================================"
