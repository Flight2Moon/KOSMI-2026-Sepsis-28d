#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CONFIG=""
T1_ID=""
T2_ID=""
T3_ID=""
T4_ID=""
INCLUDE_UNDERSAMPLING=0
REFREEZE=0
SKIP_TESTS=0
SKIP_HARDWARE_SMOKE=0
ALLOW_ENVIRONMENT_MISMATCH=0

usage(){ cat <<'EOF'
Usage: bash run_exp123_v8_horizon.sh --config PATH --t1-run-id ID --t2-run-id ID --t3-run-id ID [options]
  --t4-run-id ID --include-undersampling
  --refreeze
  --skip-tests
  --skip-hardware-smoke
  --allow-environment-mismatch

Runs one prespecified V8 outcome horizon sequentially:
T1 Natural Baseline -> T2 Optuna -> T3 Class-C SMOTE -> optional T4 Undersampling.
Also generates V8 reporting overlays for T1/T2 and complete EXP2/EXP3 reports for T3/T4.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="${2:?}"; shift 2;;
    --t1-run-id) T1_ID="${2:?}"; shift 2;;
    --t2-run-id) T2_ID="${2:?}"; shift 2;;
    --t3-run-id) T3_ID="${2:?}"; shift 2;;
    --t4-run-id) T4_ID="${2:?}"; shift 2;;
    --include-undersampling) INCLUDE_UNDERSAMPLING=1; shift;;
    --refreeze) REFREEZE=1; shift;;
    --skip-tests) SKIP_TESTS=1; shift;;
    --skip-hardware-smoke) SKIP_HARDWARE_SMOKE=1; shift;;
    --allow-environment-mismatch) ALLOW_ENVIRONMENT_MISMATCH=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 2;;
  esac
done
for value in CONFIG T1_ID T2_ID T3_ID; do
  [[ -n "${!value}" ]] || { echo "--${value,,} is required" >&2; exit 2; }
done
[[ "$CONFIG" = /* ]] || CONFIG="$ROOT/$CONFIG"
CONFIG="$(python -c 'import pathlib,sys;print(pathlib.Path(sys.argv[1]).resolve())' "$CONFIG")"
python -m json.tool "$CONFIG" >/dev/null
HORIZON="$(python - "$CONFIG" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1]))['outcomes']['outcome_horizon_days']))
PY
)"
[[ "$HORIZON" == "28" ]] || { echo "[ERROR] Phase 6 supports LOS28 only; got LOS${HORIZON}." >&2; exit 2; }
COMMON_ARGS=(--config "$CONFIG")
[[ "$SKIP_TESTS" -eq 1 ]] && COMMON_ARGS+=(--skip-tests)
[[ "$SKIP_HARDWARE_SMOKE" -eq 1 ]] && COMMON_ARGS+=(--skip-hardware-smoke)
[[ "$ALLOW_ENVIRONMENT_MISMATCH" -eq 1 ]] && COMMON_ARGS+=(--allow-environment-mismatch)

echo "[STEP] LOS${HORIZON} T1 Natural Baseline"
T1=(bash run_exp123_v6_series.sh "${COMMON_ARGS[@]}" --phase baseline --run-id "$T1_ID")
[[ "$REFREEZE" -eq 1 ]] && T1+=(--refreeze)
"${T1[@]}"
python scripts/workflows/run_exp123_v8_reporting.py --config "$CONFIG" --run-id "$T1_ID" --strategy T1_baseline

echo "[STEP] LOS${HORIZON} T2 Optuna"
T2_ARGS=(--config "$CONFIG" --skip-tests --skip-hardware-smoke)
[[ "$ALLOW_ENVIRONMENT_MISMATCH" -eq 1 ]] && T2_ARGS+=(--allow-environment-mismatch)
bash run_exp123_v6_series.sh "${T2_ARGS[@]}" --phase optimize --baseline-run-id "$T1_ID" --run-id "$T2_ID"
BACKEND="$(python - "$CONFIG" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['resources']['accelerator']['backend'])
PY
)"
TARGETED_CONFIG="$ROOT/generated_configs/${T1_ID}_targeted_${BACKEND}.json"
[[ -f "$TARGETED_CONFIG" ]] || { echo "[ERROR] Targeted config missing: $TARGETED_CONFIG" >&2; exit 2; }
python scripts/workflows/run_exp123_v8_reporting.py --config "$TARGETED_CONFIG" --run-id "$T2_ID" --strategy T2_optuna

echo "[STEP] LOS${HORIZON} T3 preflight"
bash preflight_exp123.sh --config "$TARGETED_CONFIG" --run-id "$T3_ID"
echo "[STEP] LOS${HORIZON} T3 Class-C SMOTE x2"
python scripts/workflows/run_exp123_v8_smote.py --config "$TARGETED_CONFIG" --run-id "$T3_ID" --optimized-run-id "$T2_ID"

T4_STATUS="not_requested"
if [[ "$INCLUDE_UNDERSAMPLING" -eq 1 ]]; then
  [[ -n "$T4_ID" ]] || { echo "--t4-run-id is required with --include-undersampling" >&2; exit 2; }
  echo "[STEP] LOS${HORIZON} T4 preflight"
  bash preflight_exp123.sh --config "$TARGETED_CONFIG" --run-id "$T4_ID"
  echo "[STEP] LOS${HORIZON} T4 optional Majority Undersampling"
  python scripts/workflows/run_exp123_v8_undersampling.py --config "$TARGETED_CONFIG" --run-id "$T4_ID" --optimized-run-id "$T2_ID" --confirm-optional-track
  T4_STATUS="completed"
fi

OUT="$ROOT/series/${T1_ID}_horizon_manifest.json"
python - "$OUT" "$CONFIG" "$TARGETED_CONFIG" "$HORIZON" "$T1_ID" "$T2_ID" "$T3_ID" "$T4_ID" "$T4_STATUS" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
out=Path(sys.argv[1]); out.parent.mkdir(parents=True,exist_ok=True)
p={
 'schema_version':'exp123_v8_horizon_execution_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
 'base_config':sys.argv[2],'targeted_config':sys.argv[3],'outcome_horizon_days':int(sys.argv[4]),
 'runs':[
   {'strategy':'T1_baseline','run_id':sys.argv[5],'status':'completed'},
   {'strategy':'T2_optuna','run_id':sys.argv[6],'status':'completed'},
   {'strategy':'T3_smote','run_id':sys.argv[7],'status':'completed'},
   {'strategy':'T4_undersampling','run_id':sys.argv[8] or None,'status':sys.argv[9]},
 ],
 'confirmatory_splits_accessed':False,
}
out.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'[OK] Horizon execution manifest: {out}')
PY

echo "[OK] LOS${HORIZON} V8 Discovery horizon completed. Confirmatory splits remain locked."
