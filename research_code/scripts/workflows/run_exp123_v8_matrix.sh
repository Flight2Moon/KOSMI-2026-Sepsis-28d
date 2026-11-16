#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
INCLUDE=0
EXECUTE=0
REFREEZE=0
SKIP_TESTS=0
SKIP_HARDWARE_SMOKE=0
ALLOW_ENVIRONMENT_MISMATCH=0
BASE_CONFIG="configs/experiment_cuda_v8_los28.json"
RESOURCE_PROFILE="maximum_throughput"
AUTOTUNE=1
while [[ $# -gt 0 ]]; do
 case "$1" in
  --config) BASE_CONFIG="${2:?}"; AUTOTUNE=0; shift 2;;
  --include-undersampling) INCLUDE=1; shift;;
  --execute) EXECUTE=1; shift;;
  --plan-only) EXECUTE=0; shift;;
  --refreeze) REFREEZE=1; shift;;
  --skip-tests) SKIP_TESTS=1; shift;;
  --skip-hardware-smoke) SKIP_HARDWARE_SMOKE=1; shift;;
  --allow-environment-mismatch) ALLOW_ENVIRONMENT_MISMATCH=1; shift;;
  --resource-profile) RESOURCE_PROFILE="${2:?}"; shift 2;;
  --no-autotune) AUTOTUNE=0; shift;;
  --horizons)
    [[ "${2:?}" == "28" ]] || { echo "[ERROR] Phase 6 is LOS28-only. --horizons ${2} is not supported." >&2; exit 2; }
    shift 2;;
  -h|--help)
    cat <<'EOF'
EXP123 V8 Phase 6 LOS28-only matrix orchestrator.
Default: plan-only, resource profile=maximum_throughput, resource autotuning enabled.

Options:
  --execute                       Run T1 -> T2 -> T3 and optional T4.
  --include-undersampling         Include optional T4.
  --resource-profile PROFILE      maximum_throughput|balanced|conservative.
  --no-autotune                   Use the supplied/base config resources as-is.
  --config PATH                   Explicit config; also disables auto-tuning unless run separately.
  --refreeze                      Refresh the release freeze before Discovery.
  --skip-tests / --skip-hardware-smoke
  --allow-environment-mismatch    Testing/development only.
EOF
    exit 0;;
  *) echo "Unknown option: $1" >&2; exit 2;;
 esac
done

[[ "$BASE_CONFIG" = /* ]] || BASE_CONFIG="$ROOT/$BASE_CONFIG"
if [[ "$AUTOTUNE" -eq 1 ]]; then
  AUTOTUNED="$ROOT/generated_configs/experiment_v8_los28_autotuned_${RESOURCE_PROFILE}.json"
  MANIFEST="$ROOT/generated_configs/experiment_v8_los28_autotuned_${RESOURCE_PROFILE}_resource_manifest.json"
  python scripts/ops/autotune_exp123_resources.py --config "$BASE_CONFIG" --profile "$RESOURCE_PROFILE" --output "$AUTOTUNED" --manifest "$MANIFEST"
  BASE_CONFIG="$AUTOTUNED"
fi

ARGS=(python scripts/workflows/plan_exp123_v8_matrix.py --config "$BASE_CONFIG" --horizons 28)
[[ "$INCLUDE" -eq 1 ]] && ARGS+=(--include-undersampling)
PLAN_OUTPUT="$("${ARGS[@]}" | tee /tmp/exp123_v8_plan.$$ | awk -F': ' '/\[OK\] V8 matrix plan:/{print $2}' | tail -1)"
[[ -n "$PLAN_OUTPUT" ]] || { cat /tmp/exp123_v8_plan.$$; rm -f /tmp/exp123_v8_plan.$$; echo "[ERROR] Could not resolve matrix plan path" >&2; exit 2; }
rm -f /tmp/exp123_v8_plan.$$
PLAN_OUTPUT="$(python -c 'import pathlib,sys;print(pathlib.Path(sys.argv[1]).resolve())' "$PLAN_OUTPUT")"
if [[ "$EXECUTE" -eq 0 ]]; then
  echo "[OK] LOS28 plan-only mode. Matrix plan: $PLAN_OUTPUT"
  echo "[OK] Active config: $BASE_CONFIG"
  exit 0
fi

readarray -t IDS < <(python - "$PLAN_OUTPUT" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); h=28
for strategy in ('T1_baseline','T2_optuna','T3_smote','T4_undersampling'):
 r=next(x for x in p['runs'] if int(x['outcome_horizon_days'])==h and x['strategy']==strategy)
 print(r['run_id'] or '')
PY
)
CMD=(bash scripts/workflows/run_exp123_v8_horizon.sh --config "$BASE_CONFIG" --t1-run-id "${IDS[0]}" --t2-run-id "${IDS[1]}" --t3-run-id "${IDS[2]}")
[[ "$REFREEZE" -eq 1 ]] && CMD+=(--refreeze)
[[ "$SKIP_TESTS" -eq 1 ]] && CMD+=(--skip-tests)
[[ "$SKIP_HARDWARE_SMOKE" -eq 1 ]] && CMD+=(--skip-hardware-smoke)
[[ "$ALLOW_ENVIRONMENT_MISMATCH" -eq 1 ]] && CMD+=(--allow-environment-mismatch)
if [[ "$INCLUDE" -eq 1 ]]; then
  CMD+=(--include-undersampling --t4-run-id "${IDS[3]}")
fi
"${CMD[@]}"

EXEC_MANIFEST="${PLAN_OUTPUT%.json}_execution.json"
python - "$PLAN_OUTPUT" "$EXEC_MANIFEST" "$ROOT/runs" <<'PY'
import json,sys
from pathlib import Path
plan=json.load(open(sys.argv[1])); runs_dir=Path(sys.argv[3]);
for r in plan['runs']:
 if not r.get('enabled',True): r['status']='not_requested'; continue
 r['status']='completed' if r.get('run_id') and (runs_dir/r['run_id']).is_dir() else 'missing'
plan['schema_version']='exp123_v8_phase6_los28_matrix_execution_v1'
Path(sys.argv[2]).write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(sys.argv[2])
PY
python scripts/workflows/compare_exp123_v8_matrix.py --config "$BASE_CONFIG" --matrix-manifest "$EXEC_MANIFEST"
echo "[OK] V8 Phase 6 LOS28 Discovery matrix completed. Confirmatory splits remain locked."
