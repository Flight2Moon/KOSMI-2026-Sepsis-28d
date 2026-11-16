#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
CONFIG="configs/experiment_cuda_v7.json"
REFREEZE=0
SKIP_TESTS=0
SKIP_HARDWARE_SMOKE=0
ALLOW_ENVIRONMENT_MISMATCH=0
START_DASHBOARD=0

usage(){ cat <<'EOF'
Usage: bash run_exp123_v7_series.sh [options]
  --config PATH
  --refreeze
  --skip-tests
  --skip-hardware-smoke
  --allow-environment-mismatch
  --start-dashboard

Runs Track 1 Baseline, Track 2 Optuna Optimized, then Track 3 fold-wise
majority undersampling. Internal Test and eICU confirmatory evaluation remain locked.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="${2:?}"; shift 2;;
    --refreeze) REFREEZE=1; shift;;
    --skip-tests) SKIP_TESTS=1; shift;;
    --skip-hardware-smoke) SKIP_HARDWARE_SMOKE=1; shift;;
    --allow-environment-mismatch) ALLOW_ENVIRONMENT_MISMATCH=1; shift;;
    --start-dashboard) START_DASHBOARD=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 2;;
  esac
done
[[ "$CONFIG" = /* ]] || CONFIG="$ROOT/$CONFIG"
CONFIG="$(python -c 'import pathlib,sys;print(pathlib.Path(sys.argv[1]).resolve())' "$CONFIG")"
python -m json.tool "$CONFIG" >/dev/null
BACKEND="$(python - "$CONFIG" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['resources']['accelerator']['backend'])
PY
)"
STAMP="$(date +%Y_%m%d_%H%M%S)"
SERIES_ID="exp123_v7_series_${BACKEND}_${STAMP}"
BASELINE_ID="exp123_v7_${BACKEND}_track1_baseline_${STAMP}"
OPTIMIZED_ID="exp123_v7_${BACKEND}_track2_optimized_${STAMP}"
UNDERSAMPLED_ID="exp123_v7_${BACKEND}_track3_undersampled_${STAMP}"
SERIES_DIR="$ROOT/series/$SERIES_ID"
mkdir -p "$SERIES_DIR"

COMMON_ARGS=(--config "$CONFIG")
[[ "$SKIP_TESTS" -eq 1 ]] && COMMON_ARGS+=(--skip-tests)
[[ "$SKIP_HARDWARE_SMOKE" -eq 1 ]] && COMMON_ARGS+=(--skip-hardware-smoke)
[[ "$ALLOW_ENVIRONMENT_MISMATCH" -eq 1 ]] && COMMON_ARGS+=(--allow-environment-mismatch)

if [[ "$START_DASHBOARD" -eq 1 ]]; then
  nohup python exp123_observatory.py --config "$CONFIG" >"$SERIES_DIR/dashboard.log" 2>&1 &
  echo $! > "$SERIES_DIR/dashboard.pid"
  echo "[INFO] Observatory: http://127.0.0.1:8080"
fi

echo "[STEP] Track 1 Natural Baseline"
T1=(bash run_exp123_v6_series.sh "${COMMON_ARGS[@]}" --phase baseline --run-id "$BASELINE_ID")
[[ "$REFREEZE" -eq 1 ]] && T1+=(--refreeze)
"${T1[@]}"

echo "[STEP] Track 2 Optuna Optimized"
bash run_exp123_v6_series.sh "${COMMON_ARGS[@]}" --phase optimize --baseline-run-id "$BASELINE_ID" --run-id "$OPTIMIZED_ID"

TARGETED_CONFIG="$ROOT/generated_configs/${BASELINE_ID}_targeted_${BACKEND}.json"
if [[ ! -f "$TARGETED_CONFIG" ]]; then
  echo "[ERROR] Targeted config missing: $TARGETED_CONFIG" >&2; exit 2
fi

echo "[STEP] Track 3 preflight and dataset lock"
bash preflight_exp123.sh --config "$TARGETED_CONFIG" --run-id "$UNDERSAMPLED_ID"

echo "[STEP] Track 3 Fold-wise Majority Undersampling"
python run_exp123_v7_undersampling.py \
  --config "$TARGETED_CONFIG" \
  --run-id "$UNDERSAMPLED_ID" \
  --optimized-run-id "$OPTIMIZED_ID"

python - "$SERIES_DIR" "$SERIES_ID" "$CONFIG" "$TARGETED_CONFIG" "$BASELINE_ID" "$OPTIMIZED_ID" "$UNDERSAMPLED_ID" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
out=Path(sys.argv[1])
p={
 "schema_version":"exp123_v7_three_track_series_v1","series_id":sys.argv[2],
 "base_config":sys.argv[3],"targeted_config":sys.argv[4],
 "track1_baseline_run_id":sys.argv[5],"track2_optimized_run_id":sys.argv[6],
 "track3_undersampled_run_id":sys.argv[7],"confirmatory_splits_accessed":False,
 "created_at_utc":datetime.now(timezone.utc).isoformat(),
}
(out/'series_manifest.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
PY
printf '%s\n' "$BASELINE_ID" > "$SERIES_DIR/track1_run_id.txt"
printf '%s\n' "$OPTIMIZED_ID" > "$SERIES_DIR/track2_run_id.txt"
printf '%s\n' "$UNDERSAMPLED_ID" > "$SERIES_DIR/track3_run_id.txt"
python compare_exp123_v7_tracks.py --config "$TARGETED_CONFIG" --series-id "$SERIES_ID"
echo "[OK] V7 three-track Discovery series completed: $SERIES_ID"
echo "[INFO] Confirmatory evaluation is intentionally not started."
