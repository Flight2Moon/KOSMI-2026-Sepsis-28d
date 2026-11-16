#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="configs/experiment_cuda_staged_7800x3d_48gb.json"
PHASE="all"
BASELINE_RUN_ID=""
RUN_ID=""
REFREEZE=0
SKIP_TESTS=0
SKIP_HARDWARE_SMOKE=0
RERUN_COMPLETED=0
RESUME=0
ALLOW_ENVIRONMENT_MISMATCH=0

usage() {
  cat <<'EOF'
Usage:
  bash run_exp123_v6_series.sh [options]

Options:
  --config PATH                 Base staged experiment config.
  --phase baseline|optimize|all Execution phase. Default: all.
  --baseline-run-id RUN_ID      Required for optimize phase.
  --run-id RUN_ID               Explicit Run ID for one phase.
  --refreeze                    Rebuild release freeze before a new phase.
  --resume                      Resume an interrupted single phase.
  --rerun-completed             Rerun completed scheduler jobs.
  --skip-tests                  Skip regression tests.
  --skip-hardware-smoke         Skip hardware smoke tests.
  --allow-environment-mismatch  Allow release freeze despite dependency mismatch.
  -h, --help                    Show help.

Examples:
  bash run_exp123_v6_series.sh --phase all --refreeze
  bash run_exp123_v6_series.sh --phase baseline --refreeze
  bash run_exp123_v6_series.sh --phase optimize \
    --baseline-run-id exp123_cuda_baseline_2026_0722_010000
  bash run_exp123_v6_series.sh --phase baseline --resume \
    --run-id exp123_cuda_baseline_2026_0722_010000
  bash run_exp123_v6_series.sh --phase optimize --resume \
    --baseline-run-id exp123_cuda_baseline_2026_0722_010000 \
    --run-id exp123_cuda_optimized_2026_0722_030000
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="${2:?Missing value for --config}"; shift 2 ;;
    --phase) PHASE="${2:?Missing value for --phase}"; shift 2 ;;
    --baseline-run-id) BASELINE_RUN_ID="${2:?Missing value}"; shift 2 ;;
    --run-id) RUN_ID="${2:?Missing value}"; shift 2 ;;
    --refreeze) REFREEZE=1; shift ;;
    --resume) RESUME=1; shift ;;
    --rerun-completed) RERUN_COMPLETED=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --skip-hardware-smoke) SKIP_HARDWARE_SMOKE=1; shift ;;
    --allow-environment-mismatch) ALLOW_ENVIRONMENT_MISMATCH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$PHASE" != "baseline" && "$PHASE" != "optimize" && "$PHASE" != "all" ]]; then
  echo "[ERROR] --phase must be baseline, optimize, or all." >&2
  exit 2
fi
if [[ "$RESUME" -eq 1 ]]; then
  if [[ "$PHASE" == "all" ]]; then
    echo "[ERROR] --resume supports one phase at a time." >&2
    exit 2
  fi
  if [[ -z "$RUN_ID" ]]; then
    echo "[ERROR] --resume requires --run-id." >&2
    exit 2
  fi
  if [[ "$REFREEZE" -eq 1 ]]; then
    echo "[ERROR] Do not combine --resume and --refreeze." >&2
    exit 2
  fi
fi

if [[ "$CONFIG_PATH" != /* ]]; then CONFIG_PATH="$PROJECT_ROOT/$CONFIG_PATH"; fi
CONFIG_PATH="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$CONFIG_PATH")"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config not found: $CONFIG_PATH" >&2
  exit 2
fi
python -m json.tool "$CONFIG_PATH" >/dev/null

BACKEND="$(python - "$CONFIG_PATH" <<'PY_BACKEND'
import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(config.get("resources", {}).get("accelerator", {}).get("backend", "auto"))
PY_BACKEND
)"
TIMESTAMP="$(date +%Y_%m%d_%H%M%S)"
SERIES_ID="exp123_series_${BACKEND}_${TIMESTAMP}"
SERIES_DIR="$PROJECT_ROOT/series/$SERIES_ID"
mkdir -p "$SERIES_DIR"

resolve_runs_root() {
  local config="$1"
  PYTHONPATH=src python - "$config" <<'PY_RUNS'
from pathlib import Path
import sys
from los_baseline.config import load_config
config = load_config(Path(sys.argv[1]))
value = Path(str(config["paths"].get("runs_dir", "runs"))).expanduser()
if not value.is_absolute():
    value = Path(str(config["_project_root"])) / value
print(value.resolve())
PY_RUNS
}

run_checks() {
  local config="$1"
  if [[ "$SKIP_TESTS" -eq 0 ]]; then
    echo "[STEP] Running regression tests."
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q
  fi
  echo "[STEP] Validating config and dataset paths."
  python validate_exp123_setup.py --config "$config" --check-data
  if [[ "$SKIP_HARDWARE_SMOKE" -eq 0 ]]; then
    echo "[STEP] Running hardware smoke checks."
    if [[ "$BACKEND" == "cuda" ]]; then
      python scripts/ops/check_exp123_hardware.py --config "$config" --xgboost-smoke
    else
      python scripts/ops/check_exp123_hardware.py --config "$config"
    fi
  fi
}

prepare_release_and_preflight() {
  local config="$1"
  local run_id="$2"
  local force_refreeze="$3"
  if [[ "$force_refreeze" -eq 1 ]]; then
    echo "[STEP] Rebuilding release freeze for $run_id"
    rm -rf "$PROJECT_ROOT/exp123_release_freeze"
    local freeze_args=(--config "$config" --overwrite)
    if [[ "$ALLOW_ENVIRONMENT_MISMATCH" -eq 1 ]]; then
      freeze_args+=(--allow-environment-mismatch)
    fi
    bash scripts/ops/freeze_exp123_release.sh "${freeze_args[@]}"
  fi
  local verify_args=()
  if [[ "$ALLOW_ENVIRONMENT_MISMATCH" -eq 1 ]]; then
    verify_args+=(--skip-environment)
  fi
  if ! bash scripts/ops/verify_exp123_release.sh "${verify_args[@]}"; then
    echo "[ERROR] Release freeze does not match current source/config." >&2
    echo "[INFO] Retry with --refreeze." >&2
    exit 2
  fi
  local runs_root
  runs_root="$(resolve_runs_root "$config")"
  if [[ -d "$runs_root/$run_id" ]] && find "$runs_root/$run_id" -mindepth 1 -print -quit | grep -q .; then
    echo "[ERROR] New phase Run ID already has outputs: $runs_root/$run_id" >&2
    exit 2
  fi
  bash preflight_exp123.sh --config "$config" --run-id "$run_id"
}

run_baseline_phase() {
  local run_id="$1"
  local config="$2"
  if [[ "$RESUME" -eq 1 ]]; then
    python validate_exp123_setup.py --config "$config" --check-release --run-id "$run_id"
  else
    prepare_release_and_preflight "$config" "$run_id" "$REFREEZE"
  fi
  local extra=()
  if [[ "$RERUN_COMPLETED" -eq 1 ]]; then extra+=(--rerun-completed); fi
  python run_exp123_staged_protocol.py baseline "$run_id" --config "$config" "${extra[@]}"
  echo "$run_id" > "$SERIES_DIR/baseline_run_id.txt"
  local runs_root
  runs_root="$(resolve_runs_root "$config")"
  echo "[OK] Baseline candidate plan: $runs_root/$run_id/common/staged_baseline/candidate_plan.json"
}

run_optimize_phase() {
  local baseline_run_id="$1"
  local optimized_run_id="$2"
  local base_config="$3"
  local runs_root
  runs_root="$(resolve_runs_root "$base_config")"
  local candidate_plan="$runs_root/$baseline_run_id/common/staged_baseline/candidate_plan.json"
  if [[ ! -f "$candidate_plan" ]]; then
    echo "[ERROR] Baseline candidate plan not found: $candidate_plan" >&2
    exit 2
  fi
  local targeted_config="$PROJECT_ROOT/generated_configs/${baseline_run_id}_targeted_${BACKEND}.json"
  python build_exp123_targeted_config.py \
    --base-config "$base_config" \
    --candidate-plan "$candidate_plan" \
    --output "$targeted_config"

  # A new targeted Run receives its own release freeze. An interrupted targeted
  # Run resumes against the existing release and preflight contract.
  if [[ "$RESUME" -eq 1 ]]; then
    python validate_exp123_setup.py --config "$targeted_config" --check-release --run-id "$optimized_run_id"
  else
    prepare_release_and_preflight "$targeted_config" "$optimized_run_id" 1
  fi

  local extra=()
  if [[ "$RERUN_COMPLETED" -eq 1 ]]; then extra+=(--rerun-completed); fi
  python run_exp123_staged_protocol.py optimize "$optimized_run_id" --config "$targeted_config" "${extra[@]}"
  echo "$optimized_run_id" > "$SERIES_DIR/optimized_run_id.txt"
  echo "$targeted_config" > "$SERIES_DIR/targeted_config_path.txt"
}

run_checks "$CONFIG_PATH"

case "$PHASE" in
  baseline)
    if [[ -z "$RUN_ID" ]]; then RUN_ID="exp123_${BACKEND}_baseline_${TIMESTAMP}"; fi
    run_baseline_phase "$RUN_ID" "$CONFIG_PATH"
    BASELINE_RUN_ID="$RUN_ID"
    ;;
  optimize)
    if [[ -z "$BASELINE_RUN_ID" ]]; then
      echo "[ERROR] optimize phase requires --baseline-run-id." >&2
      exit 2
    fi
    if [[ -z "$RUN_ID" ]]; then RUN_ID="exp123_${BACKEND}_optimized_${TIMESTAMP}"; fi
    run_optimize_phase "$BASELINE_RUN_ID" "$RUN_ID" "$CONFIG_PATH"
    ;;
  all)
    if [[ -n "$RUN_ID" ]]; then
      echo "[ERROR] --run-id is ambiguous with --phase all. Omit it." >&2
      exit 2
    fi
    BASELINE_RUN_ID="exp123_${BACKEND}_baseline_${TIMESTAMP}"
    run_baseline_phase "$BASELINE_RUN_ID" "$CONFIG_PATH"
    OPTIMIZED_RUN_ID="exp123_${BACKEND}_optimized_$(date +%Y_%m%d_%H%M%S)"
    run_optimize_phase "$BASELINE_RUN_ID" "$OPTIMIZED_RUN_ID" "$CONFIG_PATH"
    RUN_ID="$OPTIMIZED_RUN_ID"
    ;;
esac

python - "$SERIES_DIR" "$SERIES_ID" "$BACKEND" "$PHASE" "$CONFIG_PATH" "${BASELINE_RUN_ID:-}" "${RUN_ID:-}" <<'PY_MANIFEST'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
series_dir = Path(sys.argv[1])
payload = {
    "schema_version": "exp123_staged_series_v1",
    "series_id": sys.argv[2],
    "backend": sys.argv[3],
    "requested_phase": sys.argv[4],
    "base_config": sys.argv[5],
    "baseline_run_id": sys.argv[6] or None,
    "optimized_run_id": sys.argv[7] or None,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
(series_dir / "series_manifest.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY_MANIFEST

echo "[OK] Series manifest: $SERIES_DIR/series_manifest.json"
