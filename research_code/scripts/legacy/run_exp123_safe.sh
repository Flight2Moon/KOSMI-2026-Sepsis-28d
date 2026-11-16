#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="configs/experiment_cuda_max_gpu_throughput_7800x3d_48gb.json"
MODE="discovery"
RUN_ID=""
REFREEZE=0
SKIP_TESTS=0
SKIP_HARDWARE_SMOKE=0
RESUME=0

usage() {
  cat <<'EOF'
Usage:
  bash run_exp123_safe.sh [options]

Options:
  --config PATH             Experiment config path.
  --mode MODE               discovery or confirmatory. Default: discovery.
  --run-id RUN_ID           Explicit run ID. Required for --resume and confirmatory.
  --refreeze                Rebuild the release freeze from the selected config.
  --resume                  Resume an existing run using the same config and release.
  --skip-tests              Skip pytest.
  --skip-hardware-smoke     Skip check_exp123_hardware.py.
  -h, --help                Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:?Missing value for --config}"
      shift 2
      ;;
    --mode)
      MODE="${2:?Missing value for --mode}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:?Missing value for --run-id}"
      shift 2
      ;;
    --refreeze)
      REFREEZE=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --skip-tests)
      SKIP_TESTS=1
      shift
      ;;
    --skip-hardware-smoke)
      SKIP_HARDWARE_SMOKE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "discovery" && "$MODE" != "confirmatory" ]]; then
  echo "[ERROR] --mode must be discovery or confirmatory." >&2
  exit 2
fi

if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$PROJECT_ROOT/$CONFIG_PATH"
fi
CONFIG_PATH="$(readlink -f "$CONFIG_PATH" 2>/dev/null || true)"

if [[ -z "$CONFIG_PATH" || ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config file was not found." >&2
  echo "[INFO] Available configs:" >&2
  find "$PROJECT_ROOT/configs" -maxdepth 1 -type f -name '*.json' -printf '  - %P\n' | sort >&2
  exit 2
fi

if [[ "$RESUME" -eq 1 && -z "$RUN_ID" ]]; then
  echo "[ERROR] --resume requires --run-id." >&2
  exit 2
fi
if [[ "$MODE" == "confirmatory" && -z "$RUN_ID" ]]; then
  echo "[ERROR] Confirmatory execution requires --run-id." >&2
  exit 2
fi
if [[ "$RESUME" -eq 1 && "$REFREEZE" -eq 1 ]]; then
  echo "[ERROR] --resume and --refreeze cannot be used together." >&2
  exit 2
fi

python -m json.tool "$CONFIG_PATH" >/dev/null

BACKEND="$(python - "$CONFIG_PATH" <<'PY'
import json
import sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(config.get("resources", {}).get("accelerator", {}).get("backend", "auto"))
PY
)"

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="exp123_${BACKEND}_$(date +%Y_%m%d_%H%M%S)"
fi
export RUN_ID

printf '[INFO] Project: %s\n' "$PROJECT_ROOT"
printf '[INFO] Config:  %s\n' "$CONFIG_PATH"
printf '[INFO] Backend: %s\n' "$BACKEND"
printf '[INFO] Mode:    %s\n' "$MODE"
printf '[INFO] Run ID:  %s\n' "$RUN_ID"

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  echo "[STEP] Running regression tests."
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q
fi

echo "[STEP] Validating config and dataset paths."
python validate_exp123_setup.py --config "$CONFIG_PATH" --check-data

if [[ "$SKIP_HARDWARE_SMOKE" -eq 0 ]]; then
  echo "[STEP] Running hardware smoke checks."
  if [[ "$BACKEND" == "cuda" ]]; then
    python check_exp123_hardware.py --config "$CONFIG_PATH" --xgboost-smoke
  else
    python check_exp123_hardware.py --config "$CONFIG_PATH"
  fi
fi

if [[ "$REFREEZE" -eq 1 ]]; then
  if [[ -d "$PROJECT_ROOT/runs/$RUN_ID" ]] && find "$PROJECT_ROOT/runs/$RUN_ID" -mindepth 1 -print -quit | grep -q .; then
    echo "[ERROR] Refusing to rebuild the release while the selected Run ID already has outputs." >&2
    echo "[INFO] Use a new Run ID when --refreeze is requested." >&2
    exit 2
  fi
  echo "[STEP] Rebuilding release freeze."
  rm -rf "$PROJECT_ROOT/exp123_release_freeze"
  bash freeze_exp123_release.sh --config "$CONFIG_PATH" --overwrite
fi

if ! bash verify_exp123_release.sh; then
  echo "[ERROR] The release freeze is missing or does not match the current source/config." >&2
  echo "[INFO] Start a new run with --refreeze." >&2
  exit 2
fi

if [[ "$RESUME" -eq 0 && "$MODE" == "discovery" ]]; then
  echo "[STEP] Running preflight."
  bash preflight_exp123.sh --config "$CONFIG_PATH" --run-id "$RUN_ID"
else
  echo "[STEP] Verifying existing preflight contract."
  python validate_exp123_setup.py --config "$CONFIG_PATH" --check-release --run-id "$RUN_ID"
fi

set +e
if [[ "$MODE" == "discovery" ]]; then
  bash run_exp123_discovery_protocol.sh "$RUN_ID" "$CONFIG_PATH"
  EXIT_CODE=$?
else
  bash run_exp123_confirmatory_protocol.sh "$RUN_ID" "$CONFIG_PATH"
  EXIT_CODE=$?
fi
set -e

if [[ "$EXIT_CODE" -ne 0 ]]; then
  echo "[STEP] Generating automatic failure diagnosis."
  python diagnose_exp123_run.py --config "$CONFIG_PATH" --run-id "$RUN_ID" --mode "$MODE" || true
  exit "$EXIT_CODE"
fi

echo "[OK] EXP123 $MODE completed for $RUN_ID"
