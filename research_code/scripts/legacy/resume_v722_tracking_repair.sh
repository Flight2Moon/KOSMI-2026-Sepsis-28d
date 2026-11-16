#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CONFIG="configs/experiment_cuda_v7.json"
RUN_ID=""

usage() {
  cat <<'EOF'
Usage:
  bash resume_v722_tracking_repair.sh --run-id RUN_ID [--config PATH]

This command resumes a V7.2.1 baseline Run that stopped only because the
shared tracking summary temporary file collided across worker processes.
It records a source-code amendment inside the Run before resuming.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="${2:?Missing RUN_ID}"; shift 2 ;;
    --config) CONFIG="${2:?Missing config path}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  echo "[ERROR] --run-id is required." >&2
  exit 2
fi

CONFIG="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$CONFIG")"
RUNS_DIR="$(PYTHONPATH=src python - "$CONFIG" <<'PY'
from pathlib import Path
import sys
from los_baseline.config import load_config, resolve_path
cfg = load_config(Path(sys.argv[1]))
print(resolve_path(cfg, cfg["paths"]["runs_dir"]))
PY
)"
RUN_DIR="$RUNS_DIR/$RUN_ID"

if [[ ! -f "$RUN_DIR/common/preflight/PREFLIGHT_PASSED.lock.json" ]]; then
  echo "[ERROR] Existing preflight lock not found: $RUN_DIR" >&2
  exit 2
fi

python - "$RUN_DIR" "$CONFIG" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
from los_baseline.utils import atomic_write_json, sha256_file

run_dir = Path(sys.argv[1]).resolve()
config = Path(sys.argv[2]).resolve()
amendments = run_dir / "common" / "amendments"
amendments.mkdir(parents=True, exist_ok=True)
files = [
    Path("src/los_baseline/utils.py"),
    Path("src/los_baseline/experiment_tracking.py"),
    Path("src/los_baseline/progress.py"),
]
payload = {
    "schema_version": "exp123_runtime_amendment_v1",
    "run_id": run_dir.name,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "Resume after multiprocess tracking summary temporary-file collision.",
    "scope": "Observability persistence only; no cohort, feature, outcome, split, model, hyperparameter, calibration, or evaluation definition changed.",
    "config_path": str(config),
    "patched_files": [
        {"path": str(path), "sha256": sha256_file(path)} for path in files
    ],
}
atomic_write_json(amendments / "v7_2_2_tracking_concurrency_hotfix.json", payload)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

bash verify_v722_tracking_fix.sh

# Direct protocol resume intentionally preserves the original release/preflight
# contract and records the observability-only amendment above. Existing valid
# completion locks are reused; incomplete model directories are rebuilt.
python run_exp123_staged_protocol.py baseline "$RUN_ID" --config "$CONFIG"
