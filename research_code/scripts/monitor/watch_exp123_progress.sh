#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RUN_ID [RUNS_DIR]" >&2
  exit 2
fi
exec python watch_exp123_progress.py --run-id "$1" --runs-dir "${2:-runs}"
