#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PROJECT_ROOT/run_exp123_safe.sh" \
  --config "$PROJECT_ROOT/configs/experiment_cpu_7800x3d_48gb.json" \
  "$@"
