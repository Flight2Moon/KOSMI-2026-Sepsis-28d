#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/run_exp123_v6_series.sh" \
  --config "$ROOT/configs/experiment_cuda_staged_7800x3d_48gb.json" \
  "$@"
