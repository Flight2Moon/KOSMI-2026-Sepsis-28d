#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/run_exp123_v6_series.sh" \
  --config "$ROOT/configs/experiment_mps_staged_m1max_32gb.json" \
  "$@"
