#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/run_exp123_v7_series.sh" --config "$ROOT/configs/experiment_cuda_v7.json" "$@"
