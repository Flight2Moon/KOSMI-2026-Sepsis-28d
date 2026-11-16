#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
python check_exp123_hardware.py \
  --config configs/experiment_mps_m1max_32gb_mortality_los.json \
  --xgboost-smoke
