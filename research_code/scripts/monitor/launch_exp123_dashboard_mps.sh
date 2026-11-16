#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$PROJECT_ROOT/launch_exp123_dashboard.sh" \
  "$PROJECT_ROOT/configs/experiment_mps_m1max_32gb_mortality_los.json"
