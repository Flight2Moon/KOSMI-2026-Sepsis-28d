#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/experiment_resource_cuda.json}"
HOST="${EXP123_DASHBOARD_HOST:-127.0.0.1}"
PORT="${EXP123_DASHBOARD_PORT:-8080}"
python exp123_dashboard.py --config "$CONFIG" --host "$HOST" --port "$PORT"
