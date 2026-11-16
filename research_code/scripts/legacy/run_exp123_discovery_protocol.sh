#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RUN_ID [CONFIG_PATH]" >&2
  exit 2
fi
RUN_ID="$1"
CONFIG_PATH="${2:-configs/experiment.json}"
exec python run_exp123_resource_protocol.py discovery "$RUN_ID" --config "$CONFIG_PATH"
