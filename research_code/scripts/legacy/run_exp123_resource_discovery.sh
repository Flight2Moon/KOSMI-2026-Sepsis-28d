#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:?Usage: bash run_exp123_resource_discovery.sh RUN_ID [CONFIG]}"
CONFIG="${2:-configs/experiment_resource_cuda.json}"
python run_exp123_resource_protocol.py discovery "$RUN_ID" --config "$CONFIG"
