#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/experiment_cuda_v7.json}"
exec python exp123_observatory.py --config "$CONFIG"
