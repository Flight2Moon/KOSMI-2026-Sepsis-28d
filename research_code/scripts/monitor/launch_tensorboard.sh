#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash launch_tensorboard.sh runs/RUN_ID" >&2
    exit 1
fi

RUN_DIR="$1"
tensorboard --logdir "$RUN_DIR/tensorboard" --port "${TENSORBOARD_PORT:-6006}"
