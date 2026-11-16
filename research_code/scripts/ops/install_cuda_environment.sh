#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.10.0}"
PYTORCH_CUDA_INDEX_URL="${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
CONFIG_PATH="${CONFIG_PATH:-configs/experiment_cuda_dual_gpu.json}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi was not found. Install a compatible NVIDIA driver first." >&2
  exit 1
fi

nvidia-smi

TEMP_REQUIREMENTS="$(mktemp)"
trap 'rm -f "$TEMP_REQUIREMENTS"' EXIT

grep -vE '^torch==' requirements.txt > "$TEMP_REQUIREMENTS"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r "$TEMP_REQUIREMENTS"
"$PYTHON_BIN" -m pip install \
  "torch==${PYTORCH_VERSION}" \
  --index-url "$PYTORCH_CUDA_INDEX_URL"

"$PYTHON_BIN" scripts/ops/check_exp123_hardware.py \
  --config "$CONFIG_PATH" \
  --xgboost-smoke

cat <<EOF
[OK] CUDA environment installation and smoke checks completed.
[INFO] PyTorch index: ${PYTORCH_CUDA_INDEX_URL}
[INFO] Hardware profile: ${CONFIG_PATH}
EOF
