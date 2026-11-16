#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[ERROR] This setup script is intended for macOS." >&2
  exit 2
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "[ERROR] Apple Silicon arm64 is required for this profile." >&2
  exit 2
fi

if command -v brew >/dev/null 2>&1; then
  if ! brew list libomp >/dev/null 2>&1; then
    echo "[STEP] Installing libomp with Homebrew."
    brew install libomp
  fi
else
  echo "[WARNING] Homebrew is unavailable. Install libomp through conda-forge or Homebrew before using XGBoost." >&2
fi

python -m pip install --upgrade pip
python -m pip install -r requirements_macos_mps.txt
python -m pip check

python - <<'PY'
import platform
import torch

print("Platform:", platform.platform())
print("Machine:", platform.machine())
print("PyTorch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
if not torch.backends.mps.is_available():
    raise SystemExit("MPS is not available in the current Python/PyTorch environment.")
tensor = torch.arange(4096, device="mps", dtype=torch.float32)
value = float((tensor * tensor).sum().cpu())
print("MPS smoke value:", value)
if value <= 0:
    raise SystemExit("MPS smoke test failed.")
PY

echo "[OK] macOS MPS environment is ready."
