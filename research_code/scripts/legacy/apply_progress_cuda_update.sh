#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -d exp123_release_freeze ]]; then
  rm -rf exp123_release_freeze
  echo "[OK] Removed the stale EXP123 release freeze."
fi

find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf .pytest_cache

cat <<'NOTICE'
[IMPORTANT] Source code and hardware configuration changed.
[REQUIRED] Run the following commands before a new clinical run:
  bash freeze_exp123_release.sh --config configs/experiment_cuda_dual_gpu.json --overwrite
  bash verify_exp123_release.sh
[REQUIRED] Use a new RUN_ID. Do not resume a clinical discovery run created with the previous release manifest.
NOTICE
