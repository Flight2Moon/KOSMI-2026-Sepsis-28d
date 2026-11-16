#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[NOTICE] run_exp123_v5_cuda.sh is a compatibility alias for V6." >&2
exec bash "$ROOT/run_exp123_v6_cuda.sh" "$@"
