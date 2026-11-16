#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
rm -rf exp123_release_freeze
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
rm -rf .pytest_cache
cat <<'EOF'
[OK] Resource-aware scheduler and Web dashboard files are applied.
[REQUIRED] Install requirements, select a hardware config, and create a new release freeze.
[REQUIRED] Use a new Run ID; do not continue a Run created with the previous source release.
EOF
