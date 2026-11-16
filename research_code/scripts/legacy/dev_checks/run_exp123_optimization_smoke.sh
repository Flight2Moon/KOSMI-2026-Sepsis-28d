#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="exp123_optimization_smoke"
OVERWRITE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

OUTPUT="${PROJECT_ROOT}/runs/${RUN_ID}/common/optimization_smoke"
ARGS=(--output "${OUTPUT}")
if [[ "${OVERWRITE}" == "true" ]]; then
  ARGS+=(--overwrite)
fi

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
python "${PROJECT_ROOT}/tests/run_exp123_optimization_smoke.py" "${ARGS[@]}"
