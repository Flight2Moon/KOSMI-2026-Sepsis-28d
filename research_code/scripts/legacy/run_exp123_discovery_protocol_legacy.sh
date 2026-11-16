#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RUN_ID [CONFIG_PATH]" >&2
  exit 2
fi

RUN_ID="$1"
CONFIG_PATH="${2:-configs/experiment.json}"

python run_exp123.py --config "$CONFIG_PATH" --run-id "$RUN_ID" --step outcomes
python run_exp123.py --config "$CONFIG_PATH" --run-id "$RUN_ID" --step splits

python run_exp123.py --config "$CONFIG_PATH" --run-id "$RUN_ID" --step exp1-optimize
python run_exp123.py --config "$CONFIG_PATH" --run-id "$RUN_ID" --step exp1-train

python run_exp123.py --config "$CONFIG_PATH" --run-id "$RUN_ID" --step stage1-optimize
python run_exp123.py --config "$CONFIG_PATH" --run-id "$RUN_ID" --step stage1-train

python run_exp123.py --config "$CONFIG_PATH" --run-id "$RUN_ID" --step stage2-optimize
python run_exp123.py --config "$CONFIG_PATH" --run-id "$RUN_ID" --step stage2-train

python run_exp123.py \
  --config "$CONFIG_PATH" \
  --run-id "$RUN_ID" \
  --step stage1-heterogeneity-discovery \
  --split validation
python run_exp123.py \
  --config "$CONFIG_PATH" \
  --run-id "$RUN_ID" \
  --step stage2-heterogeneity-discovery \
  --split validation

python run_exp123.py --config "$CONFIG_PATH" --run-id "$RUN_ID" --step stage1-super-learner
python run_exp123.py --config "$CONFIG_PATH" --run-id "$RUN_ID" --step stage2-super-learner

python run_exp123.py --config "$CONFIG_PATH" --run-id "$RUN_ID" --step status

echo "[OK] Discovery protocol completed."
echo "[NEXT] Review Validation outputs, then run freeze_exp123_confirmatory.sh --run-id ${RUN_ID}."
