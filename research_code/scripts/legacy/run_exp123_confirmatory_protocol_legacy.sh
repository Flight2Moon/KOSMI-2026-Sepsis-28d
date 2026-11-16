#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RUN_ID [CONFIG_PATH]" >&2
  exit 2
fi

RUN_ID="$1"
CONFIG_PATH="${2:-configs/experiment.json}"
STAGE1_FROZEN="runs/${RUN_ID}/EXP2_two_stage/heterogeneity_analysis/stage1_mortality/discovery_validation/frozen_design.json"
STAGE2_FROZEN="runs/${RUN_ID}/EXP2_two_stage/heterogeneity_analysis/stage2_long_stay/discovery_validation/frozen_design.json"

python run_exp123.py \
  --config "$CONFIG_PATH" \
  --run-id "$RUN_ID" \
  --step stage1-heterogeneity-confirmatory \
  --split internal_test \
  --frozen-design "$STAGE1_FROZEN"
python run_exp123.py \
  --config "$CONFIG_PATH" \
  --run-id "$RUN_ID" \
  --step stage2-heterogeneity-confirmatory \
  --split internal_test \
  --frozen-design "$STAGE2_FROZEN"
python run_exp123.py \
  --config "$CONFIG_PATH" \
  --run-id "$RUN_ID" \
  --step oracle-pipeline-confirmatory \
  --split internal_test
python run_exp123.py \
  --config "$CONFIG_PATH" \
  --run-id "$RUN_ID" \
  --step final-comparison-confirmatory \
  --split internal_test

python run_exp123.py \
  --config "$CONFIG_PATH" \
  --run-id "$RUN_ID" \
  --step stage1-heterogeneity-confirmatory \
  --split external_eicu \
  --frozen-design "$STAGE1_FROZEN"
python run_exp123.py \
  --config "$CONFIG_PATH" \
  --run-id "$RUN_ID" \
  --step stage2-heterogeneity-confirmatory \
  --split external_eicu \
  --frozen-design "$STAGE2_FROZEN"
python run_exp123.py \
  --config "$CONFIG_PATH" \
  --run-id "$RUN_ID" \
  --step oracle-pipeline-confirmatory \
  --split external_eicu
python run_exp123.py \
  --config "$CONFIG_PATH" \
  --run-id "$RUN_ID" \
  --step final-comparison-confirmatory \
  --split external_eicu

python run_exp123.py --config "$CONFIG_PATH" --run-id "$RUN_ID" --step finalize-run
python run_exp123.py --config "$CONFIG_PATH" --run-id "$RUN_ID" --step status

echo "[OK] Confirmatory protocol and final run freeze completed."
