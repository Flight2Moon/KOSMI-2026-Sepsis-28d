#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUN_ID=""
BASELINE_RUN_ID=""
CONFIG=""
VERIFY_DATASET_HASHES=0

usage() {
  cat <<'EOF'
Usage:
  bash resume_v73_hpo_repair.sh \
    --run-id TRACK2_RUN_ID \
    [--baseline-run-id TRACK1_RUN_ID] \
    [--config TARGETED_CONFIG] \
    [--verify-dataset-hashes]

Repairs only the V7.2.2 Track 2 Run affected by:
- SVC(probability=False) predict_proba interface failure
- EXP1 XGBoost binary logloss with multi:softprob

Existing valid completion artifacts are reused. The original preflight is not
rewritten; a checksum-locked source amendment is recorded instead.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="${2:?Missing Run ID}"; shift 2 ;;
    --baseline-run-id) BASELINE_RUN_ID="${2:?Missing baseline Run ID}"; shift 2 ;;
    --config) CONFIG="${2:?Missing config path}"; shift 2 ;;
    --verify-dataset-hashes) VERIFY_DATASET_HASHES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$RUN_ID" ]] || { echo "[ERROR] --run-id is required." >&2; exit 2; }

if [[ -z "$CONFIG" ]]; then
  CONFIG="$(python - "$RUN_ID" <<'PY'
import json,sys
from pathlib import Path
run_id=sys.argv[1]
candidates=list(Path('runs').glob(f'*/common/preflight/PREFLIGHT_PASSED.lock.json'))
path=Path('runs')/run_id/'common/preflight/PREFLIGHT_PASSED.lock.json'
if not path.is_file():
    raise SystemExit(f'[ERROR] Preflight lock not found: {path}')
print(json.loads(path.read_text())['config_path'])
PY
)"
fi
CONFIG="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$CONFIG")"
RUNS_DIR="$(PYTHONPATH=src python - "$CONFIG" <<'PY'
from pathlib import Path
import sys
from los_baseline.config import load_config, resolve_path
cfg=load_config(Path(sys.argv[1]))
print(resolve_path(cfg,cfg['paths']['runs_dir']))
PY
)"
RUN_DIR="$RUNS_DIR/$RUN_ID"

if [[ -z "$BASELINE_RUN_ID" ]]; then
  BASELINE_RUN_ID="$(python - "$RUN_DIR" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'common/staged_optimized/BASELINE_LINK_VERIFIED.lock.json'
print(json.loads(p.read_text())['baseline_run_id'])
PY
)"
fi
BASELINE_RUN_DIR="$RUNS_DIR/$BASELINE_RUN_ID"

AMENDMENTS_DIR="$RUN_DIR/common/amendments"
mkdir -p "$AMENDMENTS_DIR"
AUDIT_PATH="$AMENDMENTS_DIR/v7_3_repair_input_audit.json"
python audit_exp123_v73_repair_inputs.py \
  --baseline-run-dir "$BASELINE_RUN_DIR" \
  --optimized-run-dir "$RUN_DIR" \
  --output "$AUDIT_PATH" >/dev/null

echo "[OK] Track 1/Track 2 repair eligibility verified."

bash verify_v73_hpo_fix.sh

AMENDMENT_PATH="$AMENDMENTS_DIR/v7_3_hpo_protocol_correction.json"
python - "$RUN_DIR" "$CONFIG" "$AUDIT_PATH" "$AMENDMENT_PATH" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path

sys.path.insert(0,str(Path.cwd()/'src'))
from los_baseline.utils import atomic_write_json, sha256_file

run_dir=Path(sys.argv[1]).resolve()
config=Path(sys.argv[2]).resolve()
audit_path=Path(sys.argv[3]).resolve()
out=Path(sys.argv[4]).resolve()
preflight_path=run_dir/'common/preflight/PREFLIGHT_PASSED.lock.json'
preflight=json.loads(preflight_path.read_text())
audit=json.loads(audit_path.read_text())
patched=[
 'src/los_baseline/amendments.py',
 'src/los_baseline/config.py',
 'src/los_baseline/failure_diagnostics.py',
 'src/los_baseline/hpo/optuna_engine.py',
 'src/los_baseline/hpo/search_space.py',
 'src/los_baseline/models.py',
 'src/los_baseline/optimization.py',
 'src/los_baseline/optimization_objectives.py',
 'src/los_baseline/training_engine.py',
 'run_exp123.py',
 'run_exp123_job.py',
 'run_exp123_amended_staged_protocol.py',
]
payload={
 'schema_version':'exp123_source_amendment_v2',
 'amendment_id':'v7_3_hpo_protocol_correction',
 'run_id':run_dir.name,
 'created_at_utc':datetime.now(timezone.utc).isoformat(),
 'base_release_manifest_sha256':preflight['release_manifest_sha256'],
 'preflight_lock_sha256':sha256_file(preflight_path),
 'config_path':str(config),
 'config_sha256':sha256_file(config),
 'shared_split_hash':audit['shared_split_hash'],
 'fixed_cv_hashes':audit['fixed_cv_hashes'],
 'confirmatory_splits_accessed':False,
 'affected_jobs':audit['expected_direct_root_failures'],
 'completed_jobs_reused':int(audit['counts']['completed']),
 'old_hpo_protocol':'legacy_v7_mixed_svm_probability_and_outer_fold_xgb_early_stopping',
 'new_hpo_protocol':'exp123_hpo_v7_3',
 'corrections':[
   'svm_hpo_uses_decision_function_with_probability_false',
   'raw_svm_margins_use_rank_metrics_without_probability_scores',
   'selected_svm_is_oof_calibrated_before_probability_evaluation',
   'exp1_xgboost_forces_multi_softprob_and_mlogloss',
   'xgboost_early_stopping_uses_inner_group_holdout_then_refits_outer_train',
   'failed_optuna_trials_do_not_consume_trial_budget',
   'new_hpo_study_name_isolated_from_failed_legacy_trials',
 ],
 'unchanged_contracts':[
   'dataset_files_and_fingerprints',
   'outcome_definition_and_28_day_endpoint',
   'patient_level_shared_split',
   'fixed_cv_fold_manifests',
   'feature_set',
   'class_weight_none_policy',
   'internal_test_and_external_eicu_remain_locked',
 ],
 'patched_files':[
   {'path':path,'sha256':sha256_file(Path(path))} for path in patched
 ],
 'clean_v73_series_required_for_final_confirmatory_claims':True,
}
atomic_write_json(out,payload)
out.with_suffix(out.suffix+'.sha256').write_text(sha256_file(out)+'\n')
print(json.dumps(payload,ensure_ascii=False,indent=2))
PY

export EXP123_AMENDMENT_MANIFEST="$AMENDMENT_PATH"
python run_exp123_amended_staged_protocol.py optimize "$RUN_ID" \
  --config "$CONFIG" \
  --amendment-manifest "$AMENDMENT_PATH"

echo "[OK] Track 2 amended resume completed."
echo "[INFO] Completion: $RUN_DIR/common/amendments/AMENDED_DISCOVERY_COMPLETED.lock.json"
echo "[IMPORTANT] Use a clean V7.3 three-track Run for final confirmatory claims."
