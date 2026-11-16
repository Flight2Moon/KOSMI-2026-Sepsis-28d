# EXP123 V9 Alpha 7 — Study28 Reproducible Maximum-Throughput Execution

V9 Alpha 7 connects hardware detection and a frozen resource plan to the full Study28 pipeline. It aggressively uses CPU/RAM while preventing nested OpenMP/BLAS oversubscription, parallelizes independent outer folds/external sites/sensitivity settings, parallelizes bootstrap metric evaluation without changing the seeded resamples, and records actual resource usage. The scientific Study28 protocol and frozen primary analysis definitions are unchanged.

## One-command full experiment

```bash
bash run_study28_full_pipeline.sh \
  --dataset-root /path/to/final_dataset \
  --run-root runs/study28_full_main
```

The default `maximum_throughput` profile targets near-full machine utilization. The first run writes `hardware_execution_plan.json`; resumes with the same `--run-root` reuse that exact plan. Use `--replan-resources` only when intentionally regenerating it. NVIDIA GPUs are detected and logged, but the confirmatory primary LightGBM stays on the CPU backend to preserve the established professor-result implementation path. See `docs/V9_ALPHA7_STUDY28_RESOURCE_AUTOTUNE.md`.

# EXP123 V9 Alpha 6 — Study28 Manuscript Evidence & Final Reporting

V9 Alpha 6 is a reporting-only layer. It preserves the frozen Alpha 2 primary model and Alpha 3 raw external validation, then assembles Alpha 2–5 aggregate evidence into role-locked main/supplementary tables, figures, an evidence index, finalization checklist, and reproducibility manifest. It does not train or retune a model and it blocks patient-level reporting artifacts.

## Run Alpha 6

```bash
PYTHONPATH=src:. python tools/run_study28_alpha6.py \
  --alpha2-output-dir runs/study28_alpha2 \
  --alpha3-output-dir runs/study28_alpha3 \
  --alpha4-output-dir runs/study28_alpha4 \
  --alpha5-output-dir runs/study28_alpha5 \
  --output-dir runs/study28_alpha6
```

For release-code verification without patient-level run artifacts, use `--use-release-verification-snapshots`. See `docs/V9_ALPHA6_STUDY28_MANUSCRIPT_EVIDENCE_REPORTING.md`.

# EXP123 V9 Alpha 5 — Study28 Robustness & Interpretability

V9 Alpha 5 is strictly downstream of the frozen Alpha 2 primary model and completed Alpha 3/4 analyses. It adds primary FS_NOAPS SHAP, no-`pre_icu_los_d` robustness, external feature-availability sensitivity, observed missingness-burden analysis, HFD28 secondary modeling, and an explicit home-discharge reproducibility guard. It does not retune or replace the primary model.

## Run Alpha 5

```bash
PYTHONPATH=src:. python tools/run_study28_alpha5.py \
  --dataset-root /path/to/final_dataset \
  --alpha2-output-dir runs/study28_alpha2 \
  --alpha3-output-dir runs/study28_alpha3 \
  --alpha4-output-dir runs/study28_alpha4 \
  --output-dir runs/study28_alpha5
```

Model-heavy settings run as isolated one-thread workers with resumable `result.json` checkpoints. The pre-specified 1,000 paired/bootstrap repeats are preserved. Home-discharge sensitivity is blocked when `discharge_location` is unavailable; labels are never inferred. See `docs/V9_ALPHA5_STUDY28_ROBUSTNESS_INTERPRETABILITY.md` and `docs/v9_alpha5_verification.json`.

# EXP123 V9 Alpha 4 — Study28 Sensitivity & Clinical Utility

V9 Alpha 4 is strictly downstream of the completed Alpha 3 raw external validation. It adds professor-grounded reverse validation, 20-seed local site recalibration, DCA, subgroup analysis, and eICU hospital-level heterogeneity without changing the Alpha 2 frozen MIMIC primary model or overwriting Alpha 3 raw external results.

## Run Alpha 4

Prerequisites: completed Alpha 2 and Alpha 3 outputs plus the professor-provided frozen final-dataset directory.

```bash
PYTHONPATH=src:. python tools/run_study28_alpha4.py \
  --dataset-root /path/to/final_dataset \
  --alpha2-output-dir runs/study28_alpha2 \
  --alpha3-output-dir runs/study28_alpha3 \
  --output-dir runs/study28_alpha4
```

The reverse-validation directions are executed as isolated LightGBM workers and checkpointed. If an OS/OpenMP runtime interrupts the parent process after a completed worker, rerun the same command; completed pair checkpoints are reused and no completed analysis is recomputed.

### Alpha 4 analyses

- Reverse sensitivity: eICU-alt → MIMIC and eICU-primary → MIMIC, using FS_NOAPS and locked LightGBM hyperparameters.
- Local adaptation: 20% local calibration / 80% held-out evaluation, 20 seeds, professor-compatible per-class logistic recalibration.
- DCA: death and prolonged one-vs-rest, threshold probabilities 0.01–0.60.
- Subgroups: eICU-alt professor reference definitions; other external datasets are explicitly exploratory extensions.
- Hospital heterogeneity: eICU-alt only, N≥50 and events≥5, Hanley-McNeil SE + DerSimonian-Laird random effects.

Patient-level reverse probabilities and reverse model artifacts are local-run sensitive outputs and must not be distributed. See `docs/V9_ALPHA4_STUDY28_SENSITIVITY_CLINICAL_UTILITY.md` and `docs/v9_alpha4_verification.json`.

# EXP123 V9 Alpha 2 — Study28 LightGBM Development (prerequisite/history)

Alpha 2 reproduces the professor 5×3 LightGBM nested-CV lock, evaluates the primary `FS_NOAPS` model on the MIMIC temporal split, and freezes the full-MIMIC model before Alpha 3 external validation. See `docs/V9_ALPHA2_STUDY28_DEVELOPMENT.md`.


# EXP123 V8 Phase 6 — LOS28 High-Throughput

Phase 6 is the active execution surface. The primary workflow is **LOS28 only**.
Historical multi-horizon/LOS14 materials are retained under `legacy_reference/` for audit only.

## Quick start

```bash
./exp123 plan
./exp123 autotune --profile maximum_throughput
./exp123 run --refreeze
# optional T4
./exp123 run --refreeze --include-undersampling
```

The root directory is intentionally minimal. Active workflow commands live in
`scripts/workflows/`, operations in `scripts/ops/`, monitoring tools in
`scripts/monitor/`, and historical entrypoints in `scripts/legacy/`.

---

# EXP123 V7.3 — Cohort-First Three-Track Clinical Experiment Platform

EXP123 V7.3은 MIMIC-IV에서 28일 상태 예측 모델을 개발하고 eICU에서 외부 검증하기 위한 재현 가능한 임상 머신러닝 실행 패키지입니다.

V7은 이전 결과 분석에서 확인된 코호트 중복과 시간축 오류를 먼저 차단한 뒤, 동일한 환자 코호트와 Split에서 세 가지 학습 Track을 비교합니다.

```text
Track 1 · Natural Baseline
Track 2 · Optuna Optimized
Track 3 · Fold-wise Majority Undersampling

각 Track에서:
EXP1 Flat A/B/C + EXP2 Two-stage + EXP3 Stage-2 Oracle
```

## V7.3 HPO Protocol Correction

- SVM HPO는 모든 후보에서 `probability=False`와 OVR `decision_function`을 일관되게 사용합니다.
- Raw SVM margin에는 AUROC/AUPRC만 적용하고 Brier score와 log loss를 확률 지표로 오용하지 않습니다.
- 선정된 SVM은 Track 2·Track 3 최종 학습에서 환자 단위 OOF calibration을 거친 뒤 확률 지표를 계산합니다.
- EXP1 XGBoost는 `multi:softprob + mlogloss + num_class=3`를 강제하고, Binary task는 `binary:logistic + logloss`를 강제합니다.
- XGBoost Early Stopping은 외부 CV validation fold가 아니라 외부 Train fold 내부의 Group-disjoint holdout에서 수행한 뒤 전체 외부 Train fold로 재학습합니다.
- Optuna Study 이름에 `exp123_hpo_v7_3`를 포함해 기존 실패 Study와 분리하고, FAIL Trial은 탐색 예산을 차감하지 않습니다.
- 모델 인터페이스 오류와 모델 설정 오류가 단순 완료 파일 누락으로 잘못 분류되지 않도록 실패 진단 우선순위를 수정했습니다.
- 실제 Track 2 실패 Run의 25개 완료 작업을 재사용하는 checksum-locked 복구 경로 `resume_v73_hpo_repair.sh`를 제공합니다.
- 최종 논문 및 Confirmatory 결과에는 새 Release freeze로 시작한 Clean V7.3 three-track Run을 사용합니다.

세부 설계와 복구 절차는 `V73_HPO_PROTOCOL_README.md`를 참고하십시오.

## V7.2.2 Multiprocess Tracking Hotfix

- 병렬 Worker가 동일한 `common/tracking/summary.json.tmp`를 공유하던 경쟁 조건을 제거했습니다.
- 모든 JSON 원자적 쓰기는 대상 디렉터리 안의 고유 임시 파일을 사용한 뒤 `os.replace`로 게시합니다.
- Tracking summary의 `history append → summary read → merge → atomic publish` 전체 구간을 프로세스 간 파일 잠금으로 보호합니다.
- 로컬 Tracking/W&B 보조 기록 오류는 기본적으로 모델 학습을 중단하지 않고 `common/progress/tracking_errors.jsonl`에 기록합니다. 엄격한 차단이 필요하면 `tracking.fail_open=false`로 설정할 수 있습니다.
- 기존 V7.2.1 Run에서 유효한 28개 완료 작업을 유지하고 미완료 작업만 다시 실행하는 기록형 복구 스크립트 `resume_v722_tracking_repair.sh`를 제공합니다.
- 코호트, Outcome, Split, Feature, 모델, Hyperparameter, Calibration 및 평가 정의는 V7.2.1과 동일합니다.

## V7.2.1 CLI Hotfix

- `run_exp123.py`의 `status` 분기 안에 있던 지역 `import json`을 제거했습니다.
- Python이 `main()` 전체의 `json`을 지역 변수로 오인해 `outcomes` 분기에서 발생하던 `UnboundLocalError`를 수정했습니다.
- Outcome 감사 산출물이 정상 생성된 뒤 CLI가 Manifest를 읽고 다음 단계로 넘어가는 실제 실행 경로를 회귀 테스트합니다.
- 데이터 계약, 코호트, Outcome 정의와 세 Track 실험 프로토콜은 V7.2와 동일합니다.

## V7.2 핵심 변경

- 동일 `stay_id`의 충돌 중복을 즉시 차단하고, 완전히 동일한 중복만 감사 후 제거합니다.
- 환자별 최초 ICU stay 한 건만 분석합니다.
- LOS와 28일 Outcome의 원점을 `icu_intime`으로 고정합니다.
- 첫 24시간 Feature 사용에 맞춰 24시간 Landmark까지 재원·생존한 환자만 포함합니다.
- eICU 사망 플래그가 0/1 두 클래스를 갖지 않으면 외부 검증을 차단합니다.
- Track 1·2는 자연 분포와 무가중 학습을 사용합니다.
- Track 3은 소수 클래스 전부를 보존하고 Train fold의 다수 클래스만 줄입니다.
- 모든 HPO 후보가 동일한 환자 단위 고정 CV fold를 사용합니다.
- Coarse/Fine Random Search를 Optuna Multivariate TPE로 교체했습니다.
- 모든 모델의 Baseline 설정을 HPO 후보에 포함하고, 1-SE 규칙으로 안정적인 후보를 선택합니다.
- XGBoost는 CV fold별 early stopping을 사용합니다.
- SVM HPO는 `probability=False`와 decision score를 사용합니다.
- Logistic Regression은 sklearn SAGA 기반 `C`·`l1_ratio` 정규화를 최적화합니다.
- 최종 모델은 환자 단위 OOF sigmoid calibration을 공통 적용합니다.
- Terminal과 Web에서 Trial·Fold·작업·자원·Artifact 진행 상황을 확인합니다.
- 로컬 Observatory와 선택적 W&B offline/online 기록을 동시에 지원합니다.


## V7.2 Outcome 감사 및 실행 제어 수정

V7.2는 코호트 선정으로 계획적으로 제외된 행과 실제 데이터 품질 오류로 제외된 행을 완전히 분리합니다. 환자별 최초 ICU stay 선택, exact duplicate 제거, 24시간 Landmark 이전 Outcome 제외는 연구 코호트 정의에 따른 `cohort_eligibility` 제외이며 `max_excluded_rows` 또는 `max_excluded_fraction` 차단 한도에 포함되지 않습니다. 음수 LOS 등 명시적으로 허용된 오류 행만 `data_quality` 제외 한도에 포함됩니다.

MIMIC과 eICU의 준비 상태도 분리됩니다.

```text
MIMIC ready + eICU blocked
→ Track 1/2/3 MIMIC Discovery 계속 실행
→ eICU external confirmatory만 on_hold
```

Audit가 차단되더라도 LOS와 mortality 값이 구조적으로 유효하면 A/B/C 분포와 assignment 파일은 진단 목적으로 생성됩니다. Confirmatory 직전 eICU guard는 원본 CSV 전체가 아니라 환자별 최초 stay와 Landmark가 적용된 `eicu_outcome_assignments.csv.gz`를 우선 검사합니다.

Outcome 산출물은 임시 디렉터리에서 완성된 뒤 원자적으로 게시됩니다. 완성된 Manifest가 존재하는 동일 Step 재호출은 안전하게 재사용하고, 부분 폴더만 남은 경우에는 명시적인 `--overwrite` 없이는 진행하지 않습니다. Scheduler는 OOM, exit 137 등 자원 관련 실패만 재시도하며 일반 Python·데이터·설정 오류는 최초 원인을 보존한 채 즉시 실패합니다.

## V7.2.1 실패 Run 복구

V7.2.1에서 `summary.json.tmp` 충돌로 일부 Baseline 작업만 실패한 경우, 기존 Release freeze와 완료 Artifact를 보존한 프로젝트 디렉터리에 V7.2.2 파일을 덮어쓴 뒤 다음을 실행합니다.

```bash
bash resume_v722_tracking_repair.sh \
  --run-id exp123_v7_cuda_track1_baseline_YYYY_MMDD_HHMMSS \
  --config configs/experiment_cuda_v7.json
```

이 경로는 `common/amendments/v7_2_2_tracking_concurrency_hotfix.json`을 생성하여 소스 수정 범위와 해시를 기록합니다. 완료 Lock이 유효한 작업은 재사용하고, 미완성 모델 디렉터리는 제거 후 다시 학습합니다. 가장 엄격한 재현성이 필요하면 새 Release freeze와 새 Run ID로 전체 실행을 시작하십시오.

## 설치

```bash
conda activate medical_26
python -m pip install -r requirements.txt
python -m pip check
```

CUDA PyTorch wheel은 시스템 드라이버와 CUDA 환경에 맞게 설치하십시오.

## 데이터 경로

다음 설정 파일 중 하나의 `paths`를 수정합니다.

| 환경 | 설정 파일 |
|---|---|
| NVIDIA CUDA | `configs/experiment_cuda_v7.json` |
| Apple Silicon MPS | `configs/experiment_mps_v7.json` |
| CPU | `configs/experiment_v7.json` |

필수 컬럼에는 다음이 포함됩니다.

```text
subject_id, hadm_id, stay_id, icu_intime, dischtime,
hospital_expire_flag, 19개 Feature
```

## eICU CSV-only external validation

V7.2에서는 eICU 외부 검증에 Parquet 파일이나 Harmonizer audit 파일이 필요하지 않습니다. 필수 입력은 `paths.eicu_csv` 하나입니다. 이 CSV에서 직접 식별자, 중복, 시간축, 24시간 Landmark, ICU 기준 LOS, 사망 Flag, A/B/C Outcome, Feature 호환성을 검사합니다.

Harmonizer 값 출처 분석이 필요한 경우에만 CSV/CSV.GZ provenance를 선택적으로 활성화합니다.

```json
{
  "audit": {
    "eicu_harmonizer_provenance": {
      "enabled": false,
      "path": null,
      "required": false,
      "auto_discovery": false,
      "imputed_feature_denominator": 18
    }
  }
}
```

기본 설정에서는 provenance 파일을 검색하거나 읽지 않으며, `pyarrow`도 설치할 필요가 없습니다.

## 전체 Discovery 실행

### CUDA

```bash
bash run_exp123_v7_cuda.sh --refreeze --start-dashboard
```

### Apple MPS

```bash
bash run_exp123_v7_mps.sh --refreeze --start-dashboard
```

### CPU

```bash
bash run_exp123_v7_cpu.sh --refreeze --start-dashboard
```

실행 순서:

```text
Release freeze / Preflight
→ Track 1 Baseline
→ Track 2 Optuna HPO 및 최종 학습
→ Track 3 fold-wise undersampling
→ 세 Track Validation 비교 HTML/CSV
```

Internal Test와 eICU confirmatory 평가는 자동으로 시작하지 않습니다.

## 로컬 Experiment Observatory

별도 터미널에서 다음을 실행할 수 있습니다.

```bash
bash launch_exp123_observatory.sh configs/experiment_cuda_v7.json
```

브라우저:

```text
http://127.0.0.1:8080
```

표시 항목:

- 현재·과거 Run과 Track
- 전체 및 계층형 진행률
- 활성 Trial과 CV Fold
- Metric history
- 모델 비교표
- Optuna Trial 파라미터와 점수
- CPU·RAM·GPU 상태
- 로그와 오류
- Artifact lineage
- 실행 Config

원격 서버에서는 SSH tunnel을 권장합니다.

```bash
ssh -L 8080:127.0.0.1:8080 USER@SERVER
```

## 선택적 W&B 기록

로컬 Observatory는 기본적으로 외부 서비스 없이 작동합니다. W&B 기록도 함께 남기려면 설정 파일에서 다음을 변경합니다.

```json
{
  "tracking": {
    "wandb": {
      "enabled": true,
      "mode": "offline",
      "project": "exp123-clinical-v7"
    }
  }
}
```

Offline 기록을 나중에 동기화할 때는 W&B CLI의 sync 기능을 사용할 수 있습니다. 온라인 전송을 원할 때만 `mode`를 `online`으로 변경하고 로그인하십시오.

## 데이터 감사 및 차단 조건

MIMIC Discovery를 차단하는 조건:

- 같은 `stay_id` 그룹에 서로 다른 값이 존재
- 최초 ICU stay 선택 후 `subject_id` 또는 `stay_id`가 고유하지 않음
- ICU 입·퇴실 시간 파싱 실패
- 모델링 대상 행에 LOS 또는 mortality 결측·비정상 값이 남음
- 데이터 품질 제외 건수 또는 비율이 동결된 한도를 초과
- MIMIC 사망 Endpoint가 0/1 두 클래스를 포함하지 않거나 최소 건수·비율 미달
- Dataset, Outcome, Split 또는 CV hash 불일치

다음은 계획된 코호트 제외이므로 건수가 많아도 그 자체로 데이터 오류가 아닙니다.

- exact duplicate stay의 중복 행 제거
- 환자별 최초 ICU stay 이후의 stay 제외
- 24시간 Landmark 이전 퇴원·사망 제외

각 제외 사유는 mortality 값과 함께 별도 CSV로 기록됩니다. eICU 사망 Endpoint가 기준을 통과하지 못하면 MIMIC Discovery는 계속되고, eICU 외부 Confirmatory만 `on_hold`가 됩니다.

## Track별 차이

| Track | Hyperparameter | Class weight | Train sampling | 평가 분포 |
|---|---|---|---|---|
| Track 1 | 기본값 | 없음 | 없음 | 자연 분포 |
| Track 2 | Optuna TPE | 없음 | 없음 | 자연 분포 |
| Track 3 | Track 2에서 상속 | 없음 | 다수 클래스만 축소 | 자연 분포 |

Track 3은 Validation, Internal Test, eICU를 절대 undersampling하지 않습니다.

## 주요 결과

```text
series/SERIES_ID/
├── series_manifest.json
└── track_comparison/
    ├── track_model_metrics.csv
    ├── track_model_metrics.json
    └── track_comparison.html

runs/RUN_ID/
├── common/
│   ├── outcomes/
│   ├── splits/
│   ├── fixed_cv/
│   ├── progress/
│   └── tracking/
├── optimization/
├── EXP1_flat_3class/
├── EXP2_two_stage/
└── TRACK3_undersampled/
```

## 검증

```bash
PYTHONPATH=src pytest -q
python -m compileall -q .
```

상세 문서:

- `docs/guides/EXP123_V7_THREE_TRACK_GUIDE.md`
- `docs/guides/EXP123_V7_OBSERVATORY_AND_WANDB.md`
- `docs/design/EXP123_V7_COHORT_AND_TEMPORAL_CONTRACT.md`
- `docs/changelog/EXP123_V7_CHANGELOG.md`
- `docs/verification/EXP123_V7_IMPLEMENTATION_VERIFICATION.md`

## Outcome 감사 결과 확인

실행 중 또는 완료 후 다음 명령으로 MIMIC/eICU 상태와 제외 사유를 요약할 수 있습니다.

```bash
python diagnose_exp123_outcomes.py \
  runs/<RUN_ID>
```

이번 V7.2 수정 사항만 빠르게 재검증하려면 다음을 실행합니다.

```bash
bash verify_v72_outcome_fix.sh
```

## EXP123 V8 Phase 1 checkpoint

V8 introduces a new experiment-matrix namespace without reinterpreting frozen V7.3 artifacts.
The V8 strategy order is `T1_baseline -> T2_optuna -> T3_smote`, with
`T4_undersampling` optional. The default matrix therefore contains six core runs
(LOS28 x T1/T2/T3); `--include-undersampling` expands it to four planned runs.

Phase 1 is intentionally planning-only: `run_exp123_v8_matrix.sh` creates and validates
the matrix but does not launch T3/T4 training until the Phase 2 sampling engine is implemented.
See `V8_PHASE1_SCOPE.md`.

## EXP123 V8 Phase 2 checkpoint

Phase 2 makes sampling a first-class training strategy while preserving all
natural discovery/evaluation splits. The V8 order is now:

| Strategy | Hyperparameters | Model-fit distribution | Evaluation distribution | Default |
|---|---|---|---|---|
| T1 Baseline | defaults | natural | natural | required |
| T2 Optuna | TPE selected | natural | natural | required |
| T3 SMOTE | inherits T2 | Class C x2, train only | natural | required |
| T4 Undersampling | inherits T2 | majority reduced, train only | natural | optional |

T3 final synthetic pools are created once per seed and shared by EXP1 and
Stage2/EXP3. Stage1 remains the natural T2 model. Synthetic rows never enter
`TaskSplitData`, validation, internal test, or external eICU splits.

Run sampled tracks after a completed V8 T2 run:

```bash
bash run_exp123_v8_sampling_tracks.sh \
  --config configs/experiment_cuda_v8_los28.json \
  --optimized-run-id <LOS28_T2_RUN_ID> \
  --t3-run-id <LOS28_T3_RUN_ID>
```

Include optional T4 only when it was prespecified before confirmatory access:

```bash
bash run_exp123_v8_sampling_tracks.sh \
  --config configs/experiment_cuda_v8_los28.json \
  --optimized-run-id <LOS28_T2_RUN_ID> \
  --t3-run-id <LOS28_T3_RUN_ID> \
  --include-undersampling \
  --t4-run-id <LOS28_T4_RUN_ID>
```

Phase 2 deliberately leaves the generic T3/T4 EXP2 combined evaluator,
calibration slope/intercept, threshold-policy expansion and DCA for the next
checkpoint. See `V8_PHASE2_SCOPE.md`.

## EXP123 V8 Phase 3

Phase 3 completes the Discovery reporting layer for the V8 matrix. New entry
points are:

- `run_exp123_v8_reporting.py` — V8 reporting overlay for natural T1/T2 runs;
- `run_exp123_v8_horizon.sh` — one LOS horizon T1→T2→T3→optional T4;
- `./exp123 run` — LOS28-only T1→T2→T3 matrix with hardware auto-tuning; add `--include-undersampling` for optional T4;
- `compare_exp123_v8_matrix.py` — common Matrix comparison;
- `verify_v8_phase3_reporting.sh` — Phase 3 regression verification.

`run_exp123_v8_matrix.sh` remains **plan-only by default**. Long-running model
execution requires the explicit `--execute` flag. Optional T4 undersampling
still requires `--include-undersampling` before confirmatory access.

See `V8_PHASE3_SCOPE.md` for the exact reporting and clinical-evaluation
contract.

## EXP123 V8 Phase 4-A — APS III / feature-set ablation

Phase 4-A adds a prespecified LOS28 + T1 Baseline ablation in which the **only**
varied axis is the feature set. The target registry is `configs/feature_sets.json`:

- `FS_FULL` — 57 reference-study variables;
- `FS_NOAPS` — 56 variables (`FS_FULL` minus `apsiii`);
- `FS_APSONLY` — APS III only.

Historical `FS_LEGACY*` entries exist only for backward-compatible smoke tests
and must not be used as a silent replacement for the target 57-feature study.
The runner performs CSV-header readiness checks before any execution, records a
feature-set SHA-256 in configs/preflight/model artifacts, and requires identical
patient split hashes across all feature sets.

Plan only:

```bash
bash run_exp123_v8_ablation.sh \
  --base-config configs/experiment_cuda_v8_los28.json
```

Execute after the target feature schema is ready:

```bash
bash run_exp123_v8_ablation.sh \
  --base-config configs/experiment_cuda_v8_los28.json \
  --execute
```

Interrupted series can be continued with `--series-id ... --resume --execute`.
Validation-only EXP1 statistics use paired patient-cluster bootstrap plus
class-specific DeLong with FDR correction. Confirmatory splits remain locked.

See `V8_PHASE4A_SCOPE.md`.

## EXP123 V8 Phase 4-B — SVM runtime optimization

Phase 4-B keeps `svm` as one model family but lets Optuna compare two explicitly
recorded implementations:

- `exact_rbf` — exact sklearn RBF `SVC`, with narrowed `C/gamma`, finite
  `max_iter`, and no hidden `probability=True` calibration;
- `nystroem_linear` — deterministic RBF Nystroem mapping followed by
  `LinearSVC`.

Both HPO candidates use raw `decision_function` scores for AUROC/AUPRC and the
selected final model receives the existing patient-group OOF sigmoid
calibration. A completed HPO fold that exceeds the configured SVM runtime
budget prunes that trial before later folds are run. The resource scheduler also
limits concurrent SVM jobs to two by default.

A train-only fixed-CV benchmark can be planned with:

```bash
python benchmark_exp123_v8_svm.py \
  --config configs/experiment_cuda_v8_los28.json \
  --source-run-id <T1_OR_T2_RUN_ID> \
  --task exp1
```

Add `--execute` to run the benchmark. It never accesses project validation,
internal test, or eICU.

See `V8_PHASE4B_SCOPE.md`.


## V8 Phase 5 — Confirmatory & External Validation

Phase 5 adds a validation-only design freeze, single-use Internal Test/eICU access locks, immutable fitted-artifact verification, frozen-threshold confirmatory evaluation, and 20%/80% ×20 patient-group site recalibration. Raw eICU transfer remains the primary external result; recalibration is reported separately as local adaptation. See `V8_PHASE5_SCOPE.md`.

Typical lifecycle:

```bash
python prepare_exp123_v8_confirmatory.py --matrix-manifest <matrix_execution.json>
python run_exp123_v8_confirmatory.py --design-freeze <confirmatory_design_freeze.json> --split internal_test --confirm-open-internal-test
python run_exp123_v8_confirmatory.py --design-freeze <confirmatory_design_freeze.json> --split external_eicu --confirm-open-eicu
python compare_exp123_v8_confirmatory.py --design-freeze <confirmatory_design_freeze.json>
```

Do not open eICU until the external mortality/data guard passes. Do not delete/recreate confirmatory locks to rerun a completed split.

---

## V9 Alpha 1: Study28 frozen dataset contract

V9 Alpha 1 preserves the V8 execution framework but adds a separate scientific input contract for the professor-provided Sepsis-3 28-day final datasets. No new manuscript-grade model training is enabled by this milestone yet.

Run the dataset gate before further development:

```bash
PYTHONPATH=src python tools/audit_study28.py \
  --dataset-root /path/to/final_dataset
```

The command must pass for MIMIC, eICU-alt, eICU-primary, SICdb, and NWICU. Source CSVs are never modified. See `docs/study28/RESEARCH_CONTRACT_V9.md` and `docs/study28/INITIAL_DATASET_AUDIT.md`.

## One-command Study28 full pipeline

Place the frozen professor-provided final datasets in one directory:

```text
/path/to/final_dataset/
├── feature_list.csv
├── mimic_final.csv
├── eicu_final.csv
├── sicdb_final.csv
└── nwicu_final.csv
```

Install dependencies once:

```bash
python -m pip install -r requirements.txt
```

Then run the entire Study28 research pipeline from the project root:

```bash
bash run_study28_full_pipeline.sh \
  --dataset-root /path/to/final_dataset \
  --run-root runs/study28_full
```

This executes, in order: frozen-data audit; Alpha 2 development/nested CV/temporal evaluation/model freeze; Alpha 3 mandatory raw validation on eICU-alt, eICU-primary, SICdb, and NWICU; Alpha 4 sensitivity/clinical utility; Alpha 5 robustness/SHAP/HFD28; and Alpha 6 manuscript evidence/reporting. The run finishes successfully only if the Alpha 6 finalization checklist passes and all four mandatory external cohorts are present.

To resume after interruption, run the exact same command with the same `--run-root`; verified checkpoints are reused where supported.

---

## Study28 EXP1/EXP2/EXP3 — baseline first, optimization second

The original multi-model EXP123 structure is now connected to the professor-provided Study28 final datasets and all mandatory external cohorts without changing the frozen `study28_full_main` reference analysis.

Fast meeting-ready baseline:

```bash
./exp123 study28-baseline \
  --dataset-root /path/to/final_dataset \
  --run-root runs/study28_exp123_main
```

Then continue the longer HPO/downsampling optimization:

```bash
./exp123 study28-optimize \
  --dataset-root /path/to/final_dataset \
  --run-root runs/study28_exp123_main
```

Or run both sequentially; baseline files are emitted first and preserved while optimization continues:

```bash
./exp123 study28-all \
  --dataset-root /path/to/final_dataset \
  --run-root runs/study28_exp123_main
```

Status:

```bash
./exp123 study28-status --run-root runs/study28_exp123_main
```

See `docs/STUDY28_EXP123_TWO_STAGE.md` for the exact EXP1/EXP2/EXP3 definitions, model matrix, external-validation rules, and output layout.
