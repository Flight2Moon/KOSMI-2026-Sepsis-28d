from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from los_baseline.experiment_scheduler import ResourceAwareScheduler, ScheduledJob
from los_baseline.outcomes import audit_outcome_source
from los_baseline.resource_manager import ResourceRequest
from run_exp123 import run_outcome_step


def _cohort_contract() -> dict:
    return {
        "enabled": True,
        "analysis_unit": "first_icu_stay_per_patient",
        "subject_id_column": "subject_id",
        "admission_id_column": "hadm_id",
        "stay_id_column": "stay_id",
        "icu_intime_column": "icu_intime",
        "discharge_time_column": "dischtime",
        "mortality_column": "hospital_expire_flag",
        "patient_selection_order": ["icu_intime", "hadm_id", "stay_id"],
        "strict_unique_stay": True,
        "strict_unique_subject": True,
        "deduplicate_exact_stays": True,
        "prediction_landmark_hours": 24,
        "require_alive_and_admitted_at_landmark": True,
        "allow_missing_identifiers": False,
    }


def _los_derivation() -> dict:
    return {
        "enabled": True,
        "datasets": ["mimic", "eicu"],
        "admission_time_column": "icu_intime",
        "discharge_time_column": "dischtime",
        "in_hospital_death_time_column": "dischtime",
        "overwrite_existing_los": True,
        "allow_existing_los_fallback": False,
        "discrepancy_tolerance_days": 0.05,
        "fail_on_discrepancy": False,
        "round_decimals": 6,
    }


def _write_dataset(path: Path, *, include_death: bool) -> None:
    rows = [
        # Subject 1: retained first stay, later stay excluded by cohort contract.
        {
            "subject_id": 1,
            "hadm_id": 11,
            "stay_id": 111,
            "icu_intime": "2026-01-01 00:00:00",
            "dischtime": "2026-01-06 00:00:00",
            "hospital_expire_flag": 0,
            "LOS": 999,
        },
        {
            "subject_id": 1,
            "hadm_id": 12,
            "stay_id": 112,
            "icu_intime": "2026-02-01 00:00:00",
            "dischtime": "2026-03-10 00:00:00",
            "hospital_expire_flag": 0,
            "LOS": 999,
        },
        # Subject 2: death by day 28 when requested.
        {
            "subject_id": 2,
            "hadm_id": 21,
            "stay_id": 211,
            "icu_intime": "2026-01-01 00:00:00",
            "dischtime": "2026-01-11 00:00:00",
            "hospital_expire_flag": 1 if include_death else 0,
            "LOS": 999,
        },
        # Subject 3: still admitted at day 28.
        {
            "subject_id": 3,
            "hadm_id": 31,
            "stay_id": 311,
            "icu_intime": "2026-01-01 00:00:00",
            "dischtime": "2026-02-10 00:00:00",
            "hospital_expire_flag": 0,
            "LOS": 999,
        },
        # Subject 4: planned exclusion before 24-hour landmark.
        {
            "subject_id": 4,
            "hadm_id": 41,
            "stay_id": 411,
            "icu_intime": "2026-01-01 00:00:00",
            "dischtime": "2026-01-01 12:00:00",
            "hospital_expire_flag": 0,
            "LOS": 999,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_planned_cohort_exclusions_do_not_trigger_quality_limits(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mimic.csv"
    _write_dataset(path, include_death=True)
    result = audit_outcome_source(
        dataset_name="mimic",
        csv_path=path,
        mortality_column="hospital_expire_flag",
        los_column="LOS",
        landmark_days=28,
        candidate_identifier_columns=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "icu_intime",
            "dischtime",
        ],
        exclusion_policy={
            "enabled": True,
            "datasets": ["mimic"],
            "rules": {"negative_los": True},
            "max_excluded_rows": 0,
            "max_excluded_fraction": 0.0,
            "fail_if_limit_exceeded": True,
        },
        los_derivation_config=_los_derivation(),
        cohort_contract_config=_cohort_contract(),
        mortality_guard_config={"enabled": False},
    )
    audit, assignments = result
    assert audit.status == "ready"
    assert audit.cohort_excluded_row_count == 2
    assert audit.quality_excluded_row_count == 0
    assert audit.retained_row_count == 3
    assert audit.cohort_exclusion_reason_counts == {
        "later_icu_stay_for_patient": 1,
        "outcome_before_prediction_landmark": 1,
    }
    assert assignments is not None
    assert set(assignments["three_class_outcome"]) == {0, 1, 2}


def test_endpoint_block_still_writes_diagnostic_labels(tmp_path: Path) -> None:
    path = tmp_path / "eicu.csv"
    _write_dataset(path, include_death=False)
    result = audit_outcome_source(
        dataset_name="eicu",
        csv_path=path,
        mortality_column="hospital_expire_flag",
        los_column="LOS",
        landmark_days=28,
        candidate_identifier_columns=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "icu_intime",
            "dischtime",
        ],
        exclusion_policy={
            "enabled": True,
            "datasets": ["eicu"],
            "rules": {"negative_los": True},
            "max_excluded_rows": 0,
            "max_excluded_fraction": 0.0,
            "fail_if_limit_exceeded": True,
        },
        los_derivation_config=_los_derivation(),
        cohort_contract_config=_cohort_contract(),
        mortality_guard_config={
            "enabled": True,
            "minimum_death_count": 1,
            "minimum_death_rate": 0.001,
            "maximum_death_rate": 0.5,
            "fail_on_missing": True,
        },
    )
    audit, assignments = result
    assert audit.status == "blocked"
    assert audit.label_derivation_status == "ready"
    assert assignments is not None
    assert sum(audit.class_counts.values()) == audit.retained_row_count
    assert audit.class_counts["B_died"] == 0


def test_both_dataset_step_allows_mimic_discovery_and_holds_eicu(
    tmp_path: Path,
) -> None:
    mimic = tmp_path / "mimic.csv"
    eicu = tmp_path / "eicu.csv"
    _write_dataset(mimic, include_death=True)
    _write_dataset(eicu, include_death=False)
    config = {
        "paths": {
            "mimic_csv": str(mimic),
            "eicu_csv": str(eicu),
            "runs_dir": str(tmp_path / "runs"),
        },
        "data": {
            "target_column": "LOS",
            "identifier_columns": [
                "subject_id",
                "hadm_id",
                "stay_id",
                "icu_intime",
                "dischtime",
            ],
        },
        "outcomes": {
            "mortality_column": "hospital_expire_flag",
            "los_column": "LOS",
            "landmark_days": 28,
            "los_derivation": _los_derivation(),
        },
        "cohort_contract": _cohort_contract(),
        "data_quality": {
            "outcome_row_exclusion": {
                "enabled": True,
                "datasets": ["mimic", "eicu"],
                "rules": {"negative_los": True},
                "max_excluded_rows": 0,
                "max_excluded_fraction": 0.0,
                "fail_if_limit_exceeded": True,
            },
            "internal_mortality_guard": {
                "enabled": True,
                "minimum_death_count": 1,
                "minimum_death_rate": 0.001,
                "maximum_death_rate": 0.9,
                "fail_on_missing": True,
            },
        },
        "external_validation": {
            "mortality_guard": {
                "enabled": True,
                "minimum_death_count": 1,
                "minimum_death_rate": 0.001,
                "maximum_death_rate": 0.9,
                "fail_on_missing": True,
            }
        },
        "development": {"model_training_enabled": True},
    }
    output_dir, issue_count = run_outcome_step(
        config=config,
        run_id="test-run",
        dataset_selection="both",
        overwrite=False,
    )
    manifest = json.loads(
        (output_dir / "outcome_step_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert issue_count == 0
    assert manifest["status"] == "ready"
    assert manifest["discovery_status"] == "ready"
    assert manifest["external_validation_status"] == "on_hold"
    assert manifest["discovery_training_enabled"] is True
    assert manifest["eicu_confirmatory_enabled"] is False
    assert (output_dir / "mimic_outcome_assignments.csv.gz").is_file()
    assert (output_dir / "eicu_outcome_assignments.csv.gz").is_file()
    assert (output_dir / "DISCOVERY_OUTCOMES_READY.lock.json").is_file()
    assert (output_dir / "EXTERNAL_VALIDATION_ON_HOLD.lock.json").is_file()

    # Re-running an already complete step is idempotent and does not overwrite.
    reused_dir, reused_issues = run_outcome_step(
        config=config,
        run_id="test-run",
        dataset_selection="both",
        overwrite=False,
    )
    assert reused_dir == output_dir
    assert reused_issues == 0


def _scheduler_config() -> dict:
    return {
        "hardware": {
            "accelerator": {"backend": "cpu"},
            "cpu": {"threads_per_worker": 1},
            "cuda": {"mode": "disabled", "gpu_ids": []},
        },
        "resources": {
            "accelerator": {"backend": "cpu"},
            "scheduler": {
                "enabled": True,
                "max_concurrent_jobs": 1,
                "poll_interval_seconds": 0.02,
            },
            "cpu": {
                "total_thread_limit": 1,
                "reserve_threads": 0,
                "default_threads_per_job": 1,
                "max_threads_per_job": 1,
            },
            "memory": {
                "total_limit_gb": 4,
                "max_fraction": 1.0,
                "reserve_gb": 0,
                "minimum_free_gb_to_launch": 0,
                "breach_grace_seconds": 1,
            },
            "cuda": {
                "gpu_ids": [],
                "max_jobs_per_gpu": 1,
                "minimum_free_memory_gb": 0,
                "reserve_memory_gb": 0,
                "pytorch_memory_fraction": 0.5,
            },
            "mps": {
                "enabled": False,
                "max_concurrent_jobs": 1,
                "minimum_free_unified_memory_gb": 0,
                "pytorch_memory_fraction": 0.5,
            },
            "job_resources": {},
            "resource_failure_policy": {
                "auto_retry": True,
                "max_retries": 3,
            },
        },
    }


def test_python_exception_is_not_retried(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_dir = project_root / "runs" / "run"
    project_root.mkdir(parents=True)
    result = ResourceAwareScheduler(
        jobs=[
            ScheduledJob(
                name="python-failure",
                command=(
                    sys.executable,
                    "-c",
                    "raise RuntimeError('ordinary deterministic failure')",
                ),
                request=ResourceRequest(cpu_threads=1, ram_gb=0.1),
            )
        ],
        config=_scheduler_config(),
        project_root=project_root,
        run_dir=run_dir,
        run_id="run",
    ).run()
    assert result["counts"]["direct_failed"] == 1
    state = json.loads(
        (run_dir / "common" / "progress" / "scheduler_state.json")
        .read_text(encoding="utf-8")
    )
    assert state["attempts"]["python-failure"] == 1


def test_external_guard_uses_eligible_outcome_cohort(tmp_path: Path) -> None:
    from los_baseline.external_validation_guard import (
        assert_external_mortality_ready,
    )

    raw_eicu = tmp_path / "raw_eicu.csv"
    pd.DataFrame(
        {
            "hospital_expire_flag": [0, 0, 1, 1],
        }
    ).to_csv(raw_eicu, index=False)
    run_dir = tmp_path / "runs" / "run"
    outcomes_dir = run_dir / "common" / "outcomes"
    outcomes_dir.mkdir(parents=True)
    # The modelling-eligible cohort has no deaths even though the raw file does.
    pd.DataFrame(
        {
            "hospital_expire_flag": [0, 0, 0],
        }
    ).to_csv(
        outcomes_dir / "eicu_outcome_assignments.csv.gz",
        index=False,
        compression="gzip",
    )
    config = {
        "paths": {"eicu_csv": str(raw_eicu)},
        "outcomes": {"mortality_column": "hospital_expire_flag"},
        "external_validation": {
            "mortality_guard": {
                "enabled": True,
                "block_confirmatory_on_failure": True,
                "minimum_death_count": 1,
                "minimum_death_rate": 0.001,
                "maximum_death_rate": 0.9,
                "fail_on_missing": True,
                "chunksize": 100,
            }
        },
    }
    try:
        assert_external_mortality_ready(config=config, run_dir=run_dir)
    except RuntimeError as exc:
        assert "death_count_below_minimum" in str(exc)
    else:
        raise AssertionError("Eligible eICU cohort with zero deaths must block.")
    payload = json.loads(
        (run_dir / "common" / "data_quality" / "eicu_mortality_flag_audit.json")
        .read_text(encoding="utf-8")
    )
    assert payload["source_scope"] == "eligible_eicu_outcome_cohort"
    assert payload["one_count"] == 0


def _write_balanced_discovery_dataset(
    path: Path,
    *,
    include_deaths: bool,
    subject_offset: int,
) -> None:
    rows: list[dict] = []
    base = pd.Timestamp("2026-01-01 00:00:00")
    subject_id = subject_offset
    for class_code in (0, 1, 2):
        for index in range(20):
            subject_id += 1
            if class_code == 0:
                los_days = 5 + (index % 10)
                mortality = 0
            elif class_code == 1:
                los_days = 3 + (index % 15)
                mortality = 1 if include_deaths else 0
            else:
                los_days = 35 + (index % 10)
                mortality = 0
            intime = base + pd.to_timedelta(subject_id, unit="h")
            rows.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": 100000 + subject_id,
                    "stay_id": 200000 + subject_id,
                    "icu_intime": intime,
                    "dischtime": intime + pd.to_timedelta(los_days, unit="D"),
                    "hospital_expire_flag": mortality,
                    "LOS": 999,
                }
            )
    # Planned cohort exclusions: later stays and stays ending before 24 hours.
    for original in rows[:5]:
        later = dict(original)
        later["hadm_id"] += 500000
        later["stay_id"] += 500000
        later["icu_intime"] = pd.Timestamp(later["icu_intime"]) + pd.Timedelta(days=100)
        later["dischtime"] = pd.Timestamp(later["icu_intime"]) + pd.Timedelta(days=8)
        rows.append(later)
    for index in range(5):
        subject_id += 1
        intime = base + pd.to_timedelta(subject_id, unit="h")
        rows.append(
            {
                "subject_id": subject_id,
                "hadm_id": 100000 + subject_id,
                "stay_id": 200000 + subject_id,
                "icu_intime": intime,
                "dischtime": intime + pd.Timedelta(hours=12),
                "hospital_expire_flag": 0,
                "LOS": 999,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_mimic_split_proceeds_when_eicu_is_on_hold(tmp_path: Path) -> None:
    from los_baseline.exp123_splits import run_common_split_step

    mimic = tmp_path / "mimic_balanced.csv"
    eicu = tmp_path / "eicu_no_deaths.csv"
    _write_balanced_discovery_dataset(
        mimic, include_deaths=True, subject_offset=0
    )
    _write_balanced_discovery_dataset(
        eicu, include_deaths=False, subject_offset=10000
    )
    config = {
        "_project_root": str(tmp_path),
        "paths": {
            "mimic_csv": str(mimic),
            "eicu_csv": str(eicu),
            "runs_dir": str(tmp_path / "runs"),
        },
        "data": {
            "target_column": "LOS",
            "patient_id_column": "subject_id",
            "identifier_columns": [
                "subject_id",
                "hadm_id",
                "stay_id",
                "icu_intime",
                "dischtime",
            ],
        },
        "split": {
            "train_ratio": 0.6,
            "validation_ratio": 0.2,
            "test_ratio": 0.2,
            "seed": 42,
            "max_attempts": 500,
        },
        "outcomes": {
            "mortality_column": "hospital_expire_flag",
            "los_column": "LOS",
            "landmark_days": 28,
            "los_derivation": _los_derivation(),
        },
        "cohort_contract": _cohort_contract(),
        "data_quality": {
            "outcome_row_exclusion": {
                "enabled": True,
                "datasets": ["mimic", "eicu"],
                "rules": {"negative_los": True},
                "max_excluded_rows": 0,
                "max_excluded_fraction": 0.0,
                "fail_if_limit_exceeded": True,
            },
            "internal_mortality_guard": {
                "enabled": True,
                "minimum_death_count": 1,
                "minimum_death_rate": 0.001,
                "maximum_death_rate": 0.9,
                "fail_on_missing": True,
            },
        },
        "external_validation": {
            "mortality_guard": {
                "enabled": True,
                "minimum_death_count": 1,
                "minimum_death_rate": 0.001,
                "maximum_death_rate": 0.9,
                "fail_on_missing": True,
            }
        },
        "development": {"model_training_enabled": True},
    }
    _, issue_count = run_outcome_step(
        config=config,
        run_id="split-run",
        dataset_selection="both",
        overwrite=False,
    )
    assert issue_count == 0
    result = run_common_split_step(
        config=config,
        run_id="split-run",
        overwrite=False,
    )
    assert result.integrity["all_classes_present_per_split"] is True
    manifest = json.loads(
        (
            tmp_path
            / "runs"
            / "split-run"
            / "common"
            / "outcomes"
            / "outcome_step_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["external_validation_status"] == "on_hold"


def test_resource_exit_code_is_retried(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_dir = project_root / "runs" / "run"
    project_root.mkdir(parents=True)
    config = _scheduler_config()
    config["resources"]["resource_failure_policy"]["max_retries"] = 2
    result = ResourceAwareScheduler(
        jobs=[
            ScheduledJob(
                name="resource-failure",
                command=(sys.executable, "-c", "raise SystemExit(137)"),
                request=ResourceRequest(cpu_threads=1, ram_gb=0.1),
            )
        ],
        config=config,
        project_root=project_root,
        run_dir=run_dir,
        run_id="run",
    ).run()
    assert result["counts"]["direct_failed"] == 1
    state = json.loads(
        (run_dir / "common" / "progress" / "scheduler_state.json")
        .read_text(encoding="utf-8")
    )
    assert state["attempts"]["resource-failure"] == 3
    assert state["resource_retry_levels"]["resource-failure"] == 2
