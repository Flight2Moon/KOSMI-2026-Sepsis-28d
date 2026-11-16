from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from los_baseline.cohort_contract import apply_cohort_contract, validate_mortality_endpoint
from los_baseline.exp123_data import TaskDataBundle, TaskSplitData
from los_baseline.hpo.fixed_cv import build_fixed_group_folds, load_fixed_folds, save_fixed_folds
from los_baseline.hpo.optuna_engine import run_optuna_optimization_study
from los_baseline.imbalance import majority_undersample_indices
from los_baseline.task_spec import STAGE1_MORTALITY_TASK


def _split(name: str, n: int, offset: int) -> TaskSplitData:
    y = np.tile([0, 0, 0, 1], n // 4)
    rng = np.random.default_rng(offset + 3)
    x = pd.DataFrame({"x1": y + rng.normal(0, 0.2, len(y)), "x2": rng.normal(size=len(y))})
    meta = pd.DataFrame({
        "source_row_index": np.arange(offset, offset + len(y)),
        "subject_id": [f"{name}_{i}" for i in range(len(y))],
        "split": name,
    })
    return TaskSplitData(name, x, y.astype(np.int64), meta)


def _bundle() -> TaskDataBundle:
    return TaskDataBundle(
        task=STAGE1_MORTALITY_TASK,
        feature_names=("x1", "x2"),
        splits={
            "train": _split("train", 80, 0),
            "validation": _split("validation", 32, 1000),
        },
        shared_split_hash="v7_test_hash",
    )


def test_cohort_contract_selects_first_stay_and_landmark() -> None:
    frame = pd.DataFrame({
        "_source_row_index": [0, 1, 2, 3],
        "subject_id": [1, 1, 2, 3],
        "hadm_id": [10, 11, 20, 30],
        "stay_id": [100, 101, 200, 300],
        "icu_intime": ["2026-01-01", "2026-02-01", "2026-01-01", "2026-01-01"],
        "dischtime": ["2026-01-10", "2026-02-10", "2026-01-01 12:00", "2026-01-05"],
        "hospital_expire_flag": [0, 0, 1, 0],
    })
    filtered, audit, excluded = apply_cohort_contract(
        frame,
        dataset_name="mimic",
        config={
            "enabled": True,
            "analysis_unit": "first_icu_stay_per_patient",
            "strict_unique_stay": True,
            "strict_unique_subject": True,
            "prediction_landmark_hours": 24,
            "require_alive_and_admitted_at_landmark": True,
        },
    )
    assert filtered["subject_id"].tolist() == [1, 3]
    assert filtered["subject_id"].is_unique
    assert filtered["stay_id"].is_unique
    assert audit.removed_later_patient_stays == 1
    assert audit.removed_before_landmark_rows == 1
    assert set(excluded["cohort_exclusion_reason"]) == {
        "later_icu_stay_for_patient", "outcome_before_prediction_landmark"
    }


def test_external_zero_deaths_is_blocked() -> None:
    issues = validate_mortality_endpoint(
        pd.DataFrame({"hospital_expire_flag": [0, 0, 0]}),
        mortality_column="hospital_expire_flag",
        config={"enabled": True, "minimum_death_count": 1, "minimum_death_rate": 0.001, "maximum_death_rate": 0.5},
    )
    assert any("both" in issue.lower() for issue in issues)
    assert any("death count" in issue.lower() for issue in issues)


def test_undersampling_preserves_every_minority_row() -> None:
    labels = np.array([0] * 20 + [1] * 5 + [2] * 8)
    plan = majority_undersample_indices(labels, ratio=2, seed=42)
    selected = labels[list(plan.selected_indices)]
    assert int((selected == 1).sum()) == 5
    assert int((selected == 2).sum()) == 8
    assert int((selected == 0).sum()) == 16
    assert plan.majority_class == 0


def test_fixed_cv_roundtrip_uses_same_hash(tmp_path: Path) -> None:
    bundle = _bundle()
    y = bundle.splits["train"].labels
    groups = bundle.splits["train"].metadata["subject_id"].to_numpy()
    folds = build_fixed_group_folds(labels=y, groups=groups, n_splits=2, seeds=[42, 1042])
    _, split_hash = save_fixed_folds(tmp_path, folds=folds, seeds=[42, 1042], n_splits=2, sample_count=len(y))
    loaded, manifest = load_fixed_folds(tmp_path)
    assert len(loaded) == 4
    assert manifest["split_hash"] == split_hash


def test_optuna_engine_writes_compatible_selection(tmp_path: Path) -> None:
    bundle = _bundle()
    config = {
        "data": {"patient_id_column": "subject_id"},
        "optimization_objectives": {
            "stage1_mortality": {
                "primary": "auprc", "direction": "maximize",
                "secondary": "brier_score", "secondary_direction": "minimize",
            }
        },
        "optimization_v2": {
            "engine": "optuna_tpe", "seed": 42, "cv_folds": 2,
            "finalist_repeat_seeds": [42, 1042], "startup_trials": 1,
            "multivariate": False, "group": False,
            "pruner_startup_trials": 10, "pruner_warmup_folds": 2,
            "finalist_top_k": 2, "stability_penalty": 0.5,
            "overfit_penalty": 0.5, "trial_budgets": {"decision_tree": 2},
        },
    }
    output = tmp_path / "run" / "optimization" / "stage1_mortality" / "decision_tree"
    stages, selection = run_optuna_optimization_study(
        task_data=bundle,
        model_name="decision_tree",
        base_model_config={"parameters": {"max_depth": 3}, "training_policy": {"class_weight_mode": "none"}},
        config=config,
        output_dir=output,
    )
    payload = json.loads((output / "selected_hyperparameters.json").read_text())
    assert payload["optimizer"].startswith("optuna")
    assert payload["confirmatory_splits_accessed"] is False
    assert payload["fixed_cv_hash"]
    assert selection.selected_model_config
    assert stages[0].to_dict()["completed_trials"] >= 1


def test_sklearn_logistic_oof_calibration_artifact_roundtrip(tmp_path: Path) -> None:
    from los_baseline.artifacts import load_task_artifact
    from los_baseline.training_engine import train_and_evaluate_single_model

    bundle = _bundle()
    output = tmp_path / "calibrated_lr"
    result = train_and_evaluate_single_model(
        task_data=bundle,
        model_name="logistic_regression",
        model_config={
            "implementation": "sklearn",
            "parameters": {
                "solver": "saga",
                "C": 1.0,
                "l1_ratio": 0.0,
                "max_iter": 2000,
                "tol": 1e-5,
            },
            "training_policy": {"class_weight_mode": "none"},
        },
        output_dir=output,
        seed=42,
        project_config={
            "data": {"patient_id_column": "subject_id"},
            "calibration": {
                "enabled": True,
                "method": "sigmoid",
                "cv_folds": 2,
                "seed": 123,
                "group_column": "subject_id",
            },
            "explainability": {},
        },
        integrity_config={"require_decision_equality": True},
    )
    metadata = json.loads((output / "artifact" / "metadata.json").read_text())
    assert metadata["model_implementation"] == "sklearn"
    assert metadata["model_serialization"] == "joblib"
    assert metadata["calibration"]["enabled"] is True
    loaded = load_task_artifact(output / "artifact")
    probabilities = loaded.predict_proba(bundle.splits["validation"].raw_features)
    assert probabilities.shape == (len(bundle.splits["validation"].labels),)
    assert np.isfinite(probabilities).all()
    assert result.prediction_integrity["validation"].matches


def test_track3_ensemble_writes_metrics(tmp_path: Path) -> None:
    from los_baseline.imbalance import run_undersampling_search

    bundle = _bundle()
    fixed = tmp_path / "fixed"
    labels = bundle.splits["train"].labels
    groups = bundle.splits["train"].metadata["subject_id"].to_numpy()
    folds = build_fixed_group_folds(labels=labels, groups=groups, n_splits=2, seeds=[42])
    save_fixed_folds(fixed, folds=folds, seeds=[42], n_splits=2, sample_count=len(labels))
    config = {
        "data": {"patient_id_column": "subject_id"},
        "optimization_objectives": {
            "stage1_mortality": {
                "primary": "auprc",
                "direction": "maximize",
                "secondary": "brier_score",
                "secondary_direction": "minimize",
            }
        },
        "undersampling": {
            "ratios": [1, 2],
            "seeds": [42, 142],
            "stability_penalty": 0.5,
        },
        "calibration": {"enabled": False},
        "explainability": {},
        "artifact_integrity": {"require_decision_equality": True},
    }
    payload = run_undersampling_search(
        task_data=bundle,
        model_name="decision_tree",
        model_config={
            "parameters": {"max_depth": 3, "min_samples_leaf": 2},
            "training_policy": {"class_weight_mode": "none"},
        },
        project_config=config,
        fixed_cv_dir=fixed,
        output_dir=tmp_path / "track3",
    )
    assert payload["preserve_all_minority_samples"] is True
    assert payload["ensemble"]["validation"]["member_count"] == 2
    assert (tmp_path / "track3" / "ensemble" / "validation" / "metrics.json").is_file()
