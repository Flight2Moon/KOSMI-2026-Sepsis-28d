from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from los_baseline.exp123_data import TaskDataBundle, TaskSplitData
from los_baseline.failure_diagnostics import classify_failure
from los_baseline.hpo.optuna_engine import (
    HPO_PROTOCOL_VERSION,
    budget_consuming_trial_count,
    run_optuna_optimization_study,
)
from los_baseline.models import normalize_task_model_config
from los_baseline.optimization import _fit_predict_partition
from los_baseline.task_spec import (
    EXP1_FLAT_3CLASS_TASK,
    STAGE1_MORTALITY_TASK,
)


def _frame(labels: np.ndarray, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.asarray(labels, dtype=np.int64)
    return pd.DataFrame(
        {
            "x1": y.astype(float) + rng.normal(0.0, 0.25, len(y)),
            "x2": rng.normal(0.0, 1.0, len(y)),
            "x3": (y % 2).astype(float) + rng.normal(0.0, 0.15, len(y)),
        }
    )


def _split(name: str, labels: np.ndarray, *, seed: int, offset: int) -> TaskSplitData:
    y = np.asarray(labels, dtype=np.int64)
    return TaskSplitData(
        split_name=name,
        raw_features=_frame(y, seed=seed),
        labels=y,
        metadata=pd.DataFrame(
            {
                "source_row_index": np.arange(offset, offset + len(y)),
                "subject_id": [f"{name}_{index}" for index in range(len(y))],
                "split": name,
            }
        ),
    )


def _binary_bundle() -> TaskDataBundle:
    return TaskDataBundle(
        task=STAGE1_MORTALITY_TASK,
        feature_names=("x1", "x2", "x3"),
        splits={
            "train": _split("train", np.tile([0, 0, 1, 0, 1], 24), seed=11, offset=0),
            "validation": _split("validation", np.tile([0, 1], 20), seed=12, offset=1000),
        },
        shared_split_hash="v73_binary_hash",
    )


def _multiclass_arrays() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    train_y = np.tile([0, 1, 2], 30).astype(np.int64)
    valid_y = np.tile([0, 1, 2], 10).astype(np.int64)
    return (
        _frame(train_y, seed=21),
        train_y,
        _frame(valid_y, seed=22),
        valid_y,
        np.asarray([f"p_{index}" for index in range(len(train_y))]),
    )


def _multiclass_bundle() -> TaskDataBundle:
    train_y = np.tile([0, 1, 2], 30).astype(np.int64)
    valid_y = np.tile([0, 1, 2], 10).astype(np.int64)
    return TaskDataBundle(
        task=EXP1_FLAT_3CLASS_TASK,
        feature_names=("x1", "x2", "x3"),
        splits={
            "train": _split("train", train_y, seed=31, offset=0),
            "validation": _split("validation", valid_y, seed=32, offset=1000),
        },
        shared_split_hash="v73_multiclass_hash",
    )


def test_svm_binary_hpo_uses_decision_scores_without_probability_metrics() -> None:
    bundle = _binary_bundle()
    train = bundle.splits["train"]
    valid = bundle.splits["validation"]
    _, metrics, diagnostics, effective = _fit_predict_partition(
        model_name="svm",
        model_config={
            "parameters": {"kernel": "linear", "C": 1.0, "probability": False},
            "training_policy": {"class_weight_mode": "none"},
        },
        task=bundle.task,
        feature_names=bundle.feature_names,
        train_features=train.raw_features,
        train_labels=train.labels,
        evaluation_features=valid.raw_features,
        evaluation_labels=valid.labels,
        seed=42,
        train_groups=train.metadata["subject_id"].to_numpy(),
    )
    assert diagnostics["prediction_kind"] == "decision_score"
    assert "auprc" in metrics and "auroc" in metrics
    assert "brier_score" not in metrics
    assert effective["parameters"]["probability"] is False
    assert effective["parameters"]["decision_function_shape"] == "ovr"


def test_svm_multiclass_hpo_uses_ovr_decision_scores() -> None:
    train_x, train_y, valid_x, valid_y, groups = _multiclass_arrays()
    predictions, metrics, diagnostics, _ = _fit_predict_partition(
        model_name="svm",
        model_config={
            "parameters": {"kernel": "linear", "C": 0.5, "probability": False},
            "training_policy": {"class_weight_mode": "none"},
        },
        task=EXP1_FLAT_3CLASS_TASK,
        feature_names=("x1", "x2", "x3"),
        train_features=train_x,
        train_labels=train_y,
        evaluation_features=valid_x,
        evaluation_labels=valid_y,
        seed=43,
        train_groups=groups,
    )
    assert predictions.shape == (len(valid_y), 3)
    assert diagnostics["prediction_kind"] == "decision_score"
    assert "macro_auprc" in metrics and "macro_auroc" in metrics
    assert "multiclass_brier" not in metrics
    assert "multiclass_log_loss" not in metrics


def test_multiclass_xgboost_overrides_shared_binary_metric() -> None:
    normalized = normalize_task_model_config(
        "xgboost",
        {
            "parameters": {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "num_class": 99,
            }
        },
        task_type="multiclass",
        num_classes=3,
        hpo_mode=True,
    )
    parameters = normalized["parameters"]
    assert parameters["objective"] == "multi:softprob"
    assert parameters["eval_metric"] == "mlogloss"
    assert parameters["num_class"] == 3


def test_xgboost_early_stopping_is_nested_and_refit_before_outer_evaluation() -> None:
    train_x, train_y, valid_x, valid_y, groups = _multiclass_arrays()
    _, metrics, diagnostics, effective = _fit_predict_partition(
        model_name="xgboost",
        model_config={
            "parameters": {
                "n_estimators": 24,
                "early_stopping_rounds": 3,
                "max_depth": 3,
                "learning_rate": 0.1,
                "tree_method": "hist",
                "device": "cpu",
                "n_jobs": 1,
                "verbosity": 0,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            },
            "training_policy": {"class_weight_mode": "none"},
        },
        task=EXP1_FLAT_3CLASS_TASK,
        feature_names=("x1", "x2", "x3"),
        train_features=train_x,
        train_labels=train_y,
        evaluation_features=valid_x,
        evaluation_labels=valid_y,
        seed=44,
        train_groups=groups,
    )
    assert metrics["macro_auprc"] > 0.0
    assert diagnostics["early_stopping_outer_validation_accessed"] is False
    assert diagnostics["early_stopping_protocol"].startswith("nested_group_holdout")
    assert diagnostics["selected_n_estimators"] >= 1
    assert diagnostics["early_stopping_split_hash"]
    assert effective["parameters"]["eval_metric"] == "mlogloss"


def test_failed_trials_do_not_consume_optuna_budget() -> None:
    trials = [
        SimpleNamespace(state=SimpleNamespace(name="COMPLETE")),
        SimpleNamespace(state=SimpleNamespace(name="PRUNED")),
        SimpleNamespace(state=SimpleNamespace(name="FAIL")),
        SimpleNamespace(state=SimpleNamespace(name="RUNNING")),
    ]
    assert budget_consuming_trial_count(trials) == 2


def test_svm_optuna_protocol_is_consistent_and_records_secondary_fallback(tmp_path: Path) -> None:
    bundle = _binary_bundle()
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
        "optimization_v2": {
            "engine": "optuna_tpe",
            "seed": 42,
            "cv_folds": 2,
            "finalist_repeat_seeds": [42, 1042],
            "startup_trials": 1,
            "multivariate": False,
            "group": False,
            "pruner_startup_trials": 20,
            "pruner_warmup_folds": 2,
            "finalist_top_k": 2,
            "stability_penalty": 0.5,
            "overfit_penalty": 0.5,
            "trial_budgets": {"svm": 2},
        },
    }
    output = tmp_path / "run" / "optimization" / "stage1_mortality" / "svm"
    run_optuna_optimization_study(
        task_data=bundle,
        model_name="svm",
        base_model_config={
            "parameters": {"kernel": "linear", "C": 1.0, "probability": True},
            "training_policy": {"class_weight_mode": "none"},
        },
        config=config,
        output_dir=output,
    )
    payload = json.loads((output / "selected_hyperparameters.json").read_text())
    assert payload["hpo_protocol_version"] == HPO_PROTOCOL_VERSION
    assert HPO_PROTOCOL_VERSION in payload["study_name"]
    assert payload["model_config"]["parameters"]["probability"] is False
    guards = payload["selection_guards"]
    assert guards["secondary_metric_applied"] is False
    assert guards["selection_fallback_when_secondary_unavailable"] == (
        "primary_stability_then_model_complexity"
    )
    summary = json.loads((output / "study_state_summary.json").read_text())
    assert summary["failed_trials_do_not_consume_budget"] is True
    assert summary["consumed_trial_count"] == 2


def test_failure_diagnostics_prioritize_model_errors_over_missing_lock() -> None:
    message = "exit_code=1; completion=missing_completion_artifact"
    svm_log = "AttributeError: This 'SVC' has no attribute 'predict_proba'"
    xgb_log = "XGBoostError: label and prediction size not match; use mlogloss"
    assert classify_failure(message, svm_log) == "model_interface_error"
    assert classify_failure(message, xgb_log) == "model_configuration_error"


def test_selected_svm_probability_false_trains_after_oof_calibration(tmp_path: Path) -> None:
    from los_baseline.training_engine import train_and_evaluate_single_model

    bundle = _multiclass_bundle()
    result = train_and_evaluate_single_model(
        task_data=bundle,
        model_name="svm",
        model_config={
            "parameters": {
                "kernel": "linear",
                "C": 1.0,
                "probability": False,
                "decision_function_shape": "ovr",
                "cache_size": 256,
            },
            "training_policy": {"class_weight_mode": "none"},
        },
        output_dir=tmp_path / "calibrated_svm",
        seed=42,
        project_config={
            "data": {"patient_id_column": "subject_id"},
            "calibration": {
                "enabled": True,
                "method": "sigmoid",
                "cv_folds": 2,
                "seed": 4242,
                "group_column": "subject_id",
            },
            "explainability": {},
        },
        integrity_config={"require_decision_equality": True},
    )
    assert result.split_metrics["validation"]["macro_auprc"] > 0.0
    metadata = json.loads(
        (tmp_path / "calibrated_svm" / "artifact" / "metadata.json").read_text()
    )
    assert metadata["calibration"]["enabled"] is True
    assert metadata["model_serialization"] == "joblib"


def test_selected_exp1_xgboost_trains_with_multiclass_metric_and_calibration(tmp_path: Path) -> None:
    from los_baseline.training_engine import train_and_evaluate_single_model

    bundle = _multiclass_bundle()
    result = train_and_evaluate_single_model(
        task_data=bundle,
        model_name="xgboost",
        model_config={
            "parameters": {
                "n_estimators": 8,
                "max_depth": 2,
                "learning_rate": 0.15,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "device": "cpu",
                "n_jobs": 1,
                "verbosity": 0,
            },
            "training_policy": {"class_weight_mode": "none"},
        },
        output_dir=tmp_path / "calibrated_xgb",
        seed=42,
        project_config={
            "data": {"patient_id_column": "subject_id"},
            "calibration": {
                "enabled": True,
                "method": "sigmoid",
                "cv_folds": 2,
                "seed": 4242,
                "group_column": "subject_id",
            },
            "explainability": {},
        },
        integrity_config={"require_decision_equality": True},
    )
    assert result.split_metrics["validation"]["macro_auprc"] > 0.0
    metadata = json.loads(
        (tmp_path / "calibrated_xgb" / "artifact" / "metadata.json").read_text()
    )
    effective = metadata["model_config_effective"]
    assert effective["objective"] == "multi:softprob"
    assert effective["eval_metric"] == "mlogloss"



def test_track3_svm_uses_hpo_selected_probability_false_with_oof_calibration(
    tmp_path: Path,
) -> None:
    from los_baseline.hpo.fixed_cv import build_fixed_group_folds, save_fixed_folds
    from los_baseline.imbalance import run_undersampling_search

    bundle = _binary_bundle()
    fixed = tmp_path / "fixed_cv"
    train = bundle.splits["train"]
    groups = train.metadata["subject_id"].to_numpy()
    folds = build_fixed_group_folds(
        labels=train.labels,
        groups=groups,
        n_splits=2,
        seeds=[42],
    )
    save_fixed_folds(
        fixed,
        folds=folds,
        seeds=[42],
        n_splits=2,
        sample_count=len(train.labels),
    )
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
            "ratios": [1],
            "seeds": [42],
            "stability_penalty": 0.5,
        },
        "calibration": {
            "enabled": True,
            "method": "sigmoid",
            "cv_folds": 2,
            "seed": 4242,
            "group_column": "subject_id",
        },
        "explainability": {},
        "artifact_integrity": {"require_decision_equality": True},
    }
    payload = run_undersampling_search(
        task_data=bundle,
        model_name="svm",
        model_config={
            "parameters": {
                "kernel": "linear",
                "C": 0.5,
                "probability": False,
                "decision_function_shape": "ovr",
            },
            "training_policy": {"class_weight_mode": "none"},
        },
        project_config=config,
        fixed_cv_dir=fixed,
        output_dir=tmp_path / "track3_svm",
    )
    assert payload["selected_ratio"] == 1.0
    assert payload["ensemble"]["validation"]["member_count"] == 1
    member_metadata = json.loads(
        (
            tmp_path
            / "track3_svm"
            / "members"
            / "seed_42"
            / "artifact"
            / "metadata.json"
        ).read_text()
    )
    assert member_metadata["calibration"]["enabled"] is True
    assert (
        tmp_path
        / "track3_svm"
        / "ensemble"
        / "validation"
        / "metrics.json"
    ).is_file()
