from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from los_baseline.exp123_data import TaskDataBundle, TaskSplitData
from los_baseline.optimization import (
    OptimizationSettings,
    evaluate_candidate_cv,
    finalize_validation_selection,
    run_optimization_stage,
)
from los_baseline.optimization_objectives import (
    ObjectiveDefinition,
    select_validation_candidate,
)
from los_baseline.optimization_registry import OptimizationRegistry
from los_baseline.search_spaces import (
    extract_tunable_parameters,
    get_search_space,
    sample_model_config,
)
from los_baseline.task_spec import STAGE1_MORTALITY_TASK


def _split(name: str, labels: np.ndarray, offset: int) -> TaskSplitData:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(offset + 42)
    features = pd.DataFrame(
        {
            "x1": labels + rng.normal(0, 0.25, len(labels)),
            "x2": rng.normal(0, 1, len(labels)),
        }
    )
    metadata = pd.DataFrame(
        {
            "source_row_index": np.arange(offset, offset + len(labels)),
            "subject_id": [f"{name}_{index // 2}" for index in range(len(labels))],
            "split": name,
        }
    )
    return TaskSplitData(name, features, labels, metadata)


def _bundle() -> TaskDataBundle:
    return TaskDataBundle(
        task=STAGE1_MORTALITY_TASK,
        feature_names=("x1", "x2"),
        splits={
            "train": _split("train", np.tile([0, 1, 0, 1], 20), 0),
            "validation": _split("validation", np.tile([0, 1], 16), 1000),
            "internal_test": _split("internal_test", np.tile([0, 1], 16), 2000),
        },
        shared_split_hash="unit_test_hash",
    )


def _settings() -> OptimizationSettings:
    return OptimizationSettings(
        seed=42,
        cv_folds=2,
        top_k_for_validation=2,
        stability_penalty=0.5,
        practical_tie_margin=0.005,
        auroc_margin_guard=0.02,
        prefer_simpler_model_on_tie=True,
        trial_timeout_seconds=60,
        study_timeout_seconds=600,
        resume=True,
        patient_id_column="subject_id",
    )


def test_search_sampling_is_deterministic() -> None:
    first = sample_model_config(
        "xgboost", {}, "stage1_mortality", "coarse", 3, 42
    )
    second = sample_model_config(
        "xgboost", {}, "stage1_mortality", "coarse", 3, 42
    )
    assert first == second


def test_fine_search_is_local_to_anchor() -> None:
    anchor = {
        "max_depth": 5,
        "min_samples_split": 10,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "ccp_alpha": 0.001,
    }
    space = get_search_space("decision_tree", "fine", anchor)
    assert set(space["max_depth"].values) == {3, 5, 7}
    assert set(space["ccp_alpha"].values) == {0.0001, 0.001, 0.01}


def test_registry_resume_contract(tmp_path: Path) -> None:
    path = tmp_path / "study.sqlite"
    with OptimizationRegistry(path) as registry:
        registry.register_study(
            "study", "stage1_mortality", "decision_tree", 42, "hash", {}
        )
        registry.upsert_trial(
            study_id="study",
            search_stage="smoke",
            trial_number=0,
            status="completed",
            tunable_parameters={"max_depth": 3},
            model_config={"parameters": {"max_depth": 3}},
            fold_metrics=[{"auroc": 0.7}],
            aggregate_metrics={"mean_auroc": 0.7, "std_auroc": 0.0},
            selection_score=0.7,
            duration_seconds=1.0,
        )
        assert registry.trial_exists("study", "smoke", 0)
        assert len(registry.completed_trials("study")) == 1


def test_cv_uses_group_disjoint_train_folds() -> None:
    result = evaluate_candidate_cv(
        task_data=_bundle(),
        model_name="decision_tree",
        model_config={"parameters": {"max_depth": 3}},
        objective=ObjectiveDefinition(
            "auprc", "maximize", "brier_score", "minimize"
        ),
        cv_folds=2,
        seed=42,
        stability_penalty=0.5,
        patient_id_column="subject_id",
    )
    assert len(result.fold_metrics) == 2
    assert np.isfinite(result.selection_score)
    assert all(row["train_group_count"] > 0 for row in result.fold_metrics)


def test_validation_selection_applies_auroc_guard() -> None:
    selected, rows = select_validation_candidate(
        candidates=[
            {
                "candidate_id": "low_brier_low_auc",
                "auprc": 0.7,
                "auroc": 0.60,
                "brier_score": 0.10,
                "complexity_score": 1.0,
            },
            {
                "candidate_id": "eligible",
                "auprc": 0.699,
                "auroc": 0.75,
                "brier_score": 0.11,
                "complexity_score": 2.0,
            },
        ],
        task=STAGE1_MORTALITY_TASK,
        objective=ObjectiveDefinition(
            "auprc", "maximize", "brier_score", "minimize"
        ),
        practical_tie_margin=0.005,
        auroc_margin_guard=0.02,
    )
    assert selected["candidate_id"] == "eligible"
    rejected = next(row for row in rows if row["candidate_id"] == "low_brier_low_auc")
    assert rejected["auroc_margin_guard_passed"] is False


def test_stage_resume_and_selection_artifact(tmp_path: Path) -> None:
    output = tmp_path / "study"
    objective = ObjectiveDefinition(
        "auprc", "maximize", "brier_score", "minimize"
    )
    first = run_optimization_stage(
        task_data=_bundle(),
        model_name="decision_tree",
        base_model_config={"parameters": {"max_depth": 3}},
        objective=objective,
        settings=_settings(),
        output_dir=output,
        search_stage="smoke",
        trial_count=2,
        cv_folds=2,
    )
    second = run_optimization_stage(
        task_data=_bundle(),
        model_name="decision_tree",
        base_model_config={"parameters": {"max_depth": 3}},
        objective=objective,
        settings=_settings(),
        output_dir=output,
        search_stage="smoke",
        trial_count=2,
        cv_folds=2,
    )
    assert first.completed_trials == 2
    assert second.skipped_existing_trials == 2
    selection = finalize_validation_selection(
        task_data=_bundle(),
        model_name="decision_tree",
        objective=objective,
        settings=_settings(),
        output_dir=output,
        preferred_search_stages=("smoke",),
    )
    payload = json.loads(Path(selection.output_path).read_text())
    assert payload["confirmatory_splits_accessed"] is False
    assert (output / "selected_hyperparameters.json.sha256").is_file()
    parameters = extract_tunable_parameters(
        "decision_tree", selection.selected_model_config
    )
    assert "max_depth" in parameters
