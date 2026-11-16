from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from los_baseline.stacking import (
    SuperLearnerArtifact,
    create_validation_subsplit,
    enumerate_library_subsets,
    fit_nnls_weights,
    load_aligned_validation_predictions,
    load_super_learner_artifact,
    save_super_learner_artifact,
)
from los_baseline.task_spec import STAGE1_MORTALITY_TASK
from los_baseline.utils import atomic_write_json


def _validation_frame(n: int = 80) -> pd.DataFrame:
    y = np.asarray(([0, 1] * (n // 2)), dtype=np.int64)
    return pd.DataFrame(
        {
            "source_row_index": np.arange(n),
            "subject_id": [f"patient_{index}" for index in range(n)],
            "y_true": y,
            "prob_model_a": np.clip(0.2 + 0.6 * y, 0, 1),
            "prob_model_b": np.clip(0.3 + 0.4 * y, 0, 1),
        }
    )


def test_validation_subsplit_is_patient_disjoint_and_reproducible() -> None:
    frame = _validation_frame()
    first = create_validation_subsplit(frame, seed=42)
    second = create_validation_subsplit(frame, seed=42)
    assert first.split_hash == second.split_hash
    assert np.array_equal(first.meta_mask, second.meta_mask)
    assert not np.any(first.meta_mask & first.threshold_mask)
    assert np.all(first.meta_mask | first.threshold_mask)
    assert set(first.subject_id[first.meta_mask]).isdisjoint(
        set(first.subject_id[first.threshold_mask])
    )
    assert np.unique(first.y_true[first.meta_mask]).size == 2
    assert np.unique(first.y_true[first.threshold_mask]).size == 2


def test_library_subset_enumeration_excludes_singletons_when_multiple_models() -> None:
    subsets = enumerate_library_subsets(["a", "b", "c"])
    assert len(subsets) == 4
    assert ("a", "b") in subsets
    assert ("a", "b", "c") in subsets
    assert all(len(value) >= 2 for value in subsets)
    assert enumerate_library_subsets(["a"]) == [("a",)]


def test_nnls_weights_are_nonnegative_and_sum_to_one() -> None:
    y = np.array([0, 0, 1, 1], dtype=float)
    matrix = np.array(
        [
            [0.1, 0.3],
            [0.2, 0.4],
            [0.8, 0.6],
            [0.9, 0.7],
        ],
        dtype=float,
    )
    weights, residual, status = fit_nnls_weights(matrix, y)
    assert status == "completed"
    assert residual >= 0
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)


def test_super_learner_artifact_roundtrip(tmp_path: Path) -> None:
    artifact = SuperLearnerArtifact(
        artifact_schema_version="exp123_super_learner_v1",
        task_name=STAGE1_MORTALITY_TASK.task_name,
        meta_learner_method="nnls",
        library_members=("a", "b"),
        weights=(0.25, 0.75),
        validation_decision_threshold=0.4,
        fit_split="validation_meta",
        threshold_selection_split="validation_threshold",
        shared_split_hash="a" * 64,
        validation_subsplit_hash="b" * 64,
        base_prediction_hashes={"a": "c" * 64, "b": "d" * 64},
        heterogeneity_frozen_design_sha256="e" * 64,
        library_candidates_sha256="f" * 64,
        selection_metric="auprc",
        selection_metric_value=0.8,
        auroc_margin_guard=0.02,
        practical_tie_margin=0.005,
        seed=42,
    )
    path = tmp_path / "artifact.json"
    digest = save_super_learner_artifact(path, artifact)
    assert len(digest) == 64
    loaded = load_super_learner_artifact(path)
    prediction = loaded.predict(
        {
            "a": np.array([0.2, 0.8]),
            "b": np.array([0.4, 0.6]),
        }
    )
    assert np.allclose(prediction, np.array([0.35, 0.65]))


def _write_prediction(
    run_dir: Path,
    model_name: str,
    source_rows: np.ndarray,
    y_true: np.ndarray,
    probability: np.ndarray,
) -> None:
    path = (
        run_dir
        / "EXP2_two_stage"
        / "stage1_mortality"
        / model_name
        / "evaluation"
        / "validation"
        / "predictions.csv.gz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "source_row_index": source_rows,
            "subject_id": [f"p_{value}" for value in source_rows],
            "task_name": STAGE1_MORTALITY_TASK.task_name,
            "model_name": model_name,
            "y_true": y_true,
            "positive_probability": probability,
        }
    ).to_csv(path, index=False, compression="gzip")


def test_aligned_prediction_loader_detects_patient_mismatch(tmp_path: Path) -> None:
    y = np.array([0, 1, 0, 1], dtype=int)
    _write_prediction(tmp_path, "a", np.arange(4), y, np.array([0.1, 0.8, 0.2, 0.7]))
    _write_prediction(tmp_path, "b", np.array([0, 1, 2, 9]), y, np.array([0.2, 0.7, 0.3, 0.6]))
    with pytest.raises(ValueError, match="identical validation patients"):
        load_aligned_validation_predictions(
            tmp_path,
            STAGE1_MORTALITY_TASK.task_name,
            ["a", "b"],
        )


def test_subset_selection_applies_auroc_guard_before_brier() -> None:
    from los_baseline.stacking import ValidationSubsplit, _subset_rows

    y_half = np.array([1] * 5 + [0] * 45, dtype=np.int64)
    y = np.concatenate([y_half, y_half])
    n = len(y)
    aligned = pd.DataFrame(
        {
            "source_row_index": np.arange(n),
            "subject_id": [f"p_{index}" for index in range(n)],
            "y_true": y,
            "prob_low_brier_low_auc": np.full(n, 0.1),
            "prob_high_auc": np.where(y == 1, 0.4, 0.3),
        }
    )
    meta_mask = np.zeros(n, dtype=bool)
    meta_mask[:50] = True
    threshold_mask = ~meta_mask
    subsplit = ValidationSubsplit(
        source_row_index=aligned["source_row_index"].to_numpy(),
        subject_id=aligned["subject_id"].to_numpy(),
        y_true=y,
        meta_mask=meta_mask,
        threshold_mask=threshold_mask,
        split_hash="a" * 64,
    )
    frame, selected, _ = _subset_rows(
        aligned=aligned,
        subsplit=subsplit,
        subsets=[("low_brier_low_auc",), ("high_auc",)],
        task_name=STAGE1_MORTALITY_TASK.task_name,
        primary_metric="auprc",
        auroc_margin_guard=0.02,
        practical_tie_margin=0.005,
    )
    low_row = frame.loc[
        frame["library_members"] == "low_brier_low_auc"
    ].iloc[0]
    assert not bool(low_row["eligible_by_auroc_guard"])
    assert selected["library_members"] == "high_auc"
