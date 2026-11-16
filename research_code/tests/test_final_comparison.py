from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from los_baseline.final_comparison import (
    FinalComparisonSettings,
    align_flat_and_pipeline_predictions,
    delong_roc_test,
    mcnemar_correctness_test,
    paired_cluster_bootstrap_differences,
)


def test_delong_matches_sklearn_auc_and_identical_predictions() -> None:
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    probability = np.array([0.1, 0.2, 0.4, 0.6, 0.7, 0.9])
    result = delong_roc_test(
        y,
        probability,
        probability,
        class_code=1,
        class_name="positive",
    )
    expected = roc_auc_score(y, probability)
    assert result.flat_auroc == pytest.approx(expected)
    assert result.pipeline_auroc == pytest.approx(expected)
    assert result.difference_pipeline_minus_flat == pytest.approx(0.0)
    assert result.p_raw == pytest.approx(1.0)


def test_delong_detects_better_pipeline_auc() -> None:
    y = np.array(([0] * 60) + ([1] * 60), dtype=np.int64)
    rng = np.random.default_rng(42)
    flat = rng.normal(0.5, 0.18, len(y))
    pipeline = 0.2 + 0.6 * y + rng.normal(0.0, 0.08, len(y))
    result = delong_roc_test(
        y,
        flat,
        pipeline,
        class_code=1,
        class_name="positive",
    )
    assert result.pipeline_auroc > result.flat_auroc
    assert result.difference_pipeline_minus_flat > 0
    assert result.p_raw < 0.05


def test_prediction_alignment_checks_indices_and_labels() -> None:
    flat = pd.DataFrame(
        {
            "source_row_index": [2, 1],
            "y_true": [1, 0],
        }
    )
    pipeline = pd.DataFrame(
        {
            "source_row_index": [1, 2],
            "y_true_3class": [0, 1],
        }
    )
    aligned_flat, aligned_pipeline = align_flat_and_pipeline_predictions(
        flat, pipeline
    )
    assert aligned_flat["source_row_index"].tolist() == [1, 2]
    assert aligned_pipeline["source_row_index"].tolist() == [1, 2]

    pipeline.loc[0, "y_true_3class"] = 2
    with pytest.raises(ValueError, match="y_true"):
        align_flat_and_pipeline_predictions(flat, pipeline)


def test_mcnemar_correctness_counts_discordant_pairs() -> None:
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    flat = np.array([0, 1, 0, 0, 2, 2], dtype=np.int64)
    pipeline = np.array([0, 0, 2, 0, 1, 2], dtype=np.int64)
    result = mcnemar_correctness_test(y, flat, pipeline)
    assert result["flat_only_correct"] == 1
    assert result["pipeline_only_correct"] == 2
    assert result["both_correct"] == 3
    assert result["discordant_pairs"] == 3
    assert 0.0 <= result["p_raw"] <= 1.0


def test_paired_cluster_bootstrap_uses_subject_clusters() -> None:
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    flat_prob = np.full((len(y), 3), 0.2)
    flat_prob[np.arange(len(y)), y] = 0.6
    pipeline_prob = np.full((len(y), 3), 0.1)
    pipeline_prob[np.arange(len(y)), y] = 0.8
    flat_prob /= flat_prob.sum(axis=1, keepdims=True)
    pipeline_prob /= pipeline_prob.sum(axis=1, keepdims=True)
    metadata = pd.DataFrame(
        {
            "source_row_index": np.arange(len(y)),
            "subject_id": [f"p_{index // 3}" for index in range(len(y))],
        }
    )
    table, manifest = paired_cluster_bootstrap_differences(
        y_true=y,
        flat_probabilities=flat_prob,
        flat_predicted_classes=np.argmax(flat_prob, axis=1),
        pipeline_probabilities=pipeline_prob,
        pipeline_predicted_classes=np.argmax(pipeline_prob, axis=1),
        metadata=metadata,
        class_names=("A", "B", "C"),
        settings=FinalComparisonSettings(
            bootstrap_repeats=20,
            minimum_successful_bootstrap_repeats=5,
            comparison_metrics=("multiclass_brier", "macro_f1"),
        ),
    )
    assert manifest["bootstrap_unit"] == "subject_id"
    assert manifest["cluster_count"] == 3
    brier = table.loc[table["metric"] == "multiclass_brier"].iloc[0]
    assert brier["improvement_positive_favors_pipeline"] > 0
