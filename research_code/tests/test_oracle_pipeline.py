from __future__ import annotations

import numpy as np
import pandas as pd

from los_baseline.oracle_pipeline import (
    _error_propagation_frames,
    derive_pipeline_classes,
    restore_three_class_probabilities,
)


def test_restore_three_class_probabilities_sum_to_one() -> None:
    stage1 = np.array([0.1, 0.7, 0.2], dtype=float)
    stage2 = np.array([0.2, 0.8, 0.5], dtype=float)
    restored = restore_three_class_probabilities(stage1, stage2)
    expected = np.array(
        [
            [0.72, 0.10, 0.18],
            [0.06, 0.70, 0.24],
            [0.40, 0.20, 0.40],
        ]
    )
    assert np.allclose(restored, expected)
    assert np.allclose(restored.sum(axis=1), 1.0)


def test_pipeline_classes_follow_frozen_stage_thresholds() -> None:
    stage1 = np.array([0.2, 0.8, 0.3, 0.4])
    stage2 = np.array([0.1, 0.9, 0.8, 0.2])
    predicted = derive_pipeline_classes(
        stage1,
        stage2,
        stage1_threshold=0.6,
        stage2_threshold=0.5,
    )
    assert predicted.tolist() == [0, 1, 2, 0]


def test_error_propagation_separates_stage1_and_stage2_errors() -> None:
    frame = pd.DataFrame(
        {
            "source_row_index": np.arange(6),
            "split": "internal_test",
            "y_true_3class": [0, 1, 2, 0, 1, 2],
            "stage1_predicted_death": [0, 0, 0, 1, 0, 0],
            "stage2_predicted_long_stay": [0, 0, 0, 1, 1, 1],
            "pipeline_predicted_class": [0, 0, 0, 1, 2, 2],
        }
    )
    detail, summary, transition = _error_propagation_frames(frame)
    categories = detail["error_category"].tolist()
    assert categories == [
        "correct_pipeline",
        "stage1_false_negative_B_to_A",
        "stage2_false_negative_C_to_A",
        "stage1_false_positive_survivor_to_B",
        "stage1_false_negative_B_to_C",
        "correct_pipeline",
    ]
    assert int(detail["stage1_error"].sum()) == 3
    assert int(detail["stage2_error_among_stage1_non_death"].sum()) == 1
    assert int(summary["count"].sum()) == len(frame)
    assert int(transition.to_numpy().sum()) == len(frame)
