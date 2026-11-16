from __future__ import annotations

import numpy as np
import pandas as pd

from los_baseline.study28_exp123.core import (
    TASK_EXP1,
    TASK_STAGE1,
    TASK_STAGE2,
    majority_downsample_indices,
    multiclass_metrics,
    task_frame,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"y": [0, 1, 2, 0, 2, 1, 0, 2]})


def test_exp123_task_mapping_matches_original_semantics():
    frame = _frame()
    _, y1 = task_frame(frame, TASK_EXP1)
    _, ys1 = task_frame(frame, TASK_STAGE1)
    f2, ys2 = task_frame(frame, TASK_STAGE2)
    assert y1.tolist() == frame.y.tolist()
    assert ys1.tolist() == [0, 1, 0, 0, 0, 1, 0, 0]
    assert f2.y.tolist() == [0, 2, 0, 2, 0, 2]
    assert ys2.tolist() == [0, 1, 0, 1, 0, 1]


def test_majority_downsampling_is_train_only_index_selector():
    y = np.array([0] * 100 + [1] * 20)
    idx = majority_downsample_indices(y, max_majority_ratio=2.0, seed=2026)
    selected = y[idx]
    counts = np.bincount(selected)
    assert counts.tolist() == [40, 20]
    assert len(np.unique(idx)) == len(idx)


def test_multiclass_metrics_normalize_near_probability_rows():
    y = np.array([0, 1, 2, 0, 1, 2])
    p = np.array([
        [0.8, 0.1, 0.1000001], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7],
        [0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7],
    ])
    metrics = multiclass_metrics(y, p)
    assert 0.9 <= metrics["macro_auroc"] <= 1.0
    assert np.isfinite(metrics["log_loss"])
