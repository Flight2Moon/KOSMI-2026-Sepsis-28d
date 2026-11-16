from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from los_baseline.study28.calibration import (
    study28_calibration_bins,
    study28_calibration_summary,
)


def _example():
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2] * 10, dtype=int)
    rng = np.random.default_rng(2026)
    raw = rng.uniform(0.05, 1.0, size=(len(y), 3))
    raw[np.arange(len(y)), y] += 1.25
    p = raw / raw.sum(axis=1, keepdims=True)
    return y, p


def test_calibration_summary_matches_professor_logit_logistic_method():
    y, p = _example()
    summary = study28_calibration_summary(
        y,
        p,
        ("early_dc", "death", "prolonged"),
        clip_probability=1e-6,
        logistic_C=1e6,
        max_iter=1000,
    )
    clipped = np.clip(p[:, 1], 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    expected = LogisticRegression(C=1e6, max_iter=1000)
    expected.fit(logit, (y == 1).astype(int))
    death = summary.loc[summary.class_code == 1].iloc[0]
    np.testing.assert_allclose(death.calibration_slope, expected.coef_[0][0], atol=1e-12)
    np.testing.assert_allclose(death.calibration_intercept, expected.intercept_[0], atol=1e-12)


def test_calibration_bins_cover_every_row_for_every_class():
    y, p = _example()
    bins = study28_calibration_bins(
        y, p, ("early_dc", "death", "prolonged"), bins=10
    )
    for class_code in range(3):
        assert bins.loc[bins.class_code == class_code, "n"].sum() == len(y)
