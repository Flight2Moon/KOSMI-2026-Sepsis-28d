from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def _validate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64)
    names = tuple(str(value) for value in class_names)
    if p.shape != (len(y), len(names)):
        raise ValueError(f"Invalid probability shape {p.shape}; expected {(len(y), len(names))}.")
    if not np.isfinite(p).all():
        raise ValueError("Calibration probabilities contain non-finite values.")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("Calibration probabilities do not sum to one.")
    if set(np.unique(y).tolist()) != set(range(len(names))):
        raise ValueError("Calibration requires all Study28 classes to be present.")
    return y, p, names


def study28_calibration_summary(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    *,
    clip_probability: float = 1e-6,
    logistic_C: float = 1e6,
    max_iter: int = 1000,
) -> pd.DataFrame:
    """Professor-compatible one-vs-rest logistic recalibration statistics.

    This reproduces ``main_28d_analysis.py``: fit an approximately unregularized
    logistic regression of binary class membership on logit(p). The fitted
    coefficient is the calibration slope and the fitted intercept is the
    calibration intercept.
    """
    y, p, names = _validate(y_true, probabilities, class_names)
    rows: list[dict[str, float | int | str]] = []
    for class_code, class_name in enumerate(names):
        binary = (y == class_code).astype(np.int64)
        clipped = np.clip(p[:, class_code], clip_probability, 1.0 - clip_probability)
        logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        regression = LogisticRegression(C=logistic_C, max_iter=max_iter)
        regression.fit(logit, binary)
        rows.append(
            {
                "class_code": class_code,
                "class_name": class_name,
                "calibration_slope": float(regression.coef_[0][0]),
                "calibration_intercept": float(regression.intercept_[0]),
                "ovr_brier": float(brier_score_loss(binary, clipped)),
                "mean_predicted_probability": float(clipped.mean()),
                "prevalence": float(binary.mean()),
            }
        )
    return pd.DataFrame(rows)


def study28_calibration_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    *,
    bins: int = 10,
    clip_probability: float = 1e-6,
) -> pd.DataFrame:
    """Professor-compatible quantile calibration bins using ``pandas.qcut``."""
    y, p, names = _validate(y_true, probabilities, class_names)
    if bins < 2:
        raise ValueError("Calibration bins must be at least two.")
    rows: list[dict[str, float | int | str]] = []
    for class_code, class_name in enumerate(names):
        clipped = np.clip(p[:, class_code], clip_probability, 1.0 - clip_probability)
        binary = (y == class_code).astype(np.int64)
        quantiles = pd.qcut(clipped, bins, labels=False, duplicates="drop")
        quantiles = np.asarray(quantiles)
        for bin_code in sorted(int(value) for value in np.unique(quantiles)):
            mask = quantiles == bin_code
            rows.append(
                {
                    "class_code": class_code,
                    "class_name": class_name,
                    "bin": bin_code,
                    "mean_predicted_probability": float(clipped[mask].mean()),
                    "observed_positive_fraction": float(binary[mask].mean()),
                    "n": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)
