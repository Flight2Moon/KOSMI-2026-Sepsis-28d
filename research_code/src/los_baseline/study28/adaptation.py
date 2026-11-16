from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..evaluation_multiclass import calculate_multiclass_metrics
from .calibration import study28_calibration_summary


@dataclass(frozen=True)
class PlattMap:
    intercepts: tuple[float, ...]
    slopes: tuple[float, ...]
    class_names: tuple[str, ...]
    clip_probability: float


def _validate_probability_matrix(P: np.ndarray, n_classes: int) -> np.ndarray:
    values = np.asarray(P, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != n_classes:
        raise ValueError(f"Expected probability matrix with {n_classes} columns, got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Probabilities contain non-finite values.")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("Probabilities do not sum to one.")
    return values


def fit_platt_map(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    class_names: Sequence[str],
    *,
    clip_probability: float = 1e-6,
    logistic_C: float = 1e6,
    max_iter: int = 1000,
) -> PlattMap:
    names = tuple(str(x) for x in class_names)
    P = _validate_probability_matrix(probabilities, len(names))
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if len(y) != len(P):
        raise ValueError("Probability and label lengths differ.")
    intercepts: list[float] = []
    slopes: list[float] = []
    for class_code in range(len(names)):
        binary = (y == class_code).astype(np.int64)
        if np.unique(binary).size < 2:
            raise ValueError(f"Calibration subset lacks both classes for class_code={class_code}.")
        p = np.clip(P[:, class_code], clip_probability, 1.0 - clip_probability)
        logit = np.log(p / (1.0 - p)).reshape(-1, 1)
        lr = LogisticRegression(C=logistic_C, max_iter=max_iter)
        lr.fit(logit, binary)
        slopes.append(float(lr.coef_[0][0]))
        intercepts.append(float(lr.intercept_[0]))
    return PlattMap(
        intercepts=tuple(intercepts),
        slopes=tuple(slopes),
        class_names=names,
        clip_probability=float(clip_probability),
    )


def apply_platt_map(probabilities: np.ndarray, mapping: PlattMap) -> np.ndarray:
    P = _validate_probability_matrix(probabilities, len(mapping.class_names))
    Q = np.zeros_like(P, dtype=np.float64)
    for class_code, (intercept, slope) in enumerate(zip(mapping.intercepts, mapping.slopes)):
        p = np.clip(P[:, class_code], mapping.clip_probability, 1.0 - mapping.clip_probability)
        logit = np.log(p / (1.0 - p))
        z = intercept + slope * logit
        Q[:, class_code] = 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))
    row_sum = Q.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0) or not np.isfinite(row_sum).all():
        raise ValueError("Recalibrated probabilities cannot be normalized.")
    Q = Q / row_sum
    if not np.allclose(Q.sum(axis=1), 1.0, atol=1e-8, rtol=1e-8):
        raise RuntimeError("Recalibrated probabilities do not sum to one after normalization.")
    return Q


def split_calibration_application_indices(
    y_true: np.ndarray,
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Professor-compatible random 20/80 split (not stratified)."""
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_cal = int(calibration_fraction * len(idx))
    if n_cal <= 0 or n_cal >= len(idx):
        raise ValueError("Calibration split must leave non-empty calibration and application subsets.")
    return idx[:n_cal], idx[n_cal:]


def repeated_local_recalibration(
    *,
    site: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    calibration_fraction: float,
    seeds: int,
    base_seed: int,
    clip_probability: float,
    logistic_C: float,
    max_iter: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Repeat professor-style 20/80 local recalibration over multiple random seeds.

    Returns per-seed class calibration/metric rows, aggregated mean/SD rows, and
    representative artifacts keyed by seed: (application_indices, raw_P, recalibrated_P).
    """
    names = tuple(str(x) for x in class_names)
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    P = _validate_probability_matrix(probabilities, len(names))
    rows: list[dict[str, object]] = []
    seed_artifacts: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for offset in range(seeds):
        seed = int(base_seed + offset)
        cal_idx, app_idx = split_calibration_application_indices(
            y, calibration_fraction=calibration_fraction, seed=seed
        )
        try:
            mapping = fit_platt_map(
                P[cal_idx], y[cal_idx], names,
                clip_probability=clip_probability,
                logistic_C=logistic_C,
                max_iter=max_iter,
            )
        except ValueError:
            continue
        Q = apply_platt_map(P[app_idx], mapping)
        seed_artifacts[seed] = (app_idx, P[app_idx], Q)
        for model_name, current in (("raw", P[app_idx]), ("recalibrated", Q)):
            metrics = calculate_multiclass_metrics(y[app_idx], current, names)
            cal = study28_calibration_summary(
                y[app_idx], current, names,
                clip_probability=clip_probability,
                logistic_C=logistic_C,
                max_iter=max_iter,
            )
            for _, crow in cal.iterrows():
                code = int(crow["class_code"])
                rows.append(
                    {
                        "site": site,
                        "seed": seed,
                        "model": model_name,
                        "class_code": code,
                        "class_name": str(crow["class_name"]),
                        "n_application": int(len(app_idx)),
                        "calibration_slope": float(crow["calibration_slope"]),
                        "calibration_intercept": float(crow["calibration_intercept"]),
                        "ovr_brier": float(crow["ovr_brier"]),
                        "mean_predicted_probability": float(crow["mean_predicted_probability"]),
                        "prevalence": float(crow["prevalence"]),
                        "class_auroc": float(metrics[f"class_{code}_auroc"]),
                        "class_auprc": float(metrics[f"class_{code}_auprc"]),
                        "macro_auroc": float(metrics["macro_auroc"]),
                        "macro_auprc": float(metrics["macro_auprc"]),
                        "multiclass_brier": float(metrics["multiclass_brier"]),
                        "multiclass_log_loss": float(metrics["multiclass_log_loss"]),
                    }
                )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, raw.copy(), seed_artifacts
    numeric = [
        "n_application", "calibration_slope", "calibration_intercept", "ovr_brier",
        "mean_predicted_probability", "prevalence", "class_auroc", "class_auprc",
        "macro_auroc", "macro_auprc", "multiclass_brier", "multiclass_log_loss",
    ]
    agg = raw.groupby(["site", "model", "class_code", "class_name"], as_index=False)[numeric].agg(["mean", "std"])
    agg.columns = ["_".join([str(x) for x in col if str(x)]) for col in agg.columns.to_flat_index()]
    return raw, agg, seed_artifacts
