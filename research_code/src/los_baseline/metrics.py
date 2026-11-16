from __future__ import annotations

from typing import Any

import numpy as np

from .progress import progress_iter
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def validate_probabilities(
    probabilities: np.ndarray,
    expected_rows: int,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(probabilities) != expected_rows:
        raise ValueError(
            "Prediction count mismatch: "
            f"{len(probabilities)} != {expected_rows}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities contain non-finite values.")
    return np.clip(probabilities, 0.0, 1.0)


def select_youden_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    probability = validate_probabilities(probability, len(y_true))
    if np.unique(y_true).size < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    finite_mask = np.isfinite(thresholds)
    if not finite_mask.any():
        return 0.5
    finite_indices = np.where(finite_mask)[0]
    best_local = int(np.argmax((tpr - fpr)[finite_mask]))
    selected = float(
        np.clip(thresholds[finite_indices[best_local]], 0.0, 1.0)
    )

    # sklearn ROC thresholds are observed probability values. Using one of
    # those values directly makes the hard decision numerically fragile after
    # model serialization: a sub-ULP probability change can move a sample to
    # the other side of ``>= threshold``. Move the threshold to the open gap
    # between the selected positive set and the nearest lower probability.
    # This preserves every validation decision while giving saved/reloaded
    # artifacts a deterministic decision margin.
    unique_probability = np.unique(probability)
    selected_positive = unique_probability[unique_probability >= selected]
    selected_negative = unique_probability[unique_probability < selected]
    if selected_positive.size == 0:
        return 1.0
    positive_edge = float(selected_positive.min())
    if selected_negative.size == 0:
        return 0.0
    negative_edge = float(selected_negative.max())
    midpoint = negative_edge + (positive_edge - negative_edge) / 2.0
    if not negative_edge < midpoint < positive_edge:
        midpoint = float(np.nextafter(positive_edge, negative_edge))
    return float(np.clip(midpoint, 0.0, 1.0))


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> tuple[float, float]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    maximum_error = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (
                (probability >= edges[index])
                & (probability <= edges[index + 1])
            )
        else:
            mask = (
                (probability >= edges[index])
                & (probability < edges[index + 1])
            )
        if not mask.any():
            continue
        confidence = float(probability[mask].mean())
        observed = float(y_true[mask].mean())
        error = abs(confidence - observed)
        ece += float(mask.mean()) * error
        maximum_error = max(maximum_error, error)
    return float(ece), float(maximum_error)


def calculate_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    decision_threshold: float,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    probability = validate_probabilities(probability, len(y_true))
    prediction = (probability >= decision_threshold).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(
        y_true, prediction, labels=[0, 1]
    ).ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) else 0.0

    positive_likelihood_ratio = (
        float(sensitivity / (1.0 - specificity))
        if specificity < 1.0
        else float("inf")
    )
    negative_likelihood_ratio = (
        float((1.0 - sensitivity) / specificity)
        if specificity > 0.0
        else float("inf")
    )
    diagnostic_odds_ratio = (
        float(positive_likelihood_ratio / negative_likelihood_ratio)
        if np.isfinite(positive_likelihood_ratio)
        and np.isfinite(negative_likelihood_ratio)
        and negative_likelihood_ratio != 0
        else float("inf")
    )

    if np.unique(y_true).size >= 2:
        auroc = float(roc_auc_score(y_true, probability))
        auprc = float(average_precision_score(y_true, probability))
        safe_log_loss = float(
            log_loss(y_true, np.column_stack([1 - probability, probability]))
        )
    else:
        auroc = float("nan")
        auprc = float("nan")
        safe_log_loss = float("nan")

    ece, maximum_calibration_error = expected_calibration_error(
        y_true, probability, bins=calibration_bins
    )

    return {
        "sample_count": int(len(y_true)),
        "positive_count": int((y_true == 1).sum()),
        "negative_count": int((y_true == 0).sum()),
        "positive_rate": float((y_true == 1).mean()),
        "decision_threshold": float(decision_threshold),
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "f1_score": float(f1_score(y_true, prediction, zero_division=0)),
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "ppv_precision": ppv,
        "npv": npv,
        "matthews_correlation_coefficient": float(
            matthews_corrcoef(y_true, prediction)
        ),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": safe_log_loss,
        "expected_calibration_error": ece,
        "maximum_calibration_error": maximum_calibration_error,
        "positive_likelihood_ratio": positive_likelihood_ratio,
        "negative_likelihood_ratio": negative_likelihood_ratio,
        "diagnostic_odds_ratio": diagnostic_odds_ratio,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def calculate_threshold_views(
    y_true: np.ndarray,
    probability: np.ndarray,
    selected_threshold: float,
    default_threshold: float,
    calibration_bins: int,
) -> dict[str, dict[str, Any]]:
    return {
        "default_0_5": calculate_metrics(
            y_true,
            probability,
            default_threshold,
            calibration_bins,
        ),
        "validation_selected": calculate_metrics(
            y_true,
            probability,
            selected_threshold,
            calibration_bins,
        ),
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    probability: np.ndarray,
    decision_threshold: float,
    repeats: int,
    seed: int,
    calibration_bins: int = 10,
) -> dict[str, dict[str, float]]:
    if repeats <= 0:
        return {}

    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    probability = validate_probabilities(probability, len(y_true))
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "auroc": [],
        "auprc": [],
        "sensitivity_recall": [],
        "specificity": [],
        "f1_score": [],
        "brier_score": [],
    }

    for _ in progress_iter(
        range(repeats), total=repeats, description="Binary bootstrap",
        unit="sample", leave=False,
    ):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sampled_y = y_true[indices]
        sampled_probability = probability[indices]
        if np.unique(sampled_y).size < 2:
            continue
        metrics = calculate_metrics(
            sampled_y,
            sampled_probability,
            decision_threshold,
            calibration_bins,
        )
        for name in values:
            value = float(metrics[name])
            if np.isfinite(value):
                values[name].append(value)

    output: dict[str, dict[str, float]] = {}
    for name, metric_values in values.items():
        if not metric_values:
            output[name] = {
                "lower_95": float("nan"),
                "upper_95": float("nan"),
                "valid_repeats": 0,
            }
            continue
        array = np.asarray(metric_values, dtype=np.float64)
        output[name] = {
            "lower_95": float(np.quantile(array, 0.025)),
            "upper_95": float(np.quantile(array, 0.975)),
            "valid_repeats": int(len(array)),
        }
    return output
