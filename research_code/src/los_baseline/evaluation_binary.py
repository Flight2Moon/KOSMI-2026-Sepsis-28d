from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from .metrics import (
    bootstrap_confidence_intervals,
    calculate_threshold_views,
    validate_probabilities,
)
from .plots import create_evaluation_figures, save_figures
from .utils import atomic_write_json


@dataclass(frozen=True)
class BinaryBootstrapSettings:
    repeats: int = 500
    seed: int = 42
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.repeats < 0:
            raise ValueError("repeats must be non-negative.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probabilities_from_binary_prediction_frame(frame: pd.DataFrame) -> np.ndarray:
    if "positive_probability" not in frame.columns:
        raise KeyError("Binary prediction frame is missing positive_probability.")
    return validate_probabilities(
        frame["positive_probability"].to_numpy(dtype=np.float64),
        len(frame),
    )


def _threshold_metrics_frame(
    split_name: str,
    threshold_views: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for view_name, metrics in threshold_views.items():
        rows.append(
            {
                "split": split_name,
                "threshold_view": view_name,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _confusion_tables(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    raw = pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_values = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums > 0,
    )
    normalized = pd.DataFrame(
        normalized_values,
        index=raw.index,
        columns=raw.columns,
    )
    return raw, normalized


def _curve_tables(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    calibration_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fpr, tpr, roc_thresholds = roc_curve(y_true, probabilities)
    roc_frame = pd.DataFrame(
        {
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "threshold": roc_thresholds,
        }
    )
    precision, recall, pr_thresholds = precision_recall_curve(
        y_true, probabilities
    )
    pr_frame = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.append(pr_thresholds, np.nan),
        }
    )
    observed, predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=calibration_bins,
        strategy="quantile",
    )
    calibration_frame = pd.DataFrame(
        {
            "mean_predicted_probability": predicted,
            "observed_positive_fraction": observed,
        }
    )
    return roc_frame, pr_frame, calibration_frame


def _bootstrap_frame(
    confidence_intervals: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric, values in confidence_intervals.items():
        rows.append({"metric": metric, **values})
    return pd.DataFrame(rows)


def write_binary_evaluation_artifacts(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    selected_threshold: float,
    default_threshold: float,
    output_dir: Path,
    split_name: str,
    model_name: str,
    positive_class_name: str,
    calibration_bins: int,
    bootstrap_settings: BinaryBootstrapSettings | None = None,
    save_plots: bool = True,
) -> dict[str, Any]:
    """Write detailed binary evaluation artifacts without changing predictions."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probability = validate_probabilities(probabilities, len(labels))
    if np.unique(labels).size != 2:
        raise ValueError(
            f"Binary evaluation requires both classes for {split_name}."
        )

    threshold_views = calculate_threshold_views(
        labels,
        probability,
        selected_threshold,
        default_threshold,
        calibration_bins,
    )
    confidence_intervals: dict[str, dict[str, float]] = {}
    if bootstrap_settings is not None and bootstrap_settings.repeats > 0:
        confidence_intervals = bootstrap_confidence_intervals(
            labels,
            probability,
            selected_threshold,
            bootstrap_settings.repeats,
            bootstrap_settings.seed,
            calibration_bins,
        )

    raw_matrix, normalized_matrix = _confusion_tables(
        labels,
        probability,
        selected_threshold,
    )
    roc_frame, pr_frame, calibration_frame = _curve_tables(
        labels,
        probability,
        calibration_bins,
    )

    _threshold_metrics_frame(split_name, threshold_views).to_csv(
        output_dir / "threshold_metrics.csv",
        index=False,
    )
    raw_matrix.to_csv(output_dir / "confusion_matrix_raw.csv")
    normalized_matrix.to_csv(output_dir / "confusion_matrix_normalized.csv")
    roc_frame.to_csv(output_dir / "roc_curve.csv", index=False)
    pr_frame.to_csv(output_dir / "precision_recall_curve.csv", index=False)
    calibration_frame.to_csv(output_dir / "calibration_curve.csv", index=False)

    if confidence_intervals:
        _bootstrap_frame(confidence_intervals).to_csv(
            output_dir / "bootstrap_confidence_intervals.csv",
            index=False,
        )
        atomic_write_json(
            output_dir / "bootstrap_confidence_intervals.json",
            confidence_intervals,
        )

    payload = {
        "split": split_name,
        "model_name": model_name,
        "positive_class_name": positive_class_name,
        "selected_threshold_source": "validation_youden",
        "selected_threshold": float(selected_threshold),
        "default_threshold": float(default_threshold),
        "threshold_views": threshold_views,
        "bootstrap_95_confidence_intervals": confidence_intervals,
        "bootstrap_settings": (
            bootstrap_settings.to_dict() if bootstrap_settings is not None else None
        ),
    }
    atomic_write_json(output_dir / "binary_metrics.json", payload)

    manifest = {
        "artifact_type": "binary_evaluation",
        "split": split_name,
        "model_name": model_name,
        "sample_count": int(len(labels)),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
        "selected_threshold": float(selected_threshold),
        "threshold_free_primary_metrics": ["auroc", "auprc", "brier_score"],
        "threshold_dependent_metrics_are_secondary": True,
        "prediction_file_modified": False,
    }
    atomic_write_json(output_dir / "binary_evaluation_manifest.json", manifest)

    figures: dict[str, Any] = {}
    if save_plots:
        figures = create_evaluation_figures(
            labels,
            probability,
            selected_threshold,
            f"{model_name} {split_name}",
            calibration_bins,
        )
        save_figures(figures, output_dir / "figures")
    for figure in figures.values():
        plt.close(figure)
    return payload
