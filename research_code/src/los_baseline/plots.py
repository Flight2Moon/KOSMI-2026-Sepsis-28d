from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def create_evaluation_figures(
    y_true: np.ndarray,
    probability: np.ndarray,
    decision_threshold: float,
    title_prefix: str,
    calibration_bins: int,
) -> dict[str, Any]:
    figures: dict[str, Any] = {}

    fpr, tpr, thresholds = roc_curve(y_true, probability)
    figure = plt.figure(figsize=(7, 6))
    axis = figure.add_subplot(111)
    auc_value = float(roc_auc_score(y_true, probability))
    axis.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    finite = np.isfinite(thresholds)
    if finite.any():
        finite_indices = np.where(finite)[0]
        selected_index = finite_indices[
            int(np.argmin(np.abs(thresholds[finite] - decision_threshold)))
        ]
        axis.scatter(
            fpr[selected_index],
            tpr[selected_index],
            label=f"Threshold={decision_threshold:.4f}",
        )
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(f"{title_prefix} ROC Curve")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figures["roc_curve"] = figure

    precision, recall, _ = precision_recall_curve(y_true, probability)
    figure = plt.figure(figsize=(7, 6))
    axis = figure.add_subplot(111)
    axis.plot(recall, precision, label="Precision-Recall")
    axis.axhline(float(np.mean(y_true)), linestyle="--", label="Prevalence")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(f"{title_prefix} Precision-Recall Curve")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figures["precision_recall_curve"] = figure

    observed, predicted = calibration_curve(
        y_true,
        probability,
        n_bins=calibration_bins,
        strategy="quantile",
    )
    figure = plt.figure(figsize=(7, 6))
    axis = figure.add_subplot(111)
    axis.plot(predicted, observed, marker="o", label="Model")
    axis.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    axis.set_xlabel("Mean Predicted Probability")
    axis.set_ylabel("Observed Positive Fraction")
    axis.set_title(f"{title_prefix} Calibration Curve")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figures["calibration_curve"] = figure

    prediction = (probability >= decision_threshold).astype(np.int32)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    figure = plt.figure(figsize=(6, 5))
    axis = figure.add_subplot(111)
    image = axis.imshow(matrix)
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(["Predicted 0", "Predicted 1"])
    axis.set_yticklabels(["Actual 0", "Actual 1"])
    axis.set_title(f"{title_prefix} Confusion Matrix")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figures["confusion_matrix"] = figure

    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(111)
    axis.hist(probability[y_true == 0], bins=40, alpha=0.5, label="Actual 0")
    axis.hist(probability[y_true == 1], bins=40, alpha=0.5, label="Actual 1")
    axis.axvline(decision_threshold, linestyle="--", label="Decision threshold")
    axis.set_xlabel("Predicted Positive Probability")
    axis.set_ylabel("Count")
    axis.set_title(f"{title_prefix} Probability Distribution")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figures["probability_distribution"] = figure

    observed, predicted = calibration_curve(
        y_true,
        probability,
        n_bins=calibration_bins,
        strategy="quantile",
    )
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    figure = plt.figure(figsize=(14, 6))
    calibration_axis = figure.add_subplot(121)
    calibration_axis.plot(predicted, observed, marker="o", label=title_prefix)
    calibration_axis.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    calibration_axis.set_xlabel("Mean Predicted Probability")
    calibration_axis.set_ylabel("Observed Positive Fraction")
    calibration_axis.set_title(f"Calibration Curve ({title_prefix})")
    calibration_axis.legend()
    calibration_axis.grid(True, alpha=0.25)

    roc_axis = figure.add_subplot(122)
    roc_axis.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    roc_axis.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    finite = np.isfinite(thresholds)
    if finite.any():
        finite_indices = np.where(finite)[0]
        selected_index = finite_indices[
            int(np.argmin(np.abs(thresholds[finite] - decision_threshold)))
        ]
        roc_axis.scatter(
            fpr[selected_index],
            tpr[selected_index],
            label=f"Best Thresh={decision_threshold:.4f}",
        )
    roc_axis.set_xlabel("False Positive Rate")
    roc_axis.set_ylabel("True Positive Rate")
    roc_axis.set_title(f"ROC Curve ({title_prefix})")
    roc_axis.legend()
    roc_axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figures["calibration_roc_summary"] = figure

    return figures


def save_figures(figures: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        figure.savefig(output_dir / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
