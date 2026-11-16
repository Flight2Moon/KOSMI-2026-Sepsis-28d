from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .utils import atomic_write_json
from .progress import progress_iter


@dataclass(frozen=True)
class BootstrapSettings:
    repeats: int = 500
    confidence_level: float = 0.95
    seed: int = 42
    minimum_successful_repeats: int = 100

    def __post_init__(self) -> None:
        if self.repeats < 1:
            raise ValueError("Bootstrap repeats must be positive.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one.")
        if self.minimum_successful_repeats < 1:
            raise ValueError("minimum_successful_repeats must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BootstrapInterval:
    metric: str
    point_estimate: float
    lower: float
    upper: float
    successful_repeats: int
    requested_repeats: int
    confidence_level: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_inputs(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    values = np.asarray(probabilities, dtype=np.float64)
    names = tuple(str(name) for name in class_names)
    if values.ndim != 2:
        raise ValueError(f"Multiclass probabilities must be two-dimensional: {values.shape}")
    if len(labels) != len(values):
        raise ValueError("y_true and probabilities must contain the same samples.")
    if values.shape[1] != len(names):
        raise ValueError(
            "Probability columns and class_names do not match: "
            f"{values.shape[1]} != {len(names)}"
        )
    if len(names) < 2:
        raise ValueError("At least two classes are required.")
    expected = set(range(len(names)))
    observed = set(np.unique(labels).tolist())
    invalid = sorted(observed - expected)
    if invalid:
        raise ValueError(f"Invalid multiclass labels: {invalid}")
    if not np.isfinite(values).all():
        raise ValueError("Multiclass probabilities contain non-finite values.")
    if ((values < -1e-8) | (values > 1.0 + 1e-8)).any():
        raise ValueError("Multiclass probabilities must be inside [0, 1].")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError(
            "Multiclass probabilities must sum to one. "
            f"Maximum error: {float(np.max(np.abs(row_sums - 1.0)))}"
        )
    values = values / row_sums[:, np.newaxis]
    return labels, values, names


def _resolve_predicted_classes(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predicted_classes: np.ndarray | None,
    num_classes: int,
) -> np.ndarray:
    if predicted_classes is None:
        return np.argmax(probabilities, axis=1).astype(np.int64)
    predictions = np.asarray(predicted_classes, dtype=np.int64).reshape(-1)
    if len(predictions) != len(labels):
        raise ValueError(
            "predicted_classes and y_true must contain the same samples."
        )
    invalid = sorted(set(np.unique(predictions).tolist()) - set(range(num_classes)))
    if invalid:
        raise ValueError(f"Invalid predicted classes: {invalid}")
    return predictions


def calculate_multiclass_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    predicted_classes: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    labels, values, names = _validate_inputs(y_true, probabilities, class_names)
    predictions = _resolve_predicted_classes(
        labels, values, predicted_classes, len(names)
    )
    one_hot = np.eye(len(names), dtype=np.float64)[labels]

    metrics: dict[str, float | int | None] = {
        "sample_count": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(labels, predictions, average="micro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "multiclass_brier": float(
            np.mean(np.sum((values - one_hot) ** 2, axis=1))
        ),
        "multiclass_log_loss": float(
            log_loss(labels, values, labels=list(range(len(names))))
        ),
    }

    valid_aurocs: list[float] = []
    valid_auprcs: list[float] = []
    for class_code, class_name in enumerate(names):
        binary = (labels == class_code).astype(np.int64)
        count = int(binary.sum())
        metrics[f"class_{class_code}_name"] = class_name
        metrics[f"class_{class_code}_count"] = count
        metrics[f"class_{class_code}_rate"] = float(count / max(len(labels), 1))
        if np.unique(binary).size == 2:
            auroc = float(roc_auc_score(binary, values[:, class_code]))
            auprc = float(average_precision_score(binary, values[:, class_code]))
            metrics[f"class_{class_code}_auroc"] = auroc
            metrics[f"class_{class_code}_auprc"] = auprc
            valid_aurocs.append(auroc)
            valid_auprcs.append(auprc)
        else:
            metrics[f"class_{class_code}_auroc"] = None
            metrics[f"class_{class_code}_auprc"] = None

    metrics["macro_auroc"] = (
        float(np.mean(valid_aurocs)) if len(valid_aurocs) == len(names) else None
    )
    metrics["macro_auprc"] = (
        float(np.mean(valid_auprcs)) if len(valid_auprcs) == len(names) else None
    )
    return metrics


def create_class_metrics_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    labels, values, names = _validate_inputs(y_true, probabilities, class_names)
    rows: list[dict[str, Any]] = []
    for class_code, class_name in enumerate(names):
        binary = (labels == class_code).astype(np.int64)
        row: dict[str, Any] = {
            "class_code": class_code,
            "class_name": class_name,
            "positive_count": int(binary.sum()),
            "negative_count": int((binary == 0).sum()),
            "positive_rate": float(binary.mean()),
            "random_auprc_baseline": float(binary.mean()),
            "auroc": np.nan,
            "auprc": np.nan,
        }
        if np.unique(binary).size == 2:
            row["auroc"] = float(roc_auc_score(binary, values[:, class_code]))
            row["auprc"] = float(
                average_precision_score(binary, values[:, class_code])
            )
        rows.append(row)
    return pd.DataFrame(rows)


def create_confusion_matrix_tables(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    predicted_classes: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels, values, names = _validate_inputs(y_true, probabilities, class_names)
    predictions = _resolve_predicted_classes(
        labels, values, predicted_classes, len(names)
    )
    raw = confusion_matrix(labels, predictions, labels=list(range(len(names))))
    denominators = raw.sum(axis=1, keepdims=True)
    normalized = np.divide(
        raw,
        denominators,
        out=np.zeros_like(raw, dtype=np.float64),
        where=denominators > 0,
    )
    raw_frame = pd.DataFrame(raw, index=names, columns=names)
    normalized_frame = pd.DataFrame(normalized, index=names, columns=names)
    raw_frame.index.name = "actual"
    raw_frame.columns.name = "predicted"
    normalized_frame.index.name = "actual"
    normalized_frame.columns.name = "predicted"
    return raw_frame, normalized_frame


def create_calibration_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    labels, values, names = _validate_inputs(y_true, probabilities, class_names)
    if n_bins < 2:
        raise ValueError("n_bins must be at least two.")
    rows: list[dict[str, Any]] = []
    for class_code, class_name in enumerate(names):
        binary = (labels == class_code).astype(np.int64)
        if np.unique(binary).size < 2:
            continue
        observed, predicted = calibration_curve(
            binary,
            values[:, class_code],
            n_bins=n_bins,
            strategy="quantile",
        )
        for bin_index, (mean_prediction, observed_fraction) in enumerate(
            zip(predicted, observed), start=1
        ):
            rows.append(
                {
                    "class_code": class_code,
                    "class_name": class_name,
                    "bin_index": bin_index,
                    "mean_predicted_probability": float(mean_prediction),
                    "observed_positive_fraction": float(observed_fraction),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_metric_values(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    settings: BootstrapSettings,
    predicted_classes: np.ndarray | None = None,
) -> tuple[dict[str, list[float]], int]:
    labels, values, names = _validate_inputs(y_true, probabilities, class_names)
    predictions = (
        None
        if predicted_classes is None
        else _resolve_predicted_classes(labels, values, predicted_classes, len(names))
    )
    rng = np.random.default_rng(settings.seed)
    collected: dict[str, list[float]] = {}
    successful = 0
    expected_classes = set(range(len(names)))
    try:
        workers = max(1, int(os.environ.get("EXP123_ASSIGNED_CPU_THREADS", "1")))
    except ValueError:
        workers = 1

    def evaluate_indices(indices: np.ndarray) -> dict[str, float] | None:
        sampled_labels = labels[indices]
        if set(np.unique(sampled_labels).tolist()) != expected_classes:
            return None
        sampled_probabilities = values[indices]
        sampled_predictions = None if predictions is None else predictions[indices]
        current = calculate_multiclass_metrics(
            sampled_labels, sampled_probabilities, names, predicted_classes=sampled_predictions
        )
        return {
            metric: float(value)
            for metric, value in current.items()
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)
        }

    # Preserve the exact legacy bootstrap sample sequence by generating every
    # resample index from the single seeded RNG in the parent thread.  Metric
    # evaluation only is parallelized, so confidence intervals remain identical
    # to the sequential implementation while CPU utilization scales with the
    # frozen per-worker thread budget.
    if workers <= 1:
        iterator = progress_iter(
            range(settings.repeats), total=settings.repeats,
            description="Multiclass bootstrap", unit="sample", leave=False,
        )
        for _ in iterator:
            result = evaluate_indices(rng.integers(0, len(labels), size=len(labels), endpoint=False))
            if result is None:
                continue
            for metric, value in result.items():
                collected.setdefault(metric, []).append(value)
            successful += 1
        return collected, successful

    batch_size = max(workers, workers * 2)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bootstrap-metric") as pool:
        done = 0
        while done < settings.repeats:
            count = min(batch_size, settings.repeats - done)
            indices_batch = [
                rng.integers(0, len(labels), size=len(labels), endpoint=False)
                for _ in range(count)
            ]
            for result in pool.map(evaluate_indices, indices_batch):
                if result is None:
                    continue
                for metric, value in result.items():
                    collected.setdefault(metric, []).append(value)
                successful += 1
            done += count
    return collected, successful

def bootstrap_multiclass_confidence_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    settings: BootstrapSettings,
    predicted_classes: np.ndarray | None = None,
) -> list[BootstrapInterval]:
    labels, values, names = _validate_inputs(y_true, probabilities, class_names)
    predictions = (
        None
        if predicted_classes is None
        else _resolve_predicted_classes(labels, values, predicted_classes, len(names))
    )
    point = calculate_multiclass_metrics(
        labels, values, names, predicted_classes=predictions
    )
    collected, successful = _bootstrap_metric_values(
        labels,
        values,
        names,
        settings,
        predicted_classes=predictions,
    )
    if successful < min(settings.minimum_successful_repeats, settings.repeats):
        raise RuntimeError(
            "Too few valid multiclass bootstrap samples: "
            f"{successful}/{settings.repeats}"
        )
    alpha = 1.0 - settings.confidence_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    intervals: list[BootstrapInterval] = []
    for metric, samples in sorted(collected.items()):
        point_value = point.get(metric)
        if not isinstance(point_value, (int, float, np.integer, np.floating)):
            continue
        sample_array = np.asarray(samples, dtype=np.float64)
        intervals.append(
            BootstrapInterval(
                metric=metric,
                point_estimate=float(point_value),
                lower=float(np.percentile(sample_array, lower_q)),
                upper=float(np.percentile(sample_array, upper_q)),
                successful_repeats=int(len(sample_array)),
                requested_repeats=settings.repeats,
                confidence_level=settings.confidence_level,
            )
        )
    return intervals


def _plot_confusion_matrix(
    matrix: pd.DataFrame,
    title: str,
    value_format: str,
) -> plt.Figure:
    figure = plt.figure(figsize=(7, 6))
    axis = figure.add_subplot(111)
    image = axis.imshow(matrix.to_numpy())
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_yticks(range(len(matrix.index)))
    axis.set_xticklabels(matrix.columns, rotation=30, ha="right")
    axis.set_yticklabels(matrix.index)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title(title)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            axis.text(
                column,
                row,
                format(matrix.iloc[row, column], value_format),
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    return figure


def _plot_calibration(
    calibration: pd.DataFrame,
    title: str,
) -> plt.Figure:
    figure = plt.figure(figsize=(7, 6))
    axis = figure.add_subplot(111)
    for class_name, group in calibration.groupby("class_name", sort=False):
        axis.plot(
            group["mean_predicted_probability"],
            group["observed_positive_fraction"],
            marker="o",
            label=str(class_name),
        )
    axis.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed positive fraction")
    axis.set_title(title)
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    return figure


def _plot_ovr_curves(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    curve_type: str,
    title: str,
) -> plt.Figure:
    labels, values, names = _validate_inputs(y_true, probabilities, class_names)
    figure = plt.figure(figsize=(7, 6))
    axis = figure.add_subplot(111)
    for class_code, class_name in enumerate(names):
        binary = (labels == class_code).astype(np.int64)
        if np.unique(binary).size < 2:
            continue
        if curve_type == "roc":
            x_values, y_values, _ = roc_curve(binary, values[:, class_code])
            score = roc_auc_score(binary, values[:, class_code])
            label = f"{class_name} AUROC={score:.3f}"
            axis.set_xlabel("False Positive Rate")
            axis.set_ylabel("True Positive Rate")
        elif curve_type == "pr":
            y_values, x_values, _ = precision_recall_curve(
                binary, values[:, class_code]
            )
            score = average_precision_score(binary, values[:, class_code])
            label = f"{class_name} AUPRC={score:.3f}"
            axis.set_xlabel("Recall")
            axis.set_ylabel("Precision")
        else:
            raise ValueError(f"Unknown curve_type: {curve_type}")
        axis.plot(x_values, y_values, label=label)
    if curve_type == "roc":
        axis.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    axis.set_title(title)
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    return figure


def write_multiclass_evaluation_artifacts(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    output_dir: Path,
    split_name: str,
    model_name: str,
    calibration_bins: int = 10,
    bootstrap_settings: BootstrapSettings | None = None,
    save_plots: bool = True,
    predicted_classes: np.ndarray | None = None,
    decision_rule: str = "probability_argmax",
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = calculate_multiclass_metrics(
        y_true, probabilities, class_names, predicted_classes=predicted_classes
    )
    atomic_write_json(output_dir / "multiclass_metrics.json", metrics)

    class_metrics = create_class_metrics_table(y_true, probabilities, class_names)
    class_metrics.to_csv(output_dir / "class_metrics.csv", index=False)

    raw, normalized = create_confusion_matrix_tables(
        y_true,
        probabilities,
        class_names,
        predicted_classes=predicted_classes,
    )
    raw.to_csv(output_dir / "confusion_matrix_raw.csv")
    normalized.to_csv(output_dir / "confusion_matrix_normalized.csv")

    calibration = create_calibration_table(
        y_true, probabilities, class_names, n_bins=calibration_bins
    )
    calibration.to_csv(output_dir / "calibration_curve.csv", index=False)

    bootstrap_payload: list[dict[str, Any]] = []
    if bootstrap_settings is not None:
        intervals = bootstrap_multiclass_confidence_intervals(
            y_true,
            probabilities,
            class_names,
            bootstrap_settings,
            predicted_classes=predicted_classes,
        )
        bootstrap_payload = [interval.to_dict() for interval in intervals]
        pd.DataFrame(bootstrap_payload).to_csv(
            output_dir / "bootstrap_confidence_intervals.csv", index=False
        )
        atomic_write_json(
            output_dir / "bootstrap_confidence_intervals.json",
            {
                "settings": bootstrap_settings.to_dict(),
                "intervals": bootstrap_payload,
            },
        )

    if save_plots:
        figures = {
            "confusion_matrix_raw": _plot_confusion_matrix(
                raw,
                f"{model_name} {split_name} confusion matrix",
                ",d",
            ),
            "confusion_matrix_normalized": _plot_confusion_matrix(
                normalized,
                f"{model_name} {split_name} row-normalized confusion matrix",
                ".3f",
            ),
            "calibration_curve_ovr": _plot_calibration(
                calibration,
                f"{model_name} {split_name} one-vs-rest calibration",
            ),
            "roc_curve_ovr": _plot_ovr_curves(
                y_true,
                probabilities,
                class_names,
                "roc",
                f"{model_name} {split_name} one-vs-rest ROC",
            ),
            "precision_recall_curve_ovr": _plot_ovr_curves(
                y_true,
                probabilities,
                class_names,
                "pr",
                f"{model_name} {split_name} one-vs-rest precision-recall",
            ),
        }
        figure_dir = output_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            figure.savefig(
                figure_dir / f"{name}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(figure)

    summary = {
        "split": split_name,
        "model_name": model_name,
        "class_names": list(class_names),
        "metrics": metrics,
        "bootstrap_enabled": bootstrap_settings is not None,
        "bootstrap_interval_count": len(bootstrap_payload),
        "calibration_bins": int(calibration_bins),
        "plots_saved": bool(save_plots),
        "decision_rule": str(decision_rule),
        "custom_predicted_classes": predicted_classes is not None,
    }
    atomic_write_json(output_dir / "multiclass_evaluation_manifest.json", summary)
    return summary


def probabilities_from_prediction_frame(
    prediction_frame: pd.DataFrame,
    num_classes: int,
) -> np.ndarray:
    columns = [f"prob_class_{class_code}" for class_code in range(num_classes)]
    missing = sorted(set(columns) - set(prediction_frame.columns))
    if missing:
        raise KeyError(f"Prediction frame is missing probability columns: {missing}")
    return prediction_frame.loc[:, columns].to_numpy(dtype=np.float64)
