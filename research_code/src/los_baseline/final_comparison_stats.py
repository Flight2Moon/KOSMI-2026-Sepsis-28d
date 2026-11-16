from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from .evaluation_multiclass import calculate_multiclass_metrics
from .progress import progress_iter


LOWER_IS_BETTER = {"multiclass_brier", "multiclass_log_loss"}


class ComparisonSettingsProtocol(Protocol):
    bootstrap_repeats: int
    confidence_level: float
    seed: int
    fdr_method: str
    fdr_alpha: float
    minimum_successful_bootstrap_repeats: int
    comparison_metrics: tuple[str, ...]


@dataclass(frozen=True)
class DelongTestResult:
    class_code: int
    class_name: str
    flat_auroc: float
    pipeline_auroc: float
    difference_pipeline_minus_flat: float
    standard_error: float
    z_statistic: float
    p_raw: float
    sample_count: int
    positive_count: int
    negative_count: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compute_midrank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[index:end] = 0.5 * (index + end - 1) + 1.0
        index = end
    output = np.empty(len(values), dtype=np.float64)
    output[order] = ranks
    return output


def _fast_delong(
    predictions_sorted_transposed: np.ndarray,
    positive_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(predictions_sorted_transposed, dtype=np.float64)
    if predictions.ndim != 2:
        raise ValueError("DeLong predictions must be two-dimensional.")
    model_count, sample_count = predictions.shape
    m = int(positive_count)
    n = sample_count - m
    if m < 1 or n < 1:
        raise ValueError("DeLong test requires positive and negative samples.")
    positive = predictions[:, :m]
    negative = predictions[:, m:]
    tx = np.empty((model_count, m), dtype=np.float64)
    ty = np.empty((model_count, n), dtype=np.float64)
    tz = np.empty((model_count, m + n), dtype=np.float64)
    for model_index in range(model_count):
        tx[model_index] = _compute_midrank(positive[model_index])
        ty[model_index] = _compute_midrank(negative[model_index])
        tz[model_index] = _compute_midrank(predictions[model_index])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01, bias=False))
    sy = np.atleast_2d(np.cov(v10, bias=False))
    covariance = sx / m + sy / n
    return aucs, covariance


def delong_roc_test(
    y_true_binary: np.ndarray,
    flat_probability: np.ndarray,
    pipeline_probability: np.ndarray,
    *,
    class_code: int,
    class_name: str,
) -> DelongTestResult:
    labels = np.asarray(y_true_binary, dtype=np.int64).reshape(-1)
    flat = np.asarray(flat_probability, dtype=np.float64).reshape(-1)
    pipeline = np.asarray(pipeline_probability, dtype=np.float64).reshape(-1)
    if len(labels) != len(flat) or len(labels) != len(pipeline):
        raise ValueError("DeLong inputs differ in length.")
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    if positive_count < 2 or negative_count < 2:
        return DelongTestResult(
            class_code=class_code,
            class_name=class_name,
            flat_auroc=float("nan"),
            pipeline_auroc=float("nan"),
            difference_pipeline_minus_flat=float("nan"),
            standard_error=float("nan"),
            z_statistic=float("nan"),
            p_raw=float("nan"),
            sample_count=len(labels),
            positive_count=positive_count,
            negative_count=negative_count,
            status="insufficient_data",
        )
    order = np.argsort(-labels, kind="mergesort")
    predictions = np.vstack([flat[order], pipeline[order]])
    aucs, covariance = _fast_delong(predictions, positive_count)
    contrast = np.array([-1.0, 1.0], dtype=np.float64)
    variance = float(contrast @ covariance @ contrast.T)
    variance = max(variance, 0.0)
    difference = float(aucs[1] - aucs[0])
    if variance <= np.finfo(np.float64).eps:
        if abs(difference) <= 1e-15:
            standard_error = 0.0
            z_statistic = 0.0
            p_value = 1.0
        else:
            standard_error = 0.0
            z_statistic = float(np.sign(difference) * np.inf)
            p_value = 0.0
    else:
        standard_error = float(np.sqrt(variance))
        z_statistic = float(difference / standard_error)
        p_value = float(2.0 * norm.sf(abs(z_statistic)))
    return DelongTestResult(
        class_code=class_code,
        class_name=class_name,
        flat_auroc=float(aucs[0]),
        pipeline_auroc=float(aucs[1]),
        difference_pipeline_minus_flat=difference,
        standard_error=standard_error,
        z_statistic=z_statistic,
        p_raw=p_value,
        sample_count=len(labels),
        positive_count=positive_count,
        negative_count=negative_count,
        status="completed",
    )


def _cluster_index_groups(
    metadata: pd.DataFrame,
) -> tuple[list[np.ndarray], str, int]:
    if "subject_id" in metadata.columns and metadata["subject_id"].notna().any():
        cluster_column = "subject_id"
        cluster_values = metadata["subject_id"].astype(str).to_numpy()
    else:
        cluster_column = "source_row_index"
        cluster_values = metadata["source_row_index"].astype(str).to_numpy()
    groups: list[np.ndarray] = []
    for value in pd.unique(cluster_values):
        groups.append(np.flatnonzero(cluster_values == value))
    return groups, cluster_column, len(groups)


def paired_cluster_bootstrap_differences(
    *,
    y_true: np.ndarray,
    flat_probabilities: np.ndarray,
    flat_predicted_classes: np.ndarray,
    pipeline_probabilities: np.ndarray,
    pipeline_predicted_classes: np.ndarray,
    metadata: pd.DataFrame,
    class_names: Sequence[str],
    settings: ComparisonSettingsProtocol,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    flat_prob = np.asarray(flat_probabilities, dtype=np.float64)
    pipeline_prob = np.asarray(pipeline_probabilities, dtype=np.float64)
    flat_pred = np.asarray(flat_predicted_classes, dtype=np.int64).reshape(-1)
    pipeline_pred = np.asarray(pipeline_predicted_classes, dtype=np.int64).reshape(-1)
    if not (
        len(labels)
        == len(flat_prob)
        == len(pipeline_prob)
        == len(flat_pred)
        == len(pipeline_pred)
        == len(metadata)
    ):
        raise ValueError("Paired bootstrap inputs differ in length.")

    flat_point = calculate_multiclass_metrics(
        labels, flat_prob, class_names, predicted_classes=flat_pred
    )
    pipeline_point = calculate_multiclass_metrics(
        labels, pipeline_prob, class_names, predicted_classes=pipeline_pred
    )
    clusters, cluster_column, cluster_count = _cluster_index_groups(metadata)
    rng = np.random.default_rng(settings.seed)
    collected: dict[str, list[float]] = {
        metric: [] for metric in settings.comparison_metrics
    }
    successful = 0
    for _ in progress_iter(
        range(settings.bootstrap_repeats), total=settings.bootstrap_repeats,
        description="Paired cluster bootstrap", unit="sample", leave=False,
    ):
        sampled_cluster_positions = rng.integers(
            0, cluster_count, size=cluster_count
        )
        sampled_indices = np.concatenate(
            [clusters[position] for position in sampled_cluster_positions]
        )
        sampled_y = labels[sampled_indices]
        if set(np.unique(sampled_y).tolist()) != set(range(len(class_names))):
            continue
        sampled_flat_prob = flat_prob[sampled_indices]
        sampled_pipeline_prob = pipeline_prob[sampled_indices]
        sampled_flat_pred = flat_pred[sampled_indices]
        sampled_pipeline_pred = pipeline_pred[sampled_indices]
        flat_metrics = calculate_multiclass_metrics(
            sampled_y,
            sampled_flat_prob,
            class_names,
            predicted_classes=sampled_flat_pred,
        )
        pipeline_metrics = calculate_multiclass_metrics(
            sampled_y,
            sampled_pipeline_prob,
            class_names,
            predicted_classes=sampled_pipeline_pred,
        )
        valid = True
        for metric in settings.comparison_metrics:
            flat_value = flat_metrics.get(metric)
            pipeline_value = pipeline_metrics.get(metric)
            if flat_value is None or pipeline_value is None:
                valid = False
                break
            if not np.isfinite(float(flat_value)) or not np.isfinite(
                float(pipeline_value)
            ):
                valid = False
                break
        if not valid:
            continue
        for metric in settings.comparison_metrics:
            flat_value = float(flat_metrics[metric])
            pipeline_value = float(pipeline_metrics[metric])
            improvement = (
                flat_value - pipeline_value
                if metric in LOWER_IS_BETTER
                else pipeline_value - flat_value
            )
            collected[metric].append(float(improvement))
        successful += 1

    required = min(
        settings.minimum_successful_bootstrap_repeats,
        settings.bootstrap_repeats,
    )
    if successful < required:
        raise RuntimeError(
            "Too few successful paired bootstrap samples: "
            f"{successful}/{settings.bootstrap_repeats}"
        )
    alpha = 1.0 - settings.confidence_level
    rows: list[dict[str, Any]] = []
    for metric in settings.comparison_metrics:
        flat_value = float(flat_point[metric])
        pipeline_value = float(pipeline_point[metric])
        raw_difference = pipeline_value - flat_value
        point_improvement = (
            flat_value - pipeline_value
            if metric in LOWER_IS_BETTER
            else raw_difference
        )
        samples = np.asarray(collected[metric], dtype=np.float64)
        lower = float(np.quantile(samples, alpha / 2.0))
        upper = float(np.quantile(samples, 1.0 - alpha / 2.0))
        less_or_equal = int((samples <= 0.0).sum())
        greater_or_equal = int((samples >= 0.0).sum())
        p_raw = float(
            min(
                1.0,
                2.0
                * min(
                    (less_or_equal + 1.0) / (len(samples) + 1.0),
                    (greater_or_equal + 1.0) / (len(samples) + 1.0),
                ),
            )
        )
        rows.append(
            {
                "test_type": "paired_cluster_bootstrap",
                "metric": metric,
                "flat_point_estimate": flat_value,
                "pipeline_point_estimate": pipeline_value,
                "difference_pipeline_minus_flat": raw_difference,
                "improvement_positive_favors_pipeline": point_improvement,
                "improvement_ci_lower": lower,
                "improvement_ci_upper": upper,
                "p_raw": p_raw,
                "successful_repeats": int(len(samples)),
                "requested_repeats": settings.bootstrap_repeats,
                "confidence_level": settings.confidence_level,
                "favorable_direction": (
                    "lower" if metric in LOWER_IS_BETTER else "higher"
                ),
            }
        )
    manifest = {
        "bootstrap_unit": cluster_column,
        "cluster_count": cluster_count,
        "sample_count": len(labels),
        "successful_repeats": successful,
        "requested_repeats": settings.bootstrap_repeats,
        "seed": settings.seed,
        "paired_resampling": True,
    }
    return pd.DataFrame(rows), manifest


def mcnemar_correctness_test(
    y_true: np.ndarray,
    flat_predicted_classes: np.ndarray,
    pipeline_predicted_classes: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    flat = np.asarray(flat_predicted_classes, dtype=np.int64).reshape(-1)
    pipeline = np.asarray(pipeline_predicted_classes, dtype=np.int64).reshape(-1)
    if len(labels) != len(flat) or len(labels) != len(pipeline):
        raise ValueError("McNemar inputs differ in length.")
    flat_correct = flat == labels
    pipeline_correct = pipeline == labels
    both_incorrect = int((~flat_correct & ~pipeline_correct).sum())
    flat_only_correct = int((flat_correct & ~pipeline_correct).sum())
    pipeline_only_correct = int((~flat_correct & pipeline_correct).sum())
    both_correct = int((flat_correct & pipeline_correct).sum())
    table = np.array(
        [[both_incorrect, pipeline_only_correct], [flat_only_correct, both_correct]],
        dtype=np.int64,
    )
    discordant = flat_only_correct + pipeline_only_correct
    exact = discordant < 25
    result = mcnemar(table, exact=exact, correction=not exact)
    return {
        "test_type": "mcnemar_correctness",
        "metric": "overall_correctness",
        "both_incorrect": both_incorrect,
        "flat_only_correct": flat_only_correct,
        "pipeline_only_correct": pipeline_only_correct,
        "both_correct": both_correct,
        "discordant_pairs": discordant,
        "statistic": float(result.statistic),
        "p_raw": float(result.pvalue),
        "exact": bool(exact),
        "correction": bool(not exact),
        "difference_accuracy_pipeline_minus_flat": float(
            pipeline_correct.mean() - flat_correct.mean()
        ),
    }


def _apply_fdr(
    tests: pd.DataFrame,
    *,
    method: str,
    alpha: float,
    family_id: str,
) -> pd.DataFrame:
    output = tests.copy()
    output["p_fdr"] = np.nan
    output["reject_fdr"] = False
    output["fdr_method"] = method
    output["fdr_alpha"] = alpha
    output["fdr_family_id"] = family_id
    valid = pd.to_numeric(output["p_raw"], errors="coerce").notna()
    if valid.any():
        rejected, corrected, _, _ = multipletests(
            output.loc[valid, "p_raw"].astype(float).to_numpy(),
            alpha=alpha,
            method=method,
        )
        output.loc[valid, "p_fdr"] = corrected
        output.loc[valid, "reject_fdr"] = rejected
    output["fdr_family_size"] = int(valid.sum())
    return output
