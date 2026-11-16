from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import (
    bootstrap_confidence_intervals,
    calculate_threshold_views,
)
from .plots import create_evaluation_figures, save_figures
from .tensorboard_logging import (
    log_figure_dictionary,
    log_metric_dictionary,
)
from .utils import atomic_write_json


def _flatten_metrics(
    split_name: str,
    views: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view_name, metrics in views.items():
        rows.append(
            {
                "split": split_name,
                "threshold_view": view_name,
                **metrics,
            }
        )
    return rows


def evaluate_and_save_split(
    split_name: str,
    y_true: np.ndarray,
    probability: np.ndarray,
    selected_threshold: float,
    default_threshold: float,
    calibration_bins: int,
    output_dir: Path,
    prediction_metadata: pd.DataFrame,
    writer: Any,
    bootstrap_repeats: int = 0,
    bootstrap_seed: int = 42,
    save_plots: bool = True,
    inference_seconds: float | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    views = calculate_threshold_views(
        y_true,
        probability,
        selected_threshold,
        default_threshold,
        calibration_bins,
    )

    confidence_intervals = bootstrap_confidence_intervals(
        y_true,
        probability,
        selected_threshold,
        bootstrap_repeats,
        bootstrap_seed,
        calibration_bins,
    )

    payload = {
        "split": split_name,
        "threshold_views": views,
        "bootstrap_95_confidence_intervals": confidence_intervals,
        "timing": {
            "inference_seconds": (
                float(inference_seconds)
                if inference_seconds is not None
                else None
            ),
            "evaluation_seconds": None,
        },
    }
    atomic_write_json(output_dir / "metrics.json", payload)
    pd.DataFrame(_flatten_metrics(split_name, views)).to_csv(
        output_dir / "metrics.csv",
        index=False,
    )

    prediction_output = prediction_metadata.copy()
    prediction_output["y_true"] = np.asarray(y_true, dtype=np.int32)
    prediction_output["positive_probability"] = probability
    prediction_output["prediction_default_0_5"] = (
        probability >= default_threshold
    ).astype(np.int32)
    prediction_output["prediction_validation_selected"] = (
        probability >= selected_threshold
    ).astype(np.int32)
    prediction_output.to_csv(
        output_dir / "predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    figures = {}
    if save_plots:
        figures = create_evaluation_figures(
            y_true,
            probability,
            selected_threshold,
            split_name,
            calibration_bins,
        )
        save_figures(figures, output_dir / "figures")

    for view_name, metrics in views.items():
        log_metric_dictionary(
            writer,
            split_name,
            view_name,
            metrics,
            step=0,
        )
    if figures:
        log_figure_dictionary(writer, split_name, figures, step=0)

    for figure in figures.values():
        plt.close(figure)

    return payload


def evaluate_external_audit_subgroups(
    prediction_metadata: pd.DataFrame,
    y_true: np.ndarray,
    probability: np.ndarray,
    selected_threshold: float,
    default_threshold: float,
    calibration_bins: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate external performance by harmonizer fallback burden group."""
    group_column = "fallback_group"

    if group_column not in prediction_metadata.columns:
        return {
            "status": "not_available",
            "reason": "fallback_group was not attached to eICU rows",
        }

    metadata = prediction_metadata.reset_index(drop=True)
    labels = np.asarray(y_true, dtype=np.int32).reshape(-1)
    probabilities = np.asarray(probability, dtype=np.float64).reshape(-1)

    if not (
        len(metadata) == len(labels) == len(probabilities)
    ):
        raise ValueError(
            "External audit subgroup inputs have inconsistent row counts."
        )

    rows: list[dict[str, Any]] = []
    normalized_groups = metadata[group_column].astype("string").fillna("unknown")

    for group_name in sorted(normalized_groups.unique().tolist()):
        mask = normalized_groups.eq(group_name).to_numpy()
        views = calculate_threshold_views(
            labels[mask],
            probabilities[mask],
            selected_threshold,
            default_threshold,
            calibration_bins,
        )

        for threshold_view, metrics in views.items():
            rows.append(
                {
                    "fallback_group": str(group_name),
                    "threshold_view": threshold_view,
                    **metrics,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fallback_subgroup_metrics.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)

    payload = {
        "status": "completed",
        "group_column": group_column,
        "group_count": int(len(set(normalized_groups.tolist()))),
        "output_csv": str(output_path),
    }
    atomic_write_json(
        output_dir / "fallback_subgroup_metrics.json",
        payload,
    )
    return payload
