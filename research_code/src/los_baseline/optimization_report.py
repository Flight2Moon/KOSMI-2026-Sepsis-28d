from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .optimization_registry import OptimizationRegistry
from .utils import atomic_write_json


def _flatten_trial(trial: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        key: value
        for key, value in trial.items()
        if key not in {"tunable_parameters", "model_config", "fold_metrics", "aggregate_metrics"}
    }
    for key, value in (trial.get("tunable_parameters") or {}).items():
        row[f"param__{key}"] = value
    for key, value in (trial.get("aggregate_metrics") or {}).items():
        row[f"metric__{key}"] = value
    return row


def _parameter_importance(completed: pd.DataFrame) -> pd.DataFrame:
    parameter_columns = [column for column in completed.columns if column.startswith("param__")]
    rows: list[dict[str, Any]] = []
    if "selection_score" not in completed.columns:
        return pd.DataFrame(columns=["parameter", "importance", "method", "sample_count"])
    for column in parameter_columns:
        values = completed[column]
        valid = values.notna() & completed["selection_score"].notna()
        subset = completed.loc[valid, [column, "selection_score"]]
        if len(subset) < 3 or subset[column].nunique(dropna=True) < 2:
            continue
        numeric = pd.to_numeric(subset[column], errors="coerce")
        if numeric.notna().sum() == len(subset):
            correlation = numeric.corr(subset["selection_score"], method="spearman")
            importance = abs(float(correlation)) if pd.notna(correlation) else 0.0
            method = "absolute_spearman"
        else:
            grouped = subset.groupby(column, dropna=False)["selection_score"].mean()
            importance = float(grouped.max() - grouped.min())
            method = "group_mean_range"
        rows.append(
            {
                "parameter": column.removeprefix("param__"),
                "importance": importance,
                "method": method,
                "sample_count": int(len(subset)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["parameter", "importance", "method", "sample_count"])
    return pd.DataFrame(rows).sort_values("importance", ascending=False, kind="mergesort")


def _plot_trial_scores(frame: pd.DataFrame, output_path: Path) -> None:
    completed = frame.loc[frame["status"] == "completed"].copy()
    if completed.empty:
        return
    completed = completed.reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(np.arange(1, len(completed) + 1), completed["selection_score"], marker="o")
    axis.set_xlabel("Completed trial order")
    axis.set_ylabel("Robust CV selection score")
    axis.set_title("Optimization trial scores")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_parameter_importance(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        return
    ordered = frame.sort_values("importance", ascending=True, kind="mergesort")
    figure, axis = plt.subplots(figsize=(9, max(4, 0.4 * len(ordered))))
    axis.barh(ordered["parameter"], ordered["importance"])
    axis.set_xlabel("Importance")
    axis.set_title("Parameter importance")
    axis.grid(True, axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_optimization_reports(
    registry: OptimizationRegistry,
    study_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = registry.all_trials(study_id)
    frame = pd.DataFrame([_flatten_trial(trial) for trial in trials])
    if frame.empty:
        frame = pd.DataFrame(
            columns=["study_id", "search_stage", "trial_number", "status"]
        )
    frame.to_csv(output_dir / "trials.csv", index=False)
    failed = frame.loc[frame.get("status", pd.Series(dtype=str)) != "completed"].copy()
    failed.to_csv(output_dir / "failed_trials.csv", index=False)
    completed = frame.loc[frame.get("status", pd.Series(dtype=str)) == "completed"].copy()
    importance = _parameter_importance(completed)
    importance.to_csv(output_dir / "parameter_importance.csv", index=False)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _plot_trial_scores(frame, figures / "trial_scores.png")
    _plot_parameter_importance(importance, figures / "parameter_importance.png")
    summary = {
        "study_id": study_id,
        "total_trials": int(len(frame)),
        "completed_trials": int(len(completed)),
        "failed_or_incomplete_trials": int(len(failed)),
        "best_selection_score": (
            float(completed["selection_score"].max())
            if not completed.empty
            else None
        ),
    }
    atomic_write_json(output_dir / "optimization_summary.json", summary)
    return summary


def write_validation_candidates(
    candidates: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = {
            key: value
            for key, value in candidate.items()
            if key not in {"model_config", "metrics"}
        }
        metrics = candidate.get("metrics")
        if isinstance(metrics, Mapping):
            for key, value in metrics.items():
                row[key] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)
