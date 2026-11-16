from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _plot_model_metric_comparison(
    model_table: pd.DataFrame,
    output_path: Path,
) -> None:
    selected = model_table.loc[
        model_table["metric"].isin(
            ["macro_auroc", "macro_auprc", "macro_f1", "balanced_accuracy"]
        )
    ].copy()
    metrics = selected["metric"].tolist()
    x = np.arange(len(metrics), dtype=float)
    width = 0.36
    figure = plt.figure(figsize=(10, 6))
    axis = figure.add_subplot(111)
    axis.bar(x - width / 2, selected["flat"], width, label="EXP1 Flat")
    axis.bar(x + width / 2, selected["pipeline"], width, label="EXP2 Pipeline")
    axis.set_xticks(x)
    axis.set_xticklabels(metrics, rotation=25, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Metric value")
    axis.set_title("EXP1 Flat vs EXP2 Pipeline")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_paired_differences(
    bootstrap_table: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = bootstrap_table.copy().reset_index(drop=True)
    y = np.arange(len(frame), dtype=float)
    point = frame["improvement_positive_favors_pipeline"].to_numpy(dtype=float)
    lower = frame["improvement_ci_lower"].to_numpy(dtype=float)
    upper = frame["improvement_ci_upper"].to_numpy(dtype=float)
    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(111)
    axis.errorbar(
        point,
        y,
        xerr=np.vstack([point - lower, upper - point]),
        fmt="o",
        capsize=4,
    )
    axis.axvline(0.0, linestyle="--")
    axis.set_yticks(y)
    axis.set_yticklabels(frame["metric"])
    axis.set_xlabel("Improvement; positive favors EXP2 Pipeline")
    axis.set_title("Paired cluster-bootstrap metric differences")
    axis.grid(True, axis="x", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
