from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def recalibration_change_summary(summary: pd.DataFrame) -> pd.DataFrame:
    required = {"site", "model", "class_code", "macro_auroc_mean", "macro_auprc_mean",
                "multiclass_brier_mean", "multiclass_log_loss_mean"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise KeyError(f"Recalibration summary is missing columns: {missing}")
    global_rows = summary.loc[summary["class_code"] == 0].copy()
    rows = []
    for site, group in global_rows.groupby("site"):
        by_model = group.set_index("model")
        if not {"raw", "recalibrated"}.issubset(by_model.index):
            continue
        raw = by_model.loc["raw"]
        rec = by_model.loc["recalibrated"]
        rows.append({
            "site": site,
            "raw_macro_auroc": float(raw["macro_auroc_mean"]),
            "recalibrated_macro_auroc": float(rec["macro_auroc_mean"]),
            "delta_macro_auroc": float(rec["macro_auroc_mean"] - raw["macro_auroc_mean"]),
            "raw_macro_auprc": float(raw["macro_auprc_mean"]),
            "recalibrated_macro_auprc": float(rec["macro_auprc_mean"]),
            "delta_macro_auprc": float(rec["macro_auprc_mean"] - raw["macro_auprc_mean"]),
            "raw_multiclass_brier": float(raw["multiclass_brier_mean"]),
            "recalibrated_multiclass_brier": float(rec["multiclass_brier_mean"]),
            "delta_multiclass_brier": float(rec["multiclass_brier_mean"] - raw["multiclass_brier_mean"]),
            "raw_log_loss": float(raw["multiclass_log_loss_mean"]),
            "recalibrated_log_loss": float(rec["multiclass_log_loss_mean"]),
            "delta_log_loss": float(rec["multiclass_log_loss_mean"] - raw["multiclass_log_loss_mean"]),
        })
    return pd.DataFrame(rows)


def dca_dominance_summary(dca: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setting, class_name, model), group in dca.groupby(["setting", "class", "model"], sort=False):
        g = group.sort_values("threshold")
        baseline = np.maximum(g["nb_treat_all"].to_numpy(float), 0.0)
        better = g["net_benefit"].to_numpy(float) > baseline
        thresholds = g.loc[better, "threshold"]
        rows.append({
            "setting": setting,
            "class": class_name,
            "model": model,
            "n_thresholds_better_than_treat_all_and_none": int(better.sum()),
            "threshold_min_better": float(thresholds.min()) if len(thresholds) else np.nan,
            "threshold_max_better": float(thresholds.max()) if len(thresholds) else np.nan,
            "max_net_benefit": float(g["net_benefit"].max()),
        })
    return pd.DataFrame(rows)


def plot_dca_by_setting(dca: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for setting, group in dca.groupby("setting", sort=False):
        classes = [c for c in ("death", "prolonged") if c in set(group["class"])]
        if not classes:
            continue
        fig, axes = plt.subplots(1, len(classes), figsize=(6 * len(classes), 4), squeeze=False)
        for idx, class_name in enumerate(classes):
            ax = axes[0, idx]
            sub = group[group["class"] == class_name]
            for model, current in sub.groupby("model", sort=False):
                ax.plot(current["threshold"], current["net_benefit"], label=model)
            base = sub.iloc[: len(sub["threshold"].unique())]
            ax.plot(base["threshold"], base["nb_treat_all"], linestyle=":", label="treat_all")
            ax.axhline(0.0, linewidth=0.8, label="treat_none")
            ax.set_title(class_name)
            ax.set_xlabel("Threshold probability")
            ax.set_ylabel("Net benefit")
            ax.legend(fontsize=8)
        fig.suptitle(setting)
        fig.tight_layout()
        safe = setting.replace("->", "_to_").replace("/", "_").replace(" ", "_")
        path = output_dir / f"dca_{safe}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_hospital_forest(levels: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for class_name in summary["class"].tolist():
        current = levels[levels["class"] == class_name].sort_values("auc").reset_index(drop=True)
        pooled = summary[summary["class"] == class_name].iloc[0]
        if current.empty:
            continue
        y = np.arange(len(current))
        low = np.maximum(0.0, current["auc"] - 1.96 * current["auc_se"])
        high = np.minimum(1.0, current["auc"] + 1.96 * current["auc_se"])
        fig, ax = plt.subplots(figsize=(8, max(5, 0.14 * len(current) + 2)))
        ax.errorbar(current["auc"], y,
                    xerr=np.vstack([current["auc"] - low, high - current["auc"]]),
                    fmt="o", markersize=3, linewidth=0.8)
        ax.axvline(float(pooled["pooled_auc"]), linestyle="--", linewidth=1.0, label="pooled RE")
        ax.axvspan(float(pooled["prediction_interval_lower"]),
                   float(pooled["prediction_interval_upper"]), alpha=0.1, label="95% prediction interval")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Hospital-level AUROC")
        ax.set_ylabel("Eligible hospitals (sorted)")
        ax.set_yticks([])
        ax.set_title(f"eICU-alt hospital heterogeneity — {class_name}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = output_dir / f"hospital_forest_{class_name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths
