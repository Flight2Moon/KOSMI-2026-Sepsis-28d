from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .constants import OUTCOME_CLASS_LABELS
from .final_comparison_stats import (
    _apply_fdr,
    delong_roc_test,
    paired_cluster_bootstrap_differences,
)


@dataclass(frozen=True)
class AblationComparisonSettings:
    bootstrap_repeats: int = 500
    confidence_level: float = 0.95
    seed: int = 42
    fdr_method: str = "fdr_bh"
    fdr_alpha: float = 0.05
    minimum_successful_bootstrap_repeats: int = 100
    comparison_metrics: tuple[str, ...] = (
        "macro_auprc",
        "macro_auroc",
        "multiclass_brier",
        "macro_f1",
    )


def _load_exp1_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="infer", low_memory=False)
    required = {
        "source_row_index",
        "subject_id",
        "y_true",
        "predicted_class",
        "prob_class_0",
        "prob_class_1",
        "prob_class_2",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"EXP1 prediction file is missing columns {missing}: {path}")
    if frame["source_row_index"].duplicated().any():
        raise ValueError(f"EXP1 prediction contains duplicated source_row_index: {path}")
    return frame.sort_values("source_row_index", kind="mergesort").reset_index(drop=True)


def _aligned_predictions(full_path: Path, ablation_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = _load_exp1_prediction(full_path)
    ablation = _load_exp1_prediction(ablation_path)
    full_ids = full["source_row_index"].to_numpy()
    ablation_ids = ablation["source_row_index"].to_numpy()
    if not np.array_equal(full_ids, ablation_ids):
        raise ValueError("Ablation comparison requires identical validation source_row_index ordering.")
    if not np.array_equal(full["y_true"].to_numpy(), ablation["y_true"].to_numpy()):
        raise ValueError("Ablation comparison labels differ between feature sets.")
    if not np.array_equal(full["subject_id"].astype(str).to_numpy(), ablation["subject_id"].astype(str).to_numpy()):
        raise ValueError("Ablation comparison subject IDs differ between feature sets.")
    return full, ablation


def compare_exp1_feature_sets(
    *,
    runs_dir: Path,
    full_run_id: str,
    ablation_run_id: str,
    full_feature_set_id: str,
    ablation_feature_set_id: str,
    model_names: Sequence[str],
    settings: AblationComparisonSettings,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_tables: list[pd.DataFrame] = []
    delong_rows: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    class_names = tuple(OUTCOME_CLASS_LABELS[index] for index in range(3))

    for model_name in model_names:
        full_path = (
            runs_dir / full_run_id / "EXP1_flat_3class" / model_name
            / "evaluation" / "validation" / "predictions.csv.gz"
        )
        ablation_path = (
            runs_dir / ablation_run_id / "EXP1_flat_3class" / model_name
            / "evaluation" / "validation" / "predictions.csv.gz"
        )
        if not full_path.is_file() or not ablation_path.is_file():
            continue
        full, ablation = _aligned_predictions(full_path, ablation_path)
        y = full["y_true"].to_numpy(dtype=np.int64)
        p_full = full[["prob_class_0", "prob_class_1", "prob_class_2"]].to_numpy(dtype=np.float64)
        p_ab = ablation[["prob_class_0", "prob_class_1", "prob_class_2"]].to_numpy(dtype=np.float64)
        pred_full = full["predicted_class"].to_numpy(dtype=np.int64)
        pred_ab = ablation["predicted_class"].to_numpy(dtype=np.int64)
        metadata = full[["source_row_index", "subject_id"]].copy()

        table, manifest = paired_cluster_bootstrap_differences(
            y_true=y,
            flat_probabilities=p_full,
            flat_predicted_classes=pred_full,
            pipeline_probabilities=p_ab,
            pipeline_predicted_classes=pred_ab,
            metadata=metadata,
            class_names=class_names,
            settings=settings,
        )
        table.insert(0, "model_name", model_name)
        table.insert(1, "reference_feature_set_id", full_feature_set_id)
        table.insert(2, "ablation_feature_set_id", ablation_feature_set_id)
        table = table.rename(columns={
            "flat_point_estimate": "full_point_estimate",
            "pipeline_point_estimate": "ablation_point_estimate",
            "difference_pipeline_minus_flat": "difference_ablation_minus_full",
            "improvement_positive_favors_pipeline": "improvement_positive_favors_ablation",
        })
        bootstrap_tables.append(table)
        manifests[model_name] = manifest

        for class_code, class_name in enumerate(class_names):
            result = delong_roc_test(
                (y == class_code).astype(np.int64),
                p_full[:, class_code],
                p_ab[:, class_code],
                class_code=class_code,
                class_name=class_name,
            ).to_dict()
            result.update({
                "model_name": model_name,
                "reference_feature_set_id": full_feature_set_id,
                "ablation_feature_set_id": ablation_feature_set_id,
            })
            result["full_auroc"] = result.pop("flat_auroc")
            result["ablation_auroc"] = result.pop("pipeline_auroc")
            result["difference_ablation_minus_full"] = result.pop("difference_pipeline_minus_flat")
            delong_rows.append(result)

    bootstrap = pd.concat(bootstrap_tables, ignore_index=True) if bootstrap_tables else pd.DataFrame()
    if not bootstrap.empty:
        bootstrap = _apply_fdr(
            bootstrap,
            method=settings.fdr_method,
            alpha=settings.fdr_alpha,
            family_id="validation_LOS28_EXP1_feature_ablation_bootstrap",
        )
    delong = pd.DataFrame(delong_rows)
    if not delong.empty:
        delong = _apply_fdr(
            delong,
            method=settings.fdr_method,
            alpha=settings.fdr_alpha,
            family_id="validation_LOS28_EXP1_feature_ablation_delong",
        )

    bootstrap.to_csv(output_dir / "exp1_paired_bootstrap_vs_full.csv", index=False)
    delong.to_csv(output_dir / "exp1_class_delong_vs_full.csv", index=False)
    manifest = {
        "schema_version": "exp123_v8_feature_ablation_statistics_v1",
        "reference_feature_set_id": full_feature_set_id,
        "ablation_feature_set_id": ablation_feature_set_id,
        "full_run_id": full_run_id,
        "ablation_run_id": ablation_run_id,
        "comparison_split": "validation",
        "confirmatory_splits_accessed": False,
        "paired_by": "subject_id/source_row_index",
        "bootstrap": {
            "requested_repeats": settings.bootstrap_repeats,
            "confidence_level": settings.confidence_level,
            "seed": settings.seed,
            "comparison_metrics": list(settings.comparison_metrics),
            "model_manifests": manifests,
        },
        "fdr": {
            "method": settings.fdr_method,
            "alpha": settings.fdr_alpha,
            "bootstrap_family": "validation_LOS28_EXP1_feature_ablation_bootstrap",
            "delong_family": "validation_LOS28_EXP1_feature_ablation_delong",
        },
    }
    return manifest
