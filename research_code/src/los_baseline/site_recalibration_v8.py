from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .clinical_evaluation_v8 import multiclass_calibration_slope_intercept
from .evaluation_multiclass import calculate_multiclass_metrics
from .utils import atomic_write_json


@dataclass(frozen=True)
class SiteRecalibrationSettings:
    calibration_fraction: float = 0.20
    repeats: int = 20
    seed: int = 2026
    minimum_successful_repeats: int = 15
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must be in (0,1).")
        if self.repeats < 1:
            raise ValueError("repeats must be positive.")
        if self.minimum_successful_repeats < 1 or self.minimum_successful_repeats > self.repeats:
            raise ValueError("minimum_successful_repeats must be between 1 and repeats.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def per_class_platt_recalibrate(
    *,
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    apply_probabilities: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    p_cal = np.asarray(calibration_probabilities, dtype=np.float64)
    p_apply = np.asarray(apply_probabilities, dtype=np.float64)
    y = np.asarray(calibration_labels, dtype=np.int64).reshape(-1)
    if p_cal.ndim != 2 or p_apply.ndim != 2 or p_cal.shape[1] != p_apply.shape[1]:
        raise ValueError("Multiclass probability shape mismatch for site recalibration.")
    q = np.zeros_like(p_apply)
    for class_code in range(p_cal.shape[1]):
        target = (y == class_code).astype(np.int64)
        if np.unique(target).size < 2:
            raise ValueError(f"Calibration subset lacks both OvR classes for class {class_code}.")
        model = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
        model.fit(_logit(p_cal[:, class_code], epsilon).reshape(-1, 1), target)
        q[:, class_code] = model.predict_proba(
            _logit(p_apply[:, class_code], epsilon).reshape(-1, 1)
        )[:, 1]
    row_sum = q.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise RuntimeError("Recalibrated probabilities have non-positive row sum.")
    return q / row_sum


def _group_split_indices(groups: Sequence[Any], *, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    group_values = np.asarray(groups)
    unique = np.unique(group_values)
    if len(unique) < 2:
        raise ValueError("Site recalibration requires at least two patient groups.")
    rng = np.random.default_rng(int(seed))
    shuffled = unique.copy(); rng.shuffle(shuffled)
    n_cal = max(1, min(len(shuffled) - 1, int(round(float(fraction) * len(shuffled)))))
    cal_groups = set(shuffled[:n_cal].tolist())
    cal_mask = np.array([value in cal_groups for value in group_values], dtype=bool)
    return np.flatnonzero(cal_mask), np.flatnonzero(~cal_mask)


def run_multiseed_site_recalibration(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    groups: Sequence[Any],
    class_names: Sequence[str],
    output_dir: Path,
    settings: SiteRecalibrationSettings,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(y), len(class_names)):
        raise ValueError("Site recalibration probability shape mismatch.")
    if len(groups) != len(y):
        raise ValueError("Site recalibration group count mismatch.")
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    successful = 0
    for repeat in range(settings.repeats):
        seed = int(settings.seed + repeat)
        cal_idx, eval_idx = _group_split_indices(groups, fraction=settings.calibration_fraction, seed=seed)
        try:
            if set(np.unique(y[cal_idx]).tolist()) != set(range(len(class_names))):
                raise ValueError("Calibration subset does not contain all outcome classes.")
            q = per_class_platt_recalibrate(
                calibration_probabilities=p[cal_idx],
                calibration_labels=y[cal_idx],
                apply_probabilities=p[eval_idx],
                epsilon=settings.epsilon,
            )
            successful += 1
        except Exception as exc:
            failures.append({"repeat": repeat, "seed": seed, "error_type": type(exc).__name__, "message": str(exc)})
            continue
        for tag, values in (("raw", p[eval_idx]), ("recalibrated", q)):
            metrics = calculate_multiclass_metrics(y[eval_idx], values, class_names)
            metric_rows.append({
                "repeat": repeat,
                "seed": seed,
                "model": tag,
                "calibration_rows": int(len(cal_idx)),
                "evaluation_rows": int(len(eval_idx)),
                **metrics,
            })
            cal = multiclass_calibration_slope_intercept(y[eval_idx], values, class_names)
            for row in cal.to_dict(orient="records"):
                calibration_rows.append({
                    "repeat": repeat,
                    "seed": seed,
                    "model": tag,
                    "calibration_rows": int(len(cal_idx)),
                    "evaluation_rows": int(len(eval_idx)),
                    **row,
                })
    if successful < settings.minimum_successful_repeats:
        raise RuntimeError(
            f"Only {successful}/{settings.repeats} site-recalibration repeats succeeded; "
            f"minimum required is {settings.minimum_successful_repeats}."
        )
    metrics_frame = pd.DataFrame(metric_rows)
    cal_frame = pd.DataFrame(calibration_rows)
    metrics_frame.to_csv(output_dir / "recalibration_multiseed_metrics.csv", index=False)
    cal_frame.to_csv(output_dir / "recalibration_multiseed_calibration.csv", index=False)

    metric_cols = [c for c in ("macro_auroc", "macro_auprc", "macro_f1", "balanced_accuracy", "multiclass_brier", "multiclass_log_loss") if c in metrics_frame.columns]
    metrics_agg = metrics_frame.groupby("model")[metric_cols].agg(["mean", "std"]).reset_index()
    metrics_agg.columns = ["model" if c[0] == "model" else "_".join(c).strip("_") for c in metrics_agg.columns.to_flat_index()]
    metrics_agg.to_csv(output_dir / "recalibration_multiseed_metrics_summary.csv", index=False)

    cal_cols = [c for c in ("calibration_slope", "calibration_intercept", "recalibration_intercept", "mean_predicted_probability", "observed_prevalence") if c in cal_frame.columns]
    cal_agg = cal_frame.groupby(["model", "class_code", "class_name"])[cal_cols].agg(["mean", "std"]).reset_index()
    flat_cols = []
    for col in cal_agg.columns.to_flat_index():
        if col[0] in {"model", "class_code", "class_name"}:
            flat_cols.append(col[0])
        else:
            flat_cols.append("_".join(col).strip("_"))
    cal_agg.columns = flat_cols
    cal_agg.to_csv(output_dir / "recalibration_multiseed_calibration_summary.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "recalibration_failed_repeats.csv", index=False)

    manifest = {
        "schema_version": "exp123_v8_site_recalibration_v1",
        "method": "per_class_logistic_recalibration_on_logit_probability_then_renormalize",
        "analysis_role": "local_site_adaptation_secondary_to_raw_external_validation",
        "raw_external_validation_remains_primary": True,
        "settings": settings.to_dict(),
        "successful_repeats": successful,
        "failed_repeats": len(failures),
        "group_split": "patient_group_level",
        "confirmatory_thresholds_reselected": False,
    }
    atomic_write_json(output_dir / "site_recalibration_manifest.json", manifest)
    return manifest
