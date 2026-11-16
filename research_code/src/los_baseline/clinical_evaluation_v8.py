from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .metrics import calculate_metrics, select_youden_threshold, validate_probabilities
from .utils import atomic_write_json


@dataclass(frozen=True)
class ThresholdSelection:
    policy: str
    threshold: float
    sensitivity_target: float | None
    selected_on_split: str
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_midpoint_threshold(probability: np.ndarray, edge: float) -> float:
    p = np.unique(np.asarray(probability, dtype=np.float64))
    positive = p[p >= edge]
    negative = p[p < edge]
    if positive.size == 0:
        return 1.0
    positive_edge = float(positive.min())
    if negative.size == 0:
        return 0.0
    negative_edge = float(negative.max())
    midpoint = negative_edge + (positive_edge - negative_edge) / 2.0
    if not negative_edge < midpoint < positive_edge:
        midpoint = float(np.nextafter(positive_edge, negative_edge))
    return float(np.clip(midpoint, 0.0, 1.0))


def _threshold_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    metrics = calculate_metrics(y_true, probability, threshold)
    metrics["flag_rate"] = float((np.asarray(probability) >= threshold).mean())
    return metrics


def select_threshold_by_policy(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    policy: str,
    sensitivity_target: float = 0.70,
    selected_on_split: str = "validation",
) -> ThresholdSelection:
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = validate_probabilities(probability, len(labels))
    if np.unique(labels).size < 2:
        threshold = 0.5
    elif policy == "fixed_0_5":
        threshold = 0.5
    elif policy == "youden":
        threshold = float(select_youden_threshold(labels, p))
    elif policy == "f1":
        precision, recall, thresholds = precision_recall_curve(labels, p)
        if len(thresholds) == 0:
            threshold = 0.5
        else:
            denominator = precision[:-1] + recall[:-1]
            scores = np.divide(
                2.0 * precision[:-1] * recall[:-1],
                denominator,
                out=np.zeros_like(denominator, dtype=np.float64),
                where=denominator > 0,
            )
            best = np.flatnonzero(scores == np.nanmax(scores))
            # Deterministic tie-break: prefer the larger threshold so an F1 tie
            # does not unnecessarily increase the flagged population.
            index = int(best[-1])
            threshold = _stable_midpoint_threshold(p, float(thresholds[index]))
    elif policy == "sensitivity_target":
        target = float(sensitivity_target)
        if not 0.0 < target <= 1.0:
            raise ValueError("sensitivity_target must be in (0, 1].")
        fpr, tpr, thresholds = roc_curve(labels, p)
        candidates: list[tuple[float, float, float]] = []
        for fp_rate, sensitivity, edge in zip(fpr, tpr, thresholds):
            if not np.isfinite(edge) or sensitivity + 1e-12 < target:
                continue
            specificity = 1.0 - float(fp_rate)
            candidates.append((specificity, float(edge), float(sensitivity)))
        if not candidates:
            threshold = 0.0
        else:
            _, edge, _ = max(candidates, key=lambda item: (item[0], item[1], item[2]))
            threshold = _stable_midpoint_threshold(p, edge)
    else:
        raise ValueError(f"Unknown threshold policy: {policy}")

    metrics = _threshold_metrics(labels, p, threshold)
    return ThresholdSelection(
        policy=policy,
        threshold=float(threshold),
        sensitivity_target=(float(sensitivity_target) if policy == "sensitivity_target" else None),
        selected_on_split=str(selected_on_split),
        metrics=metrics,
    )


def calibration_slope_intercept(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Return Cox calibration slope and intercept for a binary outcome.

    ``calibration_intercept`` is the intercept-in-the-large estimated with the
    prediction logit as an offset (slope fixed at 1). ``recalibration_intercept``
    and ``calibration_slope`` are the two coefficients of a joint logistic
    recalibration model. Keeping both avoids ambiguity in reporting.
    """
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = validate_probabilities(probability, len(y))
    p = np.clip(p, epsilon, 1.0 - epsilon)
    observed = float(y.mean()) if len(y) else float("nan")
    predicted = float(p.mean()) if len(p) else float("nan")
    base = {
        "mean_predicted_probability": predicted,
        "observed_prevalence": observed,
        "overestimation_ratio": float(predicted / max(observed, epsilon)) if len(y) else float("nan"),
    }
    if len(y) == 0 or np.unique(y).size < 2:
        return {
            **base,
            "calibration_slope": float("nan"),
            "calibration_intercept": float("nan"),
            "recalibration_intercept": float("nan"),
        }

    logit = np.log(p / (1.0 - p))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PerfectSeparationWarning)
            joint = sm.GLM(
                y,
                sm.add_constant(logit, prepend=True),
                family=sm.families.Binomial(),
            ).fit(disp=0)
        joint_intercept = float(joint.params[0])
        slope = float(joint.params[1])
    except Exception:
        joint_intercept = float("nan")
        slope = float("nan")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PerfectSeparationWarning)
            intercept_only = sm.GLM(
                y,
                np.ones((len(y), 1), dtype=np.float64),
                family=sm.families.Binomial(),
                offset=logit,
            ).fit(disp=0)
        calibration_intercept = float(intercept_only.params[0])
    except Exception:
        calibration_intercept = float("nan")

    return {
        **base,
        "calibration_slope": slope,
        "calibration_intercept": calibration_intercept,
        "recalibration_intercept": joint_intercept,
    }


def multiclass_calibration_slope_intercept(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(labels), len(class_names)):
        raise ValueError("Multiclass calibration probability shape mismatch.")
    rows: list[dict[str, Any]] = []
    for code, name in enumerate(class_names):
        binary = (labels == code).astype(np.int64)
        rows.append({
            "class_code": int(code),
            "class_name": str(name),
            **calibration_slope_intercept(binary, values[:, code]),
        })
    return pd.DataFrame(rows)


def decision_curve(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = validate_probabilities(probability, len(y))
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.50, 50)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    thresholds = thresholds[(thresholds > 0.0) & (thresholds < 1.0)]
    n = max(len(y), 1)
    prevalence = float(y.mean()) if len(y) else 0.0
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        pred = (p >= threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        odds = float(threshold / (1.0 - threshold))
        model_nb = float(tp / n - fp / n * odds)
        treat_all_nb = float(prevalence - (1.0 - prevalence) * odds)
        rows.append({
            "threshold": float(threshold),
            "net_benefit_model": model_nb,
            "net_benefit_treat_all": treat_all_nb,
            "net_benefit_treat_none": 0.0,
            "flag_rate": float(pred.mean()),
        })
    return pd.DataFrame(rows)



def dca_thresholds_from_config(config: Mapping[str, Any] | None) -> np.ndarray | None:
    cfg = dict(config or {})
    if not cfg:
        return None
    low = float(cfg.get("dca_threshold_min", 0.01))
    high = float(cfg.get("dca_threshold_max", 0.50))
    points = int(cfg.get("dca_threshold_points", 50))
    if not 0.0 < low < high < 1.0:
        raise ValueError("DCA threshold range must satisfy 0 < min < max < 1.")
    if points < 2:
        raise ValueError("dca_threshold_points must be at least 2.")
    return np.linspace(low, high, points)

def write_binary_clinical_report(
    *,
    y_true: np.ndarray,
    probability: np.ndarray,
    output_dir: Path,
    model_name: str,
    split_name: str = "validation",
    policies: Sequence[str] = ("fixed_0_5", "youden", "f1", "sensitivity_target"),
    sensitivity_target: float = 0.70,
    dca_thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selections = [
        select_threshold_by_policy(
            y_true,
            probability,
            policy=policy,
            sensitivity_target=sensitivity_target,
            selected_on_split=split_name,
        )
        for policy in policies
    ]
    threshold_frame = pd.DataFrame([
        {
            "policy": item.policy,
            "threshold": item.threshold,
            "sensitivity_target": item.sensitivity_target,
            **item.metrics,
        }
        for item in selections
    ])
    threshold_frame.to_csv(output_dir / "threshold_policy_metrics.csv", index=False)
    calibration = calibration_slope_intercept(y_true, probability)
    atomic_write_json(output_dir / "calibration_slope_intercept.json", calibration)
    dca = decision_curve(y_true, probability, thresholds=dca_thresholds)
    dca.to_csv(output_dir / "decision_curve.csv", index=False)
    payload = {
        "schema_version": "exp123_v8_binary_clinical_evaluation_v1",
        "model_name": model_name,
        "split": split_name,
        "threshold_selection_split": split_name,
        "thresholds_are_frozen_for_confirmatory_application": True,
        "threshold_policies": [item.to_dict() for item in selections],
        "calibration": calibration,
        "dca_threshold_count": int(len(dca)),
        "confirmatory_splits_accessed": False,
    }
    atomic_write_json(output_dir / "clinical_evaluation_manifest.json", payload)
    return payload


def write_multiclass_clinical_report(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    output_dir: Path,
    model_name: str,
    split_name: str = "validation",
    dca_thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = multiclass_calibration_slope_intercept(y_true, probabilities, class_names)
    calibration.to_csv(output_dir / "calibration_slope_intercept_ovr.csv", index=False)
    dca_rows: list[pd.DataFrame] = []
    for code, name in enumerate(class_names):
        binary = (np.asarray(y_true, dtype=np.int64) == code).astype(np.int64)
        frame = decision_curve(binary, np.asarray(probabilities)[:, code], thresholds=dca_thresholds)
        frame.insert(0, "class_name", str(name))
        frame.insert(0, "class_code", int(code))
        dca_rows.append(frame)
    dca = pd.concat(dca_rows, ignore_index=True) if dca_rows else pd.DataFrame()
    dca.to_csv(output_dir / "decision_curve_ovr.csv", index=False)
    payload = {
        "schema_version": "exp123_v8_multiclass_clinical_evaluation_v1",
        "model_name": model_name,
        "split": split_name,
        "calibration_method": "one_vs_rest_cox_recalibration",
        "dca_method": "one_vs_rest",
        "confirmatory_splits_accessed": False,
    }
    atomic_write_json(output_dir / "clinical_evaluation_manifest.json", payload)
    return payload
