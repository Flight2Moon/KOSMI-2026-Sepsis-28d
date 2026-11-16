from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .artifacts import load_task_artifact
from .clinical_evaluation_v8 import (
    calibration_slope_intercept,
    decision_curve,
    multiclass_calibration_slope_intercept,
)
from .evaluation_binary import BinaryBootstrapSettings, write_binary_evaluation_artifacts
from .evaluation_multiclass import BootstrapSettings, write_multiclass_evaluation_artifacts
from .metrics import calculate_metrics
from .oracle_pipeline import derive_pipeline_classes, load_confirmatory_cohort, restore_three_class_probabilities
from .site_recalibration_v8 import SiteRecalibrationSettings, run_multiseed_site_recalibration
from .task_spec import EXP1_FLAT_3CLASS_TASK
from .utils import atomic_write_json, sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _predict_from_spec(spec: Mapping[str, Any], raw_features: pd.DataFrame) -> np.ndarray:
    kind = str(spec.get("kind"))
    if kind == "standard_artifact":
        artifact_dir = Path(str(spec.get("artifact_dir"))).expanduser().resolve()
        artifact = load_task_artifact(artifact_dir)
        return np.asarray(artifact.predict_proba(raw_features), dtype=np.float64)
    if kind == "member_ensemble":
        member_dirs = [Path(str(value)).expanduser().resolve() for value in spec.get("member_artifact_dirs", [])]
        if not member_dirs:
            raise ValueError("member_ensemble predictor has no member artifacts.")
        values = []
        for artifact_dir in member_dirs:
            artifact = load_task_artifact(artifact_dir)
            values.append(np.asarray(artifact.predict_proba(raw_features), dtype=np.float64))
        first = values[0]
        for value in values[1:]:
            if value.shape != first.shape:
                raise ValueError("Sampled ensemble member probability shape mismatch.")
        mean = np.mean(values, axis=0)
        if mean.ndim == 2:
            denom = mean.sum(axis=1, keepdims=True)
            if np.all(denom > 0):
                mean = mean / denom
        return mean
    raise ValueError(f"Unsupported predictor kind: {kind}")


def _binary_probability(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 1]
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    raise ValueError(f"Expected binary probability vector, got {arr.shape}")


def _write_frozen_binary_clinical(
    *,
    y_true: np.ndarray,
    probability: np.ndarray,
    policies: list[dict[str, Any]],
    fallback_threshold: float,
    output_dir: Path,
    model_name: str,
    split_name: str,
    dca_thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if policies:
        for item in policies:
            if "threshold" not in item:
                continue
            threshold = float(item["threshold"])
            metrics = calculate_metrics(y_true, probability, threshold)
            rows.append({
                "policy": str(item.get("policy", "validation_policy")),
                "threshold": threshold,
                "selected_on_split": str(item.get("selected_on_split", "validation")),
                "sensitivity_target": item.get("sensitivity_target"),
                "flag_rate": float((probability >= threshold).mean()),
                **metrics,
            })
    if not rows:
        threshold = float(fallback_threshold)
        rows.append({
            "policy": "validation_decision_threshold",
            "threshold": threshold,
            "selected_on_split": "validation",
            "sensitivity_target": None,
            "flag_rate": float((probability >= threshold).mean()),
            **calculate_metrics(y_true, probability, threshold),
        })
    pd.DataFrame(rows).to_csv(output_dir / "frozen_threshold_policy_metrics.csv", index=False)
    calibration = calibration_slope_intercept(y_true, probability)
    atomic_write_json(output_dir / "calibration_slope_intercept.json", calibration)
    decision_curve(y_true, probability, thresholds=dca_thresholds).to_csv(output_dir / "decision_curve.csv", index=False)
    manifest = {
        "schema_version": "exp123_v8_confirmatory_binary_clinical_v1",
        "split": split_name,
        "model_name": model_name,
        "thresholds_selected_on": "validation",
        "threshold_reselection_performed": False,
        "threshold_policy_count": len(rows),
        "calibration": calibration,
    }
    atomic_write_json(output_dir / "clinical_evaluation_manifest.json", manifest)
    return manifest


def _write_confirmatory_multiclass_clinical(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
    model_name: str,
    split_name: str,
    dca_thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = multiclass_calibration_slope_intercept(y_true, probabilities, EXP1_FLAT_3CLASS_TASK.class_names)
    calibration.to_csv(output_dir / "calibration_slope_intercept_ovr.csv", index=False)
    frames = []
    for code, name in enumerate(EXP1_FLAT_3CLASS_TASK.class_names):
        binary = (np.asarray(y_true, dtype=np.int64) == code).astype(np.int64)
        frame = decision_curve(binary, np.asarray(probabilities)[:, code], thresholds=dca_thresholds)
        frame.insert(0, "class_name", name)
        frame.insert(0, "class_code", code)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(output_dir / "decision_curve_ovr.csv", index=False)
    manifest = {
        "schema_version": "exp123_v8_confirmatory_multiclass_clinical_v1",
        "split": split_name,
        "model_name": model_name,
        "threshold_reselection_performed": False,
        "calibration_method": "one_vs_rest_cox_recalibration",
    }
    atomic_write_json(output_dir / "clinical_evaluation_manifest.json", manifest)
    return manifest


def _dca_thresholds(config: Mapping[str, Any]) -> np.ndarray | None:
    cfg = dict(config.get("clinical_evaluation", {}))
    if not cfg:
        return None
    low = float(cfg.get("dca_threshold_min", 0.01)); high = float(cfg.get("dca_threshold_max", 0.50)); points = int(cfg.get("dca_threshold_points", 50))
    if 0 < low < high < 1 and points >= 2:
        return np.linspace(low, high, points)
    return None


def _multiclass_bootstrap_settings(repeats: int, seed: int, confidence: float):
    if repeats <= 0:
        return None
    return BootstrapSettings(
        repeats=repeats, seed=seed, confidence_level=confidence,
        minimum_successful_repeats=min(100, repeats),
    )


def _binary_bootstrap_settings(repeats: int, seed: int, confidence: float):
    if repeats <= 0:
        return None
    return BinaryBootstrapSettings(repeats=repeats, seed=seed, confidence_level=confidence)


def run_strategy_confirmatory(
    *,
    selection: Mapping[str, Any],
    config: Mapping[str, Any],
    runs_dir: Path,
    split_name: str,
    output_root: Path,
    site_recalibration_settings: SiteRecalibrationSettings | None = None,
) -> dict[str, Any]:
    if split_name not in {"internal_test", "external_eicu"}:
        raise ValueError(f"Unsupported confirmatory split: {split_name}")
    run_id = str(selection["run_id"])
    strategy = str(selection["strategy"])
    cohort_source_run_id = str(selection.get("cohort_source_run_id", run_id))
    cohort = load_confirmatory_cohort(
        config=config,
        run_dir=(runs_dir / cohort_source_run_id),
        split_name=split_name,
    )
    output_dir = output_root / f"LOS{int(float(selection['outcome_horizon_days']))}" / strategy
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Confirmatory strategy output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    evaluation = dict(config.get("evaluation", {}))
    clinical_cfg = dict(config.get("clinical_evaluation", {}))
    repeats = int(evaluation.get("external_bootstrap_repeats" if split_name == "external_eicu" else "multiclass_bootstrap_repeats", 500))
    confidence = float(evaluation.get("bootstrap_confidence_level", 0.95))
    seed = int(dict(config.get("oracle_pipeline", {})).get("seed", 42))
    default_threshold = float(evaluation.get("default_decision_threshold", 0.5))
    dca_thresholds = _dca_thresholds(config)

    exp1_spec = dict(selection["exp1"])
    p_exp1 = np.asarray(_predict_from_spec(exp1_spec, cohort.raw_features), dtype=np.float64)
    if p_exp1.shape != (len(cohort.y_true_3class), 3):
        raise ValueError(f"EXP1 probability shape mismatch: {p_exp1.shape}")
    p_exp1 = p_exp1 / p_exp1.sum(axis=1, keepdims=True)
    exp1_pred = cohort.metadata.copy().reset_index(drop=True)
    exp1_pred["y_true"] = cohort.y_true_3class
    for code in range(3): exp1_pred[f"prob_class_{code}"] = p_exp1[:, code]
    exp1_pred["predicted_class"] = np.argmax(p_exp1, axis=1)
    exp1_dir = output_dir / "EXP1_flat_3class"
    exp1_dir.mkdir(parents=True, exist_ok=True)
    exp1_pred.to_csv(exp1_dir / "predictions.csv.gz", index=False, compression="gzip")
    exp1_summary = write_multiclass_evaluation_artifacts(
        y_true=cohort.y_true_3class,
        probabilities=p_exp1,
        class_names=EXP1_FLAT_3CLASS_TASK.class_names,
        output_dir=exp1_dir / "evaluation",
        split_name=split_name,
        model_name=str(exp1_spec["model_name"]),
        calibration_bins=int(evaluation.get("calibration_bins", 10)),
        bootstrap_settings=_multiclass_bootstrap_settings(repeats, seed, confidence),
        save_plots=bool(evaluation.get("save_plots", True)),
    )
    _write_confirmatory_multiclass_clinical(
        y_true=cohort.y_true_3class,
        probabilities=p_exp1,
        output_dir=exp1_dir / "clinical",
        model_name=str(exp1_spec["model_name"]),
        split_name=split_name,
        dca_thresholds=dca_thresholds,
    )

    stage1_spec = dict(selection["exp2"]["stage1"])
    stage2_spec = dict(selection["exp2"]["stage2"])
    p1 = _binary_probability(_predict_from_spec(stage1_spec, cohort.raw_features))
    p2 = _binary_probability(_predict_from_spec(stage2_spec, cohort.raw_features))
    y1 = (cohort.y_true_3class == 1).astype(np.int64)
    survivors = cohort.y_true_3class != 1
    y2 = (cohort.y_true_3class[survivors] == 2).astype(np.int64)
    t1 = float(stage1_spec.get("validation_threshold") if stage1_spec.get("validation_threshold") is not None else default_threshold)
    t2 = float(stage2_spec.get("validation_threshold") if stage2_spec.get("validation_threshold") is not None else default_threshold)

    stage1_dir = output_dir / "EXP2_two_stage" / "stage1_mortality"
    stage1_summary = write_binary_evaluation_artifacts(
        y_true=y1, probabilities=p1, selected_threshold=t1, default_threshold=default_threshold,
        output_dir=stage1_dir / "evaluation", split_name=split_name,
        model_name=str(stage1_spec["model_name"]), positive_class_name="B_died",
        calibration_bins=int(evaluation.get("calibration_bins", 10)),
        bootstrap_settings=_binary_bootstrap_settings(repeats, seed, confidence),
        save_plots=bool(evaluation.get("save_plots", True)),
    )
    _write_frozen_binary_clinical(
        y_true=y1, probability=p1,
        policies=list(stage1_spec.get("validation_threshold_policies", [])), fallback_threshold=t1,
        output_dir=stage1_dir / "clinical", model_name=str(stage1_spec["model_name"]), split_name=split_name,
        dca_thresholds=dca_thresholds,
    )

    stage2_dir = output_dir / "EXP2_two_stage" / "stage2_long_stay"
    stage2_summary = write_binary_evaluation_artifacts(
        y_true=y2, probabilities=p2[survivors], selected_threshold=t2, default_threshold=default_threshold,
        output_dir=stage2_dir / "evaluation", split_name=split_name,
        model_name=str(stage2_spec["model_name"]), positive_class_name="C_still_admitted",
        calibration_bins=int(evaluation.get("calibration_bins", 10)),
        bootstrap_settings=_binary_bootstrap_settings(repeats, seed, confidence),
        save_plots=bool(evaluation.get("save_plots", True)),
    )
    _write_frozen_binary_clinical(
        y_true=y2, probability=p2[survivors],
        policies=list(stage2_spec.get("validation_threshold_policies", [])), fallback_threshold=t2,
        output_dir=stage2_dir / "clinical", model_name=str(stage2_spec["model_name"]), split_name=split_name,
        dca_thresholds=dca_thresholds,
    )

    p_combined = restore_three_class_probabilities(p1, p2)
    hard = derive_pipeline_classes(p1, p2, stage1_threshold=t1, stage2_threshold=t2)
    combined_dir = output_dir / "EXP2_two_stage" / "combined"
    combined_pred = cohort.metadata.copy().reset_index(drop=True)
    combined_pred["y_true_3class"] = cohort.y_true_3class
    combined_pred["stage1_probability_B"] = p1
    combined_pred["stage2_probability_C_given_nonB"] = p2
    for code in range(3): combined_pred[f"prob_class_{code}"] = p_combined[:, code]
    combined_pred["argmax_prediction"] = np.argmax(p_combined, axis=1)
    combined_pred["hard_gate_prediction"] = hard
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_pred.to_csv(combined_dir / "predictions.csv.gz", index=False, compression="gzip")
    combined_summary = write_multiclass_evaluation_artifacts(
        y_true=cohort.y_true_3class,
        probabilities=p_combined,
        class_names=EXP1_FLAT_3CLASS_TASK.class_names,
        output_dir=combined_dir / "evaluation",
        split_name=split_name,
        model_name=f"{stage1_spec['model_name']}->{stage2_spec['model_name']}",
        calibration_bins=int(evaluation.get("calibration_bins", 10)),
        bootstrap_settings=_multiclass_bootstrap_settings(repeats, seed, confidence),
        save_plots=bool(evaluation.get("save_plots", True)),
        predicted_classes=hard,
        decision_rule="frozen_validation_stage_thresholds",
    )
    _write_confirmatory_multiclass_clinical(
        y_true=cohort.y_true_3class,
        probabilities=p_combined,
        output_dir=combined_dir / "clinical",
        model_name=f"{stage1_spec['model_name']}->{stage2_spec['model_name']}",
        split_name=split_name,
        dca_thresholds=dca_thresholds,
    )

    exp3_spec = dict(selection["exp3"])
    p_exp3_full = _binary_probability(_predict_from_spec(exp3_spec, cohort.raw_features))
    p_exp3 = p_exp3_full[survivors]
    exp3_threshold = float(exp3_spec.get("validation_threshold") if exp3_spec.get("validation_threshold") is not None else default_threshold)
    exp3_dir = output_dir / "EXP3_survivor_binary"
    exp3_summary = write_binary_evaluation_artifacts(
        y_true=y2, probabilities=p_exp3, selected_threshold=exp3_threshold, default_threshold=default_threshold,
        output_dir=exp3_dir / "evaluation", split_name=split_name,
        model_name=str(exp3_spec["model_name"]), positive_class_name="C_still_admitted",
        calibration_bins=int(evaluation.get("calibration_bins", 10)),
        bootstrap_settings=_binary_bootstrap_settings(repeats, seed, confidence),
        save_plots=bool(evaluation.get("save_plots", True)),
    )
    _write_frozen_binary_clinical(
        y_true=y2, probability=p_exp3,
        policies=list(exp3_spec.get("validation_threshold_policies", [])), fallback_threshold=exp3_threshold,
        output_dir=exp3_dir / "clinical", model_name=str(exp3_spec["model_name"]), split_name=split_name,
        dca_thresholds=dca_thresholds,
    )

    recalibration = None
    if split_name == "external_eicu" and site_recalibration_settings is not None:
        group_column = str(config.get("data", {}).get("patient_id_column", "subject_id"))
        groups = cohort.metadata[group_column].to_numpy() if group_column in cohort.metadata.columns else cohort.metadata["source_row_index"].to_numpy()
        recal_root = output_dir / "site_recalibration"
        recalibration = {
            "EXP1": run_multiseed_site_recalibration(
                y_true=cohort.y_true_3class, probabilities=p_exp1, groups=groups,
                class_names=EXP1_FLAT_3CLASS_TASK.class_names,
                output_dir=recal_root / "EXP1_flat_3class", settings=site_recalibration_settings,
            ),
            "EXP2_Combined": run_multiseed_site_recalibration(
                y_true=cohort.y_true_3class, probabilities=p_combined, groups=groups,
                class_names=EXP1_FLAT_3CLASS_TASK.class_names,
                output_dir=recal_root / "EXP2_two_stage_combined", settings=site_recalibration_settings,
            ),
        }

    manifest = {
        "schema_version": "exp123_v8_strategy_confirmatory_result_v1",
        "created_at_utc": _utc_now(),
        "run_id": run_id,
        "cohort_source_run_id": cohort_source_run_id,
        "strategy": strategy,
        "outcome_horizon_days": float(selection["outcome_horizon_days"]),
        "split": split_name,
        "sample_count": int(len(cohort.y_true_3class)),
        "survivor_count": int(survivors.sum()),
        "selection_split": "validation",
        "threshold_reselection_performed": False,
        "raw_external_validation_primary": bool(split_name == "external_eicu"),
        "site_recalibration": recalibration,
        "summaries": {
            "EXP1": exp1_summary,
            "Stage1": stage1_summary,
            "Stage2": stage2_summary,
            "EXP2_Combined": combined_summary,
            "EXP3": exp3_summary,
        },
    }
    atomic_write_json(output_dir / "confirmatory_result_manifest.json", manifest)
    return manifest
