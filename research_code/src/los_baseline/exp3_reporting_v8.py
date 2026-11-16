from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .clinical_evaluation_v8 import dca_thresholds_from_config, write_binary_clinical_report
from .utils import atomic_write_json, sha256_file


def build_exp3_report_from_stage2_predictions(
    *,
    run_id: str,
    strategy_id: str,
    stage2_prediction_paths: Mapping[str, Path],
    output_dir: Path,
    overwrite: bool = False,
    clinical_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    lock = output_dir / "EXP3_REPORT_COMPLETED.lock.json"
    if lock.is_file() and not overwrite:
        return json.loads(lock.read_text(encoding="utf-8"))
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        # The Phase-2 artifact_reference.json may already exist. It is safe to
        # add the Phase-3 report beside it, but any other Phase-3 output means
        # the operation is ambiguous.
        existing = {p.name for p in output_dir.iterdir()}
        if existing - {"artifact_reference.json"}:
            raise FileExistsError(f"EXP3 report output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    prediction_hashes: dict[str, str] = {}
    for model_name, path_value in stage2_prediction_paths.items():
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if "positive_probability" not in frame.columns or "y_true" not in frame.columns:
            raise KeyError(f"Stage2 prediction file lacks EXP3 fields: {path}")
        metrics_path = path.parent / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        probability = frame["positive_probability"].to_numpy(dtype=float)
        y_true = frame["y_true"].to_numpy(dtype=int)
        clinical_dir = output_dir / "models" / model_name / "clinical_validation"
        cfg = dict(clinical_config or {})
        clinical = write_binary_clinical_report(
            y_true=y_true,
            probability=probability,
            output_dir=clinical_dir,
            model_name=model_name,
            split_name="validation",
            policies=tuple(cfg.get("threshold_policies", ["fixed_0_5", "youden", "f1", "sensitivity_target"])),
            sensitivity_target=float(cfg.get("sensitivity_target", 0.70)),
            dca_thresholds=dca_thresholds_from_config(cfg),
        )
        rows.append({
            "strategy_id": strategy_id,
            "experiment": "EXP3_survivor_binary",
            "model_name": model_name,
            **{f"validation_{key}": value for key, value in metrics.items()},
            "calibration_slope": clinical["calibration"]["calibration_slope"],
            "calibration_intercept": clinical["calibration"]["calibration_intercept"],
            "mean_predicted_probability": clinical["calibration"]["mean_predicted_probability"],
            "observed_prevalence": clinical["calibration"]["observed_prevalence"],
        })
        prediction_hashes[model_name] = sha256_file(path)

    frame = pd.DataFrame(rows)
    # EXP3 selection follows the Stage2 AUPRC endpoint in the V8 analysis plan.
    if "validation_auprc" in frame.columns:
        frame = frame.sort_values(
            ["validation_auprc", "validation_auroc", "model_name"],
            ascending=[False, False, True],
            kind="mergesort",
        )
    summary_path = output_dir / "model_summary_validation.csv"
    frame.to_csv(summary_path, index=False)
    reference_model = str(frame.iloc[0]["model_name"]) if len(frame) else None
    payload = {
        "schema_version": "exp123_v8_exp3_report_v1",
        "run_id": run_id,
        "strategy_id": strategy_id,
        "experiment": "EXP3_survivor_binary",
        "source_task": "stage2_long_stay",
        "training_performed": False,
        "reuse_stage2_artifacts": True,
        "duplicate_training": False,
        "validation_reference_model": reference_model,
        "selection_metric": "auprc",
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "stage2_prediction_sha256": prediction_hashes,
        "confirmatory_splits_accessed": False,
    }
    atomic_write_json(lock, payload)
    return payload
