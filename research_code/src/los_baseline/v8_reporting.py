from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .clinical_evaluation_v8 import (
    dca_thresholds_from_config,
    write_binary_clinical_report,
    write_multiclass_clinical_report,
)
from .task_spec import EXP1_FLAT_3CLASS_TASK
from .utils import atomic_write_json


def write_exp1_sampled_clinical_reports(
    *,
    model_prediction_paths: Mapping[str, Path],
    output_root: Path,
    clinical_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for model_name, path in model_prediction_paths.items():
        frame = pd.read_csv(path)
        columns = [f"prob_class_{i}" for i in range(3)]
        missing = [column for column in ["y_true", *columns] if column not in frame.columns]
        if missing:
            raise KeyError(f"EXP1 sampled prediction file is missing {missing}: {path}")
        probabilities = frame[columns].to_numpy(dtype=np.float64)
        clinical_dir = output_root / model_name / "clinical_validation"
        write_multiclass_clinical_report(
            y_true=frame["y_true"].to_numpy(dtype=np.int64),
            probabilities=probabilities,
            class_names=EXP1_FLAT_3CLASS_TASK.class_names,
            output_dir=clinical_dir,
            model_name=model_name,
            dca_thresholds=dca_thresholds_from_config(clinical_config),
        )
        rows.append({"model_name": model_name, "clinical_dir": str(clinical_dir)})
    payload = {
        "schema_version": "exp123_v8_exp1_sampled_clinical_reports_v1",
        "models": rows,
        "confirmatory_splits_accessed": False,
    }
    atomic_write_json(output_root / "clinical_reports_manifest.json", payload)
    return payload


def write_stage_binary_clinical_reports(
    *,
    model_prediction_paths: Mapping[str, Path],
    output_root: Path,
    clinical_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for model_name, path in model_prediction_paths.items():
        frame = pd.read_csv(path)
        clinical_dir = output_root / model_name / "clinical_validation"
        cfg = dict(clinical_config or {})
        manifest = write_binary_clinical_report(
            y_true=frame["y_true"].to_numpy(dtype=np.int64),
            probability=frame["positive_probability"].to_numpy(dtype=float),
            output_dir=clinical_dir,
            model_name=model_name,
            policies=tuple(cfg.get("threshold_policies", ["fixed_0_5", "youden", "f1", "sensitivity_target"])),
            sensitivity_target=float(cfg.get("sensitivity_target", 0.70)),
            dca_thresholds=dca_thresholds_from_config(cfg),
        )
        rows.append({
            "model_name": model_name,
            "clinical_dir": str(clinical_dir),
            "calibration": manifest["calibration"],
        })
    payload = {
        "schema_version": "exp123_v8_binary_stage_clinical_reports_v1",
        "models": rows,
        "confirmatory_splits_accessed": False,
    }
    atomic_write_json(output_root / "clinical_reports_manifest.json", payload)
    return payload
