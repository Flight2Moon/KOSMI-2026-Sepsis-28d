from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .artifacts import load_task_artifact
from .exp123_data import TaskSplitData
from .utils import atomic_write_json, sha256_file


def infer_threshold_from_prediction_frame(frame: pd.DataFrame, default: float = 0.5) -> float:
    if "validation_decision_threshold" in frame.columns:
        values = frame["validation_decision_threshold"].dropna().unique()
        if len(values) == 1:
            return float(values[0])
    if "predicted_at_validation_threshold" in frame.columns and "positive_probability" in frame.columns:
        decision = frame["predicted_at_validation_threshold"].to_numpy(dtype=np.int64)
        probability = frame["positive_probability"].to_numpy(dtype=np.float64)
        pos = probability[decision == 1]
        neg = probability[decision == 0]
        if len(pos) and len(neg):
            low = float(neg.max())
            high = float(pos.min())
            if low < high:
                return float(low + (high - low) / 2.0)
        if len(pos) and not len(neg):
            return 0.0
        if len(neg) and not len(pos):
            return 1.0
    return float(default)


def _pipeline_truth(metadata: pd.DataFrame, *, task_kind: str) -> np.ndarray:
    if "three_class_outcome" not in metadata.columns:
        raise KeyError("Pipeline inference requires three_class_outcome metadata.")
    y3 = metadata["three_class_outcome"].to_numpy(dtype=np.int64)
    if task_kind == "stage1":
        return (y3 == 1).astype(np.int64)
    if task_kind == "stage2":
        # Stage2 is only *evaluated* among A/C survivors, but EXP2 cascade needs
        # a C-given-nonB score for every patient. B rows receive y=0 here only
        # as pipeline bookkeeping; these labels must never be used to report
        # Stage2 standalone performance.
        return (y3 == 2).astype(np.int64)
    raise ValueError(f"Unknown pipeline task kind: {task_kind}")


def materialize_standard_artifact_pipeline_predictions(
    *,
    artifact_dir: Path,
    full_validation: TaskSplitData,
    task_kind: str,
    output_path: Path,
    default_threshold: float = 0.5,
) -> Path:
    artifact = load_task_artifact(artifact_dir)
    probability = np.asarray(artifact.predict_proba(full_validation.raw_features), dtype=np.float64).reshape(-1)
    threshold = float(
        artifact.validation_decision_threshold
        if artifact.validation_decision_threshold is not None
        else default_threshold
    )
    frame = full_validation.metadata.copy().reset_index(drop=True)
    frame["y_true"] = _pipeline_truth(frame, task_kind=task_kind)
    frame["positive_probability"] = probability
    frame["validation_decision_threshold"] = threshold
    frame["predicted_at_validation_threshold"] = (probability >= threshold).astype(np.int64)
    frame["pipeline_inference_only"] = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, compression="gzip")
    atomic_write_json(output_path.with_suffix(".manifest.json"), {
        "schema_version": "exp123_v8_pipeline_input_v1",
        "task_kind": task_kind,
        "source": "single_standard_artifact",
        "artifact_dir": str(artifact_dir),
        "row_count": int(len(frame)),
        "validation_decision_threshold": threshold,
        "standalone_task_metrics_permitted": False,
        "confirmatory_splits_accessed": False,
    })
    return output_path


def materialize_member_ensemble_pipeline_predictions(
    *,
    member_dirs: Sequence[Path],
    full_validation: TaskSplitData,
    task_kind: str,
    survivor_validation_prediction_path: Path,
    output_path: Path,
    default_threshold: float = 0.5,
) -> Path:
    if not member_dirs:
        raise ValueError("Sampled pipeline inference requires at least one member artifact.")
    probabilities: list[np.ndarray] = []
    artifact_hashes: dict[str, str] = {}
    for member_dir in member_dirs:
        artifact_dir = Path(member_dir) / "artifact"
        artifact = load_task_artifact(artifact_dir)
        probabilities.append(
            np.asarray(artifact.predict_proba(full_validation.raw_features), dtype=np.float64).reshape(-1)
        )
        metadata_path = artifact_dir / "metadata.json"
        artifact_hashes[str(member_dir)] = sha256_file(metadata_path)
    probability = np.mean(probabilities, axis=0)
    survivor_frame = pd.read_csv(survivor_validation_prediction_path)
    threshold = infer_threshold_from_prediction_frame(survivor_frame, default_threshold)
    frame = full_validation.metadata.copy().reset_index(drop=True)
    frame["y_true"] = _pipeline_truth(frame, task_kind=task_kind)
    frame["positive_probability"] = probability
    frame["validation_decision_threshold"] = threshold
    frame["predicted_at_validation_threshold"] = (probability >= threshold).astype(np.int64)
    frame["pipeline_inference_only"] = True
    frame["ensemble_member_count"] = int(len(member_dirs))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, compression="gzip")
    atomic_write_json(output_path.with_suffix(".manifest.json"), {
        "schema_version": "exp123_v8_pipeline_input_v1",
        "task_kind": task_kind,
        "source": "sampled_member_artifact_probability_mean",
        "member_count": int(len(member_dirs)),
        "member_artifact_metadata_sha256": artifact_hashes,
        "survivor_validation_prediction_path": str(survivor_validation_prediction_path),
        "validation_decision_threshold": threshold,
        "row_count": int(len(frame)),
        "standalone_task_metrics_permitted": False,
        "confirmatory_splits_accessed": False,
    })
    return output_path
