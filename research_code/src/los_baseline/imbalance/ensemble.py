from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..metrics import select_youden_threshold
from ..task_spec import TaskDefinition
from ..training_engine import _safe_binary_metrics, _safe_multiclass_metrics
from ..utils import atomic_write_json


def ensemble_member_prediction_files(
    *,
    task: TaskDefinition,
    member_dirs: Sequence[Path],
    split_names: Sequence[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Average member probabilities while keeping evaluation distributions natural."""
    frames_by_split: dict[str, list[pd.DataFrame]] = {}
    for member_dir in member_dirs:
        for split_name in split_names:
            path = member_dir / "evaluation" / split_name / "predictions.csv.gz"
            if path.is_file():
                frames_by_split.setdefault(split_name, []).append(pd.read_csv(path))

    validation_threshold: float | None = None
    if task.task_type == "binary" and "validation" in frames_by_split:
        frames = frames_by_split["validation"]
        probability = np.mean(
            [frame["positive_probability"].to_numpy(dtype=float) for frame in frames],
            axis=0,
        )
        validation_threshold = float(
            select_youden_threshold(frames[0]["y_true"].to_numpy(dtype=np.int64), probability)
        )

    result: dict[str, Any] = {}
    for split_name, frames in frames_by_split.items():
        if not frames:
            continue
        base = frames[0].copy()
        expected_rows = len(base)
        for frame in frames[1:]:
            if len(frame) != expected_rows:
                raise ValueError(f"Ensemble member row mismatch for {split_name}.")
            if not np.array_equal(
                frame["y_true"].to_numpy(dtype=np.int64),
                base["y_true"].to_numpy(dtype=np.int64),
            ):
                raise ValueError(f"Ensemble member labels differ for {split_name}.")

        if task.task_type == "binary":
            probability = np.mean(
                [frame["positive_probability"].to_numpy(dtype=float) for frame in frames],
                axis=0,
            )
            assert validation_threshold is not None
            base["positive_probability"] = probability
            base["predicted_at_validation_threshold"] = (
                probability >= validation_threshold
            ).astype(np.int64)
            base["predicted_at_0_5"] = (probability >= 0.5).astype(np.int64)
            metrics = _safe_binary_metrics(
                base["y_true"].to_numpy(dtype=np.int64),
                probability,
                validation_threshold,
            )
            probability_columns = ["positive_probability"]
        else:
            probability_columns = [f"prob_class_{i}" for i in range(task.num_classes)]
            probability = np.column_stack(
                [
                    np.mean([frame[column].to_numpy(dtype=float) for frame in frames], axis=0)
                    for column in probability_columns
                ]
            )
            probability = probability / probability.sum(axis=1, keepdims=True)
            for index, column in enumerate(probability_columns):
                base[column] = probability[:, index]
                base[f"prob_{task.class_names[index]}"] = probability[:, index]
            base["predicted_class"] = np.argmax(probability, axis=1).astype(np.int64)
            metrics = _safe_multiclass_metrics(
                base["y_true"].to_numpy(dtype=np.int64), probability, task
            )

        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        base.to_csv(split_dir / "predictions.csv.gz", index=False, compression="gzip")
        atomic_write_json(split_dir / "metrics.json", metrics)
        result[split_name] = {
            "member_count": int(len(frames)),
            "probability_columns": probability_columns,
            "validation_decision_threshold": validation_threshold,
            "metrics": metrics,
            "evaluation_distribution_modified": False,
        }
    return result
