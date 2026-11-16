from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .clinical_evaluation_v8 import dca_thresholds_from_config, write_multiclass_clinical_report
from .evaluation_multiclass import calculate_multiclass_metrics
from .oracle_pipeline import derive_pipeline_classes, restore_three_class_probabilities
from .task_spec import EXP1_FLAT_3CLASS_TASK
from .utils import atomic_write_json, sha256_file


def _load_prediction(path: Path, *, binary: bool = True) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"y_true"}
    if binary:
        required.add("positive_probability")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Prediction file {path} is missing columns: {missing}")
    return frame


def _alignment_columns(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    preferred = ["source_row_index", "subject_id", "stay_id"]
    return [column for column in preferred if column in left.columns and column in right.columns]


def _assert_aligned(left: pd.DataFrame, right: pd.DataFrame, *, label: str) -> None:
    if len(left) != len(right):
        raise ValueError(f"{label} prediction row mismatch: {len(left)} != {len(right)}")
    columns = _alignment_columns(left, right)
    for column in columns:
        if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
            raise ValueError(f"{label} prediction rows are not aligned by {column}.")


def _threshold_from_frame(frame: pd.DataFrame, default_threshold: float) -> float:
    if "validation_decision_threshold" in frame.columns:
        values = frame["validation_decision_threshold"].dropna().unique()
        if len(values) == 1:
            return float(values[0])
    if "predicted_at_validation_threshold" in frame.columns:
        decision = frame["predicted_at_validation_threshold"].to_numpy(dtype=np.int64)
        probability = frame["positive_probability"].to_numpy(dtype=np.float64)
        positive = probability[decision == 1]
        negative = probability[decision == 0]
        if len(positive) and len(negative):
            lo = float(negative.max())
            hi = float(positive.min())
            if lo < hi:
                return float(lo + (hi - lo) / 2.0)
        if len(positive) and not len(negative):
            return 0.0
        if len(negative) and not len(positive):
            return 1.0
    return float(default_threshold)


def _three_class_truth(frame: pd.DataFrame) -> np.ndarray:
    for column in ("three_class_outcome", "y_true_3class"):
        if column in frame.columns:
            return frame[column].to_numpy(dtype=np.int64)
    raise KeyError("Stage prediction metadata is missing three_class_outcome.")


def _rank_rows(rows: Sequence[dict[str, Any]], primary_metric: str) -> list[dict[str, Any]]:
    if primary_metric not in {"macro_auprc", "macro_auroc"}:
        raise ValueError(f"Unsupported EXP2 combined selection metric: {primary_metric}")
    secondary = "macro_auroc" if primary_metric == "macro_auprc" else "macro_auprc"
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get(primary_metric) if row.get(primary_metric) is not None else -np.inf),
            -float(row.get(secondary) if row.get(secondary) is not None else -np.inf),
            float(row.get("multiclass_brier") if row.get("multiclass_brier") is not None else np.inf),
            float(row.get("multiclass_log_loss") if row.get("multiclass_log_loss") is not None else np.inf),
            str(row["stage1_model"]),
            str(row["stage2_model"]),
        ),
    )


def build_exp2_combined_from_prediction_files(
    *,
    run_id: str,
    strategy_id: str,
    stage1_prediction_paths: Mapping[str, Path],
    stage2_prediction_paths: Mapping[str, Path],
    output_dir: Path,
    primary_metric: str = "macro_auprc",
    default_threshold: float = 0.5,
    overwrite: bool = False,
    clinical_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate all Stage1 x Stage2 combinations using frozen validation predictions.

    This evaluator is storage-agnostic: Stage1 can come from a referenced T2
    natural model while Stage2 comes from a sampled ensemble. No model is
    refitted and no confirmatory split is accessed.
    """
    lock = output_dir / "EXP2_COMBINED_COMPLETED.lock.json"
    if lock.is_file() and not overwrite:
        return json.loads(lock.read_text(encoding="utf-8"))
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"EXP2 combined output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_frames = {name: _load_prediction(Path(path)) for name, path in stage1_prediction_paths.items()}
    stage2_frames = {name: _load_prediction(Path(path)) for name, path in stage2_prediction_paths.items()}
    if not stage1_frames or not stage2_frames:
        raise ValueError("EXP2 combined evaluation requires Stage1 and Stage2 predictions.")

    reference = next(iter(stage1_frames.values()))
    y_true = _three_class_truth(reference)
    for name, frame in stage1_frames.items():
        _assert_aligned(reference, frame, label=f"Stage1 {name}")
        if not np.array_equal(y_true, _three_class_truth(frame)):
            raise ValueError(f"Stage1 three-class labels differ for {name}.")
    for name, frame in stage2_frames.items():
        _assert_aligned(reference, frame, label=f"Stage2 {name}")
        if "three_class_outcome" in frame.columns and not np.array_equal(
            y_true, frame["three_class_outcome"].to_numpy(dtype=np.int64)
        ):
            raise ValueError(f"Stage2 three-class labels differ for {name}.")

    rows: list[dict[str, Any]] = []
    probability_cache: dict[tuple[str, str], np.ndarray] = {}
    hard_cache: dict[tuple[str, str], np.ndarray] = {}
    for stage1_model, stage1 in stage1_frames.items():
        p1 = stage1["positive_probability"].to_numpy(dtype=np.float64)
        stage1_threshold = _threshold_from_frame(stage1, default_threshold)
        if "predicted_at_validation_threshold" in stage1.columns:
            # Derive the frozen threshold decision directly when the numeric
            # threshold itself is not persisted in the prediction CSV.
            stage1_hard = stage1["predicted_at_validation_threshold"].to_numpy(dtype=np.int64)
        else:
            stage1_hard = (p1 >= stage1_threshold).astype(np.int64)

        for stage2_model, stage2 in stage2_frames.items():
            p2 = stage2["positive_probability"].to_numpy(dtype=np.float64)
            stage2_threshold = _threshold_from_frame(stage2, default_threshold)
            probabilities = restore_three_class_probabilities(p1, p2)
            probability_cache[(stage1_model, stage2_model)] = probabilities
            argmax_metrics = calculate_multiclass_metrics(
                y_true=y_true,
                probabilities=probabilities,
                class_names=EXP1_FLAT_3CLASS_TASK.class_names,
            )
            if "predicted_at_validation_threshold" in stage2.columns:
                stage2_hard = stage2["predicted_at_validation_threshold"].to_numpy(dtype=np.int64)
                hard_classes = np.where(stage1_hard == 1, 1, np.where(stage2_hard == 1, 2, 0)).astype(np.int64)
            else:
                hard_classes = derive_pipeline_classes(
                    p1, p2,
                    stage1_threshold=stage1_threshold,
                    stage2_threshold=stage2_threshold,
                )
            hard_cache[(stage1_model, stage2_model)] = hard_classes
            hard_metrics = calculate_multiclass_metrics(
                y_true=y_true,
                probabilities=probabilities,
                class_names=EXP1_FLAT_3CLASS_TASK.class_names,
                predicted_classes=hard_classes,
            )
            rows.append({
                "strategy_id": strategy_id,
                "stage1_model": stage1_model,
                "stage2_model": stage2_model,
                "stage1_threshold": stage1_threshold,
                "stage2_threshold": stage2_threshold,
                **argmax_metrics,
                "hard_gate_accuracy": hard_metrics["accuracy"],
                "hard_gate_balanced_accuracy": hard_metrics["balanced_accuracy"],
                "hard_gate_macro_f1": hard_metrics["macro_f1"],
            })

    ranked = _rank_rows(rows, primary_metric)
    csv_path = output_dir / "pairwise_pipeline_validation.csv"
    pd.DataFrame(ranked).to_csv(csv_path, index=False)
    top = ranked[0]
    top_key = (str(top["stage1_model"]), str(top["stage2_model"]))
    top_probability = probability_cache[top_key]
    top_hard = hard_cache[top_key]
    top_path = output_dir / "top_pair.json"
    atomic_write_json(top_path, {
        "run_id": run_id,
        "strategy_id": strategy_id,
        "selection_split": "validation",
        "selection_metric": primary_metric,
        "selection_rule": [
            f"maximize_{primary_metric}",
            "maximize_macro_auroc" if primary_metric == "macro_auprc" else "maximize_macro_auprc",
            "minimize_multiclass_brier",
            "minimize_multiclass_log_loss",
        ],
        "top_pair": top,
        "pair_count": len(ranked),
        "confirmatory_splits_accessed": False,
    })
    pred = reference.copy().reset_index(drop=True)
    pred["y_true_3class"] = y_true
    pred["stage1_probability_B"] = stage1_frames[top_key[0]]["positive_probability"].to_numpy(dtype=float)
    pred["stage2_probability_C_given_nonB"] = stage2_frames[top_key[1]]["positive_probability"].to_numpy(dtype=float)
    for code in range(3):
        pred[f"prob_class_{code}"] = top_probability[:, code]
    pred["argmax_prediction"] = np.argmax(top_probability, axis=1).astype(np.int64)
    pred["hard_gate_prediction"] = top_hard
    predictions_path = output_dir / "top_pair_predictions.csv.gz"
    pred.to_csv(predictions_path, index=False, compression="gzip")

    clinical_dir = output_dir / "clinical_validation"
    write_multiclass_clinical_report(
        y_true=y_true,
        probabilities=top_probability,
        class_names=EXP1_FLAT_3CLASS_TASK.class_names,
        output_dir=clinical_dir,
        model_name=f"{top_key[0]}->{top_key[1]}",
        split_name="validation",
        dca_thresholds=dca_thresholds_from_config(clinical_config),
    )
    manifest = {
        "schema_version": "exp123_v8_exp2_combined_v1",
        "run_id": run_id,
        "strategy_id": strategy_id,
        "pair_count": len(ranked),
        "selection_metric": primary_metric,
        "pairwise_csv": str(csv_path),
        "pairwise_csv_sha256": sha256_file(csv_path),
        "top_pair_path": str(top_path),
        "top_pair_sha256": sha256_file(top_path),
        "top_pair_predictions": str(predictions_path),
        "top_pair_predictions_sha256": sha256_file(predictions_path),
        "evaluation_distribution": "natural_validation",
        "confirmatory_splits_accessed": False,
    }
    atomic_write_json(lock, manifest)
    return manifest
