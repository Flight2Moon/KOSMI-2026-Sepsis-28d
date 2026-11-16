from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import load_task_artifact
from .config import resolve_path
from .constants import OUTCOME_CLASS_LABELS
from .evaluation_binary import (
    BinaryBootstrapSettings,
    write_binary_evaluation_artifacts,
)
from .evaluation_multiclass import (
    BootstrapSettings,
    write_multiclass_evaluation_artifacts,
)
from .exp123_data import load_feature_frame
from .metrics import validate_probabilities
from .stacking import (
    SuperLearnerArtifact,
    load_super_learner_artifact,
    super_learner_output_dir,
)
from .task_spec import (
    EXP1_FLAT_3CLASS_TASK,
    STAGE1_MORTALITY_TASK,
    STAGE2_LONG_STAY_TASK,
)
from .utils import atomic_write_json, environment_info, sha256_file


CONFIRMATORY_SPLITS = {"internal_test", "external_eicu"}


@dataclass(frozen=True)
class ConfirmatoryCohort:
    split_name: str
    raw_features: pd.DataFrame
    y_true_3class: np.ndarray
    metadata: pd.DataFrame
    feature_names: tuple[str, ...]
    source_dataset_path: str | None = None
    source_assignment_path: str | None = None

    def __post_init__(self) -> None:
        if self.split_name not in CONFIRMATORY_SPLITS:
            raise ValueError(f"Unsupported confirmatory split: {self.split_name}")
        labels = np.asarray(self.y_true_3class, dtype=np.int64).reshape(-1)
        if len(self.raw_features) != len(labels) or len(self.metadata) != len(labels):
            raise ValueError("Confirmatory cohort features, labels, and metadata differ in length.")
        if "source_row_index" not in self.metadata.columns:
            raise KeyError("Confirmatory metadata must contain source_row_index.")
        if self.metadata["source_row_index"].duplicated().any():
            raise ValueError("Confirmatory source_row_index values must be unique.")
        observed = set(np.unique(labels).tolist())
        expected = set(EXP1_FLAT_3CLASS_TASK.expected_classes)
        if observed != expected:
            raise ValueError(
                f"Confirmatory split must contain A/B/C classes. Observed: {sorted(observed)}"
            )
        missing = sorted(set(self.feature_names) - set(self.raw_features.columns))
        if missing:
            raise KeyError(f"Confirmatory cohort is missing features: {missing}")


@dataclass(frozen=True)
class OraclePipelineRunResult:
    split_name: str
    output_dir: str
    sample_count: int
    survivor_count: int
    stage1_threshold: float
    stage2_threshold: float
    pipeline_macro_auroc: float
    pipeline_macro_f1: float
    pipeline_multiclass_brier: float
    oracle_stage2_auroc: float
    oracle_stage2_auprc: float
    oracle_stage2_brier: float
    completed_lock_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _normalize_assignment_source_index(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "source_row_index" not in output.columns:
        if "_source_row_index" not in output.columns:
            raise KeyError("Outcome assignment must contain source_row_index.")
        output = output.rename(columns={"_source_row_index": "source_row_index"})
    output["source_row_index"] = pd.to_numeric(
        output["source_row_index"], errors="raise"
    ).astype(np.int64)
    if output["source_row_index"].duplicated().any():
        raise ValueError("Outcome assignment source_row_index values must be unique.")
    return output


def load_confirmatory_cohort(
    *,
    config: Mapping[str, Any],
    run_dir: Path,
    split_name: str,
) -> ConfirmatoryCohort:
    if split_name not in CONFIRMATORY_SPLITS:
        raise ValueError(f"Unsupported confirmatory split: {split_name}")
    feature_names = tuple(str(value) for value in config["data"]["feature_columns"])
    if split_name == "internal_test":
        dataset_path = resolve_path(dict(config), config["paths"]["mimic_csv"])
        assignment_path = run_dir / "common" / "splits" / "split_assignments.csv.gz"
        assignments = pd.read_csv(assignment_path, compression="infer", low_memory=False)
        assignments = assignments.loc[assignments["split"] == "internal_test"].copy()
    else:
        dataset_path = resolve_path(dict(config), config["paths"]["eicu_csv"])
        assignment_path = (
            run_dir / "common" / "outcomes" / "eicu_outcome_assignments.csv.gz"
        )
        assignments = pd.read_csv(assignment_path, compression="infer", low_memory=False)
        assignments["split"] = "external_eicu"

    assignments = _normalize_assignment_source_index(assignments)
    if "three_class_outcome" not in assignments.columns:
        raise KeyError(f"three_class_outcome is missing from {assignment_path}")
    assignments["three_class_outcome"] = pd.to_numeric(
        assignments["three_class_outcome"], errors="raise"
    ).astype(np.int64)
    assignments = assignments.sort_values("source_row_index", kind="mergesort").reset_index(
        drop=True
    )

    feature_frame = load_feature_frame(dataset_path, feature_names)
    feature_lookup = feature_frame.set_index("_source_row_index", drop=False)
    missing_rows = sorted(
        set(assignments["source_row_index"].tolist())
        - set(feature_lookup.index.astype(int).tolist())
    )
    if missing_rows:
        raise ValueError(
            f"Confirmatory assignment contains {len(missing_rows)} rows absent from dataset."
        )
    raw_features = feature_lookup.loc[
        assignments["source_row_index"].to_numpy(dtype=np.int64), list(feature_names)
    ].copy()
    raw_features.index = pd.RangeIndex(len(raw_features))
    metadata = assignments.reset_index(drop=True)
    return ConfirmatoryCohort(
        split_name=split_name,
        raw_features=raw_features,
        y_true_3class=metadata["three_class_outcome"].to_numpy(dtype=np.int64),
        metadata=metadata,
        feature_names=feature_names,
        source_dataset_path=str(dataset_path),
        source_assignment_path=str(assignment_path),
    )


def restore_three_class_probabilities(
    stage1_probability: np.ndarray,
    stage2_probability: np.ndarray,
) -> np.ndarray:
    p_stage1 = validate_probabilities(stage1_probability, len(stage1_probability))
    p_stage2 = validate_probabilities(stage2_probability, len(stage2_probability))
    if len(p_stage1) != len(p_stage2):
        raise ValueError("Stage1 and Stage2 probabilities differ in length.")
    p_b = p_stage1
    p_c = (1.0 - p_stage1) * p_stage2
    p_a = (1.0 - p_stage1) * (1.0 - p_stage2)
    probabilities = np.column_stack([p_a, p_b, p_c])
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Restored three-class probabilities do not sum to one.")
    return probabilities


def derive_pipeline_classes(
    stage1_probability: np.ndarray,
    stage2_probability: np.ndarray,
    *,
    stage1_threshold: float,
    stage2_threshold: float,
) -> np.ndarray:
    p_stage1 = validate_probabilities(stage1_probability, len(stage1_probability))
    p_stage2 = validate_probabilities(stage2_probability, len(stage2_probability))
    if len(p_stage1) != len(p_stage2):
        raise ValueError("Stage1 and Stage2 probabilities differ in length.")
    predicted = np.zeros(len(p_stage1), dtype=np.int64)
    death = p_stage1 >= float(stage1_threshold)
    predicted[death] = 1
    predicted[~death & (p_stage2 >= float(stage2_threshold))] = 2
    return predicted


def _artifact_file_hashes(artifact_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(artifact_dir)): sha256_file(path)
        for path in sorted(artifact_dir.rglob("*"))
        if path.is_file()
    }


def _predict_stage_base_models(
    *,
    run_dir: Path,
    task_name: str,
    members: Sequence[str],
    raw_features: pd.DataFrame,
    expected_split_hash: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, str]]]:
    stage_dir_name = (
        "stage1_mortality"
        if task_name == STAGE1_MORTALITY_TASK.task_name
        else "stage2_long_stay"
    )
    probabilities: dict[str, np.ndarray] = {}
    hashes: dict[str, dict[str, str]] = {}
    for model_name in members:
        artifact_dir = (
            run_dir / "EXP2_two_stage" / stage_dir_name / model_name / "artifact"
        )
        loaded = load_task_artifact(artifact_dir)
        if loaded.task_name != task_name or loaded.task_type != "binary":
            raise ValueError(f"Base artifact task mismatch: {artifact_dir}")
        if tuple(loaded.feature_names) != tuple(raw_features.columns):
            raise ValueError(
                f"Feature order mismatch between confirmatory cohort and {model_name}."
            )
        artifact_split_hash = loaded.metadata.get("shared_split_hash")
        if expected_split_hash is not None and artifact_split_hash != expected_split_hash:
            raise ValueError(
                f"Base artifact shared split hash mismatch for {task_name}/{model_name}."
            )
        probability = loaded.predict_proba(raw_features)
        probabilities[model_name] = validate_probabilities(probability, len(raw_features))
        hashes[model_name] = _artifact_file_hashes(artifact_dir)
    return probabilities, hashes


def _base_prediction_frame(
    cohort: ConfirmatoryCohort,
    probabilities: Mapping[str, np.ndarray],
    combined: np.ndarray,
    task_name: str,
) -> pd.DataFrame:
    columns = ["source_row_index"]
    if "subject_id" in cohort.metadata.columns:
        columns.append("subject_id")
    output = cohort.metadata.loc[:, columns].copy().reset_index(drop=True)
    output["split"] = cohort.split_name
    output["task_name"] = task_name
    output["y_true_3class"] = cohort.y_true_3class
    for model_name, values in probabilities.items():
        output[f"prob_{model_name}"] = values
    output["super_learner_probability"] = combined
    return output


def _pipeline_prediction_frame(
    cohort: ConfirmatoryCohort,
    stage1_probability: np.ndarray,
    stage2_probability: np.ndarray,
    probabilities_3class: np.ndarray,
    hard_classes: np.ndarray,
    stage1_threshold: float,
    stage2_threshold: float,
) -> pd.DataFrame:
    output = cohort.metadata.copy().reset_index(drop=True)
    output["split"] = cohort.split_name
    output["y_true_3class"] = cohort.y_true_3class
    output["stage1_probability_B"] = stage1_probability
    output["stage2_probability_C_given_survival"] = stage2_probability
    output["stage1_predicted_death"] = (
        stage1_probability >= stage1_threshold
    ).astype(np.int64)
    output["stage2_predicted_long_stay"] = (
        stage2_probability >= stage2_threshold
    ).astype(np.int64)
    for class_code, class_name in OUTCOME_CLASS_LABELS.items():
        output[f"prob_class_{class_code}"] = probabilities_3class[:, class_code]
        output[f"prob_{class_name}"] = probabilities_3class[:, class_code]
    output["probability_argmax_class"] = np.argmax(probabilities_3class, axis=1)
    output["pipeline_predicted_class"] = hard_classes
    output["pipeline_correct"] = (hard_classes == cohort.y_true_3class).astype(np.int64)
    return output


def _error_propagation_frames(
    pipeline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = pipeline["y_true_3class"].to_numpy(dtype=np.int64)
    stage1_death = pipeline["stage1_predicted_death"].to_numpy(dtype=bool)
    stage2_long = pipeline["stage2_predicted_long_stay"].to_numpy(dtype=bool)
    final = pipeline["pipeline_predicted_class"].to_numpy(dtype=np.int64)

    category = np.full(len(pipeline), "correct_pipeline", dtype=object)
    category[(y != 1) & stage1_death] = "stage1_false_positive_survivor_to_B"
    category[(y == 1) & ~stage1_death & ~stage2_long] = "stage1_false_negative_B_to_A"
    category[(y == 1) & ~stage1_death & stage2_long] = "stage1_false_negative_B_to_C"
    category[(y == 0) & ~stage1_death & stage2_long] = "stage2_false_positive_A_to_C"
    category[(y == 2) & ~stage1_death & ~stage2_long] = "stage2_false_negative_C_to_A"
    category[(y == final)] = "correct_pipeline"

    detail_columns = [
        column
        for column in ["source_row_index", "subject_id", "split"]
        if column in pipeline.columns
    ]
    detail = pipeline.loc[:, detail_columns].copy()
    detail["y_true_3class"] = y
    detail["pipeline_predicted_class"] = final
    detail["error_category"] = category
    detail["stage1_error"] = (
        ((y == 1) & ~stage1_death) | ((y != 1) & stage1_death)
    ).astype(np.int64)
    detail["stage2_error_among_stage1_non_death"] = (
        (~stage1_death) & (y != 1) & (final != y)
    ).astype(np.int64)

    summary = (
        pd.Series(category, name="error_category")
        .value_counts(dropna=False)
        .rename_axis("error_category")
        .reset_index(name="count")
    )
    summary["rate"] = summary["count"] / max(len(detail), 1)
    summary["split"] = pipeline["split"].iloc[0]

    transition = pd.crosstab(
        pd.Series(y, name="actual"),
        pd.Series(final, name="predicted"),
        dropna=False,
    ).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
    transition.index = [OUTCOME_CLASS_LABELS[index] for index in transition.index]
    transition.columns = [OUTCOME_CLASS_LABELS[index] for index in transition.columns]
    transition.index.name = "actual"
    transition.columns.name = "predicted"
    return detail, summary, transition


def _pipeline_output_dir(run_dir: Path, split_name: str) -> Path:
    return run_dir / "EXP2_two_stage" / "pipeline_evaluation" / split_name


def _write_exp3_oracle_reference(
    run_dir: Path,
    split_name: str,
    oracle_dir: Path,
    manifest_sha256: str,
) -> Path:
    output = run_dir / "EXP3_survivor_binary" / f"oracle_evaluation_reference_{split_name}.json"
    atomic_write_json(
        output,
        {
            "experiment": "EXP3_survivor_binary",
            "split": split_name,
            "source": str(oracle_dir),
            "source_manifest_sha256": manifest_sha256,
            "population": "ground_truth_survivors_A_or_C",
            "duplicate_evaluation": False,
            "duplicate_training": False,
        },
    )
    (output.with_name(output.name + ".sha256")).write_text(
        sha256_file(output) + "\n", encoding="utf-8"
    )
    return output


def run_oracle_pipeline_for_cohort(
    *,
    run_dir: Path,
    cohort: ConfirmatoryCohort,
    config: Mapping[str, Any],
    overwrite: bool = False,
) -> OraclePipelineRunResult:
    run_dir = run_dir.expanduser().resolve()
    output_dir = _pipeline_output_dir(run_dir, cohort.split_name)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Confirmatory pipeline output already exists: {output_dir}"
            )
        raise RuntimeError(
            "Confirmatory outputs must not be overwritten. Use a new Run ID."
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    stage1_path = super_learner_output_dir(
        run_dir, STAGE1_MORTALITY_TASK.task_name
    ) / "artifact.json"
    stage2_path = super_learner_output_dir(
        run_dir, STAGE2_LONG_STAY_TASK.task_name
    ) / "artifact.json"
    stage1_artifact = load_super_learner_artifact(stage1_path)
    stage2_artifact = load_super_learner_artifact(stage2_path)
    if stage1_artifact.task_name != STAGE1_MORTALITY_TASK.task_name:
        raise ValueError("Stage1 Super Learner artifact has the wrong task.")
    if stage2_artifact.task_name != STAGE2_LONG_STAY_TASK.task_name:
        raise ValueError("Stage2 Super Learner artifact has the wrong task.")
    if stage1_artifact.shared_split_hash != stage2_artifact.shared_split_hash:
        raise ValueError("Stage1 and Stage2 Super Learners use different shared splits.")

    stage1_base, stage1_hashes = _predict_stage_base_models(
        run_dir=run_dir,
        task_name=STAGE1_MORTALITY_TASK.task_name,
        members=stage1_artifact.library_members,
        raw_features=cohort.raw_features,
        expected_split_hash=stage1_artifact.shared_split_hash,
    )
    stage2_base, stage2_hashes = _predict_stage_base_models(
        run_dir=run_dir,
        task_name=STAGE2_LONG_STAY_TASK.task_name,
        members=stage2_artifact.library_members,
        raw_features=cohort.raw_features,
        expected_split_hash=stage2_artifact.shared_split_hash,
    )
    p_stage1 = stage1_artifact.predict(stage1_base)
    p_stage2 = stage2_artifact.predict(stage2_base)
    probabilities_3class = restore_three_class_probabilities(p_stage1, p_stage2)
    hard_classes = derive_pipeline_classes(
        p_stage1,
        p_stage2,
        stage1_threshold=stage1_artifact.validation_decision_threshold,
        stage2_threshold=stage2_artifact.validation_decision_threshold,
    )

    base_dir = output_dir / "base_predictions"
    base_dir.mkdir(parents=True, exist_ok=True)
    _base_prediction_frame(
        cohort, stage1_base, p_stage1, STAGE1_MORTALITY_TASK.task_name
    ).to_csv(base_dir / "stage1_base_predictions.csv.gz", index=False, compression="gzip")
    _base_prediction_frame(
        cohort, stage2_base, p_stage2, STAGE2_LONG_STAY_TASK.task_name
    ).to_csv(base_dir / "stage2_base_predictions.csv.gz", index=False, compression="gzip")

    pipeline = _pipeline_prediction_frame(
        cohort,
        p_stage1,
        p_stage2,
        probabilities_3class,
        hard_classes,
        stage1_artifact.validation_decision_threshold,
        stage2_artifact.validation_decision_threshold,
    )
    pipeline_dir = output_dir / "pipeline_3class"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    pipeline.to_csv(
        pipeline_dir / "predictions.csv.gz", index=False, compression="gzip"
    )

    oracle_mask = cohort.y_true_3class != 1
    oracle_y = (cohort.y_true_3class[oracle_mask] == 2).astype(np.int64)
    oracle_metadata = cohort.metadata.loc[oracle_mask].reset_index(drop=True)
    oracle_predictions = oracle_metadata.copy()
    oracle_predictions["split"] = cohort.split_name
    oracle_predictions["y_true"] = oracle_y
    oracle_predictions["positive_probability"] = p_stage2[oracle_mask]
    oracle_predictions["predicted_at_validation_threshold"] = (
        p_stage2[oracle_mask] >= stage2_artifact.validation_decision_threshold
    ).astype(np.int64)
    oracle_predictions["predicted_at_0_5"] = (p_stage2[oracle_mask] >= 0.5).astype(np.int64)
    oracle_dir = output_dir / "oracle_stage2"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    oracle_predictions.to_csv(
        oracle_dir / "predictions.csv.gz", index=False, compression="gzip"
    )

    evaluation = config.get("evaluation", {})
    pipeline_config = config.get("oracle_pipeline", {})
    repeats = int(
        pipeline_config.get(
            "bootstrap_repeats",
            evaluation.get("external_bootstrap_repeats", 500),
        )
    )
    seed = int(pipeline_config.get("seed", 42))
    confidence = float(evaluation.get("bootstrap_confidence_level", 0.95))
    binary_summary = write_binary_evaluation_artifacts(
        y_true=oracle_y,
        probabilities=p_stage2[oracle_mask],
        selected_threshold=stage2_artifact.validation_decision_threshold,
        default_threshold=float(evaluation.get("default_decision_threshold", 0.5)),
        output_dir=oracle_dir / "evaluation",
        split_name=cohort.split_name,
        model_name="stage2_super_learner_oracle",
        positive_class_name="C_still_admitted",
        calibration_bins=int(evaluation.get("calibration_bins", 10)),
        bootstrap_settings=BinaryBootstrapSettings(
            repeats=repeats,
            seed=seed,
            confidence_level=confidence,
        ),
        save_plots=bool(evaluation.get("save_plots", True)),
    )
    multiclass_summary = write_multiclass_evaluation_artifacts(
        y_true=cohort.y_true_3class,
        probabilities=probabilities_3class,
        class_names=EXP1_FLAT_3CLASS_TASK.class_names,
        output_dir=pipeline_dir / "evaluation",
        split_name=cohort.split_name,
        model_name="two_stage_super_learner_pipeline",
        calibration_bins=int(evaluation.get("calibration_bins", 10)),
        bootstrap_settings=BootstrapSettings(
            repeats=repeats,
            seed=seed,
            confidence_level=confidence,
        ),
        save_plots=bool(evaluation.get("save_plots", True)),
        predicted_classes=hard_classes,
        decision_rule="stage1_validation_threshold_then_stage2_validation_threshold",
    )

    detail, error_summary, transition = _error_propagation_frames(pipeline)
    error_dir = output_dir / "error_propagation"
    error_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(error_dir / "error_propagation.csv.gz", index=False, compression="gzip")
    error_summary.to_csv(error_dir / "error_propagation_summary.csv", index=False)
    transition.to_csv(error_dir / "final_transition_matrix.csv")
    atomic_write_json(
        error_dir / "error_propagation_summary.json",
        {
            "split": cohort.split_name,
            "sample_count": int(len(cohort.y_true_3class)),
            "stage1_error_count": int(detail["stage1_error"].sum()),
            "stage2_error_count_among_stage1_non_death": int(
                detail["stage2_error_among_stage1_non_death"].sum()
            ),
            "category_counts": {
                str(row.error_category): int(row.count)
                for row in error_summary.itertuples(index=False)
            },
        },
    )

    probability_integrity = {
        "sample_count": int(len(cohort.y_true_3class)),
        "all_probabilities_finite": bool(np.isfinite(probabilities_3class).all()),
        "all_probabilities_in_unit_interval": bool(
            ((probabilities_3class >= 0.0) & (probabilities_3class <= 1.0)).all()
        ),
        "maximum_probability_sum_error": float(
            np.max(np.abs(probabilities_3class.sum(axis=1) - 1.0))
        ),
        "hard_class_values": sorted(np.unique(hard_classes).astype(int).tolist()),
        "stage2_applied_to_full_cohort_for_probability_restoration": True,
        "stage2_oracle_evaluated_on_ground_truth_survivors_only": True,
    }
    atomic_write_json(output_dir / "probability_integrity.json", probability_integrity)

    manifest = {
        "step": "oracle_pipeline_confirmatory",
        "split": cohort.split_name,
        "sample_count": int(len(cohort.y_true_3class)),
        "survivor_count": int(oracle_mask.sum()),
        "class_counts": {
            OUTCOME_CLASS_LABELS[class_code]: int(
                (cohort.y_true_3class == class_code).sum()
            )
            for class_code in OUTCOME_CLASS_LABELS
        },
        "stage1_super_learner_artifact": str(stage1_path),
        "stage1_super_learner_sha256": sha256_file(stage1_path),
        "stage2_super_learner_artifact": str(stage2_path),
        "stage2_super_learner_sha256": sha256_file(stage2_path),
        "stage1_base_artifact_hashes": stage1_hashes,
        "stage2_base_artifact_hashes": stage2_hashes,
        "shared_split_hash": stage1_artifact.shared_split_hash,
        "source_dataset_path": cohort.source_dataset_path,
        "source_dataset_sha256": (
            sha256_file(Path(cohort.source_dataset_path))
            if cohort.source_dataset_path and Path(cohort.source_dataset_path).is_file()
            else None
        ),
        "source_assignment_path": cohort.source_assignment_path,
        "source_assignment_sha256": (
            sha256_file(Path(cohort.source_assignment_path))
            if cohort.source_assignment_path and Path(cohort.source_assignment_path).is_file()
            else None
        ),
        "decision_rule": {
            "stage1": "B if P(B) >= frozen validation threshold",
            "stage2": "C if Stage1 non-death and P(C|survival) >= frozen validation threshold",
            "otherwise": "A",
        },
        "probability_restoration": {
            "P_A": "(1-P_B) * (1-P_C_given_survival)",
            "P_B": "P_B",
            "P_C": "(1-P_B) * P_C_given_survival",
        },
        "confirmatory_design_frozen": True,
        "model_refitting": False,
        "threshold_refitting": False,
        "super_learner_weight_refitting": False,
        "environment": environment_info(),
    }
    atomic_write_json(output_dir / "confirmatory_manifest.json", manifest)
    manifest_sha = sha256_file(output_dir / "confirmatory_manifest.json")
    _write_exp3_oracle_reference(run_dir, cohort.split_name, oracle_dir, manifest_sha)

    pipeline_metrics = multiclass_summary["metrics"]
    oracle_metrics = binary_summary["threshold_views"]["validation_selected"]
    lock_payload = {
        "status": "completed",
        "split": cohort.split_name,
        "confirmatory_manifest_sha256": manifest_sha,
        "pipeline_predictions_sha256": sha256_file(
            pipeline_dir / "predictions.csv.gz"
        ),
        "oracle_predictions_sha256": sha256_file(
            oracle_dir / "predictions.csv.gz"
        ),
        "may_not_be_overwritten": True,
        "use_new_run_id_for_any_changed_design": True,
    }
    lock_path = output_dir / "COMPLETED.lock.json"
    atomic_write_json(lock_path, lock_payload)
    lock_sha = sha256_file(lock_path)
    (output_dir / "COMPLETED.lock.json.sha256").write_text(
        lock_sha + "\n", encoding="utf-8"
    )

    result = OraclePipelineRunResult(
        split_name=cohort.split_name,
        output_dir=str(output_dir),
        sample_count=int(len(cohort.y_true_3class)),
        survivor_count=int(oracle_mask.sum()),
        stage1_threshold=float(stage1_artifact.validation_decision_threshold),
        stage2_threshold=float(stage2_artifact.validation_decision_threshold),
        pipeline_macro_auroc=float(pipeline_metrics["macro_auroc"]),
        pipeline_macro_f1=float(pipeline_metrics["macro_f1"]),
        pipeline_multiclass_brier=float(pipeline_metrics["multiclass_brier"]),
        oracle_stage2_auroc=float(oracle_metrics["auroc"]),
        oracle_stage2_auprc=float(oracle_metrics["auprc"]),
        oracle_stage2_brier=float(oracle_metrics["brier_score"]),
        completed_lock_sha256=lock_sha,
    )
    atomic_write_json(output_dir / "oracle_pipeline_result.json", result.to_dict())
    return result


def run_oracle_pipeline_confirmatory(
    *,
    run_dir: Path,
    config: Mapping[str, Any],
    split_name: str,
) -> OraclePipelineRunResult:
    cohort = load_confirmatory_cohort(
        config=config,
        run_dir=run_dir,
        split_name=split_name,
    )
    return run_oracle_pipeline_for_cohort(
        run_dir=run_dir,
        cohort=cohort,
        config=config,
    )
