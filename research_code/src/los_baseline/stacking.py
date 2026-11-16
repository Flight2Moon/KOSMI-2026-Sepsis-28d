from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import StratifiedGroupKFold

from .evaluation_binary import (
    BinaryBootstrapSettings,
    write_binary_evaluation_artifacts,
)
from .metrics import calculate_metrics, select_youden_threshold, validate_probabilities
from .stage_heterogeneity import stage_analysis_root, stage_prediction_path
from .task_spec import STAGE1_MORTALITY_TASK, STAGE2_LONG_STAY_TASK
from .utils import atomic_write_json, environment_info, sha256_file
from .progress import progress_iter


SUPPORTED_TASKS = {
    STAGE1_MORTALITY_TASK.task_name,
    STAGE2_LONG_STAY_TASK.task_name,
}


@dataclass(frozen=True)
class ValidationSubsplit:
    source_row_index: np.ndarray
    subject_id: np.ndarray
    y_true: np.ndarray
    meta_mask: np.ndarray
    threshold_mask: np.ndarray
    split_hash: str

    @property
    def meta_count(self) -> int:
        return int(self.meta_mask.sum())

    @property
    def threshold_count(self) -> int:
        return int(self.threshold_mask.sum())


@dataclass(frozen=True)
class SuperLearnerArtifact:
    artifact_schema_version: str
    task_name: str
    meta_learner_method: str
    library_members: tuple[str, ...]
    weights: tuple[float, ...]
    validation_decision_threshold: float
    fit_split: str
    threshold_selection_split: str
    shared_split_hash: str | None
    validation_subsplit_hash: str
    base_prediction_hashes: dict[str, str]
    heterogeneity_frozen_design_sha256: str
    library_candidates_sha256: str
    selection_metric: str
    selection_metric_value: float
    auroc_margin_guard: float
    practical_tie_margin: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def predict(self, probabilities: Mapping[str, np.ndarray]) -> np.ndarray:
        missing = [name for name in self.library_members if name not in probabilities]
        if missing:
            raise KeyError(f"Missing base probabilities for Super Learner: {missing}")
        matrix = np.column_stack(
            [np.asarray(probabilities[name], dtype=np.float64) for name in self.library_members]
        )
        if matrix.ndim != 2 or matrix.shape[1] != len(self.weights):
            raise ValueError("Base probability matrix does not match artifact weights.")
        result = matrix @ np.asarray(self.weights, dtype=np.float64)
        return validate_probabilities(result, matrix.shape[0])


@dataclass(frozen=True)
class SuperLearnerRunResult:
    task_name: str
    output_dir: str
    selected_members: tuple[str, ...]
    weights: tuple[float, ...]
    validation_decision_threshold: float
    selection_metric: str
    selection_metric_value: float
    validation_subsplit_hash: str
    artifact_sha256: str
    confirmatory_splits_accessed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file was not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _shared_split_hash(run_dir: Path) -> str | None:
    path = run_dir / "common" / "splits" / "split_hash.txt"
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    if len(value) != 64:
        raise ValueError(f"Invalid shared split hash in {path}: {value}")
    return value


def super_learner_output_dir(run_dir: Path, task_name: str) -> Path:
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported Super Learner task: {task_name}")
    return run_dir.expanduser().resolve() / "EXP2_two_stage" / "super_learner" / task_name


def load_stage9_candidate_pool(
    run_dir: Path,
    task_name: str,
) -> tuple[list[str], dict[str, Any]]:
    discovery_dir = stage_analysis_root(run_dir, task_name) / "discovery_validation"
    candidate_path = discovery_dir / "library_candidates.json"
    frozen_path = discovery_dir / "frozen_design.json"
    candidate_payload = _read_json(candidate_path)
    if candidate_payload.get("selection_is_final_super_learner") is not False:
        raise ValueError("Step 8 candidate evidence must not be marked as final.")
    candidates = [str(value) for value in candidate_payload.get("candidate_models", [])]
    if not candidates:
        raise ValueError(f"No Stage9 candidate models were frozen for {task_name}.")
    if len(candidates) != len(set(candidates)):
        raise ValueError("Stage9 candidate models contain duplicates.")
    if not frozen_path.is_file():
        raise FileNotFoundError(f"Frozen heterogeneity design was not found: {frozen_path}")
    payload = {
        "candidate_path": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "frozen_design_path": str(frozen_path),
        "frozen_design_sha256": sha256_file(frozen_path),
        "candidate_payload": candidate_payload,
    }
    return candidates, payload


def load_aligned_validation_predictions(
    run_dir: Path,
    task_name: str,
    model_names: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported Super Learner task: {task_name}")
    model_names = tuple(str(value) for value in model_names)
    if not model_names:
        raise ValueError("At least one base model is required.")

    reference: pd.DataFrame | None = None
    hashes: dict[str, str] = {}
    probability_columns: list[pd.Series] = []
    for model_name in progress_iter(
        model_names, total=len(model_names), description=f"{task_name} base predictions",
        unit="model", leave=False, details={"task": task_name},
    ):
        path = stage_prediction_path(run_dir, task_name, model_name, "validation")
        if not path.is_file():
            raise FileNotFoundError(f"Validation prediction was not found: {path}")
        frame = pd.read_csv(path, compression="infer", low_memory=False)
        required = {"source_row_index", "y_true", "positive_probability"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"Missing prediction columns in {path}: {missing}")
        if "task_name" in frame.columns:
            observed = set(frame["task_name"].dropna().astype(str).unique())
            if observed != {task_name}:
                raise ValueError(f"Task mismatch in {path}: {sorted(observed)}")
        if "model_name" in frame.columns:
            observed = set(frame["model_name"].dropna().astype(str).unique())
            if observed != {model_name}:
                raise ValueError(f"Model mismatch in {path}: {sorted(observed)}")

        columns = ["source_row_index", "y_true", "positive_probability"]
        if "subject_id" in frame.columns:
            columns.insert(1, "subject_id")
        current = frame.loc[:, columns].copy()
        current["source_row_index"] = pd.to_numeric(
            current["source_row_index"], errors="raise"
        ).astype(np.int64)
        current["y_true"] = pd.to_numeric(current["y_true"], errors="raise").astype(np.int64)
        current["positive_probability"] = validate_probabilities(
            pd.to_numeric(current["positive_probability"], errors="raise").to_numpy(dtype=float),
            len(current),
        )
        if current["source_row_index"].duplicated().any():
            raise ValueError(f"Duplicate source_row_index values in {path}")
        if not set(current["y_true"].unique()).issubset({0, 1}):
            raise ValueError(f"Non-binary y_true values found in {path}")
        if "subject_id" not in current.columns:
            current["subject_id"] = current["source_row_index"].astype(str)
        current["subject_id"] = current["subject_id"].astype(str)
        current = current.sort_values("source_row_index").reset_index(drop=True)

        identity = current.loc[:, ["source_row_index", "subject_id", "y_true"]]
        if reference is None:
            reference = identity.copy()
        else:
            if not reference.equals(identity):
                raise ValueError(
                    "Base models do not share identical validation patients and labels. "
                    f"Mismatch detected for {model_name}."
                )
        probability_columns.append(
            pd.Series(
                current["positive_probability"].to_numpy(dtype=float),
                name=f"prob_{model_name}",
            )
        )
        hashes[model_name] = sha256_file(path)

    assert reference is not None
    aligned = pd.concat([reference, *probability_columns], axis=1)
    if np.unique(aligned["y_true"].to_numpy(dtype=int)).size != 2:
        raise ValueError("Validation predictions must contain both binary classes.")
    return aligned, hashes


def create_validation_subsplit(
    aligned: pd.DataFrame,
    *,
    seed: int,
) -> ValidationSubsplit:
    required = {"source_row_index", "subject_id", "y_true"}
    missing = sorted(required - set(aligned.columns))
    if missing:
        raise KeyError(f"Validation frame is missing columns: {missing}")
    y = aligned["y_true"].to_numpy(dtype=np.int64)
    groups = aligned["subject_id"].astype(str).to_numpy()
    if len(np.unique(groups)) < 4:
        raise ValueError("At least four patient groups are required for validation subsplitting.")
    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    try:
        meta_indices, threshold_indices = next(
            splitter.split(np.zeros((len(aligned), 1)), y, groups)
        )
    except ValueError as exception:
        raise ValueError(
            "Unable to create patient-level validation meta/threshold split."
        ) from exception
    meta_mask = np.zeros(len(aligned), dtype=bool)
    threshold_mask = np.zeros(len(aligned), dtype=bool)
    meta_mask[np.asarray(meta_indices, dtype=int)] = True
    threshold_mask[np.asarray(threshold_indices, dtype=int)] = True
    if np.any(meta_mask & threshold_mask) or not np.all(meta_mask | threshold_mask):
        raise RuntimeError("Validation subsplit masks are not a complete disjoint partition.")
    if set(groups[meta_mask]) & set(groups[threshold_mask]):
        raise RuntimeError("Patient overlap detected between validation subsplits.")
    for name, mask in (("validation_meta", meta_mask), ("validation_threshold", threshold_mask)):
        if np.unique(y[mask]).size != 2:
            raise ValueError(f"Both classes are required in {name}.")

    assignment = pd.DataFrame(
        {
            "source_row_index": aligned["source_row_index"].to_numpy(dtype=np.int64),
            "subject_id": groups,
            "y_true": y,
            "validation_subsplit": np.where(meta_mask, "validation_meta", "validation_threshold"),
        }
    ).sort_values("source_row_index")
    payload = assignment.to_csv(index=False).encode("utf-8")
    import hashlib

    split_hash = hashlib.sha256(payload).hexdigest()
    return ValidationSubsplit(
        source_row_index=aligned["source_row_index"].to_numpy(dtype=np.int64),
        subject_id=groups,
        y_true=y,
        meta_mask=meta_mask,
        threshold_mask=threshold_mask,
        split_hash=split_hash,
    )


def enumerate_library_subsets(model_names: Sequence[str]) -> list[tuple[str, ...]]:
    names = tuple(dict.fromkeys(str(value) for value in model_names))
    if not names:
        raise ValueError("At least one candidate model is required.")
    if len(names) == 1:
        return [names]
    subsets: list[tuple[str, ...]] = []
    for size in range(2, len(names) + 1):
        subsets.extend(itertools.combinations(names, size))
    return subsets


def fit_nnls_weights(
    probability_matrix: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, float, str]:
    matrix = np.asarray(probability_matrix, dtype=np.float64)
    labels = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != len(labels):
        raise ValueError("NNLS probability matrix and labels have incompatible shapes.")
    if matrix.shape[1] < 1:
        raise ValueError("NNLS requires at least one base model.")
    if not np.isfinite(matrix).all() or not np.isfinite(labels).all():
        raise ValueError("NNLS inputs contain non-finite values.")
    raw, residual_norm = nnls(matrix, labels)
    total = float(raw.sum())
    if total <= 1e-15:
        weights = np.full(matrix.shape[1], 1.0 / matrix.shape[1], dtype=np.float64)
        status = "uniform_fallback_zero_nnls_sum"
    else:
        weights = raw / total
        status = "completed"
    if np.any(weights < -1e-12) or not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise RuntimeError("Normalized NNLS weights violate the convex-weight contract.")
    return weights, float(residual_norm), status


def _probability_matrix(aligned: pd.DataFrame, members: Sequence[str]) -> np.ndarray:
    return np.column_stack(
        [aligned[f"prob_{model_name}"].to_numpy(dtype=np.float64) for model_name in members]
    )


def _subset_rows(
    *,
    aligned: pd.DataFrame,
    subsplit: ValidationSubsplit,
    subsets: Sequence[tuple[str, ...]],
    task_name: str,
    primary_metric: str,
    auroc_margin_guard: float,
    practical_tie_margin: float,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    full_predictions: dict[str, np.ndarray] = {}
    y = aligned["y_true"].to_numpy(dtype=np.int64)
    for subset_index, members in progress_iter(
        list(enumerate(subsets)), total=len(subsets),
        description=f"{task_name} Super Learner subsets", unit="subset",
        leave=False, details={"task": task_name},
    ):
        matrix = _probability_matrix(aligned, members)
        weights, residual_norm, fit_status = fit_nnls_weights(
            matrix[subsplit.meta_mask], y[subsplit.meta_mask]
        )
        combined = validate_probabilities(matrix @ weights, len(aligned))
        threshold = select_youden_threshold(
            y[subsplit.threshold_mask], combined[subsplit.threshold_mask]
        )
        threshold_metrics = calculate_metrics(
            y[subsplit.threshold_mask],
            combined[subsplit.threshold_mask],
            threshold,
        )
        meta_metrics = calculate_metrics(
            y[subsplit.meta_mask],
            combined[subsplit.meta_mask],
            threshold,
        )
        subset_id = f"subset_{subset_index:03d}"
        full_predictions[subset_id] = combined
        effective_members = [
            name for name, weight in zip(members, weights, strict=True) if weight > 1e-10
        ]
        rows.append(
            {
                "subset_id": subset_id,
                "task_name": task_name,
                "library_members": "|".join(members),
                "library_size": len(members),
                "effective_members": "|".join(effective_members),
                "effective_library_size": len(effective_members),
                "weights_json": json.dumps(
                    {name: float(weight) for name, weight in zip(members, weights, strict=True)},
                    sort_keys=True,
                ),
                "nnls_residual_norm": residual_norm,
                "nnls_fit_status": fit_status,
                "validation_decision_threshold": threshold,
                "meta_auroc": meta_metrics["auroc"],
                "meta_auprc": meta_metrics["auprc"],
                "meta_brier_score": meta_metrics["brier_score"],
                "threshold_auroc": threshold_metrics["auroc"],
                "threshold_auprc": threshold_metrics["auprc"],
                "threshold_brier_score": threshold_metrics["brier_score"],
                "threshold_log_loss": threshold_metrics["log_loss"],
                "threshold_balanced_accuracy": threshold_metrics["balanced_accuracy"],
                "threshold_f1": threshold_metrics["f1_score"],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No Super Learner subset candidates were evaluated.")
    best_auroc = float(frame["threshold_auroc"].max())
    frame["eligible_by_auroc_guard"] = (
        frame["threshold_auroc"] >= best_auroc - auroc_margin_guard
    )
    eligible = frame.loc[frame["eligible_by_auroc_guard"]].copy()
    if eligible.empty:
        raise RuntimeError("No Super Learner subset passed the AUROC margin guard.")
    if primary_metric not in {"auroc", "auprc"}:
        raise ValueError(f"Unsupported Super Learner primary metric: {primary_metric}")
    primary_column = f"threshold_{primary_metric}"
    best_primary = float(eligible[primary_column].max())
    eligible["within_primary_practical_tie"] = (
        eligible[primary_column] >= best_primary - practical_tie_margin
    )
    frame["within_primary_practical_tie"] = False
    frame.loc[eligible.index, "within_primary_practical_tie"] = eligible[
        "within_primary_practical_tie"
    ]
    selected_pool = eligible.loc[eligible["within_primary_practical_tie"]].copy()
    selected_pool = selected_pool.sort_values(
        ["threshold_brier_score", "effective_library_size", "library_size", "subset_id"],
        ascending=[True, True, True, True],
    )
    selected_row = selected_pool.iloc[0].to_dict()
    frame["selected"] = frame["subset_id"] == selected_row["subset_id"]
    selected_row["selection_primary_metric"] = primary_metric
    selected_row["best_threshold_auroc"] = best_auroc
    selected_row["auroc_margin_guard"] = auroc_margin_guard
    selected_row["practical_tie_margin"] = practical_tie_margin
    return frame, selected_row, full_predictions


def save_super_learner_artifact(path: Path, artifact: SuperLearnerArtifact) -> str:
    atomic_write_json(path, artifact.to_dict())
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def load_super_learner_artifact(path: Path) -> SuperLearnerArtifact:
    payload = _read_json(path)
    return SuperLearnerArtifact(
        artifact_schema_version=str(payload["artifact_schema_version"]),
        task_name=str(payload["task_name"]),
        meta_learner_method=str(payload["meta_learner_method"]),
        library_members=tuple(str(value) for value in payload["library_members"]),
        weights=tuple(float(value) for value in payload["weights"]),
        validation_decision_threshold=float(payload["validation_decision_threshold"]),
        fit_split=str(payload["fit_split"]),
        threshold_selection_split=str(payload["threshold_selection_split"]),
        shared_split_hash=(
            None if payload.get("shared_split_hash") is None else str(payload["shared_split_hash"])
        ),
        validation_subsplit_hash=str(payload["validation_subsplit_hash"]),
        base_prediction_hashes={
            str(key): str(value) for key, value in payload["base_prediction_hashes"].items()
        },
        heterogeneity_frozen_design_sha256=str(
            payload["heterogeneity_frozen_design_sha256"]
        ),
        library_candidates_sha256=str(payload["library_candidates_sha256"]),
        selection_metric=str(payload["selection_metric"]),
        selection_metric_value=float(payload["selection_metric_value"]),
        auroc_margin_guard=float(payload["auroc_margin_guard"]),
        practical_tie_margin=float(payload["practical_tie_margin"]),
        seed=int(payload["seed"]),
    )


def _write_prediction_frame(
    path: Path,
    aligned: pd.DataFrame,
    probabilities: np.ndarray,
    mask: np.ndarray,
    *,
    task_name: str,
    threshold: float,
    split_name: str,
) -> pd.DataFrame:
    selected = aligned.loc[mask, ["source_row_index", "subject_id", "y_true"]].copy()
    selected["split"] = split_name
    selected["task_name"] = task_name
    selected["model_name"] = "super_learner"
    selected["positive_probability"] = probabilities[mask]
    selected["predicted_at_validation_threshold"] = (
        selected["positive_probability"].to_numpy(dtype=float) >= threshold
    ).astype(np.int64)
    selected["predicted_at_0_5"] = (
        selected["positive_probability"].to_numpy(dtype=float) >= 0.5
    ).astype(np.int64)
    selected.to_csv(path, index=False, compression="gzip")
    return selected


def run_stage_super_learner(
    *,
    run_dir: Path,
    task_name: str,
    config: Mapping[str, Any],
    overwrite: bool = False,
) -> SuperLearnerRunResult:
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported Super Learner task: {task_name}")
    run_dir = run_dir.expanduser().resolve()
    output_dir = super_learner_output_dir(run_dir, task_name)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Super Learner output already exists: {output_dir}")
        raise RuntimeError("Super Learner outputs are immutable. Use a new Run ID.")
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_models, evidence = load_stage9_candidate_pool(run_dir, task_name)
    aligned, prediction_hashes = load_aligned_validation_predictions(
        run_dir, task_name, candidate_models
    )
    super_config = config.get("super_learner", {})
    method = str(super_config.get("meta_learner_method", "nnls"))
    if method != "nnls":
        raise ValueError(f"Unsupported Super Learner method: {method}")
    seed = int(super_config.get("seed", config.get("optimization", {}).get("seed", 42)))
    subsplit = create_validation_subsplit(aligned, seed=seed)

    assignment = pd.DataFrame(
        {
            "source_row_index": aligned["source_row_index"],
            "subject_id": aligned["subject_id"],
            "y_true": aligned["y_true"],
            "validation_subsplit": np.where(
                subsplit.meta_mask, "validation_meta", "validation_threshold"
            ),
        }
    )
    assignment.to_csv(
        output_dir / "validation_subsplit_assignments.csv.gz",
        index=False,
        compression="gzip",
    )
    group_overlap = bool(
        set(assignment.loc[assignment["validation_subsplit"] == "validation_meta", "subject_id"])
        & set(
            assignment.loc[
                assignment["validation_subsplit"] == "validation_threshold", "subject_id"
            ]
        )
    )
    atomic_write_json(
        output_dir / "validation_subsplit_manifest.json",
        {
            "task_name": task_name,
            "seed": seed,
            "method": "StratifiedGroupKFold(n_splits=2)",
            "meta_count": subsplit.meta_count,
            "threshold_count": subsplit.threshold_count,
            "patient_group_overlap": group_overlap,
            "both_classes_meta": bool(
                np.unique(subsplit.y_true[subsplit.meta_mask]).size == 2
            ),
            "both_classes_threshold": bool(
                np.unique(subsplit.y_true[subsplit.threshold_mask]).size == 2
            ),
            "validation_subsplit_hash": subsplit.split_hash,
        },
    )

    subsets = enumerate_library_subsets(candidate_models)
    objective = config["optimization_objectives"][task_name]
    primary_metric = str(objective["primary"])
    selection_config = config.get("optimization", {}).get("selection", {})
    auroc_guard = float(
        super_config.get(
            "auroc_margin_guard",
            selection_config.get("auroc_margin_guard", 0.02),
        )
    )
    tie_margin = float(
        super_config.get(
            "practical_tie_margin",
            selection_config.get("practical_tie_margin", 0.005),
        )
    )
    search_frame, selected_row, predictions_by_subset = _subset_rows(
        aligned=aligned,
        subsplit=subsplit,
        subsets=subsets,
        task_name=task_name,
        primary_metric=primary_metric,
        auroc_margin_guard=auroc_guard,
        practical_tie_margin=tie_margin,
    )
    search_frame.to_csv(output_dir / "subset_search_results.csv", index=False)
    selected_members = tuple(str(selected_row["library_members"]).split("|"))
    weight_mapping = json.loads(str(selected_row["weights_json"]))
    weights = tuple(float(weight_mapping[name]) for name in selected_members)
    selected_probabilities = predictions_by_subset[str(selected_row["subset_id"])]
    selected_threshold = float(selected_row["validation_decision_threshold"])

    artifact = SuperLearnerArtifact(
        artifact_schema_version="exp123_super_learner_v1",
        task_name=task_name,
        meta_learner_method=method,
        library_members=selected_members,
        weights=weights,
        validation_decision_threshold=selected_threshold,
        fit_split="validation_meta",
        threshold_selection_split="validation_threshold",
        shared_split_hash=_shared_split_hash(run_dir),
        validation_subsplit_hash=subsplit.split_hash,
        base_prediction_hashes={name: prediction_hashes[name] for name in selected_members},
        heterogeneity_frozen_design_sha256=str(evidence["frozen_design_sha256"]),
        library_candidates_sha256=str(evidence["candidate_sha256"]),
        selection_metric=primary_metric,
        selection_metric_value=float(selected_row[f"threshold_{primary_metric}"]),
        auroc_margin_guard=auroc_guard,
        practical_tie_margin=tie_margin,
        seed=seed,
    )
    artifact_path = output_dir / "artifact.json"
    artifact_digest = save_super_learner_artifact(artifact_path, artifact)
    loaded = load_super_learner_artifact(artifact_path)
    probability_mapping = {
        name: aligned[f"prob_{name}"].to_numpy(dtype=float)
        for name in selected_members
    }
    reloaded_probabilities = loaded.predict(probability_mapping)
    maximum_difference = float(
        np.max(np.abs(reloaded_probabilities - selected_probabilities))
    )
    integrity = {
        "artifact_reloaded": True,
        "all_validation_predictions_match": bool(maximum_difference <= 1e-12),
        "maximum_absolute_difference": maximum_difference,
        "tolerance": 1e-12,
        "weights_non_negative": bool(np.all(np.asarray(weights) >= -1e-12)),
        "weights_sum_to_one": bool(np.isclose(sum(weights), 1.0, atol=1e-10)),
        "confirmatory_splits_accessed": False,
    }
    if not integrity["all_validation_predictions_match"]:
        raise RuntimeError("Reloaded Super Learner predictions do not match.")
    atomic_write_json(output_dir / "artifact_integrity.json", integrity)

    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    meta_prediction = _write_prediction_frame(
        predictions_dir / "validation_meta.csv.gz",
        aligned,
        selected_probabilities,
        subsplit.meta_mask,
        task_name=task_name,
        threshold=selected_threshold,
        split_name="validation_meta",
    )
    threshold_prediction = _write_prediction_frame(
        predictions_dir / "validation_threshold.csv.gz",
        aligned,
        selected_probabilities,
        subsplit.threshold_mask,
        task_name=task_name,
        threshold=selected_threshold,
        split_name="validation_threshold",
    )
    full_mask = np.ones(len(aligned), dtype=bool)
    full_prediction = _write_prediction_frame(
        predictions_dir / "validation_full.csv.gz",
        aligned,
        selected_probabilities,
        full_mask,
        task_name=task_name,
        threshold=selected_threshold,
        split_name="validation_full_exploratory",
    )

    evaluation_config = config.get("evaluation", {})
    bootstrap_repeats = int(
        super_config.get(
            "bootstrap_repeats",
            evaluation_config.get("binary_bootstrap_repeats", 500),
        )
    )
    bootstrap = BinaryBootstrapSettings(
        repeats=bootstrap_repeats,
        seed=seed,
        confidence_level=float(evaluation_config.get("bootstrap_confidence_level", 0.95)),
    )
    for name, frame, role in (
        ("validation_meta", meta_prediction, "meta_fit_evaluation"),
        ("validation_threshold", threshold_prediction, "threshold_selection_evaluation"),
        ("validation_full", full_prediction, "exploratory_full_validation_evaluation"),
    ):
        evaluation_dir = output_dir / "evaluation" / name
        write_binary_evaluation_artifacts(
            y_true=frame["y_true"].to_numpy(dtype=np.int64),
            probabilities=frame["positive_probability"].to_numpy(dtype=float),
            selected_threshold=selected_threshold,
            default_threshold=float(evaluation_config.get("default_decision_threshold", 0.5)),
            output_dir=evaluation_dir,
            split_name=name,
            model_name="super_learner",
            positive_class_name=(
                "B_died" if task_name == STAGE1_MORTALITY_TASK.task_name else "C_still_admitted"
            ),
            calibration_bins=int(evaluation_config.get("calibration_bins", 10)),
            bootstrap_settings=(bootstrap if name == "validation_threshold" else None),
            save_plots=bool(evaluation_config.get("save_plots", True)),
        )
        atomic_write_json(
            evaluation_dir / "evaluation_role.json",
            {
                "role": role,
                "confirmatory": False,
                "may_be_used_to_change_internal_test_or_eicu_results": False,
            },
        )

    selected_payload = {
        "task_name": task_name,
        "selected_subset_id": selected_row["subset_id"],
        "selected_members": list(selected_members),
        "weights": {name: weight for name, weight in zip(selected_members, weights, strict=True)},
        "validation_decision_threshold": selected_threshold,
        "selection_primary_metric": primary_metric,
        "selection_primary_metric_value": float(selected_row[f"threshold_{primary_metric}"]),
        "threshold_auroc": float(selected_row["threshold_auroc"]),
        "threshold_auprc": float(selected_row["threshold_auprc"]),
        "threshold_brier_score": float(selected_row["threshold_brier_score"]),
        "auroc_margin_guard": auroc_guard,
        "practical_tie_margin": tie_margin,
        "candidate_models_from_step8": candidate_models,
        "selection_split": "validation_threshold",
        "weights_fit_split": "validation_meta",
        "confirmatory_splits_accessed": False,
        "selection_is_final_for_confirmatory_pipeline": True,
    }
    atomic_write_json(output_dir / "selected_library.json", selected_payload)
    atomic_write_json(
        output_dir / "super_learner_manifest.json",
        {
            "step": "stage_super_learner",
            "task_name": task_name,
            "meta_learner_method": method,
            "candidate_models": candidate_models,
            "candidate_subset_count": len(subsets),
            "selected_members": list(selected_members),
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_digest,
            "data_access": {
                "validation": True,
                "validation_meta": True,
                "validation_threshold": True,
                "internal_test": False,
                "external_eicu": False,
            },
            "frozen_inputs": {
                "library_candidates_sha256": evidence["candidate_sha256"],
                "heterogeneity_frozen_design_sha256": evidence["frozen_design_sha256"],
                "base_prediction_hashes": prediction_hashes,
                "validation_subsplit_hash": subsplit.split_hash,
            },
            "environment": environment_info(),
        },
    )
    result = SuperLearnerRunResult(
        task_name=task_name,
        output_dir=str(output_dir),
        selected_members=selected_members,
        weights=weights,
        validation_decision_threshold=selected_threshold,
        selection_metric=primary_metric,
        selection_metric_value=float(selected_row[f"threshold_{primary_metric}"]),
        validation_subsplit_hash=subsplit.split_hash,
        artifact_sha256=artifact_digest,
        confirmatory_splits_accessed=False,
    )
    atomic_write_json(output_dir / "super_learner_result.json", result.to_dict())
    return result


def run_all_stage_super_learners(
    *,
    run_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, SuperLearnerRunResult]:
    return {
        task_name: run_stage_super_learner(
            run_dir=run_dir,
            task_name=task_name,
            config=config,
        )
        for task_name in (
            STAGE1_MORTALITY_TASK.task_name,
            STAGE2_LONG_STAY_TASK.task_name,
        )
    }
