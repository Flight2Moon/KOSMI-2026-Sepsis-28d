from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import load_clinical_dataset
from .heterogeneity import (
    DEFAULT_MODELS,
    DEFAULT_SUBGROUP_VARIABLES,
    StrataDefinition,
    apply_fdr_to_interaction_results,
    apply_strata_definition,
    compute_subgroup_metrics,
    evaluate_model_eligibility_per_stratum,
    fit_strata_definition,
    plot_error_vs_covariate,
    plot_heterogeneity_heatmap,
    rank_models_per_stratum,
    test_model_subgroup_interaction,
)
from .task_spec import STAGE1_MORTALITY_TASK, STAGE2_LONG_STAY_TASK
from .utils import atomic_write_json, environment_info, sha256_file
from .progress import progress_iter


TASK_OUTPUT_ROOTS: dict[str, Path] = {
    STAGE1_MORTALITY_TASK.task_name: Path("EXP2_two_stage") / "stage1_mortality",
    STAGE2_LONG_STAY_TASK.task_name: Path("EXP2_two_stage") / "stage2_long_stay",
}


@dataclass(frozen=True)
class StageHeterogeneityConfig:
    run_dir: Path
    task_name: str
    discovery_split: str = "validation"
    model_names: tuple[str, ...] = DEFAULT_MODELS
    subgroup_variables: tuple[str, ...] = DEFAULT_SUBGROUP_VARIABLES
    n_bins: int = 3
    minimum_bin_size: int = 30
    minimum_class_count: int = 10
    default_probability_threshold: float = 0.5
    fdr_method: str = "fdr_bh"
    fdr_alpha: float = 0.05
    brier_tie_margin: float = 0.005
    auroc_margin_guard: float = 0.02
    random_seed: int = 42
    trend_bins: int = 50

    def __post_init__(self) -> None:
        if self.task_name not in TASK_OUTPUT_ROOTS:
            raise ValueError(f"Unsupported stage heterogeneity task: {self.task_name}")
        if self.discovery_split != "validation":
            raise ValueError("Stage heterogeneity discovery is restricted to validation.")
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2.")
        if self.minimum_bin_size < 1 or self.minimum_class_count < 1:
            raise ValueError("Minimum sample guards must be positive.")
        if self.auroc_margin_guard < 0:
            raise ValueError("auroc_margin_guard must be non-negative.")
        if not 0 < self.fdr_alpha < 1:
            raise ValueError("fdr_alpha must be between 0 and 1.")


def stage_task_root(run_dir: Path, task_name: str) -> Path:
    if task_name not in TASK_OUTPUT_ROOTS:
        raise ValueError(f"Unsupported stage heterogeneity task: {task_name}")
    return run_dir.expanduser().resolve() / TASK_OUTPUT_ROOTS[task_name]


def stage_analysis_root(run_dir: Path, task_name: str) -> Path:
    if task_name not in TASK_OUTPUT_ROOTS:
        raise ValueError(f"Unsupported stage heterogeneity task: {task_name}")
    return (
        run_dir.expanduser().resolve()
        / "EXP2_two_stage"
        / "heterogeneity_analysis"
        / task_name
    )


def stage_prediction_path(
    run_dir: Path,
    task_name: str,
    model_name: str,
    split: str,
) -> Path:
    return (
        stage_task_root(run_dir, task_name)
        / model_name
        / "evaluation"
        / split
        / "predictions.csv.gz"
    )


def _load_covariates(
    dataset_path: Path,
    subgroup_variables: Iterable[str],
    target_column: str,
    identifier_columns: Iterable[str],
) -> pd.DataFrame:
    variables = list(dict.fromkeys(str(value) for value in subgroup_variables))
    dataset = load_clinical_dataset(
        csv_path=dataset_path,
        feature_columns=variables,
        target_column=target_column,
        candidate_identifier_columns=list(identifier_columns),
    )
    frame = dataset.frame.loc[:, ["_source_row_index", *variables]].rename(
        columns={"_source_row_index": "source_row_index"}
    )
    if frame["source_row_index"].duplicated().any():
        raise ValueError("Covariate source_row_index values are not unique.")
    return frame


def load_stage_predictions_with_covariates(
    *,
    run_dir: Path,
    task_name: str,
    model_names: Iterable[str],
    split: str,
    dataset_path: Path,
    subgroup_variables: Iterable[str] = DEFAULT_SUBGROUP_VARIABLES,
    target_column: str = "LOS",
    identifier_columns: Iterable[str] = ("subject_id", "hadm_id", "stay_id"),
) -> pd.DataFrame:
    """Load Stage1 or Stage2 predictions and join prespecified covariates."""
    if split not in {"validation", "internal_test", "external_eicu"}:
        raise ValueError(f"Unsupported heterogeneity split: {split}")
    run_dir = run_dir.expanduser().resolve()
    subgroup_variables = tuple(str(value) for value in subgroup_variables)
    model_names = tuple(str(value) for value in model_names)
    covariates = _load_covariates(
        dataset_path,
        subgroup_variables,
        target_column,
        identifier_columns,
    )
    model_frames: list[pd.DataFrame] = []
    reference_indices: np.ndarray | None = None
    reference_y: np.ndarray | None = None

    for model_name in progress_iter(
        model_names, total=len(model_names),
        description=f"{task_name}/{split} heterogeneity inputs", unit="model",
        leave=False, details={"task": task_name, "split": split},
    ):
        path = stage_prediction_path(run_dir, task_name, str(model_name), split)
        if not path.is_file():
            raise FileNotFoundError(f"Prediction file was not found: {path}")
        prediction = pd.read_csv(path, compression="infer", low_memory=False)
        required = {"source_row_index", "y_true", "positive_probability"}
        missing = sorted(required - set(prediction.columns))
        if missing:
            raise KeyError(f"Missing prediction columns in {path}: {missing}")
        if "task_name" in prediction.columns:
            task_values = set(prediction["task_name"].dropna().astype(str).unique())
            if task_values != {task_name}:
                raise ValueError(
                    f"Prediction task mismatch in {path}: {sorted(task_values)}"
                )
        if "model_name" in prediction.columns:
            model_values = set(prediction["model_name"].dropna().astype(str).unique())
            if model_values != {str(model_name)}:
                raise ValueError(
                    f"Prediction model mismatch in {path}: {sorted(model_values)}"
                )

        prediction = prediction.loc[
            :, ["source_row_index", "y_true", "positive_probability"]
        ].copy()
        prediction["source_row_index"] = pd.to_numeric(
            prediction["source_row_index"], errors="raise"
        ).astype(np.int64)
        prediction["y_true"] = pd.to_numeric(
            prediction["y_true"], errors="raise"
        ).astype(np.int32)
        prediction["positive_probability"] = pd.to_numeric(
            prediction["positive_probability"], errors="raise"
        ).astype(np.float64)
        if prediction["source_row_index"].duplicated().any():
            raise ValueError(f"Duplicate source rows in {path}")
        if not set(prediction["y_true"].unique()).issubset({0, 1}):
            raise ValueError(f"Non-binary y_true values found in {path}")
        probabilities = prediction["positive_probability"].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise ValueError(f"Invalid probabilities found in {path}")
        prediction = prediction.sort_values("source_row_index").reset_index(drop=True)

        current_indices = prediction["source_row_index"].to_numpy(dtype=np.int64)
        current_y = prediction["y_true"].to_numpy(dtype=np.int32)
        if reference_indices is None:
            reference_indices = current_indices
            reference_y = current_y
        else:
            if not np.array_equal(reference_indices, current_indices):
                raise ValueError(
                    "Stage models do not share the same source_row_index set. "
                    f"Mismatch detected for {model_name}/{split}."
                )
            if not np.array_equal(reference_y, current_y):
                raise ValueError(
                    "Stage models do not share identical y_true values. "
                    f"Mismatch detected for {model_name}/{split}."
                )

        merged = prediction.merge(
            covariates,
            on="source_row_index",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        unmatched = int((merged["_merge"] != "both").sum())
        if unmatched:
            raise ValueError(
                f"Covariate join failed for {unmatched} rows in {model_name}/{split}."
            )
        merged = merged.drop(columns="_merge")
        missing_covariates = merged[list(subgroup_variables)].isna().sum()
        if int(missing_covariates.sum()) > 0:
            raise ValueError(
                "Subgroup covariates contain missing values after join: "
                f"{missing_covariates[missing_covariates > 0].to_dict()}"
            )
        merged = merged.rename(columns={"positive_probability": "y_prob"})
        merged["model_name"] = str(model_name)
        merged["task_name"] = task_name
        merged["split"] = split
        merged["squared_error"] = (
            merged["y_true"].astype(float) - merged["y_prob"]
        ) ** 2
        clipped = np.clip(merged["y_prob"].to_numpy(dtype=float), 1e-15, 1 - 1e-15)
        y = merged["y_true"].to_numpy(dtype=float)
        merged["log_loss_component"] = -(
            y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)
        )
        model_frames.append(merged)

    if not model_frames:
        raise ValueError("At least one model prediction file is required.")
    return pd.concat(model_frames, ignore_index=True)


def _shared_split_hash(run_dir: Path) -> str | None:
    path = run_dir.expanduser().resolve() / "common" / "splits" / "split_hash.txt"
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    if len(value) != 64:
        raise ValueError(f"Invalid shared split hash in {path}: {value}")
    return value


def _prediction_hashes(
    run_dir: Path,
    task_name: str,
    model_names: Iterable[str],
    split: str,
) -> dict[str, str]:
    return {
        str(model_name): sha256_file(
            stage_prediction_path(run_dir, task_name, str(model_name), split)
        )
        for model_name in model_names
    }


def _definition_from_payload(payload: Mapping[str, Any]) -> StrataDefinition:
    return StrataDefinition(
        variable=str(payload["variable"]),
        method=str(payload["method"]),
        labels=tuple(str(value) for value in payload["labels"]),
        edges=tuple(float(value) for value in payload["edges"]),
        requested_n_bins=int(payload["requested_n_bins"]),
        actual_n_bins=int(payload["actual_n_bins"]),
        fitted_on_split=str(payload["fitted_on_split"]),
        minimum_bin_size=int(payload["minimum_bin_size"]),
        minimum_class_count=int(payload["minimum_class_count"]),
        duplicate_edges_dropped=bool(payload["duplicate_edges_dropped"]),
    )


def _write_sha256_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def _library_candidate_evidence(
    eligibility: pd.DataFrame,
    rankings: pd.DataFrame,
    model_names: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    winner_counts = rankings["best_model"].value_counts().to_dict()
    tie_winner_counts = (
        rankings.loc[rankings["practical_tie"].astype(bool), "best_model"]
        .value_counts()
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        model = eligibility.loc[eligibility["model_name"] == model_name].copy()
        analyzable = model.loc[~model["insufficient_data"].astype(bool)]
        eligible = analyzable.loc[analyzable["eligible_by_auroc_guard"].astype(bool)]
        total_cells = int(len(model))
        analyzable_cells = int(len(analyzable))
        eligible_cells = int(len(eligible))
        rows.append(
            {
                "model_name": model_name,
                "total_stratum_cells": total_cells,
                "analyzable_stratum_cells": analyzable_cells,
                "eligible_stratum_cells": eligible_cells,
                "eligible_fraction": (
                    float(eligible_cells / analyzable_cells)
                    if analyzable_cells
                    else float("nan")
                ),
                "brier_winner_count": int(winner_counts.get(model_name, 0)),
                "practical_tie_winner_count": int(
                    tie_winner_counts.get(model_name, 0)
                ),
                "mean_auroc": (
                    float(analyzable["auroc"].mean())
                    if analyzable_cells
                    else float("nan")
                ),
                "mean_brier": (
                    float(analyzable["brier_mean"].mean())
                    if analyzable_cells
                    else float("nan")
                ),
                "recommended_for_stage9_subset_search": bool(eligible_cells > 0),
                "recommendation_reason": (
                    "passed_auroc_guard_in_at_least_one_prespecified_stratum"
                    if eligible_cells > 0
                    else "failed_auroc_guard_in_all_analyzable_strata"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    candidates = frame.loc[
        frame["recommended_for_stage9_subset_search"].astype(bool), "model_name"
    ].astype(str).tolist()
    fallback_used = False
    if not candidates:
        # Small or highly imbalanced validation strata can make every subgroup
        # cell ineligible even when the overall model predictions are valid.
        # An empty pool would abort the full protocol after all training work.
        # Preserve the preselected targeted library as a transparent fallback;
        # Stage9 still selects its final subset on the locked validation split.
        candidates = [str(name) for name in model_names]
        fallback_used = True
    payload = {
        "status": "validation_evidence_frozen",
        "purpose": "Stage9 Super Learner subset-search candidate pool",
        "selection_is_final_super_learner": False,
        "candidate_models": candidates,
        "excluded_models": [name for name in model_names if name not in candidates],
        "rule": (
            "include models passing the AUROC margin guard in at least one prespecified stratum"
            if not fallback_used
            else "fallback_to_all_preselected_models_when_no_subgroup_cell_passes_the_auroc_guard"
        ),
        "fallback_used": fallback_used,
        "fallback_reason": (
            "no_model_passed_the_prespecified_subgroup_auroc_guard"
            if fallback_used
            else None
        ),
        "model_evidence": frame.to_dict(orient="records"),
    }
    return frame, payload


def _write_analysis_outputs(
    *,
    output_dir: Path,
    frame: pd.DataFrame,
    config: StageHeterogeneityConfig,
    definitions: Mapping[str, StrataDefinition],
    frozen_definitions: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    all_metrics: list[pd.DataFrame] = []
    all_rankings: list[pd.DataFrame] = []
    all_eligibility: list[pd.DataFrame] = []
    interaction_results: list[dict[str, Any]] = []

    for variable in config.subgroup_variables:
        definition = definitions[variable]
        stratified = apply_strata_definition(frame, definition)
        metrics = compute_subgroup_metrics(
            stratified,
            variable,
            config.minimum_class_count,
            config.minimum_bin_size,
            config.default_probability_threshold,
        )
        eligibility = evaluate_model_eligibility_per_stratum(
            metrics,
            auroc_margin_guard=config.auroc_margin_guard,
        )
        rankings = rank_models_per_stratum(
            metrics,
            tie_margin=config.brier_tie_margin,
            auroc_margin_guard=config.auroc_margin_guard,
        )
        all_metrics.append(metrics)
        all_eligibility.append(eligibility)
        all_rankings.append(rankings)
        interaction_results.append(
            test_model_subgroup_interaction(
                stratified,
                variable,
                config.model_names,
                config.default_probability_threshold,
            )
        )
        plot_heterogeneity_heatmap(
            metrics,
            variable,
            output_dir / "figures" / f"heatmap_{variable}.png",
        )
        plot_error_vs_covariate(
            stratified,
            variable,
            output_dir / "figures" / f"error_vs_{variable}.png",
            config.trend_bins,
        )

    metrics_frame = pd.concat(all_metrics, ignore_index=True)
    eligibility_frame = pd.concat(all_eligibility, ignore_index=True)
    ranking_frame = pd.concat(all_rankings, ignore_index=True)
    interaction_results = apply_fdr_to_interaction_results(
        interaction_results,
        config.fdr_method,
        config.fdr_alpha,
    )
    metrics_frame.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    eligibility_frame.to_csv(
        output_dir / "model_eligibility_per_stratum.csv", index=False
    )
    ranking_frame.to_csv(output_dir / "best_model_per_stratum.csv", index=False)
    atomic_write_json(output_dir / "interaction_tests.json", interaction_results)
    atomic_write_json(
        output_dir / "strata_definitions.json",
        {key: asdict(value) for key, value in definitions.items()},
    )
    atomic_write_json(
        output_dir / "analysis_contract.json",
        {
            "task_name": config.task_name,
            "split": str(frame["split"].iloc[0]),
            "model_names": list(config.model_names),
            "subgroup_variables": list(config.subgroup_variables),
            "threshold_free_primary_metrics": ["auroc", "brier_mean", "log_loss_mean"],
            "threshold_dependent_metric_role": "Cochran Q reference at fixed 0.5 only",
            "auroc_margin_guard": config.auroc_margin_guard,
            "strata_refitted": not frozen_definitions,
        },
    )
    return metrics_frame, eligibility_frame, ranking_frame, interaction_results


def run_stage_heterogeneity_discovery(
    config: StageHeterogeneityConfig,
    *,
    dataset_path: Path,
    target_column: str = "LOS",
    identifier_columns: Iterable[str] = ("subject_id", "hadm_id", "stay_id"),
) -> Path:
    output_dir = stage_analysis_root(config.run_dir, config.task_name) / "discovery_validation"
    frozen_path = output_dir / "frozen_design.json"
    if frozen_path.exists():
        raise FileExistsError(
            "A frozen stage heterogeneity discovery design already exists. "
            "Use the frozen design for confirmatory analysis or create a new Run ID. "
            f"Existing design: {frozen_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_stage_predictions_with_covariates(
        run_dir=config.run_dir,
        task_name=config.task_name,
        model_names=config.model_names,
        split=config.discovery_split,
        dataset_path=dataset_path,
        subgroup_variables=config.subgroup_variables,
        target_column=target_column,
        identifier_columns=identifier_columns,
    )
    definitions: dict[str, StrataDefinition] = {}
    for variable in config.subgroup_variables:
        definitions[variable] = fit_strata_definition(
            frame,
            variable,
            fitted_on_split=config.discovery_split,
            method="quantile",
            n_bins=config.n_bins,
            minimum_bin_size=config.minimum_bin_size,
            minimum_class_count=config.minimum_class_count,
        )

    metrics, eligibility, rankings, interaction_results = _write_analysis_outputs(
        output_dir=output_dir,
        frame=frame,
        config=config,
        definitions=definitions,
        frozen_definitions=False,
    )
    evidence_frame, evidence_payload = _library_candidate_evidence(
        eligibility,
        rankings,
        list(config.model_names),
    )
    evidence_frame.to_csv(output_dir / "library_candidate_evidence.csv", index=False)
    atomic_write_json(output_dir / "library_candidates.json", evidence_payload)

    prediction_hashes = _prediction_hashes(
        config.run_dir,
        config.task_name,
        config.model_names,
        config.discovery_split,
    )
    frozen_payload = {
        "design_status": "frozen",
        "design_type": "stage_model_subgroup_heterogeneity",
        "task_name": config.task_name,
        "prediction_root": str(TASK_OUTPUT_ROOTS[config.task_name]),
        "discovery_split": config.discovery_split,
        "model_names": list(config.model_names),
        "subgroup_variables": list(config.subgroup_variables),
        "minimum_bin_size": config.minimum_bin_size,
        "minimum_class_count": config.minimum_class_count,
        "default_probability_threshold": config.default_probability_threshold,
        "fdr_method": config.fdr_method,
        "fdr_alpha": config.fdr_alpha,
        "brier_tie_margin": config.brier_tie_margin,
        "auroc_margin_guard": config.auroc_margin_guard,
        "trend_bins": config.trend_bins,
        "shared_split_hash": _shared_split_hash(config.run_dir),
        "validation_prediction_sha256": prediction_hashes,
        "strata_definitions": {
            variable: asdict(definition)
            for variable, definition in definitions.items()
        },
        "discovery_conclusions": {
            result["stratify_by"]: result.get("conclusion", "not_testable")
            for result in interaction_results
        },
        "stage9_library_candidate_evidence": evidence_payload,
        "confirmatory_splits_accessed": False,
    }
    atomic_write_json(frozen_path, frozen_payload)
    frozen_hash = _write_sha256_sidecar(frozen_path)
    atomic_write_json(
        output_dir / "run_config.json",
        {
            **asdict(config),
            "run_dir": str(config.run_dir.expanduser().resolve()),
            "dataset_path": str(dataset_path.expanduser().resolve()),
            "analysis_role": "discovery_validation_only",
            "fdr_family": f"validation_{config.task_name}",
            "frozen_design_sha256": frozen_hash,
            "confirmatory_splits_accessed": False,
            "environment": environment_info(),
        },
    )
    return output_dir


def load_frozen_stage_heterogeneity_design(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen stage heterogeneity design not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("design_status") != "frozen":
        raise ValueError("The supplied stage heterogeneity design is not frozen.")
    if payload.get("design_type") != "stage_model_subgroup_heterogeneity":
        raise ValueError("The supplied file is not a stage heterogeneity design.")
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.is_file():
        recorded = sidecar.read_text(encoding="utf-8").strip().split()[0]
        actual = sha256_file(path)
        if recorded != actual:
            raise ValueError(f"Frozen design SHA-256 mismatch: {path}")
    return payload


def run_stage_heterogeneity_confirmatory(
    *,
    run_dir: Path,
    task_name: str,
    split: str,
    dataset_path: Path,
    frozen_design_path: Path,
    target_column: str = "LOS",
    identifier_columns: Iterable[str] = ("subject_id", "hadm_id", "stay_id"),
) -> Path:
    if split not in {"internal_test", "external_eicu"}:
        raise ValueError(
            "Stage confirmatory heterogeneity is restricted to internal_test or external_eicu."
        )
    design = load_frozen_stage_heterogeneity_design(frozen_design_path)
    if str(design["task_name"]) != task_name:
        raise ValueError(
            f"Frozen task mismatch: expected {task_name}, found {design['task_name']}"
        )
    output_dir = stage_analysis_root(run_dir, task_name) / "confirmatory" / split
    if (output_dir / "run_config.json").exists():
        raise FileExistsError(
            "This stage confirmatory split has already been analysed. Re-running it "
            "would violate the single-use reporting guard. "
            f"Existing output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = tuple(str(value) for value in design["model_names"])
    subgroup_variables = tuple(str(value) for value in design["subgroup_variables"])
    config = StageHeterogeneityConfig(
        run_dir=run_dir,
        task_name=task_name,
        discovery_split="validation",
        model_names=model_names,
        subgroup_variables=subgroup_variables,
        n_bins=max(
            int(value["requested_n_bins"])
            for value in design["strata_definitions"].values()
        ),
        minimum_bin_size=int(design["minimum_bin_size"]),
        minimum_class_count=int(design["minimum_class_count"]),
        default_probability_threshold=float(design["default_probability_threshold"]),
        fdr_method=str(design["fdr_method"]),
        fdr_alpha=float(design["fdr_alpha"]),
        brier_tie_margin=float(design["brier_tie_margin"]),
        auroc_margin_guard=float(design["auroc_margin_guard"]),
        trend_bins=int(design.get("trend_bins", 50)),
    )
    frame = load_stage_predictions_with_covariates(
        run_dir=run_dir,
        task_name=task_name,
        model_names=model_names,
        split=split,
        dataset_path=dataset_path,
        subgroup_variables=subgroup_variables,
        target_column=target_column,
        identifier_columns=identifier_columns,
    )
    definitions = {
        variable: _definition_from_payload(design["strata_definitions"][variable])
        for variable in subgroup_variables
    }
    _, eligibility, rankings, _ = _write_analysis_outputs(
        output_dir=output_dir,
        frame=frame,
        config=config,
        definitions=definitions,
        frozen_definitions=True,
    )
    evidence_frame, evidence_payload = _library_candidate_evidence(
        eligibility,
        rankings,
        list(model_names),
    )
    evidence_frame.to_csv(output_dir / "library_candidate_evidence.csv", index=False)
    evidence_payload["role"] = "confirmatory_reporting_only"
    evidence_payload["must_not_change_stage9_library"] = True
    atomic_write_json(output_dir / "library_candidates.json", evidence_payload)

    atomic_write_json(
        output_dir / "run_config.json",
        {
            "analysis_role": "confirmatory_single_use",
            "task_name": task_name,
            "split": split,
            "dataset_path": str(dataset_path.expanduser().resolve()),
            "frozen_design_path": str(frozen_design_path.expanduser().resolve()),
            "frozen_design_sha256": sha256_file(
                frozen_design_path.expanduser().resolve()
            ),
            "no_refitting_of_strata": True,
            "auroc_margin_guard_refitted": False,
            "stage9_library_may_be_changed_from_this_result": False,
            "prediction_sha256": _prediction_hashes(
                run_dir, task_name, model_names, split
            ),
            "environment": environment_info(),
        },
    )
    return output_dir


def stage_heterogeneity_config_from_project(
    *,
    run_dir: Path,
    task_name: str,
    project_config: Mapping[str, Any],
) -> StageHeterogeneityConfig:
    heterogeneity = project_config.get("heterogeneity", {})
    if task_name == STAGE1_MORTALITY_TASK.task_name:
        models = project_config["experiments"]["exp2_two_stage"]["stage1_mortality"]["models"]
    elif task_name == STAGE2_LONG_STAY_TASK.task_name:
        models = project_config["experiments"]["exp2_two_stage"]["stage2_long_stay"]["models"]
    else:
        raise ValueError(f"Unsupported stage heterogeneity task: {task_name}")
    return StageHeterogeneityConfig(
        run_dir=run_dir,
        task_name=task_name,
        model_names=tuple(str(value) for value in models),
        subgroup_variables=tuple(
            str(value)
            for value in heterogeneity.get(
                "subgroup_variables", DEFAULT_SUBGROUP_VARIABLES
            )
        ),
        n_bins=int(heterogeneity.get("n_bins", 3)),
        minimum_bin_size=int(heterogeneity.get("minimum_bin_size", 30)),
        minimum_class_count=int(heterogeneity.get("minimum_class_count", 10)),
        default_probability_threshold=float(
            heterogeneity.get("default_probability_threshold", 0.5)
        ),
        fdr_method=str(heterogeneity.get("fdr_method", "fdr_bh")),
        fdr_alpha=float(heterogeneity.get("fdr_alpha", 0.05)),
        brier_tie_margin=float(heterogeneity.get("brier_tie_margin", 0.005)),
        auroc_margin_guard=float(heterogeneity.get("auroc_margin_guard", 0.02)),
        random_seed=int(project_config.get("execution", {}).get("seed", 42)),
        trend_bins=int(heterogeneity.get("trend_bins", 50)),
    )
