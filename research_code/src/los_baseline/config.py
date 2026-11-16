from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .constants import DEFAULT_FEATURE_NAMES, MODEL_ORDER


def resolve_outcome_horizon_days(config: dict[str, Any], *, required: bool = True) -> float | None:
    """Return the canonical EXP123 outcome horizon in days.

    V8 uses ``outcomes.outcome_horizon_days``. ``outcomes.landmark_days`` is
    accepted only for backward compatibility with frozen V7 releases.
    """
    outcomes = config.get("outcomes", {})
    if "outcome_horizon_days" in outcomes:
        value = float(outcomes["outcome_horizon_days"])
    elif "landmark_days" in outcomes:
        value = float(outcomes["landmark_days"])
    else:
        if not required:
            return None
        raise ValueError(
            "outcomes.outcome_horizon_days is required for EXP123 V8 "
            "(V7 compatibility also accepts outcomes.landmark_days)."
        )
    if value <= 0:
        raise ValueError("Outcome horizon must be positive.")
    return value


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file was not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    config = copy.deepcopy(config)
    validate_config(config)
    config["_config_path"] = str(path)
    config["_project_root"] = str(path.parent.parent.resolve())
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = [
        "paths",
        "data",
        "split",
        "models",
        "evaluation",
        "explainability",
        "execution",
    ]
    missing_sections = [
        section for section in required_sections if section not in config
    ]
    if missing_sections:
        raise KeyError(f"Missing configuration sections: {missing_sections}")

    data_config = config["data"]
    features = data_config.get("feature_columns", DEFAULT_FEATURE_NAMES)
    if not features:
        raise ValueError("data.feature_columns must not be empty.")
    if len(features) != len(set(features)):
        raise ValueError("Duplicate feature columns were found.")

    thresholds = data_config.get("threshold_days")
    if thresholds is not None:
        if sorted(set(thresholds)) != sorted(thresholds):
            raise ValueError("data.threshold_days must contain unique values.")
        if any(float(value) <= 0 for value in thresholds):
            raise ValueError("All LOS thresholds must be positive.")

    split = config["split"]
    ratios = [
        float(split["train_ratio"]),
        float(split["validation_ratio"]),
        float(split["test_ratio"]),
    ]
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0: {ratios}")
    if any(value <= 0 for value in ratios):
        raise ValueError("All split ratios must be greater than zero.")

    unknown_models = set(config["models"]) - set(MODEL_ORDER)
    if unknown_models:
        raise KeyError(f"Unsupported model names: {sorted(unknown_models)}")

    audit = config.get("audit", {})
    strict_columns = audit.get("strict_non_missing_columns", [])
    if not isinstance(strict_columns, list):
        raise TypeError("audit.strict_non_missing_columns must be a list.")
    provenance = audit.get("eicu_harmonizer_provenance", {})
    if provenance and not isinstance(provenance, dict):
        raise TypeError("audit.eicu_harmonizer_provenance must be an object.")
    denominator = int(
        provenance.get(
            "imputed_feature_denominator",
            audit.get("eicu_imputed_feature_denominator", 18),
        )
    )
    if denominator <= 0:
        raise ValueError(
            "eICU harmonizer provenance denominator must be positive."
        )
    provenance_path = provenance.get("path")
    if provenance_path is not None:
        lower = str(provenance_path).lower()
        if not lower.endswith((".csv", ".csv.gz", ".csv.bz2", ".csv.xz")):
            raise ValueError(
                "Optional eICU harmonizer provenance must be CSV/CSV.GZ. "
                "Parquet is not required or supported by the CSV-only release."
            )
    if bool(provenance.get("required", False)) and not bool(
        provenance.get("enabled", False)
    ):
        raise ValueError(
            "eICU harmonizer provenance cannot be required while disabled."
        )

    data_quality = config.get("data_quality", {})
    if data_quality:
        exclusion = data_quality.get("outcome_row_exclusion", {})
        if exclusion:
            datasets = exclusion.get("datasets", ["mimic", "eicu"])
            if not isinstance(datasets, list) or not datasets:
                raise ValueError(
                    "data_quality.outcome_row_exclusion.datasets must be a non-empty list."
                )
            unknown_datasets = sorted(set(map(str, datasets)) - {"mimic", "eicu"})
            if unknown_datasets:
                raise ValueError(
                    "Unknown outcome exclusion datasets: "
                    f"{unknown_datasets}"
                )
            rules = exclusion.get("rules", {})
            if not isinstance(rules, dict):
                raise TypeError(
                    "data_quality.outcome_row_exclusion.rules must be an object."
                )
            supported_rules = {
                "negative_los",
                "missing_los",
                "missing_mortality",
                "invalid_mortality",
            }
            unknown_rules = sorted(set(rules) - supported_rules)
            if unknown_rules:
                raise ValueError(
                    f"Unsupported outcome exclusion rules: {unknown_rules}"
                )
            max_fraction = float(exclusion.get("max_excluded_fraction", 0.01))
            if not 0.0 <= max_fraction <= 1.0:
                raise ValueError(
                    "data_quality.outcome_row_exclusion.max_excluded_fraction "
                    "must be between 0 and 1."
                )
            max_rows = int(exclusion.get("max_excluded_rows", 100))
            if max_rows < 0:
                raise ValueError(
                    "data_quality.outcome_row_exclusion.max_excluded_rows "
                    "must be non-negative."
                )

    outcomes = config.get("outcomes")
    if outcomes is not None:
        mortality_column = str(outcomes.get("mortality_column", "")).strip()
        los_column = str(
            outcomes.get("los_column", data_config.get("target_column", ""))
        ).strip()
        horizon_days = resolve_outcome_horizon_days(config)
        if not mortality_column:
            raise ValueError("outcomes.mortality_column must not be empty.")
        if not los_column:
            raise ValueError("outcomes.los_column must not be empty.")
        if horizon_days <= 0:
            raise ValueError("outcomes.outcome_horizon_days must be positive.")
        if "outcome_horizon_days" in outcomes and "landmark_days" in outcomes:
            if float(outcomes["outcome_horizon_days"]) != float(outcomes["landmark_days"]):
                raise ValueError(
                    "Conflicting outcome horizon definitions: "
                    "outcomes.outcome_horizon_days and outcomes.landmark_days differ."
                )
        los_derivation = outcomes.get("los_derivation", {})
        if los_derivation:
            datasets = los_derivation.get("datasets", ["mimic", "eicu"])
            if not isinstance(datasets, list) or not datasets:
                raise ValueError(
                    "outcomes.los_derivation.datasets must be a non-empty list."
                )
            unknown = sorted(set(map(str, datasets)) - {"mimic", "eicu"})
            if unknown:
                raise ValueError(
                    f"Unknown LOS derivation datasets: {unknown}"
                )
            for key in (
                "admission_time_column",
                "discharge_time_column",
                "in_hospital_death_time_column",
            ):
                value = str(los_derivation.get(key, "")).strip()
                if bool(los_derivation.get("enabled", False)) and not value:
                    raise ValueError(
                        f"outcomes.los_derivation.{key} must not be empty."
                    )
            tolerance = float(
                los_derivation.get("discrepancy_tolerance_days", 0.05)
            )
            if tolerance < 0:
                raise ValueError(
                    "outcomes.los_derivation.discrepancy_tolerance_days must "
                    "be non-negative."
                )
            decimals = int(los_derivation.get("round_decimals", 6))
            if decimals < 0 or decimals > 12:
                raise ValueError(
                    "outcomes.los_derivation.round_decimals must be between 0 and 12."
                )

    matrix = config.get("experiment_matrix", {})
    if matrix:
        strategies = matrix.get("strategies", {})
        if not isinstance(strategies, dict):
            raise TypeError("experiment_matrix.strategies must be an object.")
        required = {"T1_baseline", "T2_optuna", "T3_smote", "T4_undersampling"}
        unknown = sorted(set(strategies) - required)
        if unknown:
            raise ValueError(f"Unknown V8 strategies: {unknown}")
        missing = sorted(required - set(strategies))
        if missing:
            raise ValueError(f"Missing V8 strategy definitions: {missing}")
        for key in ("T1_baseline", "T2_optuna", "T3_smote"):
            if not bool(strategies[key].get("enabled", False)):
                raise ValueError(f"{key} is a mandatory V8 core strategy and must be enabled.")
        t4 = strategies["T4_undersampling"]
        if not bool(t4.get("optional", False)):
            raise ValueError("T4_undersampling must be marked optional in V8.")

    optimization = config.get("optimization", {})
    if optimization:
        cv_folds = int(optimization.get("cv_folds", 5))
        top_k = int(optimization.get("top_k_for_validation", 3))
        if cv_folds < 2:
            raise ValueError("optimization.cv_folds must be at least 2.")
        if top_k < 1:
            raise ValueError(
                "optimization.top_k_for_validation must be at least 1."
            )
        search_stages = optimization.get("search_stages", {})
        for stage_name in ("smoke", "coarse", "fine"):
            if stage_name not in search_stages:
                raise KeyError(
                    f"optimization.search_stages is missing {stage_name}."
                )
            stage = search_stages[stage_name]
            if int(stage.get("trials", 0)) < 1:
                raise ValueError(
                    f"optimization.search_stages.{stage_name}.trials must be positive."
                )
            if int(stage.get("cv_folds", 0)) < 2:
                raise ValueError(
                    f"optimization.search_stages.{stage_name}.cv_folds must be at least 2."
                )
        model_budgets = optimization.get("model_search_budgets", {})
        if model_budgets:
            if not isinstance(model_budgets, dict):
                raise TypeError("optimization.model_search_budgets must be an object.")
            unknown_budget_models = sorted(set(model_budgets) - set(MODEL_ORDER))
            if unknown_budget_models:
                raise KeyError(
                    "Unsupported models in optimization.model_search_budgets: "
                    f"{unknown_budget_models}"
                )
            for model_name, stages in model_budgets.items():
                if not isinstance(stages, dict):
                    raise TypeError(
                        f"optimization.model_search_budgets.{model_name} must be an object."
                    )
                for stage_name, stage in stages.items():
                    if stage_name not in {"smoke", "coarse", "fine"}:
                        raise KeyError(
                            f"Unsupported model search stage: {model_name}/{stage_name}"
                        )
                    if int(stage.get("trials", 0)) < 1 or int(stage.get("cv_folds", 0)) < 2:
                        raise ValueError(
                            f"Invalid model search budget: {model_name}/{stage_name}"
                        )

        runtime = optimization.get("runtime_limits", {})
        if float(runtime.get("trial_timeout_seconds", 0)) <= 0:
            raise ValueError(
                "optimization.runtime_limits.trial_timeout_seconds must be positive."
            )
        if float(runtime.get("study_timeout_seconds", 0)) <= 0:
            raise ValueError(
                "optimization.runtime_limits.study_timeout_seconds must be positive."
            )

    optimization_v2 = config.get("optimization_v2", {})
    if optimization_v2:
        if str(optimization_v2.get("engine", "")).lower() != "optuna_tpe":
            raise ValueError("optimization_v2.engine must be optuna_tpe.")
        if int(optimization_v2.get("cv_folds", 0)) < 2:
            raise ValueError("optimization_v2.cv_folds must be at least 2.")
        repeat_seeds = optimization_v2.get("finalist_repeat_seeds", [])
        if not isinstance(repeat_seeds, list) or not repeat_seeds:
            raise ValueError(
                "optimization_v2.finalist_repeat_seeds must be a non-empty list."
            )
        budgets = optimization_v2.get("trial_budgets", {})
        if not isinstance(budgets, dict) or not budgets:
            raise ValueError(
                "optimization_v2.trial_budgets must be a non-empty object."
            )
        unknown_budget_models = sorted(set(budgets) - set(MODEL_ORDER))
        if unknown_budget_models:
            raise KeyError(
                f"Unsupported optimization_v2 trial budget models: {unknown_budget_models}"
            )
        for model_name, budget in budgets.items():
            if int(budget) < 1:
                raise ValueError(
                    f"optimization_v2.trial_budgets.{model_name} must be positive."
                )
        if "svm" in budgets and not bool(
            config.get("calibration", {}).get("enabled", False)
        ):
            raise ValueError(
                "Optuna SVM HPO uses probability=False and therefore requires "
                "calibration.enabled=true for final model training."
            )


    svm_runtime = config.get("svm_runtime_optimization", {})
    if svm_runtime:
        if not isinstance(svm_runtime, dict):
            raise TypeError("svm_runtime_optimization must be an object.")
        backends = list(svm_runtime.get("search_backends", []))
        invalid = sorted(set(backends) - {"exact_rbf", "nystroem_linear"})
        if invalid:
            raise ValueError(f"Unsupported SVM runtime backends: {invalid}")
        if bool(svm_runtime.get("enabled", False)) and not backends:
            raise ValueError("Enabled svm_runtime_optimization requires search_backends.")
        if float(svm_runtime.get("max_fold_fit_seconds", 0.0)) < 0:
            raise ValueError("svm_runtime_optimization.max_fold_fit_seconds must be non-negative.")
        exact = dict(svm_runtime.get("exact_rbf", {}))
        approx = dict(svm_runtime.get("nystroem_linear", {}))
        for label, block in (("exact_rbf", exact), ("nystroem_linear", approx)):
            if block and float(block.get("c_min", 1e-2)) <= 0:
                raise ValueError(f"svm_runtime_optimization.{label}.c_min must be positive.")
            if block and float(block.get("c_max", 1e2)) <= float(block.get("c_min", 1e-2)):
                raise ValueError(f"svm_runtime_optimization.{label}.c_max must exceed c_min.")
        if approx and int(approx.get("n_components_max", 2048)) < int(approx.get("n_components_min", 256)):
            raise ValueError("nystroem_linear.n_components_max must be >= n_components_min.")

    calibration = config.get("calibration", {})
    if calibration:
        if int(calibration.get("cv_folds", 0)) < 2:
            raise ValueError("calibration.cv_folds must be at least 2.")
        if str(calibration.get("method", "sigmoid")).lower() not in {
            "sigmoid",
            "isotonic",
        }:
            raise ValueError("calibration.method must be sigmoid or isotonic.")
        if not str(calibration.get("group_column", "")).strip():
            raise ValueError("calibration.group_column must not be empty.")

    staged = config.get("staged_protocol", {})
    if staged:
        if not isinstance(staged, dict):
            raise TypeError("staged_protocol must be an object.")
        candidate = staged.get("candidate_selection", {})
        if candidate:
            if int(candidate.get("top_models_per_task", 1)) < 1:
                raise ValueError(
                    "staged_protocol.candidate_selection.top_models_per_task must be positive."
                )
            if int(candidate.get("top_pipeline_pairs", 1)) < 1:
                raise ValueError(
                    "staged_protocol.candidate_selection.top_pipeline_pairs must be positive."
                )
            if int(candidate.get("max_models_per_task", 1)) < 1:
                raise ValueError(
                    "staged_protocol.candidate_selection.max_models_per_task must be positive."
                )
        targeted = staged.get("targeted_optimization", {})
        if targeted:
            stages = list(targeted.get("search_stages", ["coarse", "fine"]))
            unknown = sorted(set(stages) - {"smoke", "coarse", "fine", "optuna"})
            if unknown:
                raise ValueError(
                    f"Unknown staged targeted optimization stages: {unknown}"
                )
            if not stages:
                raise ValueError(
                    "staged_protocol.targeted_optimization.search_stages must not be empty."
                )

    heterogeneity = config.get("heterogeneity", {})
    if heterogeneity:
        subgroup_variables = heterogeneity.get("subgroup_variables", [])
        if not isinstance(subgroup_variables, list) or not subgroup_variables:
            raise ValueError("heterogeneity.subgroup_variables must be a non-empty list.")
        if int(heterogeneity.get("n_bins", 3)) < 2:
            raise ValueError("heterogeneity.n_bins must be at least 2.")
        if int(heterogeneity.get("minimum_bin_size", 30)) < 1:
            raise ValueError("heterogeneity.minimum_bin_size must be positive.")
        if int(heterogeneity.get("minimum_class_count", 10)) < 1:
            raise ValueError("heterogeneity.minimum_class_count must be positive.")
        default_threshold = float(
            heterogeneity.get("default_probability_threshold", 0.5)
        )
        if not 0.0 < default_threshold < 1.0:
            raise ValueError(
                "heterogeneity.default_probability_threshold must be between 0 and 1."
            )
        if float(heterogeneity.get("auroc_margin_guard", 0.02)) < 0:
            raise ValueError("heterogeneity.auroc_margin_guard must be non-negative.")
        if float(heterogeneity.get("brier_tie_margin", 0.005)) < 0:
            raise ValueError("heterogeneity.brier_tie_margin must be non-negative.")
        fdr_alpha = float(heterogeneity.get("fdr_alpha", 0.05))
        if not 0.0 < fdr_alpha < 1.0:
            raise ValueError("heterogeneity.fdr_alpha must be between 0 and 1.")
        if int(heterogeneity.get("trend_bins", 50)) < 5:
            raise ValueError("heterogeneity.trend_bins must be at least 5.")

    final_freeze = config.get("final_freeze", {})
    if final_freeze:
        if float(final_freeze.get("minimum_free_disk_gb", 0.0)) < 0:
            raise ValueError("final_freeze.minimum_free_disk_gb must be non-negative.")
        if final_freeze.get("requirements_policy") != "exact_pins_required":
            raise ValueError(
                "final_freeze.requirements_policy must be exact_pins_required."
            )
        if not bool(final_freeze.get("confirmatory_design_freeze_required", False)):
            raise ValueError(
                "final_freeze.confirmatory_design_freeze_required must be true."
            )
        if not bool(final_freeze.get("confirmatory_order_locked", False)):
            raise ValueError("final_freeze.confirmatory_order_locked must be true.")
        if bool(final_freeze.get("confirmatory_overwrite_allowed", True)):
            raise ValueError(
                "final_freeze.confirmatory_overwrite_allowed must be false."
            )

    development = config.get("development", {})
    if bool(development.get("model_training_enabled", False)):
        completed = set(development.get("completed_steps", []))
        if "final_freeze" not in completed:
            raise ValueError(
                "model_training_enabled requires final_freeze in completed_steps."
            )

    objectives = config.get("optimization_objectives", {})
    for task_name in (
        "exp1_flat_3class",
        "stage1_mortality",
        "stage2_long_stay",
    ):
        if objectives and task_name not in objectives:
            raise KeyError(
                f"optimization_objectives is missing {task_name}."
            )
        if task_name in objectives:
            objective = objectives[task_name]
            if objective.get("direction") not in {"maximize", "minimize"}:
                raise ValueError(
                    f"Invalid optimization direction for {task_name}."
                )
            if objective.get("secondary_direction") not in {
                "maximize",
                "minimize",
            }:
                raise ValueError(
                    f"Invalid secondary optimization direction for {task_name}."
                )


def resolve_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(config["_project_root"]) / path).resolve()
