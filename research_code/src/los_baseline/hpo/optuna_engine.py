from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..exp123_data import TaskDataBundle
from ..models import normalize_task_model_config
from ..optimization import _fit_predict_partition, _groups_from_training_data
from ..optimization_objectives import load_objective_definition, robust_selection_score
from ..progress import emit_progress, progress_iter, tracked_operation
from ..config import resolve_outcome_horizon_days
from ..utils import atomic_write_json, sha256_file
from .fixed_cv import build_fixed_group_folds, load_fixed_folds, save_fixed_folds
from .search_space import suggest_model_config
from ..svm_runtime import svm_backend_label, svm_runtime_settings


HPO_PROTOCOL_VERSION = "exp123_hpo_v7_3"
SVM_PHASE4B_HPO_PROTOCOL_VERSION = "exp123_hpo_v8_phase4b_svm_runtime"


def resolve_hpo_protocol_version(model_name: str, config: Mapping[str, Any]) -> str:
    if model_name == "svm" and svm_runtime_settings(config)["enabled"]:
        return SVM_PHASE4B_HPO_PROTOCOL_VERSION
    return HPO_PROTOCOL_VERSION


def _direction_adjusted(value: float, direction: str) -> float:
    return float(value if direction == "maximize" else -value)


def budget_consuming_trial_count(trials: Sequence[Any]) -> int:
    """Count only complete/pruned trials against the requested HPO budget."""
    return sum(
        str(getattr(trial.state, "name", trial.state)).upper()
        in {"COMPLETE", "PRUNED"}
        for trial in trials
    )


def _evaluate_config(
    *,
    task_data: TaskDataBundle,
    model_name: str,
    model_config: dict[str, Any],
    folds: Sequence[Any],
    objective: Any,
    patient_id_column: str,
    stability_penalty: float,
    overfit_penalty: float,
    seed: int,
    trial: Any | None = None,
    project_config: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
    train = task_data.splits["train"]
    labels = np.asarray(train.labels, dtype=np.int64)
    groups = _groups_from_training_data(task_data, patient_id_column)
    fold_rows: list[dict[str, Any]] = []
    primary_values: list[float] = []
    train_primary_values: list[float] = []
    for step, fixed in enumerate(folds, start=1):
        fit_idx = fixed.train_indices
        valid_idx = fixed.validation_indices
        if set(groups[fit_idx]) & set(groups[valid_idx]):
            raise RuntimeError("Fixed CV group leakage detected.")
        _, metrics, diagnostics, effective = _fit_predict_partition(
            model_name=model_name,
            model_config=model_config,
            task=task_data.task,
            feature_names=task_data.feature_names,
            train_features=train.raw_features.iloc[fit_idx].reset_index(drop=True),
            train_labels=labels[fit_idx],
            evaluation_features=train.raw_features.iloc[valid_idx].reset_index(drop=True),
            evaluation_labels=labels[valid_idx],
            seed=seed + fixed.repeat * 1000 + fixed.fold,
            train_groups=groups[fit_idx],
        )
        primary = float(metrics[objective.primary])
        train_primary = float(diagnostics["train_metrics"][objective.primary])
        primary_values.append(primary)
        train_primary_values.append(train_primary)
        row = {
            "repeat": fixed.repeat,
            "fold": fixed.fold,
            **{k: float(v) for k, v in metrics.items()},
            "train_primary": train_primary,
            "prediction_kind": diagnostics.get("prediction_kind", "probability"),
            "effective_config": effective,
            "best_iteration": diagnostics.get("best_iteration"),
            "selected_n_estimators": diagnostics.get("selected_n_estimators"),
            "early_stopping_protocol": diagnostics.get("early_stopping_protocol"),
            "early_stopping_split_hash": diagnostics.get("early_stopping_split_hash"),
            "early_stopping_outer_validation_accessed": diagnostics.get(
                "early_stopping_outer_validation_accessed"
            ),
        }
        fold_rows.append(row)
        if model_name == "svm":
            runtime = svm_runtime_settings(project_config)
            fit_seconds = float(diagnostics.get("fit_seconds", 0.0) or 0.0)
            limit = float(runtime.get("max_fold_fit_seconds", 0.0) or 0.0)
            row["svm_backend"] = diagnostics.get("svm_backend", svm_backend_label(model_config))
            row["fit_seconds"] = fit_seconds
            row["svm_runtime_limit_seconds"] = limit
            row["svm_runtime_limit_exceeded"] = bool(limit > 0.0 and fit_seconds > limit)
            if trial is not None and limit > 0.0 and fit_seconds > limit:
                trial.set_user_attr("runtime_guard_triggered", True)
                trial.set_user_attr("runtime_guard_fit_seconds", fit_seconds)
                trial.set_user_attr("runtime_guard_limit_seconds", limit)
                trial.set_user_attr("svm_backend", row["svm_backend"])
                import optuna
                raise optuna.TrialPruned(
                    f"SVM fold fit exceeded runtime guard: {fit_seconds:.3f}s > {limit:.3f}s"
                )
        if trial is not None:
            running_mean = float(np.mean(primary_values))
            trial.report(_direction_adjusted(running_mean, objective.direction), step=step)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned(f"Pruned after fixed fold {step}.")
    mean_primary = float(np.mean(primary_values))
    std_primary = float(np.std(primary_values, ddof=1)) if len(primary_values) > 1 else 0.0
    mean_train = float(np.mean(train_primary_values))
    robust_aggregate = {
        f"mean_{objective.primary}": mean_primary,
        f"std_{objective.primary}": std_primary,
        f"mean_train_{objective.primary}": mean_train,
    }
    score = robust_selection_score(
        aggregate_metrics=robust_aggregate,
        objective=objective,
        stability_penalty=stability_penalty,
        overfit_penalty=overfit_penalty,
    )
    aggregate: dict[str, float] = {
        "mean_primary": mean_primary,
        "std_primary": std_primary,
        "mean_train_primary": mean_train,
        "selection_score": float(score),
    }
    metric_names = sorted({k for row in fold_rows for k, v in row.items() if isinstance(v, (int, float)) and k not in {"repeat", "fold"}})
    for metric in metric_names:
        vals = [float(row[metric]) for row in fold_rows if metric in row]
        aggregate[f"mean_{metric}"] = float(np.mean(vals))
        aggregate[f"std_{metric}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return float(score), aggregate, fold_rows


def _complexity_key(model_name: str, config: Mapping[str, Any]) -> tuple[float, ...]:
    p = dict(config.get("parameters", {}))
    if model_name == "decision_tree":
        return (float(p.get("max_depth", 999)), -float(p.get("min_samples_leaf", 1)))
    if model_name == "random_forest":
        return (float(p.get("n_estimators", 0)), float(p.get("max_depth", 999)))
    if model_name == "svm":
        backend = str(p.get("svm_backend", "legacy_svc"))
        if backend == "nystroem_linear":
            return (0.0, float(p.get("nystroem_components", 1024)), float(p.get("C", 1.0)))
        return (1.0 if p.get("kernel", "rbf") == "rbf" else 0.5, float(p.get("C", 1.0)))
    if model_name == "xgboost":
        return (float(p.get("max_depth", 999)), float(p.get("n_estimators", 0)))
    if model_name == "logistic_regression":
        return (float(dict(config.get("parameters", {})).get("l1_ratio", 0.0)), float(dict(config.get("parameters", {})).get("C", 1.0)))
    return (0.0,)


def run_optuna_optimization_study(
    *,
    task_data: TaskDataBundle,
    model_name: str,
    base_model_config: dict[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
    search_stages: Sequence[str] | None = None,
):
    import optuna

    output_dir.mkdir(parents=True, exist_ok=True)
    opt = dict(config.get("optimization_v2", {}))
    objective = load_objective_definition(task_data.task.task_name, config["optimization_objectives"])
    seed = int(opt.get("seed", 42))
    patient_id_column = str(config.get("data", {}).get("patient_id_column", "subject_id"))
    labels = np.asarray(task_data.splits["train"].labels, dtype=np.int64)
    groups = _groups_from_training_data(task_data, patient_id_column)

    fixed_dir = output_dir.parent.parent.parent / "common" / "fixed_cv" / task_data.task.task_name
    if (fixed_dir / "fixed_cv_manifest.json").is_file():
        all_folds, fixed_manifest = load_fixed_folds(fixed_dir)
    else:
        primary_seed = int(opt.get("primary_cv_seed", seed))
        repeat_seeds = [int(v) for v in opt.get("finalist_repeat_seeds", [seed, seed + 1000])]
        all_folds = build_fixed_group_folds(
            labels=labels,
            groups=groups,
            n_splits=int(opt.get("cv_folds", 5)),
            seeds=repeat_seeds,
        )
        _, split_hash = save_fixed_folds(
            fixed_dir,
            folds=all_folds,
            seeds=repeat_seeds,
            n_splits=int(opt.get("cv_folds", 5)),
            sample_count=len(labels),
            outcome_horizon_days=resolve_outcome_horizon_days(dict(config), required=False),
        )
        fixed_manifest = json.loads((fixed_dir / "fixed_cv_manifest.json").read_text())
    primary_folds = [fold for fold in all_folds if fold.repeat == 1]

    budgets = dict(opt.get("trial_budgets", {}))
    n_trials = int(budgets.get(model_name, 50))
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=int(opt.get("startup_trials", 20)),
        multivariate=bool(opt.get("multivariate", True)),
        group=bool(opt.get("group", True)),
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(opt.get("pruner_startup_trials", 20)),
        n_warmup_steps=int(opt.get("pruner_warmup_folds", 2)),
    )
    storage = f"sqlite:///{(output_dir / 'optuna_study.sqlite3').resolve()}"
    hpo_protocol_version = resolve_hpo_protocol_version(model_name, config)
    study_name = (
        f"{task_data.task.task_name}__{model_name}__{hpo_protocol_version}"
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    baseline = normalize_task_model_config(
        model_name,
        json.loads(json.dumps(base_model_config, default=str)),
        task_type=task_data.task.task_type,
        num_classes=task_data.task.num_classes,
        hpo_mode=True,
    )
    baseline.setdefault("training_policy", {})["class_weight_mode"] = "none"
    if model_name == "svm" and svm_runtime_settings(config)["enabled"]:
        runtime = svm_runtime_settings(config)
        bp = baseline.setdefault("parameters", {})
        bp["svm_backend"] = "exact_rbf"
        bp["kernel"] = "rbf"
        bp["probability"] = False
        bp["decision_function_shape"] = "ovr"
        bp["max_iter"] = int(runtime["exact_rbf"]["max_iter"])
        bp["tol"] = float(runtime["exact_rbf"]["tol"])
        bp["cache_size"] = max(1024, int(bp.get("cache_size", 2048)))
    if not study.trials:
        study.enqueue_trial({}, user_attrs={"candidate_type": "baseline"})

    trial_started = time.perf_counter()
    def objective_fn(trial: Any) -> float:
        is_baseline = trial.number == 0 and not trial.params
        model_config = (
            baseline
            if is_baseline
            else suggest_model_config(trial, model_name, baseline, project_config=config)
        )
        model_config = normalize_task_model_config(
            model_name,
            model_config,
            task_type=task_data.task.task_type,
            num_classes=task_data.task.num_classes,
            hpo_mode=True,
        )
        trial.set_user_attr("candidate_type", "baseline" if is_baseline else "tpe")
        trial.set_user_attr("model_config", model_config)
        if model_name == "svm":
            trial.set_user_attr("svm_backend", svm_backend_label(model_config))
        emit_progress(
            "hpo_trial_started", operation=study_name, current=trial.number + 1,
            total=n_trials, unit="trial", status="running",
            details={"task": task_data.task.task_name, "model": model_name, "trial": trial.number},
        )
        score, aggregate, fold_rows = _evaluate_config(
            task_data=task_data,
            model_name=model_name,
            model_config=model_config,
            folds=primary_folds,
            objective=objective,
            patient_id_column=patient_id_column,
            stability_penalty=float(opt.get("stability_penalty", 0.5)),
            overfit_penalty=float(opt.get("overfit_penalty", 0.5)),
            seed=seed + trial.number * 10000,
            trial=trial,
            project_config=config,
        )
        for key, value in aggregate.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("fold_metrics", fold_rows)
        return score

    consumed_trials = budget_consuming_trial_count(study.trials)
    remaining = max(0, n_trials - consumed_trials)
    if remaining:
        with tracked_operation(f"Optuna {study_name}", details={"trials": remaining}):
            study.optimize(objective_fn, n_trials=remaining, gc_after_trial=True, show_progress_bar=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f"No completed Optuna trials for {study_name}.")
    top_k = int(opt.get("finalist_top_k", 10))
    ranked_completed = sorted(
        completed,
        key=lambda t: t.value if t.value is not None else -math.inf,
        reverse=True,
    )
    top = ranked_completed[:top_k]
    baseline_trials = [
        trial for trial in completed
        if trial.user_attrs.get("candidate_type") == "baseline"
    ]
    if baseline_trials and all(trial.number != baseline_trials[0].number for trial in top):
        if len(top) >= top_k and top:
            top = top[:-1]
        top.append(baseline_trials[0])
    finalists: list[dict[str, Any]] = []
    for rank, trial in enumerate(top, start=1):
        model_config = dict(trial.user_attrs["model_config"])
        score, aggregate, fold_rows = _evaluate_config(
            task_data=task_data, model_name=model_name, model_config=model_config,
            folds=all_folds, objective=objective, patient_id_column=patient_id_column,
            stability_penalty=float(opt.get("stability_penalty", 0.5)),
            overfit_penalty=float(opt.get("overfit_penalty", 0.5)),
            seed=seed + 500000 + rank * 10000, trial=None,
            project_config=config,
        )
        se = float(aggregate["std_primary"] / math.sqrt(max(len(all_folds), 1)))
        finalists.append({
            "rank_from_tpe": rank,
            "trial_number": trial.number,
            "model_config": model_config,
            "selection_score": score,
            "primary_mean": aggregate["mean_primary"],
            "primary_std": aggregate["std_primary"],
            "primary_se": se,
            "aggregate": aggregate,
            "fold_metrics": fold_rows,
            "complexity_key": _complexity_key(model_name, model_config),
            "candidate_type": trial.user_attrs.get("candidate_type", "tpe"),
        })

    best_primary = max(row["primary_mean"] for row in finalists) if objective.direction == "maximize" else min(row["primary_mean"] for row in finalists)
    best_row = min(finalists, key=lambda r: abs(r["primary_mean"] - best_primary))
    one_se = best_row["primary_se"]
    if objective.direction == "maximize":
        eligible = [r for r in finalists if r["primary_mean"] >= best_primary - one_se]
    else:
        eligible = [r for r in finalists if r["primary_mean"] <= best_primary + one_se]
    secondary_key = f"mean_{objective.secondary}"
    secondary_values = [row["aggregate"].get(secondary_key) for row in eligible]
    secondary_available = all(
        value is not None and math.isfinite(float(value))
        for value in secondary_values
    )
    reverse_secondary = objective.secondary_direction == "maximize"

    def selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        if secondary_available:
            secondary = float(row["aggregate"][secondary_key])
            secondary_rank = -secondary if reverse_secondary else secondary
        else:
            secondary_rank = 0.0
        return (
            secondary_rank,
            row["primary_std"],
            row["complexity_key"],
        )

    selected = sorted(
        eligible,
        key=selection_key,
    )[0]

    trial_rows = []
    for t in study.trials:
        trial_rows.append({
            "trial_number": t.number,
            "state": t.state.name,
            "value": t.value,
            "candidate_type": t.user_attrs.get("candidate_type"),
            "runtime_guard_triggered": bool(t.user_attrs.get("runtime_guard_triggered", False)),
            "runtime_guard_fit_seconds": t.user_attrs.get("runtime_guard_fit_seconds"),
            "runtime_guard_limit_seconds": t.user_attrs.get("runtime_guard_limit_seconds"),
            "svm_backend": t.user_attrs.get("svm_backend"),
            "mean_fit_seconds": t.user_attrs.get("mean_fit_seconds"),
            "std_fit_seconds": t.user_attrs.get("std_fit_seconds"),
            **{f"param_{k}": v for k, v in t.params.items()},
        })
    pd.DataFrame(trial_rows).to_csv(output_dir / "optuna_trials.csv", index=False)
    pd.DataFrame([
        {k: v for k, v in row.items() if k not in {"model_config", "aggregate", "fold_metrics", "complexity_key"}}
        | {"model_config_json": json.dumps(row["model_config"], sort_keys=True)}
        for row in finalists
    ]).to_csv(output_dir / "finalists_repeated_cv.csv", index=False)

    final_model_config = normalize_task_model_config(
        model_name,
        json.loads(json.dumps(selected["model_config"], default=str)),
        task_type=task_data.task.task_type,
        num_classes=task_data.task.num_classes,
        hpo_mode=True,
    )
    if model_name == "svm":
        final_model_config.setdefault("parameters", {})["probability"] = False
        final_model_config.setdefault("calibration", {})["method"] = "sigmoid_oof"
    if model_name == "xgboost":
        best_iterations = [
            row.get("best_iteration")
            for row in selected["fold_metrics"]
            if row.get("best_iteration") is not None
        ]
        if best_iterations:
            final_model_config.setdefault("parameters", {})["n_estimators"] = max(1, int(np.median(best_iterations)) + 1)
        final_model_config.setdefault("parameters", {}).pop("early_stopping_rounds", None)
    payload = {
        "study_id": f"{task_data.task.task_name}:{model_name}",
        "optimizer": "optuna_multivariate_tpe_fixed_group_cv",
        "hpo_protocol_version": hpo_protocol_version,
        "task_name": task_data.task.task_name,
        "model_name": model_name,
        "selected_on_split": "train_repeated_fixed_group_cv",
        "confirmatory_splits_accessed": False,
        "forbidden_splits": ["internal_test", "external_eicu"],
        "selected_candidate_id": f"optuna_trial_{selected['trial_number']}",
        "source_stage": "optuna_tpe_repeated_cv_one_se",
        "source_trial_number": int(selected["trial_number"]),
        "model_config": final_model_config,
        "metrics": selected["aggregate"],
        "objective": objective.to_dict(),
        "selection_guards": {
            "baseline_fallback": True,
            "one_standard_error_rule": True,
            "stability_penalty": float(opt.get("stability_penalty", 0.5)),
            "overfit_penalty": float(opt.get("overfit_penalty", 0.5)),
            "secondary_metric_requested": objective.secondary,
            "secondary_metric_applied": bool(secondary_available),
            "secondary_metric_unavailable_reason": (
                None
                if secondary_available
                else "raw_decision_scores_are_not_calibrated_probabilities"
            ),
            "selection_fallback_when_secondary_unavailable": (
                "primary_stability_then_model_complexity"
            ),
        },
        "fixed_cv_hash": fixed_manifest["split_hash"],
        "shared_split_hash": task_data.shared_split_hash,
        "study_storage": storage,
        "study_name": study_name,
        "trial_budget": n_trials,
        "budget_consuming_states": ["COMPLETE", "PRUNED"],
        "failed_trials_consume_budget": False,
        "completed_trials": len(completed),
        "elapsed_seconds": time.perf_counter() - trial_started,
        "svm_runtime_optimization": (
            {
                **svm_runtime_settings(config),
                "selected_backend": svm_backend_label(final_model_config),
                "runtime_guard_pruned_trials": int(sum(
                    bool(t.user_attrs.get("runtime_guard_triggered", False))
                    for t in study.trials
                )),
            }
            if model_name == "svm" else None
        ),
    }
    selected_path = output_dir / "selected_hyperparameters.json"
    atomic_write_json(selected_path, payload)
    (output_dir / "selected_hyperparameters.json.sha256").write_text(sha256_file(selected_path) + "\n", encoding="utf-8")
    atomic_write_json(output_dir / "optuna_summary.json", payload)
    state_counts = {
        state.name.lower(): sum(t.state == state for t in study.trials)
        for state in optuna.trial.TrialState
    }
    atomic_write_json(
        output_dir / "study_state_summary.json",
        {
            "hpo_protocol_version": hpo_protocol_version,
            "study_name": study_name,
            "requested_budget": n_trials,
            "budget_consuming_states": ["complete", "pruned"],
            "consumed_trial_count": int(
                budget_consuming_trial_count(study.trials)
            ),
            "state_counts": state_counts,
            "failed_trials_do_not_consume_budget": True,
        },
    )

    # Compatibility with existing callers.
    class Stage:
        def to_dict(self_nonlocal):
            return {"search_stage": "optuna", "completed_trials": len(completed), "output_dir": str(output_dir)}
    class Selection:
        def to_dict(self_nonlocal):
            return payload
        selected_model_config = final_model_config
        selected_candidate_id = payload["selected_candidate_id"]
    return [Stage()], Selection()
