#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from los_baseline.study28.audit import audit_study_datasets
from los_baseline.study28.resource_plan import resource_environment, write_resource_plan
from los_baseline.study28_exp123.core import (
    EXP1_NAME,
    EXP2_NAME,
    EXP3_NAME,
    TASK_EXP1,
    TASK_STAGE1,
    TASK_STAGE2,
    cascade_probabilities,
    default_model_config,
    development_train_validation_indices,
    evaluate_cascade,
    evaluate_task_model,
    fit_task_model,
    load_context,
    load_fitted_task_model,
    load_json,
    model_output_row,
    multiclass_metrics,
)
from los_baseline.study28_exp123.hpo import optimize_model
from los_baseline.utils import atomic_write_json


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the restored Study28 EXP1/EXP2/EXP3 pipeline in two operational stages: "
            "fast baseline first, then long optimization."
        )
    )
    p.add_argument("stage", choices=["baseline", "optimize", "all", "status"])
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs" / "study28_exp123_main")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "study28_exp123_two_stage.json")
    p.add_argument("--include-svm", action="store_true", help="Include exact RBF SVM. Disabled by default so baseline results arrive quickly.")
    p.add_argument("--resource-profile", choices=["maximum_throughput", "balanced"], default="maximum_throughput")
    p.add_argument("--replan-resources", action="store_true")
    p.add_argument("--force", action="store_true", help="Recompute completed task/model artifacts instead of resuming.")
    return p


def _resolve_resources(run_root: Path, profile: str, replan: bool) -> dict[str, str]:
    plan_path = run_root / "hardware_execution_plan.json"
    env_path = run_root / "hardware_execution_env.sh"
    if replan or not plan_path.is_file():
        plan = write_resource_plan(json_path=plan_path, env_path=env_path, profile=profile)
    else:
        from los_baseline.study28.resource_plan import Study28ResourcePlan, StagePlan
        payload = load_json(plan_path)
        def sp(name: str) -> StagePlan:
            q = payload[name]
            return StagePlan(**q)
        plan = Study28ResourcePlan(
            schema_version=payload["schema_version"], created_at_utc=payload["created_at_utc"],
            profile=payload["profile"], hardware=payload["hardware"], policy=payload["policy"],
            cpu_thread_budget=int(payload["cpu_thread_budget"]), ram_budget_gb=float(payload["ram_budget_gb"]),
            primary_lightgbm_backend=payload["primary_lightgbm_backend"],
            primary_lightgbm_backend_reason=payload["primary_lightgbm_backend_reason"],
            serial_model_threads=int(payload["serial_model_threads"]),
            nested_cv=sp("nested_cv"), external_validation=sp("external_validation"),
            reverse_validation=sp("reverse_validation"), alpha5_workers=sp("alpha5_workers"),
        )
    env = resource_environment(plan)
    env["STUDY28_RESOURCE_PLAN_PATH"] = str(plan_path.resolve())
    for key, value in env.items():
        os.environ[key] = value
    return env


def _models(config: Mapping[str, Any], include_svm: bool) -> list[str]:
    models = list(config["baseline"]["models"])
    if include_svm:
        for model in config["baseline"].get("optional_models", []):
            if model not in models:
                models.append(model)
    return models


def _ensure_audit(dataset_root: Path, run_root: Path) -> None:
    out = run_root / "00_dataset_audit"
    result = audit_study_datasets(
        dataset_root=dataset_root,
        registry_path=PROJECT_ROOT / "configs" / "dataset_registry_28d.json",
        feature_list_path=dataset_root / "feature_list.csv",
        output_dir=out,
    )
    if not result.passed:
        raise RuntimeError(f"Study28 dataset audit failed. See {out}")


def _fit_provisional_temporal_external(
    *,
    context,
    task_name: str,
    model_name: str,
    model_config: Mapping[str, Any],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    seed: int,
    strategy: str,
    max_majority_ratio: float,
    artifact_dir: Path,
    force: bool,
):
    provisional_dir = artifact_dir / "provisional_threshold_selection"
    temporal_dir = artifact_dir / "temporal_full_development"
    external_dir = artifact_dir / "external_full_mimic"
    required = (
        provisional_dir / "model_manifest.json",
        temporal_dir / "model_manifest.json",
        external_dir / "model_manifest.json",
    )
    if not force and all(path.is_file() for path in required):
        return (
            load_fitted_task_model(provisional_dir),
            load_fitted_task_model(temporal_dir),
            load_fitted_task_model(external_dir),
        )
    provisional = fit_task_model(
        task_name=task_name, model_name=model_name, model_config=model_config,
        feature_names=context.features, train_frame=train_frame,
        threshold_frame=validation_frame if task_name != TASK_EXP1 else None,
        seed=seed, strategy=strategy, max_majority_ratio=max_majority_ratio,
    )
    provisional.save(provisional_dir)
    temporal_model = fit_task_model(
        task_name=task_name, model_name=model_name, model_config=model_config,
        feature_names=context.features, train_frame=context.development,
        threshold_frame=None, fixed_threshold=provisional.threshold,
        seed=seed, strategy=strategy, max_majority_ratio=max_majority_ratio,
    )
    temporal_model.save(temporal_dir)
    external_model = fit_task_model(
        task_name=task_name, model_name=model_name, model_config=model_config,
        feature_names=context.features, train_frame=context.mimic,
        threshold_frame=None, fixed_threshold=provisional.threshold,
        seed=seed, strategy=strategy, max_majority_ratio=max_majority_ratio,
    )
    external_model.save(external_dir)
    return provisional, temporal_model, external_model


def _write_meeting_summary(
    *,
    output_path: Path,
    task_results: pd.DataFrame,
    pair_results: pd.DataFrame,
    models: list[str],
    phase_label: str,
) -> None:
    val_exp1 = task_results.loc[(task_results.task == TASK_EXP1) & (task_results.split == "development_validation")]
    val_stage1 = task_results.loc[(task_results.task == TASK_STAGE1) & (task_results.split == "development_validation")]
    val_stage2 = task_results.loc[(task_results.task == TASK_STAGE2) & (task_results.split == "development_validation")]
    val_pairs = pair_results.loc[pair_results.split == "development_validation"]
    def best(frame: pd.DataFrame, metric: str):
        if frame.empty:
            return None
        return frame.loc[frame[metric].astype(float).idxmax()].to_dict()
    b1 = best(val_exp1, "macro_auroc")
    b_stage1 = best(val_stage1, "auprc")
    b_stage2 = best(val_stage2, "auprc")
    bp = best(val_pairs, "macro_auroc")
    lines = [
        f"# Study28 EXP123 {phase_label} results",
        "",
        "## Experimental structure",
        "- EXP1: flat 3-class prediction (early discharge / death / prolonged stay)",
        "- EXP2: two-stage cascade (Stage 1 death vs survivor; Stage 2 early vs prolonged among survivors)",
        "- EXP3: survivor-only binary prediction; reuses EXP2 Stage 2 artifacts (no duplicate training)",
        f"- Models: {', '.join(models)}",
        "- Model/candidate selection uses MIMIC development data only. Temporal/external results are evaluation-only.",
        "- Mandatory external cohorts: eICU-alt, eICU-primary, SICdb, NWICU.",
        "",
        "## Development-selection leaders",
    ]
    if b1:
        lines.append(f"- EXP1: **{b1['model']}** — validation macro-AUROC {float(b1['macro_auroc']):.4f}")
    if b_stage1:
        lines.append(f"- EXP2 Stage 1 mortality: **{b_stage1['model']}** — validation AUPRC {float(b_stage1['auprc']):.4f}")
    if b_stage2:
        lines.append(f"- EXP2 Stage 2 / EXP3: **{b_stage2['model']}** — validation AUPRC {float(b_stage2['auprc']):.4f}")
    if bp:
        lines.append(
            f"- EXP2 best pair: **{bp['stage1_model']} → {bp['stage2_model']}** — validation macro-AUROC {float(bp['macro_auroc']):.4f}"
        )
    lines += ["", "## Temporal and external results", "See `task_results.csv` and `exp2_pairwise_results.csv` for the complete model-by-site matrix."]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_baseline(args, config: Mapping[str, Any]) -> int:
    if args.dataset_root is None:
        raise SystemExit("--dataset-root is required for baseline")
    run_root = args.run_root.expanduser().resolve()
    out = run_root / "stage1_baseline"
    out.mkdir(parents=True, exist_ok=True)
    marker = out / config["reporting"]["baseline_ready_marker"]
    if marker.is_file() and not args.force:
        print(f"[BASELINE] Existing completed baseline found: {out}")
        return 0

    _ensure_audit(args.dataset_root, run_root)
    context = load_context(dataset_root=args.dataset_root, project_root=PROJECT_ROOT, feature_set=config["feature_set"])
    seed = int(config["seed"])
    train_idx, val_idx = development_train_validation_indices(
        context.development,
        validation_fraction=float(config["baseline"]["validation_fraction"]),
        seed=seed,
    )
    train_frame = context.development.iloc[train_idx].reset_index(drop=True)
    validation_frame = context.development.iloc[val_idx].reset_index(drop=True)
    models = _models(config, args.include_svm)

    provisional: dict[str, dict[str, Any]] = {TASK_EXP1: {}, TASK_STAGE1: {}, TASK_STAGE2: {}}
    temporal_models: dict[str, dict[str, Any]] = {TASK_EXP1: {}, TASK_STAGE1: {}, TASK_STAGE2: {}}
    external_models: dict[str, dict[str, Any]] = {TASK_EXP1: {}, TASK_STAGE1: {}, TASK_STAGE2: {}}
    rows: list[dict[str, Any]] = []
    for task_name in (TASK_EXP1, TASK_STAGE1, TASK_STAGE2):
        experiment = EXP1_NAME if task_name == TASK_EXP1 else (EXP2_NAME if task_name == TASK_STAGE1 else EXP3_NAME)
        for model_name in models:
            print(f"[BASELINE] {task_name} / {model_name}", flush=True)
            cfg = default_model_config(model_name, config)
            pmodel, tmodel, emodel = _fit_provisional_temporal_external(
                context=context,
                task_name=task_name,
                model_name=model_name,
                model_config=cfg,
                train_frame=train_frame,
                validation_frame=validation_frame,
                seed=seed,
                strategy="natural",
                max_majority_ratio=float(config["optimization"]["majority_downsampling"]["max_majority_to_minority_ratio"]),
                artifact_dir=out / "models" / task_name / model_name,
                force=args.force,
            )
            provisional[task_name][model_name] = pmodel
            temporal_models[task_name][model_name] = tmodel
            external_models[task_name][model_name] = emodel
            rows.append(model_output_row(
                phase="baseline", experiment=experiment, task=task_name, model_name=model_name,
                split_name="development_validation", strategy="natural",
                metrics=evaluate_task_model(pmodel, validation_frame),
            ))
            rows.append(model_output_row(
                phase="baseline", experiment=experiment, task=task_name, model_name=model_name,
                split_name="mimic_temporal", strategy="natural",
                metrics=evaluate_task_model(tmodel, context.temporal),
            ))
            for site, frame in context.external.items():
                rows.append(model_output_row(
                    phase="baseline", experiment=experiment, task=task_name, model_name=model_name,
                    split_name=site, strategy="natural", metrics=evaluate_task_model(emodel, frame),
                ))

    task_results = pd.DataFrame(rows)
    task_results.to_csv(out / "task_results.csv", index=False)
    # EXP3 is literally the Stage-2 artifact/reference.  Create an explicit view for reporting.
    exp3 = task_results.loc[task_results.task == TASK_STAGE2].copy()
    exp3["experiment"] = EXP3_NAME
    exp3.to_csv(out / "exp3_survivor_binary_results.csv", index=False)

    pair_rows: list[dict[str, Any]] = []
    for stage1_name in models:
        for stage2_name in models:
            for split_name, frame, source in [
                ("development_validation", validation_frame, provisional),
                ("mimic_temporal", context.temporal, temporal_models),
                *[(site, ext, external_models) for site, ext in context.external.items()],
            ]:
                metrics = evaluate_cascade(
                    stage1=source[TASK_STAGE1][stage1_name],
                    stage2=source[TASK_STAGE2][stage2_name],
                    frame=frame,
                )
                pair_rows.append({
                    "phase": "baseline", "experiment": EXP2_NAME,
                    "stage1_model": stage1_name, "stage2_model": stage2_name,
                    "stage1_strategy": "natural", "stage2_strategy": "natural",
                    "split": split_name, **metrics,
                })
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out / "exp2_pairwise_results.csv", index=False)
    _write_meeting_summary(
        output_path=out / "MEETING_BASELINE_SUMMARY.md",
        task_results=task_results,
        pair_results=pairs,
        models=models,
        phase_label="Baseline",
    )
    # Compact meeting table: selection is development-only, followed by temporal/external rows.
    meeting_rows: list[dict[str, Any]] = []
    exp1_val = task_results.loc[(task_results.task == TASK_EXP1) & (task_results.split == "development_validation")].copy()
    exp1_best = str(exp1_val.loc[exp1_val["macro_auroc"].astype(float).idxmax(), "model"])
    exp3_val = task_results.loc[(task_results.task == TASK_STAGE2) & (task_results.split == "development_validation")].copy()
    exp3_best = str(exp3_val.loc[exp3_val["auprc"].astype(float).idxmax(), "model"])
    pair_val = pairs.loc[pairs.split == "development_validation"].copy()
    pair_best_row = pair_val.loc[pair_val["macro_auroc"].astype(float).idxmax()]
    pair_best = (str(pair_best_row["stage1_model"]), str(pair_best_row["stage2_model"]))
    for exp_name, task_name, candidate in ((EXP1_NAME, TASK_EXP1, exp1_best), (EXP3_NAME, TASK_STAGE2, exp3_best)):
        view = task_results.loc[(task_results.task == task_name) & (task_results.model == candidate)]
        for _, row in view.iterrows():
            meeting_rows.append({
                "experiment": exp_name, "candidate": candidate, "split": row["split"],
                "macro_auroc": row.get("macro_auroc"), "macro_auprc": row.get("macro_auprc"),
                "auroc": row.get("auroc"), "auprc": row.get("auprc"),
                "balanced_accuracy": row.get("balanced_accuracy"),
            })
    view = pairs.loc[(pairs.stage1_model == pair_best[0]) & (pairs.stage2_model == pair_best[1])]
    for _, row in view.iterrows():
        meeting_rows.append({
            "experiment": EXP2_NAME, "candidate": f"{pair_best[0]} -> {pair_best[1]}", "split": row["split"],
            "macro_auroc": row.get("macro_auroc"), "macro_auprc": row.get("macro_auprc"),
            "auroc": None, "auprc": None, "balanced_accuracy": row.get("balanced_accuracy"),
        })
    pd.DataFrame(meeting_rows).to_csv(out / "BASELINE_FOR_MEETING.csv", index=False)
    payload = {
        "status": "ready",
        "stage": "baseline",
        "models": models,
        "feature_set": config["feature_set"],
        "development_n": len(context.development),
        "temporal_n": len(context.temporal),
        "external_sites": {site: len(frame) for site, frame in context.external.items()},
        "selection_policy": "MIMIC development validation only; temporal/external never select candidates",
        "meeting_summary": str((out / "MEETING_BASELINE_SUMMARY.md").resolve()),
        "task_results": str((out / "task_results.csv").resolve()),
        "exp2_pairwise_results": str((out / "exp2_pairwise_results.csv").resolve()),
        "baseline_for_meeting": str((out / "BASELINE_FOR_MEETING.csv").resolve()),
    }
    atomic_write_json(marker, payload)
    print("\n[BASELINE_READY] Baseline results are ready for the meeting.")
    print(f"[BASELINE_READY] Summary: {out / 'MEETING_BASELINE_SUMMARY.md'}")
    print(f"[BASELINE_READY] Meeting table: {out / 'BASELINE_FOR_MEETING.csv'}")
    print(f"[BASELINE_READY] Full matrix: {out / 'task_results.csv'}")
    print("[BASELINE_READY] Optimization can now continue without changing these baseline files.\n")
    return 0


def run_optimize(args, config: Mapping[str, Any]) -> int:
    if args.dataset_root is None:
        raise SystemExit("--dataset-root is required for optimize")
    run_root = args.run_root.expanduser().resolve()
    baseline_dir = run_root / "stage1_baseline"
    baseline_marker = baseline_dir / config["reporting"]["baseline_ready_marker"]
    if not baseline_marker.is_file():
        print("[OPTIMIZE] Baseline is not complete; running baseline first.")
        run_baseline(args, config)
    out = run_root / "stage2_optimization"
    out.mkdir(parents=True, exist_ok=True)
    marker = out / config["reporting"]["optimization_complete_marker"]
    if marker.is_file() and not args.force:
        print(f"[OPTIMIZE] Existing completed optimization found: {out}")
        return 0

    context = load_context(dataset_root=args.dataset_root, project_root=PROJECT_ROOT, feature_set=config["feature_set"])
    seed = int(config["seed"])
    train_idx, val_idx = development_train_validation_indices(
        context.development,
        validation_fraction=float(config["baseline"]["validation_fraction"]),
        seed=seed,
    )
    train_frame = context.development.iloc[train_idx].reset_index(drop=True)
    validation_frame = context.development.iloc[val_idx].reset_index(drop=True)
    models = _models(config, args.include_svm)
    opt_cfg = config["optimization"]
    folds = int(opt_cfg["cv_folds"])
    ratio = float(opt_cfg["majority_downsampling"]["max_majority_to_minority_ratio"])

    selected: dict[str, dict[str, Any]] = {TASK_EXP1: {}, TASK_STAGE1: {}, TASK_STAGE2: {}}
    provisional: dict[str, dict[str, Any]] = {TASK_EXP1: {}, TASK_STAGE1: {}, TASK_STAGE2: {}}
    temporal_models: dict[str, dict[str, Any]] = {TASK_EXP1: {}, TASK_STAGE1: {}, TASK_STAGE2: {}}
    external_models: dict[str, dict[str, Any]] = {TASK_EXP1: {}, TASK_STAGE1: {}, TASK_STAGE2: {}}
    optimization_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    for task_name in (TASK_EXP1, TASK_STAGE1, TASK_STAGE2):
        experiment = EXP1_NAME if task_name == TASK_EXP1 else (EXP2_NAME if task_name == TASK_STAGE1 else EXP3_NAME)
        for model_name in models:
            print(f"[OPTIMIZE] HPO {task_name} / {model_name}", flush=True)
            model_out = out / "hpo" / task_name / model_name
            result_path = model_out / "optimization_result.json"
            if result_path.is_file() and not args.force:
                optimization = load_json(result_path)
            else:
                optimization = optimize_model(
                    frame=context.development,
                    task_name=task_name,
                    model_name=model_name,
                    base_model_config=default_model_config(model_name, config),
                    feature_names=context.features,
                    output_dir=model_out,
                    seed=seed,
                    folds=folds,
                    n_trials=int(opt_cfg["trial_budgets"][model_name]),
                    max_majority_ratio=ratio,
                )
            selected[task_name][model_name] = optimization
            strategy = str(optimization["selected_strategy"])
            best_cfg = dict(optimization["best_model_config"])
            pmodel, tmodel, emodel = _fit_provisional_temporal_external(
                context=context,
                task_name=task_name,
                model_name=model_name,
                model_config=best_cfg,
                train_frame=train_frame,
                validation_frame=validation_frame,
                seed=seed,
                strategy=strategy,
                max_majority_ratio=ratio,
                artifact_dir=out / "models" / task_name / model_name,
                force=args.force,
            )
            provisional[task_name][model_name] = pmodel
            temporal_models[task_name][model_name] = tmodel
            external_models[task_name][model_name] = emodel
            optimization_rows.append({
                "task": task_name, "model": model_name,
                "objective": optimization["objective"],
                "natural_cv_score": optimization["best_natural_cv_score"],
                "downsampling_cv_score": optimization["downsampling_cv_score"],
                "selected_strategy": strategy,
                "n_trials_complete": optimization["n_trials_complete"],
            })
            for split_name, frame, mdl in [
                ("development_validation", validation_frame, pmodel),
                ("mimic_temporal", context.temporal, tmodel),
                *[(site, ext, emodel) for site, ext in context.external.items()],
            ]:
                result_rows.append(model_output_row(
                    phase="optimized", experiment=experiment, task=task_name,
                    model_name=model_name, split_name=split_name, strategy=strategy,
                    metrics=evaluate_task_model(mdl, frame),
                ))

    pd.DataFrame(optimization_rows).to_csv(out / "optimization_selection.csv", index=False)
    task_results = pd.DataFrame(result_rows)
    task_results.to_csv(out / "task_results.csv", index=False)
    exp3 = task_results.loc[task_results.task == TASK_STAGE2].copy()
    exp3["experiment"] = EXP3_NAME
    exp3.to_csv(out / "exp3_survivor_binary_results.csv", index=False)

    pair_rows: list[dict[str, Any]] = []
    for stage1_name in models:
        for stage2_name in models:
            s1_strategy = selected[TASK_STAGE1][stage1_name]["selected_strategy"]
            s2_strategy = selected[TASK_STAGE2][stage2_name]["selected_strategy"]
            for split_name, frame, source in [
                ("development_validation", validation_frame, provisional),
                ("mimic_temporal", context.temporal, temporal_models),
                *[(site, ext, external_models) for site, ext in context.external.items()],
            ]:
                metrics = evaluate_cascade(
                    stage1=source[TASK_STAGE1][stage1_name],
                    stage2=source[TASK_STAGE2][stage2_name],
                    frame=frame,
                )
                pair_rows.append({
                    "phase": "optimized", "experiment": EXP2_NAME,
                    "stage1_model": stage1_name, "stage2_model": stage2_name,
                    "stage1_strategy": s1_strategy, "stage2_strategy": s2_strategy,
                    "split": split_name, **metrics,
                })
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out / "exp2_pairwise_results.csv", index=False)

    _write_meeting_summary(
        output_path=out / "OPTIMIZED_RESULTS_SUMMARY.md",
        task_results=task_results,
        pair_results=pairs,
        models=models,
        phase_label="Optimized",
    )

    # Side-by-side baseline vs optimized table on matching task/model/split keys.
    baseline_tasks = pd.read_csv(baseline_dir / "task_results.csv")
    comparison = baseline_tasks.merge(
        task_results,
        on=["experiment", "task", "model", "split"],
        how="inner",
        suffixes=("_baseline", "_optimized"),
    )
    for metric in ("macro_auroc", "macro_auprc", "auroc", "auprc", "balanced_accuracy"):
        left, right = f"{metric}_baseline", f"{metric}_optimized"
        if left in comparison.columns and right in comparison.columns:
            comparison[f"delta_{metric}"] = comparison[right] - comparison[left]
    comparison.to_csv(out / "baseline_vs_optimized.csv", index=False)

    # Compact, selection-safe final comparison for meetings/manuscript planning.
    baseline_pairs = pd.read_csv(baseline_dir / "exp2_pairwise_results.csv")
    selected_rows: list[dict[str, Any]] = []
    def selected_model_name(frame: pd.DataFrame, task: str, metric: str) -> str:
        view = frame.loc[(frame["task"] == task) & (frame["split"] == "development_validation")].copy()
        return str(view.loc[view[metric].astype(float).idxmax(), "model"])
    def selected_pair(frame: pd.DataFrame) -> tuple[str, str]:
        view = frame.loc[frame["split"] == "development_validation"].copy()
        row = view.loc[view["macro_auroc"].astype(float).idxmax()]
        return str(row["stage1_model"]), str(row["stage2_model"])
    for phase_name, frame in (("baseline", baseline_tasks), ("optimized", task_results)):
        exp1_model = selected_model_name(frame, TASK_EXP1, "macro_auroc")
        exp3_model = selected_model_name(frame, TASK_STAGE2, "auprc")
        for exp_name, task_name, model_name, metric_name in (
            (EXP1_NAME, TASK_EXP1, exp1_model, "macro_auroc"),
            (EXP3_NAME, TASK_STAGE2, exp3_model, "auprc"),
        ):
            view = frame.loc[(frame["task"] == task_name) & (frame["model"] == model_name)].copy()
            for _, row in view.iterrows():
                selected_rows.append({
                    "phase": phase_name, "experiment": exp_name, "candidate": model_name,
                    "split": row["split"], "selection_metric": metric_name,
                    "macro_auroc": row.get("macro_auroc"), "macro_auprc": row.get("macro_auprc"),
                    "auroc": row.get("auroc"), "auprc": row.get("auprc"),
                    "balanced_accuracy": row.get("balanced_accuracy"),
                })
    for phase_name, frame in (("baseline", baseline_pairs), ("optimized", pairs)):
        s1, s2 = selected_pair(frame)
        view = frame.loc[(frame["stage1_model"] == s1) & (frame["stage2_model"] == s2)].copy()
        for _, row in view.iterrows():
            selected_rows.append({
                "phase": phase_name, "experiment": EXP2_NAME, "candidate": f"{s1} -> {s2}",
                "split": row["split"], "selection_metric": "macro_auroc",
                "macro_auroc": row.get("macro_auroc"), "macro_auprc": row.get("macro_auprc"),
                "auroc": None, "auprc": None, "balanced_accuracy": row.get("balanced_accuracy"),
            })
    pd.DataFrame(selected_rows).to_csv(out / "FINAL_COMPARISON_FOR_MEETING.csv", index=False)

    payload = {
        "status": "complete",
        "stage": "optimization",
        "models": models,
        "feature_set": config["feature_set"],
        "optimization_selection": str((out / "optimization_selection.csv").resolve()),
        "task_results": str((out / "task_results.csv").resolve()),
        "exp2_pairwise_results": str((out / "exp2_pairwise_results.csv").resolve()),
        "baseline_vs_optimized": str((out / "baseline_vs_optimized.csv").resolve()),
        "final_comparison_for_meeting": str((out / "FINAL_COMPARISON_FOR_MEETING.csv").resolve()),
        "summary": str((out / "OPTIMIZED_RESULTS_SUMMARY.md").resolve()),
        "external_sites": list(context.external),
        "selection_policy": "HPO and natural-vs-downsampling selection use MIMIC development CV only; temporal/external never select candidates",
    }
    atomic_write_json(marker, payload)
    print("\n[OPTIMIZATION_COMPLETE] Optimized EXP1/EXP2/EXP3 results are ready.")
    print(f"[OPTIMIZATION_COMPLETE] Comparison: {out / 'baseline_vs_optimized.csv'}")
    print(f"[OPTIMIZATION_COMPLETE] Summary: {out / 'OPTIMIZED_RESULTS_SUMMARY.md'}\n")
    return 0


def show_status(args, config: Mapping[str, Any]) -> int:
    root = args.run_root.expanduser().resolve()
    b = root / "stage1_baseline" / config["reporting"]["baseline_ready_marker"]
    o = root / "stage2_optimization" / config["reporting"]["optimization_complete_marker"]
    print(f"Run root: {root}")
    print(f"Baseline: {'READY' if b.is_file() else 'NOT READY'}")
    print(f"Optimization: {'COMPLETE' if o.is_file() else 'RUNNING / NOT STARTED'}")
    if b.is_file():
        print(f"Baseline meeting summary: {root / 'stage1_baseline' / 'MEETING_BASELINE_SUMMARY.md'}")
    if o.is_file():
        print(f"Optimized summary: {root / 'stage2_optimization' / 'OPTIMIZED_RESULTS_SUMMARY.md'}")
    return 0


def main() -> int:
    args = parser().parse_args()
    args.run_root = args.run_root.expanduser().resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)
    config = load_json(args.config)
    _resolve_resources(args.run_root, args.resource_profile, args.replan_resources)
    if args.stage == "status":
        return show_status(args, config)
    if args.stage == "baseline":
        return run_baseline(args, config)
    if args.stage == "optimize":
        return run_optimize(args, config)
    if args.stage == "all":
        run_baseline(args, config)
        print("[TWO_STAGE] Baseline is ready. Continuing into the long optimization stage now.")
        return run_optimize(args, config)
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
