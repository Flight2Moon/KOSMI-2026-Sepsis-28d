from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from los_baseline.config import load_config
from los_baseline.optimization import resolve_search_stage_config
from los_baseline.staged_protocol import (
    select_targeted_candidates,
    write_targeted_optimization_config,
)
from run_exp123_staged_protocol import baseline_jobs, optimized_jobs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_cuda_staged_7800x3d_48gb.json"


def _write_summary(run_dir: Path, task_key: str) -> None:
    paths = {
        "exp1": run_dir / "EXP1_flat_3class" / "model_summary_validation.csv",
        "stage1": run_dir / "EXP2_two_stage" / "stage1_mortality" / "model_summary_validation.csv",
        "stage2": run_dir / "EXP2_two_stage" / "stage2_long_stay" / "model_summary_validation.csv",
    }
    rows = []
    models = [
        "decision_tree",
        "logistic_regression",
        "random_forest",
        "svm",
        "xgboost",
    ]
    for index, model in enumerate(models):
        row = {"model_name": model}
        if task_key == "exp1":
            row.update(
                validation_macro_auroc=0.70 + index * 0.01,
                validation_multiclass_brier=0.40 - index * 0.01,
            )
        elif task_key == "stage1":
            row.update(
                validation_auprc=0.20 + index * 0.02,
                validation_brier_score=0.18 - index * 0.01,
            )
        else:
            row.update(
                validation_auroc=0.65 + index * 0.015,
                validation_brier_score=0.20 - index * 0.01,
            )
        rows.append(row)
    path = paths[task_key]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_model_specific_search_budget_resolution() -> None:
    config = load_config(CONFIG_PATH)
    resolved = resolve_search_stage_config(
        config=config,
        task_name="stage1_mortality",
        model_name="xgboost",
        search_stage="coarse",
    )
    assert resolved == {"trials": 36, "cv_folds": 5}
    svm = resolve_search_stage_config(
        config=config,
        task_name="stage2_long_stay",
        model_name="svm",
        search_stage="fine",
    )
    assert svm == {"trials": 6, "cv_folds": 5}


def test_candidate_selection_uses_baseline_and_pairwise_results(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    run_dir = tmp_path / "runs" / "baseline"
    for task in ("exp1", "stage1", "stage2"):
        _write_summary(run_dir, task)
    split_dir = run_dir / "common" / "splits"
    split_dir.mkdir(parents=True)
    split_dir.joinpath("split_hash.txt").write_text("shared-hash\n", encoding="utf-8")
    pair_dir = run_dir / "common" / "staged_baseline" / "exp2_pairwise_validation"
    pair_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "stage1_model": "svm",
                "stage2_model": "random_forest",
                "macro_auroc": 0.88,
                "macro_auprc": 0.70,
                "multiclass_brier": 0.25,
                "multiclass_log_loss": 0.60,
            },
            {
                "stage1_model": "xgboost",
                "stage2_model": "xgboost",
                "macro_auroc": 0.87,
                "macro_auprc": 0.69,
                "multiclass_brier": 0.24,
                "multiclass_log_loss": 0.59,
            },
        ]
    ).to_csv(pair_dir / "pairwise_pipeline_validation.csv", index=False)

    result = select_targeted_candidates(
        config=config,
        run_id="baseline",
        run_dir=run_dir,
    )
    assert result.split_hash == "shared-hash"
    # LR and XGBoost are always retained for interpretation and Super Learner.
    assert "logistic_regression" in result.selected_models["stage1"]
    assert "xgboost" in result.selected_models["stage1"]
    assert len(result.selected_models["stage1"]) <= 3
    assert len(result.selected_models["stage2"]) <= 3
    assert Path(result.output_path).is_file()


def test_targeted_config_filters_models_and_records_link(tmp_path: Path) -> None:
    plan = {
        "baseline_run_id": "baseline_run",
        "shared_split_hash": "split-hash",
        "selected_models": {
            "exp1": ["logistic_regression", "xgboost"],
            "stage1": ["logistic_regression", "xgboost", "svm"],
            "stage2": ["logistic_regression", "xgboost"],
        },
    }
    plan_path = tmp_path / "candidate_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "targeted.json"
    write_targeted_optimization_config(
        base_config_path=CONFIG_PATH,
        candidate_plan_path=plan_path,
        output_path=output,
    )
    config = load_config(output)
    assert config["experiments"]["exp1_flat_3class"]["models"] == [
        "logistic_regression",
        "xgboost",
    ]
    assert config["staged_protocol"]["linked_baseline"]["run_id"] == "baseline_run"
    assert config["evaluation"]["binary_bootstrap_repeats"] == 1000


def test_baseline_graph_trains_before_candidate_selection() -> None:
    config = load_config(CONFIG_PATH)
    jobs = baseline_jobs(config, CONFIG_PATH, "baseline")
    names = {job.name for job in jobs}
    assert "baseline-candidate-plan" in names
    assert not any("targeted-optimize" in name for name in names)
    candidate = next(job for job in jobs if job.name == "baseline-candidate-plan")
    assert "exp2-baseline-pairwise-validation" in candidate.dependencies


def test_optimized_graph_is_gated_by_baseline_link(tmp_path: Path) -> None:
    plan = {
        "baseline_run_id": "baseline",
        "shared_split_hash": "hash",
        "selected_models": {
            "exp1": ["logistic_regression", "xgboost"],
            "stage1": ["logistic_regression", "xgboost"],
            "stage2": ["logistic_regression", "xgboost"],
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    targeted_path = tmp_path / "targeted.json"
    write_targeted_optimization_config(
        base_config_path=CONFIG_PATH,
        candidate_plan_path=plan_path,
        output_path=targeted_path,
    )
    config = load_config(targeted_path)
    jobs = optimized_jobs(config, targeted_path, "optimized")
    optimize_jobs = [job for job in jobs if job.kind == "targeted_optimization"]
    assert len(optimize_jobs) == 6
    assert all("verify-baseline-link" in job.dependencies for job in optimize_jobs)
    assert not any(job.model_name == "svm" for job in optimize_jobs)
