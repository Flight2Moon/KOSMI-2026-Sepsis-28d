from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from los_baseline.experiment_scheduler import ResourceAwareScheduler, ScheduledJob
from los_baseline.external_validation_guard import audit_external_mortality_flag
from los_baseline.optimization_objectives import ObjectiveDefinition, robust_selection_score
from los_baseline.plots import create_evaluation_figures, save_figures
from los_baseline.resource_manager import ResourceRequest, request_for_job
from los_baseline.search_spaces import MODEL_SEARCH_SPACES


def _cuda_resource_config() -> dict:
    return {
        "hardware": {"accelerator": {"backend": "cuda"}},
        "resources": {
            "accelerator": {"backend": "cuda"},
            "cpu": {
                "total_thread_limit": 8,
                "default_threads_per_job": 2,
                "max_threads_per_job": 8,
            },
            "memory": {"total_limit_gb": 32, "max_fraction": 1.0},
            "scheduler": {"max_concurrent_jobs": 4},
            "cuda": {"gpu_ids": [0], "max_jobs_per_gpu": 2},
            "job_resources": {
                "logistic_regression": {
                    "cpu_threads": 2,
                    "ram_gb": 7.0,
                    "accelerator": "cuda",
                    "gpu_memory_gb": 2.5,
                }
            },
            "resource_failure_policy": {
                "retry_ram_scale": 1.5,
                "retry_gpu_memory_scale": 1.25,
                "retry_unified_memory_scale": 1.35,
                "max_ram_gb_per_job": 12.0,
                "max_gpu_memory_gb_per_job": 8.0,
                "max_unified_memory_gb_per_job": 10.0,
                "retry_exclusive_gpu": True,
            },
        },
    }


def test_cuda_logistic_regression_is_forced_to_cpu() -> None:
    request = request_for_job(
        _cuda_resource_config(),
        model_name="logistic_regression",
        job_kind="training",
    )
    assert request.accelerator == "cpu"
    assert request.ram_gb == pytest.approx(7.0)
    assert request.gpu_memory_gb == 0.0
    assert request.exclusive_gpu is False


def test_resource_retry_request_scales_and_respects_caps(tmp_path: Path) -> None:
    job = ScheduledJob(
        name="rf",
        command=("python", "-c", "pass"),
        request=ResourceRequest(
            cpu_threads=2,
            ram_gb=6.0,
            accelerator="cuda",
            gpu_memory_gb=6.0,
            unified_memory_gb=6.0,
        ),
    )
    scheduler = ResourceAwareScheduler(
        jobs=[job],
        config=_cuda_resource_config(),
        project_root=tmp_path,
        run_dir=tmp_path / "run",
        run_id="test",
    )
    scheduler.resource_retry_levels[job.name] = 2
    effective = scheduler._effective_request(job)
    assert effective.ram_gb == pytest.approx(12.0)
    assert effective.gpu_memory_gb == pytest.approx(8.0)
    assert effective.unified_memory_gb == pytest.approx(10.0)
    assert effective.exclusive_gpu is True


def test_rf_search_space_excludes_high_risk_options() -> None:
    space = MODEL_SEARCH_SPACES["random_forest"]
    assert None not in space["max_depth"].values
    assert 1 not in space["min_samples_leaf"].values
    assert space["bootstrap"].values == (True,)


def test_robust_score_penalizes_train_validation_gap() -> None:
    objective = ObjectiveDefinition(
        primary="auroc",
        direction="maximize",
        secondary="brier_score",
        secondary_direction="minimize",
    )
    stable = robust_selection_score(
        {"mean_auroc": 0.80, "std_auroc": 0.02, "mean_train_auroc": 0.82},
        objective,
        stability_penalty=0.1,
        overfit_penalty=0.5,
    )
    overfit = robust_selection_score(
        {"mean_auroc": 0.80, "std_auroc": 0.02, "mean_train_auroc": 0.98},
        objective,
        stability_penalty=0.1,
        overfit_penalty=0.5,
    )
    assert stable > overfit
    assert stable - overfit == pytest.approx(0.08)


def test_external_mortality_guard_detects_all_zero_mapping(tmp_path: Path) -> None:
    path = tmp_path / "eicu.csv"
    pd.DataFrame({"hospital_expire_flag": [0] * 50}).to_csv(path, index=False)
    audit = audit_external_mortality_flag(
        dataset_path=path,
        mortality_column="hospital_expire_flag",
        settings={"minimum_death_count": 2, "minimum_death_rate": 0.01},
    )
    assert audit.passed is False
    assert audit.death_rate == 0.0
    assert any(reason.startswith("death_count_below_minimum") for reason in audit.blocking_reasons)


def test_external_mortality_guard_accepts_normalized_strings(tmp_path: Path) -> None:
    path = tmp_path / "eicu.csv"
    values = ["Alive"] * 18 + [" Expired "] * 2
    pd.DataFrame({"hospital_expire_flag": values}).to_csv(path, index=False)
    audit = audit_external_mortality_flag(
        dataset_path=path,
        mortality_column="hospital_expire_flag",
        settings={
            "minimum_death_count": 2,
            "minimum_death_rate": 0.01,
            "maximum_death_rate": 0.5,
        },
    )
    assert audit.passed is True
    assert audit.death_rate == pytest.approx(0.1)


def test_reference_equivalent_combined_plot_is_saved(tmp_path: Path) -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    probability = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95], dtype=float)
    figures = create_evaluation_figures(
        y_true=y_true,
        probability=probability,
        decision_threshold=0.5,
        title_prefix="SVM",
        calibration_bins=3,
    )
    try:
        assert "calibration_roc_summary" in figures
        save_figures(figures, tmp_path)
        assert (tmp_path / "calibration_roc_summary.png").is_file()
        assert (tmp_path / "roc_curve.png").is_file()
        assert (tmp_path / "calibration_curve.png").is_file()
    finally:
        for figure in figures.values():
            plt.close(figure)
