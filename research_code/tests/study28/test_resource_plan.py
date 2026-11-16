from __future__ import annotations

from pathlib import Path

from los_baseline.study28 import resource_plan as rp


def test_maximum_throughput_plan_uses_machine_wide_budget(monkeypatch):
    monkeypatch.setattr(
        rp,
        "_hardware_snapshot",
        lambda: {
            "logical_cpu_threads": 32,
            "physical_cpu_cores": 16,
            "ram_total_gb": 128.0,
            "ram_available_gb_at_planning": 120.0,
            "gpu_count": 1,
            "gpus": [{"index": 0, "name": "GPU", "memory_total_gb": 48.0}],
        },
    )
    plan = rp.build_study28_resource_plan("maximum_throughput")
    assert plan.cpu_thread_budget == 30
    assert plan.nested_cv.workers == 5
    assert plan.nested_cv.threads_per_worker == 6
    assert plan.external_validation.workers == 4
    assert plan.external_validation.workers * plan.external_validation.threads_per_worker >= 28
    assert plan.reverse_validation.workers == 2
    assert plan.primary_lightgbm_backend == "cpu"
    assert "reproduction" in plan.primary_lightgbm_backend_reason


def test_small_machine_uses_all_cpu_threads(monkeypatch):
    monkeypatch.setattr(
        rp,
        "_hardware_snapshot",
        lambda: {
            "logical_cpu_threads": 6,
            "physical_cpu_cores": 6,
            "ram_total_gb": 32.0,
            "ram_available_gb_at_planning": 30.0,
            "gpu_count": 0,
            "gpus": [],
        },
    )
    plan = rp.build_study28_resource_plan("maximum_throughput")
    assert plan.cpu_thread_budget == 6
    assert plan.serial_model_threads == 6


def test_resource_environment_freezes_stage_parallelism(monkeypatch):
    monkeypatch.setattr(
        rp,
        "_hardware_snapshot",
        lambda: {
            "logical_cpu_threads": 24,
            "physical_cpu_cores": 12,
            "ram_total_gb": 64.0,
            "ram_available_gb_at_planning": 60.0,
            "gpu_count": 0,
            "gpus": [],
        },
    )
    plan = rp.build_study28_resource_plan()
    env = rp.resource_environment(plan)
    assert env["EXP123_ASSIGNED_CPU_THREADS"] == str(plan.serial_model_threads)
    assert env["STUDY28_NESTED_CV_WORKERS"] == str(plan.nested_cv.workers)
    assert env["OMP_DYNAMIC"] == "FALSE"


def test_write_resource_plan_creates_reusable_shell_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rp,
        "_hardware_snapshot",
        lambda: {
            "logical_cpu_threads": 8,
            "physical_cpu_cores": 4,
            "ram_total_gb": 16.0,
            "ram_available_gb_at_planning": 14.0,
            "gpu_count": 0,
            "gpus": [],
        },
    )
    json_path = tmp_path / "plan.json"
    env_path = tmp_path / "plan.sh"
    plan = rp.write_resource_plan(json_path=json_path, env_path=env_path)
    assert json_path.is_file() and env_path.is_file()
    text = env_path.read_text(encoding="utf-8")
    assert "STUDY28_RESOURCE_PLAN_PATH" in text
    assert str(plan.cpu_thread_budget) in text


def test_child_thread_environment_blocks_nested_blas_oversubscription():
    env = rp.child_thread_environment({}, 3)
    for key in (
        "EXP123_ASSIGNED_CPU_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        assert env[key] == "3"
    assert env["OMP_DYNAMIC"] == "FALSE"

def test_multiclass_bootstrap_parallelism_preserves_exact_seeded_samples(monkeypatch):
    import numpy as np
    from los_baseline.evaluation_multiclass import BootstrapSettings, bootstrap_multiclass_confidence_intervals
    y = np.array([0,1,2] * 30, dtype=int)
    p = np.full((len(y), 3), 0.1, dtype=float)
    p[np.arange(len(y)), y] = 0.8
    p /= p.sum(axis=1, keepdims=True)
    settings = BootstrapSettings(repeats=40, confidence_level=0.95, seed=2026, minimum_successful_repeats=30)
    monkeypatch.setenv('EXP123_ASSIGNED_CPU_THREADS', '1')
    serial = [x.to_dict() for x in bootstrap_multiclass_confidence_intervals(y, p, ('a','b','c'), settings)]
    monkeypatch.setenv('EXP123_ASSIGNED_CPU_THREADS', '4')
    parallel = [x.to_dict() for x in bootstrap_multiclass_confidence_intervals(y, p, ('a','b','c'), settings)]
    assert serial == parallel


def test_paired_bootstrap_parallelism_preserves_exact_seeded_samples(monkeypatch):
    import numpy as np
    from los_baseline.study28.robustness import _paired_bootstrap_macro_auc_difference
    y = np.array([0,1,2] * 30, dtype=int)
    primary = np.full((len(y), 3), 0.15); primary[np.arange(len(y)), y] = 0.70
    sensitivity = np.full((len(y), 3), 0.10); sensitivity[np.arange(len(y)), y] = 0.80
    monkeypatch.setenv('EXP123_ASSIGNED_CPU_THREADS', '1')
    serial = _paired_bootstrap_macro_auc_difference(y, primary, sensitivity, repeats=40, confidence_level=.95, seed=2026)
    monkeypatch.setenv('EXP123_ASSIGNED_CPU_THREADS', '4')
    parallel = _paired_bootstrap_macro_auc_difference(y, primary, sensitivity, repeats=40, confidence_level=.95, seed=2026)
    assert serial == parallel
