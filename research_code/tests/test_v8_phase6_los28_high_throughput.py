from __future__ import annotations

import json
from pathlib import Path

import pytest

from los_baseline.experiment_matrix import build_matrix_plan
from los_baseline.final_freeze_inventory import iter_release_source_files
from los_baseline.models import build_task_sklearn_model
from los_baseline.resource_autotune import HardwareInventory, autotune_config


ROOT = Path(__file__).resolve().parents[1]


def _base_config() -> dict:
    return json.loads((ROOT / "configs" / "experiment_cuda_v8_los28.json").read_text())


def test_phase6_matrix_is_los28_only() -> None:
    cfg = _base_config()
    plan = build_matrix_plan(cfg, backend="cuda", timestamp="2026_0822_000000")
    assert plan["requested_horizons"] == [28]
    assert [r["strategy"] for r in plan["runs"]] == [
        "T1_baseline", "T2_optuna", "T3_smote", "T4_undersampling"
    ]
    assert plan["runs"][3]["status"] == "not_requested"
    with pytest.raises(ValueError, match="LOS28 only"):
        build_matrix_plan(cfg, horizons=[14], backend="cuda", timestamp="x")


def test_phase6_autotune_cpu_fallback_is_executable() -> None:
    cfg = _base_config()
    hw = HardwareInventory(
        logical_cpus=32,
        physical_cpus=16,
        total_ram_gb=64.0,
        available_ram_gb=60.0,
        gpus=(),
    )
    tuned, manifest = autotune_config(cfg, profile="maximum_throughput", hardware=hw)
    assert tuned["resources"]["accelerator"]["backend"] == "cpu"
    assert tuned["resources"]["cpu"]["total_thread_limit"] == 30
    ram_limit = tuned["resources"]["memory"]["total_limit_gb"]
    reserve = tuned["resources"]["memory"]["reserve_gb"]
    for spec in tuned["resources"]["job_resources"].values():
        if isinstance(spec, dict) and "ram_gb" in spec:
            assert float(spec["ram_gb"]) <= ram_limit - reserve + 1e-9
        if isinstance(spec, dict):
            assert spec.get("accelerator", "cpu") != "cuda"
    assert manifest["resolved"]["backend"] == "cpu"


def test_phase6_autotune_uses_multiple_gpu_slots() -> None:
    cfg = _base_config()
    gpus = (
        {"index": 0, "name": "GPU0", "memory_total_gb": 49.0, "memory_free_gb": 48.0, "memory_used_gb": 1.0, "utilization_percent": 0.0, "temperature_c": 30.0, "power_draw_w": 40.0},
        {"index": 1, "name": "GPU1", "memory_total_gb": 49.0, "memory_free_gb": 48.0, "memory_used_gb": 1.0, "utilization_percent": 0.0, "temperature_c": 30.0, "power_draw_w": 40.0},
    )
    hw = HardwareInventory(32, 16, 128.0, 120.0, gpus)
    tuned, manifest = autotune_config(cfg, profile="maximum_throughput", hardware=hw)
    assert tuned["resources"]["accelerator"]["backend"] == "cuda"
    assert tuned["resources"]["cuda"]["gpu_ids"] == [0, 1]
    assert tuned["resources"]["cuda"]["max_jobs_per_gpu"] >= 2
    assert tuned["resources"]["scheduler"]["max_concurrent_by_model"]["xgboost"] >= 4
    assert manifest["resolved"]["scheduler_max_concurrent_jobs"] >= 4


def test_maximum_throughput_allocates_more_cpu_than_balanced() -> None:
    cfg = _base_config()
    hw = HardwareInventory(64, 32, 256.0, 240.0, ())
    maximum, _ = autotune_config(cfg, profile="maximum_throughput", hardware=hw)
    balanced, _ = autotune_config(cfg, profile="balanced", hardware=hw)
    assert maximum["resources"]["cpu"]["total_thread_limit"] > balanced["resources"]["cpu"]["total_thread_limit"]
    assert maximum["resources"]["memory"]["total_limit_gb"] > balanced["resources"]["memory"]["total_limit_gb"]


def test_scheduler_assigned_threads_reach_random_forest(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = {"parameters": {"n_estimators": 5}}
    monkeypatch.setenv("EXP123_ASSIGNED_CPU_THREADS", "3")
    model = build_task_sklearn_model(
        "random_forest", cfg, seed=42, task_type="binary", num_classes=2,
        class_weights={0: 1.0, 1: 1.0},
    )
    assert model.n_jobs == 3


def test_release_inventory_tracks_scripts_directory() -> None:
    paths = {p.relative_to(ROOT).as_posix() for p in iter_release_source_files(ROOT)}
    assert "scripts/workflows/run_exp123_v8_matrix.sh" in paths
    assert "scripts/ops/autotune_exp123_resources.py" in paths
    assert "scripts/internal/run_exp123_v8_sampling_worker.py" in paths


def test_project_root_is_decluttered() -> None:
    root_files = {p.name for p in ROOT.iterdir() if p.is_file()}
    assert "exp123" in root_files
    assert "run_exp123_v8_matrix.sh" not in root_files
    assert "verify_v8_phase5_confirmatory.sh" not in root_files
    assert len(root_files) <= 20


def test_sampling_tracks_use_resource_aware_parallel_workers() -> None:
    smote = (ROOT / "scripts" / "workflows" / "run_exp123_v8_smote.py").read_text(encoding="utf-8")
    undersampling = (ROOT / "scripts" / "workflows" / "run_exp123_v8_undersampling.py").read_text(encoding="utf-8")
    worker = ROOT / "scripts" / "internal" / "run_exp123_v8_sampling_worker.py"
    assert worker.is_file()
    assert "v8_track3_smote_parallel" in smote
    assert "smote-member" in smote
    assert "v8_track4_undersampling_parallel" in undersampling
    assert "undersampling-model" in undersampling
    assert "ResourceAwareScheduler" in smote
    assert "ResourceAwareScheduler" in undersampling
