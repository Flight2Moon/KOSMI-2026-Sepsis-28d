from __future__ import annotations

import json
import sys
from pathlib import Path

from los_baseline.experiment_scheduler import ResourceAwareScheduler, ScheduledJob
from los_baseline.failure_diagnostics import normalize_existing_result
from los_baseline.resource_manager import ResourceRequest


def _config() -> dict:
    return {
        "hardware": {
            "accelerator": {"backend": "cpu"},
            "cpu": {"threads_per_worker": 1},
            "cuda": {"mode": "disabled", "gpu_ids": [0, 1]},
        },
        "resources": {
            "accelerator": {"backend": "cpu"},
            "scheduler": {
                "enabled": True,
                "max_concurrent_jobs": 2,
                "poll_interval_seconds": 0.05,
            },
            "cpu": {
                "total_thread_limit": 2,
                "reserve_threads": 0,
                "default_threads_per_job": 1,
                "max_threads_per_job": 1,
            },
            "memory": {
                "total_limit_gb": 4,
                "max_fraction": 1.0,
                "reserve_gb": 0,
                "minimum_free_gb_to_launch": 0,
                "breach_grace_seconds": 2,
            },
            "cuda": {
                "gpu_ids": [0, 1],
                "max_jobs_per_gpu": 3,
                "minimum_free_memory_gb": 1,
                "reserve_memory_gb": 1,
                "pytorch_memory_fraction": 0.2,
            },
            "mps": {
                "enabled": False,
                "max_concurrent_jobs": 1,
                "minimum_free_unified_memory_gb": 0,
                "pytorch_memory_fraction": 0.5,
            },
            "job_resources": {},
            "resource_failure_policy": {
                "auto_retry": False,
                "max_retries": 0,
            },
        },
    }


def test_result_v2_separates_direct_failure_and_blocked(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_dir = project_root / "runs" / "test"
    project_root.mkdir(parents=True)
    jobs = [
        ScheduledJob(
            name="root-failure",
            command=(sys.executable, "-c", "raise RuntimeError('intentional root failure')"),
            request=ResourceRequest(cpu_threads=1, ram_gb=0.1),
        ),
        ScheduledJob(
            name="dependent",
            command=(sys.executable, "-c", "print('must not run')"),
            dependencies=("root-failure",),
            request=ResourceRequest(cpu_threads=1, ram_gb=0.1),
        ),
    ]
    result = ResourceAwareScheduler(
        jobs=jobs,
        config=_config(),
        project_root=project_root,
        run_dir=run_dir,
        run_id="test",
        mode="discovery",
    ).run()
    assert result["schema_version"] == "exp123_resource_protocol_result_v2"
    assert result["counts"]["direct_failed"] == 1
    assert result["counts"]["blocked"] == 1
    assert result["direct_failed_jobs"] == ["root-failure"]
    assert result["blocked_jobs"][0]["root_failed_jobs"] == ["root-failure"]
    assert result["root_failures"][0]["failure_category"] == "python_exception"


def test_gpu_reservation_is_considered_before_next_launch(tmp_path: Path) -> None:
    config = _config()
    config["hardware"]["accelerator"]["backend"] = "cuda"
    config["resources"]["accelerator"]["backend"] = "cuda"
    scheduler = ResourceAwareScheduler(
        jobs=[],
        config=config,
        project_root=tmp_path,
        run_dir=tmp_path / "run",
        run_id="gpu-reservation",
    )
    scheduler.gpu_allocations[0] = 2
    scheduler.gpu_reserved_memory_gb[0] = 12.0
    job = ScheduledJob(
        name="new-xgboost",
        command=(sys.executable, "-c", "pass"),
        request=ResourceRequest(
            cpu_threads=1,
            ram_gb=1,
            accelerator="cuda",
            gpu_memory_gb=6,
        ),
    )
    snapshot = {
        "gpu": [
            {
                "index": 0,
                "memory_free_gb": 15.5,
                "memory_total_gb": 15.9,
            },
            {
                "index": 1,
                "memory_free_gb": 15.5,
                "memory_total_gb": 15.9,
            },
        ]
    }
    assert scheduler._choose_gpu(job, snapshot) == 1


def test_old_flat_result_is_normalized(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "old"
    progress = run_dir / "common" / "progress"
    logs = progress / "step_logs"
    logs.mkdir(parents=True)
    (logs / "first.log").write_text(
        "Traceback (most recent call last):\nRuntimeError: failed\n",
        encoding="utf-8",
    )
    result = {
        "run_id": "old",
        "elapsed_seconds": 1.0,
        "status": {"first": "failed", "second": "blocked"},
        "failures": {"first": "exit_code=1", "second": "dependency_failed"},
        "completed": 0,
        "failed": 2,
    }
    manifest = {
        "jobs": [
            {"name": "first", "dependencies": [], "request": {}},
            {"name": "second", "dependencies": ["first"], "request": {}},
        ]
    }
    normalized = normalize_existing_result(
        result=result,
        manifest=manifest,
        run_dir=run_dir,
        mode="discovery",
    )
    assert normalized["counts"]["direct_failed"] == 1
    assert normalized["counts"]["blocked"] == 1
    assert normalized["blocked_jobs"][0]["root_failed_jobs"] == ["first"]


def test_required_resource_profiles_are_packaged() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = [
        "experiment_resource_cpu.json",
        "experiment_cpu_7800x3d_48gb.json",
        "experiment_cuda_max_gpu_throughput_7800x3d_48gb.json",
        "experiment_cuda_max_cpu_throughput_7800x3d_48gb.json",
    ]
    for name in expected:
        path = root / "configs" / name
        assert path.is_file(), name
        json.loads(path.read_text(encoding="utf-8"))


def test_packaged_cuda_profile_keeps_only_xgboost_on_gpu() -> None:
    from scripts.ops import check_exp123_hardware as hardware_check

    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            root
            / "configs"
            / "experiment_cuda_max_gpu_throughput_7800x3d_48gb.json"
        ).read_text(encoding="utf-8")
    )
    jobs = hardware_check._gpu_capable_jobs(config)
    assert len(jobs) == 3
    assert {job["model_name"] for job in jobs} == {"xgboost"}
    packing = hardware_check._find_gpu_packing(
        jobs,
        capacities={0: 15.061, 1: 15.061},
        max_jobs_per_gpu=3,
    )
    assert packing is not None
    assert sum(len(items) for items in packing.values()) == 3
    assert all(len(items) <= 3 for items in packing.values())
    assert sum(int(item["cpu_threads"]) for item in jobs) <= 14
    assert sum(float(item["ram_gb"]) for item in jobs) <= 48


def test_cuda_jobs_have_priority_over_cpu_jobs(tmp_path: Path) -> None:
    config = _config()
    config["hardware"]["accelerator"]["backend"] = "cuda"
    config["resources"]["accelerator"]["backend"] = "cuda"
    jobs = [
        ScheduledJob(
            name="cpu-job",
            command=(sys.executable, "-c", "pass"),
            request=ResourceRequest(cpu_threads=1, ram_gb=1, accelerator="cpu"),
        ),
        ScheduledJob(
            name="gpu-job",
            command=(sys.executable, "-c", "pass"),
            request=ResourceRequest(
                cpu_threads=1,
                ram_gb=1,
                accelerator="cuda",
                gpu_memory_gb=2,
            ),
        ),
    ]
    scheduler = ResourceAwareScheduler(
        jobs=jobs,
        config=config,
        project_root=tmp_path,
        run_dir=tmp_path / "run",
        run_id="priority",
    )
    order = sorted(scheduler.jobs, key=scheduler._job_priority)
    assert order[0] == "gpu-job"
