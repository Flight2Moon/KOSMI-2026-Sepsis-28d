from __future__ import annotations

import json
import sys
from pathlib import Path

from los_baseline.experiment_scheduler import ResourceAwareScheduler, ScheduledJob
from los_baseline.resource_manager import ResourceRequest, request_for_job, resource_config


def _config(backend: str = "cpu") -> dict:
    return {
        "hardware": {
            "accelerator": {"backend": backend},
            "cpu": {"threads_per_worker": 2},
            "cuda": {"mode": "disabled", "gpu_ids": [0, 1]},
        },
        "resources": {
            "accelerator": {"backend": backend},
            "scheduler": {"enabled": True, "max_concurrent_jobs": 2, "poll_interval_seconds": 0.05},
            "cpu": {"total_thread_limit": 4, "reserve_threads": 0, "default_threads_per_job": 1, "max_threads_per_job": 2},
            "memory": {"total_limit_gb": 8, "max_fraction": 1.0, "reserve_gb": 0, "minimum_free_gb_to_launch": 0, "breach_grace_seconds": 2},
            "cuda": {"gpu_ids": [0, 1], "max_jobs_per_gpu": 1, "minimum_free_memory_gb": 1, "reserve_memory_gb": 1, "pytorch_memory_fraction": 0.75},
            "mps": {"enabled": backend == "mps", "max_concurrent_jobs": 1, "minimum_free_unified_memory_gb": 0, "pytorch_memory_fraction": 0.5},
            "job_resources": {
                "xgboost": {"cpu_threads": 2, "ram_gb": 2, "gpu_memory_gb": 3},
                "logistic_regression": {"cpu_threads": 1, "ram_gb": 1, "unified_memory_gb": 2},
            },
            "resource_failure_policy": {"auto_retry": False, "max_retries": 0},
        },
    }


def test_resource_profiles_select_model_backends() -> None:
    cuda = request_for_job(_config("cuda"), model_name="xgboost", job_kind="optimization")
    mps = request_for_job(_config("mps"), model_name="logistic_regression", job_kind="training")
    mps_xgb = request_for_job(_config("mps"), model_name="xgboost", job_kind="training")
    assert cuda.accelerator == "cuda"
    assert cuda.gpu_memory_gb == 3
    assert mps.accelerator == "mps"
    assert mps_xgb.accelerator == "cpu"


def test_resource_config_respects_cpu_limit() -> None:
    resolved = resource_config(_config())
    assert resolved["cpu"]["total_thread_limit"] <= 4
    assert resolved["scheduler"]["max_concurrent_jobs"] == 2


def test_scheduler_runs_dependencies_and_writes_state(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_dir = project_root / "runs" / "test"
    project_root.mkdir(parents=True)
    first = run_dir / "first.lock"
    second = run_dir / "second.lock"
    jobs = [
        ScheduledJob(
            name="first",
            command=(sys.executable, "-c", f"from pathlib import Path; p=Path({str(first)!r}); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('ok')"),
            request=ResourceRequest(cpu_threads=1, ram_gb=0.2),
            completion_path="first.lock",
        ),
        ScheduledJob(
            name="second",
            command=(sys.executable, "-c", f"from pathlib import Path; p=Path({str(second)!r}); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('ok')"),
            dependencies=("first",),
            request=ResourceRequest(cpu_threads=1, ram_gb=0.2),
            completion_path="second.lock",
        ),
    ]
    result = ResourceAwareScheduler(
        jobs=jobs,
        config=_config(),
        project_root=project_root,
        run_dir=run_dir,
        run_id="test",
    ).run()
    assert result["failed"] == 0
    assert result["completed"] == 2
    assert first.is_file() and second.is_file()
    state = json.loads((run_dir / "common" / "progress" / "scheduler_state.json").read_text())
    assert state["status"]["first"] == "completed"
    assert state["status"]["second"] == "completed"


def test_completion_artifact_sha256_is_verified(tmp_path: Path) -> None:
    from los_baseline.resource_manager import verify_completion_artifact
    import hashlib

    marker = tmp_path / "completed.lock.json"
    marker.write_text('{"status":"completed"}', encoding="utf-8")
    checksum = marker.with_name(marker.name + ".sha256")
    checksum.write_text(hashlib.sha256(marker.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    assert verify_completion_artifact(marker) == (True, "valid")

    marker.write_text('{"status":"changed"}', encoding="utf-8")
    valid, reason = verify_completion_artifact(marker)
    assert valid is False
    assert reason == "sha256_mismatch"


def test_exclusive_gpu_request_does_not_share_gpu(tmp_path: Path) -> None:
    config = _config("cuda")
    config["resources"]["cuda"]["max_jobs_per_gpu"] = 2
    scheduler = ResourceAwareScheduler(
        jobs=[],
        config=config,
        project_root=tmp_path,
        run_dir=tmp_path / "run",
        run_id="exclusive",
    )
    scheduler.gpu_allocations[0] = 1
    job = ScheduledJob(
        name="exclusive-job",
        command=(sys.executable, "-c", "pass"),
        request=ResourceRequest(
            cpu_threads=1,
            ram_gb=1,
            accelerator="cuda",
            gpu_memory_gb=1,
            exclusive_gpu=True,
        ),
    )
    snapshot = {
        "gpu": [
            {"index": 0, "memory_free_gb": 15.0},
            {"index": 1, "memory_free_gb": 15.0},
        ]
    }
    assert scheduler._choose_gpu(job, snapshot) == 1
