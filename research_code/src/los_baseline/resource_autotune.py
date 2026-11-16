from __future__ import annotations

import copy
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import psutil

from .resource_manager import nvidia_snapshot


@dataclass(frozen=True)
class HardwareInventory:
    logical_cpus: int
    physical_cpus: int
    total_ram_gb: float
    available_ram_gb: float
    gpus: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gpus"] = list(self.gpus)
        return payload


def detect_hardware() -> HardwareInventory:
    memory = psutil.virtual_memory()
    logical = int(psutil.cpu_count(logical=True) or 1)
    physical = int(psutil.cpu_count(logical=False) or logical)
    return HardwareInventory(
        logical_cpus=logical,
        physical_cpus=max(1, physical),
        total_ram_gb=float(memory.total) / (1024.0 ** 3),
        available_ram_gb=float(memory.available) / (1024.0 ** 3),
        gpus=tuple(nvidia_snapshot()),
    )


_PROFILE = {
    "maximum_throughput": {
        "cpu_fraction": 0.96,
        "ram_fraction": 0.94,
        "ram_reserve_fraction": 0.04,
        "gpu_reserve_gb": 1.5,
        "gpu_min_free_gb": 1.0,
        "max_jobs_cap": 32,
        "gpu_jobs_cap": 8,
    },
    "balanced": {
        "cpu_fraction": 0.85,
        "ram_fraction": 0.85,
        "ram_reserve_fraction": 0.08,
        "gpu_reserve_gb": 2.5,
        "gpu_min_free_gb": 2.0,
        "max_jobs_cap": 20,
        "gpu_jobs_cap": 5,
    },
    "conservative": {
        "cpu_fraction": 0.70,
        "ram_fraction": 0.75,
        "ram_reserve_fraction": 0.12,
        "gpu_reserve_gb": 4.0,
        "gpu_min_free_gb": 3.0,
        "max_jobs_cap": 12,
        "gpu_jobs_cap": 3,
    },
}


def _clamp_int(value: float | int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _job_ram(total_ram_gb: float, fraction: float, *, low: float, high: float) -> float:
    return round(max(low, min(high, total_ram_gb * fraction)), 2)


def autotune_config(
    config: Mapping[str, Any],
    *,
    profile: str = "maximum_throughput",
    hardware: HardwareInventory | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a frozen high-throughput resource configuration and audit manifest.

    The tuner maximizes *aggregate* throughput, not threads per individual fit.
    CPU-thread reservations and GPU-VRAM reservations are chosen together so the
    ResourceAwareScheduler can run multiple independent model jobs without
    uncontrolled BLAS/OpenMP oversubscription.
    """
    if profile not in _PROFILE:
        raise ValueError(f"Unknown resource profile: {profile}")
    policy = _PROFILE[profile]
    hw = hardware or detect_hardware()
    tuned = copy.deepcopy(dict(config))
    resources = tuned.setdefault("resources", {})
    resources.setdefault("accelerator", {})
    scheduler = resources.setdefault("scheduler", {})
    cpu = resources.setdefault("cpu", {})
    memory = resources.setdefault("memory", {})
    cuda = resources.setdefault("cuda", {})
    job_resources = resources.setdefault("job_resources", {})
    failure = resources.setdefault("resource_failure_policy", {})

    logical = max(1, hw.logical_cpus)
    physical = max(1, hw.physical_cpus)
    thread_budget = max(1, int(math.floor(logical * float(policy["cpu_fraction"]))))
    reserve_threads = max(0, logical - thread_budget)
    ram_limit = max(1.0, hw.total_ram_gb * float(policy["ram_fraction"]))
    ram_reserve = max(1.0, hw.total_ram_gb * float(policy["ram_reserve_fraction"]))

    gpu_rows = list(hw.gpus)
    gpu_ids = [int(row["index"]) for row in gpu_rows]
    has_cuda = bool(gpu_ids)
    requested_backend = str(resources.get("accelerator", {}).get("backend", "auto")).lower()
    if requested_backend == "cuda" and not has_cuda:
        # Development machines may not have CUDA.  Persist the fallback in the
        # generated config so the actual run remains reproducible.
        resources["accelerator"]["backend"] = "cpu"
        resources["accelerator"]["required"] = False
    elif has_cuda:
        resources["accelerator"]["backend"] = "cuda"
        resources["accelerator"]["required"] = True

    cpu["total_thread_limit"] = thread_budget
    cpu["reserve_threads"] = reserve_threads
    cpu["default_threads_per_job"] = max(1, min(4, physical // 4 or 1))
    cpu["max_threads_per_job"] = max(1, min(thread_budget, max(4, physical // 2)))

    memory["total_limit_gb"] = round(ram_limit, 2)
    memory["max_fraction"] = float(policy["ram_fraction"])
    memory["reserve_gb"] = round(ram_reserve, 2)
    memory["minimum_free_gb_to_launch"] = round(max(1.0, ram_reserve * 0.5), 2)
    memory["enforce_job_ram_limit"] = True

    # Estimate XGBoost VRAM reservation from the smallest GPU so the same
    # configuration is valid on every visible device.
    if has_cuda:
        min_total_vram = min(float(row.get("memory_total_gb", 0.0)) for row in gpu_rows)
        gpu_reserve = float(policy["gpu_reserve_gb"])
        usable_vram = max(1.0, min_total_vram - gpu_reserve)
        xgb_vram = max(2.0, min(10.0, usable_vram / 4.0))
        per_gpu_jobs = max(1, min(int(policy["gpu_jobs_cap"]), int(usable_vram // xgb_vram)))
        cuda["gpu_ids"] = gpu_ids
        cuda["max_jobs_per_gpu"] = per_gpu_jobs
        cuda["minimum_free_memory_gb"] = float(policy["gpu_min_free_gb"])
        cuda["reserve_memory_gb"] = gpu_reserve
        cuda["pytorch_memory_fraction"] = round(min(0.85, max(0.10, 0.92 / per_gpu_jobs)), 4)
        cuda["assignment_strategy"] = "most_free_memory"
        cuda["allow_cpu_fallback"] = False
        cuda["enforce_process_memory_limit"] = True
        # Sharing is deliberate for throughput; the scheduler still honors
        # per-job exclusive requests on retries.
        cuda["enforce_exclusive_gpu"] = False
    else:
        xgb_vram = 0.0
        per_gpu_jobs = 0
        cuda["gpu_ids"] = []
        cuda["max_jobs_per_gpu"] = 1
        cuda["allow_cpu_fallback"] = True

    gpu_job_slots = len(gpu_ids) * per_gpu_jobs
    # CPU jobs such as LR/RF/SVM run beside GPU-bound XGBoost work.  The CPU
    # budget remains the hard limiter inside ResourceAwareScheduler.
    cpu_parallel_hint = max(1, thread_budget // max(1, int(cpu["default_threads_per_job"])))
    max_jobs = min(int(policy["max_jobs_cap"]), max(2, cpu_parallel_hint + gpu_job_slots))
    scheduler["enabled"] = True
    scheduler["mode"] = "resource_aware_autotuned"
    scheduler["max_concurrent_jobs"] = max_jobs
    scheduler["poll_interval_seconds"] = min(1.0, float(scheduler.get("poll_interval_seconds", 1.0)))
    scheduler["resource_refresh_seconds"] = min(1.0, float(scheduler.get("resource_refresh_seconds", 1.0)))
    scheduler["max_resource_wait_seconds"] = 0
    scheduler["max_concurrent_by_model"] = {
        "svm": max(1, min(4, thread_budget)),
        "random_forest": max(1, min(4, max_jobs)),
        "xgboost": max(1, gpu_job_slots) if has_cuda else max(1, min(4, max_jobs)),
    }

    # Per-fit thread counts are deliberately below the machine-wide count.
    rf_threads = max(1, min(int(cpu["max_threads_per_job"]), max(2, physical // 4)))
    xgb_threads = max(1, min(int(cpu["max_threads_per_job"]), max(2, physical // max(2, gpu_job_slots or 2))))
    lr_threads = max(1, min(int(cpu["max_threads_per_job"]), max(2, physical // 4)))
    job_resources.setdefault("decision_tree", {})
    job_resources["decision_tree"].update({"cpu_threads": 1, "ram_gb": _job_ram(hw.total_ram_gb, 0.04, low=2.0, high=8.0), "accelerator": "cpu"})
    job_resources.setdefault("logistic_regression", {})
    job_resources["logistic_regression"].update({"cpu_threads": lr_threads, "ram_gb": _job_ram(hw.total_ram_gb, 0.08, low=4.0, high=16.0), "gpu_memory_gb": 0.0, "exclusive_gpu": False, "accelerator": "cpu"})
    job_resources.setdefault("random_forest", {})
    job_resources["random_forest"].update({"cpu_threads": rf_threads, "ram_gb": _job_ram(hw.total_ram_gb, 0.12, low=6.0, high=24.0), "accelerator": "cpu"})
    job_resources.setdefault("svm", {})
    job_resources["svm"].update({"cpu_threads": 1, "ram_gb": _job_ram(hw.total_ram_gb, 0.14, low=8.0, high=24.0), "accelerator": "cpu"})
    job_resources.setdefault("xgboost", {})
    job_resources["xgboost"].update({
        "cpu_threads": xgb_threads,
        "ram_gb": _job_ram(hw.total_ram_gb, 0.12, low=6.0, high=24.0),
        "gpu_memory_gb": round(xgb_vram, 2),
        "exclusive_gpu": False,
        "accelerator": "cuda" if has_cuda else "cpu",
    })

    # Ensure every declared job can fit inside the machine-wide RAM budget.
    per_job_ram_cap = max(0.5, ram_limit - ram_reserve)
    for name, spec in list(job_resources.items()):
        if not isinstance(spec, dict):
            continue
        if "ram_gb" in spec:
            spec["ram_gb"] = round(min(float(spec["ram_gb"]), per_job_ram_cap), 2)
        if not has_cuda and str(spec.get("accelerator", "cpu")).lower() == "cuda":
            spec["accelerator"] = "cpu"
            spec["gpu_memory_gb"] = 0.0
            spec["exclusive_gpu"] = False

    # Resource retries stay bounded by the detected machine rather than stale
    # hard-coded values from a different server.
    failure["max_ram_gb_per_job"] = round(max(4.0, ram_limit - ram_reserve), 2)
    if has_cuda:
        failure["max_gpu_memory_gb_per_job"] = round(max(2.0, min(float(row["memory_total_gb"]) for row in gpu_rows) - float(policy["gpu_reserve_gb"])), 2)
    failure["auto_retry"] = True

    manifest = {
        "schema_version": "exp123_v8_phase6_resource_autotune_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "hardware": hw.to_dict(),
        "resolved": {
            "backend": resources["accelerator"]["backend"],
            "cpu_thread_budget": thread_budget,
            "ram_limit_gb": round(ram_limit, 2),
            "scheduler_max_concurrent_jobs": max_jobs,
            "gpu_ids": gpu_ids,
            "max_jobs_per_gpu": per_gpu_jobs,
            "xgboost_gpu_memory_reservation_gb": round(xgb_vram, 2),
            "job_resources": copy.deepcopy(job_resources),
        },
        "environment_overrides": {
            "EXP123_ASSIGNED_CPU_THREADS": "set per scheduled worker",
            "OMP_NUM_THREADS": "set per scheduled worker",
            "MKL_NUM_THREADS": "set per scheduled worker",
            "OPENBLAS_NUM_THREADS": "set per scheduled worker",
        },
        "note": "Generated values are persisted before execution; the run does not retune resources mid-study.",
    }
    return tuned, manifest
