from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from ..resource_manager import nvidia_snapshot
from ..utils import atomic_write_json


@dataclass(frozen=True)
class StagePlan:
    workers: int
    threads_per_worker: int
    estimated_ram_gb_per_worker: float
    cpu_threads_reserved: int
    ram_gb_reserved: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Study28ResourcePlan:
    schema_version: str
    created_at_utc: str
    profile: str
    hardware: dict[str, Any]
    policy: dict[str, Any]
    cpu_thread_budget: int
    ram_budget_gb: float
    primary_lightgbm_backend: str
    primary_lightgbm_backend_reason: str
    serial_model_threads: int
    nested_cv: StagePlan
    external_validation: StagePlan
    reverse_validation: StagePlan
    alpha5_workers: StagePlan

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


PROFILES: dict[str, dict[str, float]] = {
    # Intended for dedicated research workstations/servers.  The scheduler aims
    # to occupy almost all compute while leaving a very small OS/runtime margin.
    "maximum_throughput": {
        "cpu_fraction": 0.96,
        "ram_fraction_total": 0.94,
        "ram_fraction_available": 0.97,
        "minimum_ram_reserve_gb": 1.5,
    },
    "balanced": {
        "cpu_fraction": 0.88,
        "ram_fraction_total": 0.88,
        "ram_fraction_available": 0.92,
        "minimum_ram_reserve_gb": 3.0,
    },
}


def _cpu_counts() -> tuple[int, int]:
    logical = int(psutil.cpu_count(logical=True) or os.cpu_count() or 1)
    physical = int(psutil.cpu_count(logical=False) or logical)
    return max(1, logical), max(1, physical)


def _hardware_snapshot() -> dict[str, Any]:
    logical, physical = _cpu_counts()
    memory = psutil.virtual_memory()
    gpus = list(nvidia_snapshot())
    return {
        "logical_cpu_threads": logical,
        "physical_cpu_cores": physical,
        "ram_total_gb": round(float(memory.total) / (1024.0 ** 3), 3),
        "ram_available_gb_at_planning": round(float(memory.available) / (1024.0 ** 3), 3),
        "gpu_count": len(gpus),
        "gpus": gpus,
    }


def _thread_budget(logical: int, fraction: float) -> int:
    # Small machines lose too much throughput if an entire core is reserved.
    # On larger hosts retain ~4% for the OS/runtime while still targeting >95%.
    if logical <= 8:
        return logical
    return max(1, min(logical, int(math.floor(logical * fraction))))


def _stage_plan(
    *,
    task_count: int,
    cpu_budget: int,
    ram_budget_gb: float,
    ram_per_worker_gb: float,
    minimum_threads_per_worker: int = 1,
    maximum_workers: int | None = None,
) -> StagePlan:
    task_count = max(1, int(task_count))
    max_workers = task_count if maximum_workers is None else min(task_count, max(1, int(maximum_workers)))
    ram_slots = max(1, int(math.floor(ram_budget_gb / max(0.5, ram_per_worker_gb))))
    cpu_slots = max(1, int(cpu_budget // max(1, minimum_threads_per_worker)))
    workers = max(1, min(max_workers, ram_slots, cpu_slots))
    threads = max(1, cpu_budget // workers)
    # Use the whole frozen budget where possible.  The remaining 0..workers-1
    # threads are the unavoidable integer division slack, not a safety reserve.
    return StagePlan(
        workers=workers,
        threads_per_worker=threads,
        estimated_ram_gb_per_worker=float(ram_per_worker_gb),
        cpu_threads_reserved=max(0, cpu_budget - workers * threads),
        ram_gb_reserved=max(0.0, ram_budget_gb - workers * ram_per_worker_gb),
    )


def build_study28_resource_plan(profile: str = "maximum_throughput") -> Study28ResourcePlan:
    if profile not in PROFILES:
        raise ValueError(f"Unsupported Study28 resource profile: {profile}")
    policy = dict(PROFILES[profile])
    hw = _hardware_snapshot()
    logical = int(hw["logical_cpu_threads"])
    cpu_budget = _thread_budget(logical, float(policy["cpu_fraction"]))

    total_ram = float(hw["ram_total_gb"])
    available_ram = float(hw["ram_available_gb_at_planning"])
    reserve_floor = float(policy["minimum_ram_reserve_gb"])
    ram_budget = min(
        total_ram * float(policy["ram_fraction_total"]),
        available_ram * float(policy["ram_fraction_available"]),
        max(1.0, total_ram - reserve_floor),
    )
    ram_budget = max(1.0, ram_budget)

    # Measured on the real professor-provided MIMIC table, an isolated nested-CV
    # fold peaked at ~0.30 GB RSS.  We reserve >2x that observed peak for model
    # workers and 0.50 GB for prediction/bootstrap-only external workers.  These
    # are execution reservations, not scientific/model assumptions.
    nested = _stage_plan(
        task_count=5,
        cpu_budget=cpu_budget,
        ram_budget_gb=ram_budget,
        ram_per_worker_gb=0.75,
        minimum_threads_per_worker=2,
    )
    external = _stage_plan(
        task_count=4,
        cpu_budget=cpu_budget,
        ram_budget_gb=ram_budget,
        ram_per_worker_gb=0.50,
        minimum_threads_per_worker=1,
    )
    if logical >= 16:
        reverse = _stage_plan(
            task_count=2, cpu_budget=cpu_budget, ram_budget_gb=ram_budget,
            ram_per_worker_gb=0.75, minimum_threads_per_worker=2,
        )
    else:
        reverse = StagePlan(
            workers=1, threads_per_worker=cpu_budget,
            estimated_ram_gb_per_worker=0.75, cpu_threads_reserved=0,
            ram_gb_reserved=max(0.0, ram_budget - 0.75),
        )
    if logical >= 12:
        alpha5 = _stage_plan(
            task_count=5, cpu_budget=cpu_budget, ram_budget_gb=ram_budget,
            ram_per_worker_gb=0.75, minimum_threads_per_worker=2,
        )
    else:
        alpha5 = StagePlan(
            workers=1, threads_per_worker=cpu_budget,
            estimated_ram_gb_per_worker=0.75, cpu_threads_reserved=0,
            ram_gb_reserved=max(0.0, ram_budget - 0.75),
        )

    gpu_count = int(hw["gpu_count"])
    gpu_reason = (
        "GPU(s) detected but confirmatory Study28 LightGBM remains on the CPU backend. "
        "The professor-result reproduction was established with CPU LightGBM; changing the "
        "histogram backend is treated as a scientific implementation change rather than a "
        "resource-only optimization. GPUs are recorded for provenance and may be used only "
        "by explicitly separated exploratory analyses."
        if gpu_count
        else "No NVIDIA GPU detected. Confirmatory Study28 LightGBM uses the CPU backend."
    )

    return Study28ResourcePlan(
        schema_version="study28_resource_plan_v1",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        profile=profile,
        hardware=hw,
        policy=policy,
        cpu_thread_budget=cpu_budget,
        ram_budget_gb=round(ram_budget, 3),
        primary_lightgbm_backend="cpu",
        primary_lightgbm_backend_reason=gpu_reason,
        serial_model_threads=cpu_budget,
        nested_cv=nested,
        external_validation=external,
        reverse_validation=reverse,
        alpha5_workers=alpha5,
    )


def resource_environment(plan: Study28ResourcePlan) -> dict[str, str]:
    return {
        "STUDY28_RESOURCE_PROFILE": plan.profile,
        "STUDY28_CPU_THREAD_BUDGET": str(plan.cpu_thread_budget),
        "STUDY28_RAM_BUDGET_GB": str(plan.ram_budget_gb),
        "STUDY28_SERIAL_MODEL_THREADS": str(plan.serial_model_threads),
        "STUDY28_NESTED_CV_WORKERS": str(plan.nested_cv.workers),
        "STUDY28_NESTED_CV_THREADS": str(plan.nested_cv.threads_per_worker),
        "STUDY28_EXTERNAL_SITE_WORKERS": str(plan.external_validation.workers),
        "STUDY28_EXTERNAL_SITE_THREADS": str(plan.external_validation.threads_per_worker),
        "STUDY28_REVERSE_WORKERS": str(plan.reverse_validation.workers),
        "STUDY28_REVERSE_THREADS": str(plan.reverse_validation.threads_per_worker),
        "STUDY28_ALPHA5_WORKERS": str(plan.alpha5_workers.workers),
        "STUDY28_ALPHA5_THREADS": str(plan.alpha5_workers.threads_per_worker),
        # Parent-process fits (temporal/full-MIMIC freeze) use the whole budget.
        "EXP123_ASSIGNED_CPU_THREADS": str(plan.serial_model_threads),
        "OMP_NUM_THREADS": str(plan.serial_model_threads),
        "MKL_NUM_THREADS": str(plan.serial_model_threads),
        "OPENBLAS_NUM_THREADS": str(plan.serial_model_threads),
        "NUMEXPR_NUM_THREADS": str(plan.serial_model_threads),
        "VECLIB_MAXIMUM_THREADS": str(plan.serial_model_threads),
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
    }


def write_resource_plan(
    *,
    json_path: Path,
    env_path: Path,
    profile: str = "maximum_throughput",
) -> Study28ResourcePlan:
    plan = build_study28_resource_plan(profile)
    json_path = Path(json_path).expanduser().resolve()
    env_path = Path(env_path).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(json_path, plan.to_dict())
    env = resource_environment(plan)
    env["STUDY28_RESOURCE_PLAN_PATH"] = str(json_path)
    lines = ["# Auto-generated; frozen for this Study28 run."]
    for key in sorted(env):
        value = str(env[key]).replace("'", "'\\''")
        lines.append(f"export {key}='{value}'")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan


def read_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return max(1, int(raw))
    except ValueError:
        return int(default)


def child_thread_environment(base: dict[str, str], threads: int) -> dict[str, str]:
    env = dict(base)
    value = str(max(1, int(threads)))
    for key in (
        "EXP123_ASSIGNED_CPU_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = value
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    return env
