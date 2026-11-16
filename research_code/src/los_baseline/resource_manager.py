from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import psutil


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gb(value: int | float) -> float:
    return float(value) / (1024.0 ** 3)


@dataclass(frozen=True)
class ResourceRequest:
    cpu_threads: int = 1
    ram_gb: float = 1.0
    accelerator: str = "cpu"
    gpu_memory_gb: float = 0.0
    unified_memory_gb: float = 0.0
    exclusive_gpu: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceSnapshot:
    timestamp_utc: str
    cpu_percent: float
    logical_cpu_count: int
    available_ram_gb: float
    used_ram_gb: float
    total_ram_gb: float
    gpu: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gpu"] = list(self.gpu)
        return payload


def nvidia_snapshot() -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    query = "index,name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw"
    try:
        result = subprocess.run(
            [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except subprocess.TimeoutExpired:
        return []
    if result.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 8:
            continue
        try:
            rows.append(
                {
                    "index": int(values[0]),
                    "name": values[1],
                    "memory_used_gb": float(values[2]) / 1024.0,
                    "memory_free_gb": float(values[3]) / 1024.0,
                    "memory_total_gb": float(values[4]) / 1024.0,
                    "utilization_percent": float(values[5]),
                    "temperature_c": float(values[6]),
                    "power_draw_w": float(values[7].split()[0]),
                }
            )
        except ValueError:
            continue
    return rows


def system_snapshot(*, include_gpu: bool = True) -> ResourceSnapshot:
    memory = psutil.virtual_memory()
    return ResourceSnapshot(
        timestamp_utc=utc_now(),
        cpu_percent=float(psutil.cpu_percent(interval=None)),
        logical_cpu_count=int(psutil.cpu_count(logical=True) or 1),
        available_ram_gb=gb(memory.available),
        used_ram_gb=gb(memory.used),
        total_ram_gb=gb(memory.total),
        gpu=tuple(nvidia_snapshot()) if include_gpu else (),
    )


def resource_config(config: Mapping[str, Any]) -> dict[str, Any]:
    resources = dict(config.get("resources", {}))
    cpu = dict(resources.get("cpu", {}))
    memory = dict(resources.get("memory", {}))
    scheduler = dict(resources.get("scheduler", {}))
    cuda = dict(resources.get("cuda", {}))
    mps = dict(resources.get("mps", {}))
    backend = str(
        resources.get("accelerator", {}).get(
            "backend",
            config.get("hardware", {}).get("accelerator", {}).get("backend", "auto"),
        )
    ).lower()
    env_backend = os.environ.get("EXP123_ACCELERATOR_BACKEND", "").strip().lower()
    if env_backend in {"cpu", "cuda", "mps"}:
        backend = env_backend
    if backend == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                backend = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                backend = "mps"
            else:
                backend = "cpu"
        except Exception:
            backend = "cpu"
    logical = int(psutil.cpu_count(logical=True) or 1)
    total_ram = gb(psutil.virtual_memory().total)
    thread_limit = int(cpu.get("total_thread_limit", max(1, logical - int(cpu.get("reserve_threads", 1)))))
    thread_limit = max(1, min(thread_limit, logical))
    configured_ram = float(memory.get("total_limit_gb", total_ram * float(memory.get("max_fraction", 0.8))))
    ram_limit = min(configured_ram, total_ram * float(memory.get("max_fraction", 1.0)))
    return {
        "backend": backend,
        "scheduler": {
            "enabled": bool(scheduler.get("enabled", True)),
            "max_concurrent_jobs": max(1, int(scheduler.get("max_concurrent_jobs", 4))),
            "poll_interval_seconds": max(0.25, float(scheduler.get("poll_interval_seconds", 2.0))),
            "resource_refresh_seconds": max(0.25, float(scheduler.get("resource_refresh_seconds", 2.0))),
            "max_resource_wait_seconds": max(0.0, float(scheduler.get("max_resource_wait_seconds", 0.0))),
            "failure_policy": str(scheduler.get("failure_policy", "stop_dependents_continue_independent")),
            "max_concurrent_by_model": {
                str(name): max(1, int(limit))
                for name, limit in dict(scheduler.get("max_concurrent_by_model", {})).items()
            },
        },
        "cpu": {
            "total_thread_limit": thread_limit,
            "reserve_threads": int(cpu.get("reserve_threads", 1)),
            "default_threads_per_job": max(1, int(cpu.get("default_threads_per_job", 2))),
            "max_threads_per_job": max(1, int(cpu.get("max_threads_per_job", thread_limit))),
        },
        "memory": {
            "total_limit_gb": max(1.0, ram_limit),
            "reserve_gb": max(0.0, float(memory.get("reserve_gb", 2.0))),
            "minimum_free_gb_to_launch": max(0.0, float(memory.get("minimum_free_gb_to_launch", 2.0))),
            "breach_grace_seconds": max(1.0, float(memory.get("breach_grace_seconds", 30.0))),
            "enforce_job_ram_limit": bool(memory.get("enforce_job_ram_limit", True)),
        },
        "cuda": {
            "gpu_ids": [int(value) for value in cuda.get("gpu_ids", config.get("hardware", {}).get("cuda", {}).get("gpu_ids", [0]))],
            "max_jobs_per_gpu": max(1, int(cuda.get("max_jobs_per_gpu", 1))),
            "minimum_free_memory_gb": max(0.0, float(cuda.get("minimum_free_memory_gb", 2.0))),
            "reserve_memory_gb": max(0.0, float(cuda.get("reserve_memory_gb", 1.0))),
            "pytorch_memory_fraction": float(cuda.get("pytorch_memory_fraction", 0.75)),
            "assignment_strategy": str(cuda.get("assignment_strategy", "most_free_memory")),
            "allow_cpu_fallback": bool(cuda.get("allow_cpu_fallback", False)),
            "enforce_process_memory_limit": bool(cuda.get("enforce_process_memory_limit", True)),
            "gpu_memory_breach_grace_seconds": max(1.0, float(cuda.get("gpu_memory_breach_grace_seconds", memory.get("breach_grace_seconds", 30.0)))),
            "enforce_exclusive_gpu": bool(cuda.get("enforce_exclusive_gpu", True)),
        },
        "mps": {
            "enabled": bool(mps.get("enabled", backend in {"mps", "auto"})),
            "max_concurrent_jobs": max(1, int(mps.get("max_concurrent_jobs", 1))),
            "pytorch_memory_fraction": float(mps.get("pytorch_memory_fraction", 0.55)),
            "minimum_free_unified_memory_gb": max(0.0, float(mps.get("minimum_free_unified_memory_gb", 4.0))),
            "allow_cpu_fallback": bool(mps.get("allow_cpu_fallback", True)),
            "enforce_unified_memory_limit": bool(mps.get("enforce_unified_memory_limit", True)),
        },
        "job_resources": dict(resources.get("job_resources", {})),
        "resource_failure_policy": dict(resources.get("resource_failure_policy", {})),
    }


def request_for_job(
    config: Mapping[str, Any],
    *,
    model_name: str | None,
    job_kind: str,
) -> ResourceRequest:
    resolved = resource_config(config)
    defaults = {
        "cpu_threads": resolved["cpu"]["default_threads_per_job"],
        "ram_gb": 4.0,
        "accelerator": "cpu",
        "gpu_memory_gb": 0.0,
        "unified_memory_gb": 0.0,
        "exclusive_gpu": False,
    }
    job_resources = resolved["job_resources"]
    if job_kind in job_resources:
        defaults.update(dict(job_resources[job_kind]))
    if model_name and model_name in job_resources:
        defaults.update(dict(job_resources[model_name]))
    backend = resolved["backend"]
    explicit_accelerator = False
    if job_kind in job_resources and "accelerator" in job_resources[job_kind]:
        explicit_accelerator = True
    if model_name and model_name in job_resources and "accelerator" in job_resources[model_name]:
        explicit_accelerator = True
    if model_name == "xgboost" and backend == "cuda" and not explicit_accelerator:
        defaults["accelerator"] = "cuda"
    elif model_name == "logistic_regression" and backend == "cuda":
        # A single linear layer is faster and substantially more memory-stable on CPU.
        defaults["accelerator"] = "cpu"
        defaults["gpu_memory_gb"] = 0.0
        defaults["exclusive_gpu"] = False
    elif model_name == "logistic_regression" and backend == "mps" and not explicit_accelerator:
        defaults["accelerator"] = "mps"
    if backend == "mps" and defaults.get("accelerator") == "cuda":
        defaults["accelerator"] = "cpu"
    return ResourceRequest(
        cpu_threads=max(1, min(int(defaults["cpu_threads"]), resolved["cpu"]["max_threads_per_job"])),
        ram_gb=max(0.1, float(defaults["ram_gb"])),
        accelerator=str(defaults["accelerator"]),
        gpu_memory_gb=max(0.0, float(defaults.get("gpu_memory_gb", 0.0))),
        unified_memory_gb=max(0.0, float(defaults.get("unified_memory_gb", defaults.get("ram_gb", 0.0)))),
        exclusive_gpu=bool(defaults.get("exclusive_gpu", False)),
    )


def process_tree_rss_gb(pid: int) -> float:
    try:
        process = psutil.Process(pid)
        processes = [process, *process.children(recursive=True)]
        return gb(sum(item.memory_info().rss for item in processes if item.is_running()))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def terminate_process_group(process: subprocess.Popen[Any], timeout: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.25)
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(payload), ensure_ascii=False, default=str) + "\n")


def process_tree_pids(pid: int) -> set[int]:
    """Return the live PID set for a worker and all of its descendants."""
    try:
        process = psutil.Process(pid)
        return {process.pid, *(child.pid for child in process.children(recursive=True))}
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return set()


def nvidia_process_tree_memory_gb(pid: int, gpu_id: int | None = None) -> float:
    """Return GPU memory attributed to a process tree using nvidia-smi.

    A zero value means either no CUDA allocation is currently visible or the
    platform does not expose per-process accounting. The scheduler records this
    as telemetry and only enforces a configured limit when accounting succeeds.
    """
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return 0.0
    pid_set = process_tree_pids(pid)
    if not pid_set:
        return 0.0
    try:
        gpu_map_result = subprocess.run(
            [executable, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=5.0,
        )
    except subprocess.TimeoutExpired:
        return 0.0
    if gpu_map_result.returncode != 0:
        return 0.0
    uuid_to_index: dict[str, int] = {}
    for line in gpu_map_result.stdout.splitlines():
        values = [value.strip() for value in line.split(",", 1)]
        if len(values) != 2:
            continue
        try:
            uuid_to_index[values[1]] = int(values[0])
        except ValueError:
            continue
    try:
        process_result = subprocess.run(
            [executable, "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=5.0,
        )
    except subprocess.TimeoutExpired:
        return 0.0
    if process_result.returncode != 0:
        return 0.0
    total_mb = 0.0
    for line in process_result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 3:
            continue
        try:
            process_pid = int(values[0])
            process_gpu = uuid_to_index.get(values[1])
            used_mb = float(values[2].split()[0])
        except (ValueError, IndexError):
            continue
        if process_pid not in pid_set:
            continue
        if gpu_id is not None and process_gpu != gpu_id:
            continue
        total_mb += used_mb
    return total_mb / 1024.0


def verify_completion_artifact(path: Path) -> tuple[bool, str]:
    """Validate a completion marker and its optional SHA-256 companion."""
    if not path.is_file():
        return False, "missing_completion_artifact"
    if path.suffix.lower() == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exception:
            return False, f"invalid_json:{type(exception).__name__}"
    checksum_path = path.with_name(path.name + ".sha256")
    if checksum_path.is_file():
        expected = checksum_path.read_text(encoding="utf-8", errors="replace").strip().split()[0]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected != actual:
            return False, "sha256_mismatch"
    return True, "valid"
