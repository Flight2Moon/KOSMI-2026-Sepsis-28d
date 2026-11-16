from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .failure_diagnostics import build_protocol_result
from .progress import emit_progress
from .resource_manager import (
    ResourceRequest,
    append_jsonl,
    nvidia_process_tree_memory_gb,
    process_tree_rss_gb,
    resource_config,
    system_snapshot,
    terminate_process_group,
    utc_now,
    verify_completion_artifact,
)


@dataclass(frozen=True)
class ScheduledJob:
    name: str
    command: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    request: ResourceRequest = field(default_factory=ResourceRequest)
    completion_path: str | None = None
    model_name: str | None = None
    task_key: str | None = None
    kind: str = "generic"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["request"] = self.request.to_dict()
        return payload


@dataclass
class RunningJob:
    job: ScheduledJob
    process: subprocess.Popen[str]
    log_file: Any
    log_path: Path
    started_monotonic: float
    started_at_utc: str
    gpu_id: int | None
    attempt: int
    effective_exclusive_gpu: bool = False
    breach_started: float | None = None
    gpu_breach_started: float | None = None
    peak_rss_gb: float = 0.0
    peak_gpu_memory_gb: float = 0.0


def validate_job_graph(jobs: Sequence[ScheduledJob]) -> None:
    names = [job.name for job in jobs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate scheduler job names: {duplicates}")

    known = set(names)
    missing = {
        job.name: sorted(set(job.dependencies) - known)
        for job in jobs
        if set(job.dependencies) - known
    }
    if missing:
        raise ValueError(f"Scheduler jobs reference missing dependencies: {missing}")

    graph = {job.name: tuple(job.dependencies) for job in jobs}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str, path: tuple[str, ...]) -> None:
        if name in visited:
            return
        if name in visiting:
            cycle = " -> ".join((*path, name))
            raise ValueError(f"Scheduler dependency cycle detected: {cycle}")
        visiting.add(name)
        for dependency in graph[name]:
            visit(dependency, (*path, name))
        visiting.remove(name)
        visited.add(name)

    for name in names:
        visit(name, ())


class ResourceAwareScheduler:
    def __init__(
        self,
        *,
        jobs: Sequence[ScheduledJob],
        config: Mapping[str, Any],
        project_root: Path,
        run_dir: Path,
        run_id: str,
        rerun_completed: bool = False,
        mode: str = "unknown",
    ) -> None:
        validate_job_graph(jobs)
        self.jobs = {job.name: job for job in jobs}
        self.job_order = {job.name: index for index, job in enumerate(jobs)}
        self.config = dict(config)
        self.resolved = resource_config(config)
        self.project_root = project_root.resolve()
        self.run_dir = run_dir.resolve()
        self.run_id = run_id
        self.mode = mode
        self.rerun_completed = rerun_completed
        self.progress_dir = self.run_dir / "common" / "progress"
        self.log_dir = self.progress_dir / "step_logs"
        self.state_dir = self.progress_dir / "scheduler_jobs"
        self.state_path = self.progress_dir / "scheduler_state.json"
        self.events_path = self.progress_dir / "resource_events.jsonl"
        self.metrics_path = self.progress_dir / "resource_metrics.jsonl"
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.status: dict[str, str] = {name: "queued" for name in self.jobs}
        self.attempts: dict[str, int] = {name: 0 for name in self.jobs}
        self.resource_retry_levels: dict[str, int] = {name: 0 for name in self.jobs}
        self.running: dict[str, RunningJob] = {}
        self.failures: dict[str, str] = {}
        self.waiting_started: dict[str, float] = {}
        self.waiting_reasons: dict[str, str] = {}
        self.gpu_allocations: dict[int, int] = {
            int(gpu_id): 0 for gpu_id in self.resolved["cuda"]["gpu_ids"]
        }
        self.gpu_reserved_memory_gb: dict[int, float] = {
            int(gpu_id): 0.0 for gpu_id in self.resolved["cuda"]["gpu_ids"]
        }
        self.gpu_exclusive: dict[int, bool] = {
            int(gpu_id): False for gpu_id in self.resolved["cuda"]["gpu_ids"]
        }
        self.mps_running = 0
        self._initialize_completed()

    def _initialize_completed(self) -> None:
        for name, job in self.jobs.items():
            if self.rerun_completed:
                continue
            if not job.completion_path:
                continue
            path = self.run_dir / job.completion_path
            valid, reason = verify_completion_artifact(path)
            if valid:
                self.status[name] = "completed"
            elif path.exists():
                self.failures[name] = f"invalid_existing_completion:{reason}"

    def _write_job_state(self, name: str, **extra: Any) -> None:
        job = self.jobs[name]
        payload = {
            "timestamp_utc": utc_now(),
            "run_id": self.run_id,
            "job": job.to_dict(),
            "status": self.status[name],
            "attempt": self.attempts[name],
            **extra,
        }
        temporary = self.state_dir / f"{name}.json.tmp"
        final = self.state_dir / f"{name}.json"
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, final)

    def _write_state(self, snapshot: Mapping[str, Any] | None = None) -> None:
        statuses = [
            "queued",
            "waiting_for_resources",
            "running",
            "completed",
            "failed",
            "blocked",
            "resource_exceeded",
            "cancelled",
        ]
        payload = {
            "schema_version": "exp123_scheduler_state_v2",
            "timestamp_utc": utc_now(),
            "run_id": self.run_id,
            "mode": self.mode,
            "summary": {
                status: sum(value == status for value in self.status.values())
                for status in statuses
            },
            "status": self.status,
            "attempts": self.attempts,
            "resource_retry_levels": self.resource_retry_levels,
            "failures": self.failures,
            "waiting_reasons": self.waiting_reasons,
            "gpu_allocations": self.gpu_allocations,
            "gpu_reserved_memory_gb": self.gpu_reserved_memory_gb,
            "running": {
                name: {
                    "pid": item.process.pid,
                    "gpu_id": item.gpu_id,
                    "started_at_utc": item.started_at_utc,
                    "rss_gb": process_tree_rss_gb(item.process.pid),
                    "peak_rss_gb": item.peak_rss_gb,
                    "gpu_process_memory_gb": (
                        nvidia_process_tree_memory_gb(
                            item.process.pid,
                            item.gpu_id,
                        )
                        if item.gpu_id is not None
                        else 0.0
                    ),
                    "peak_gpu_process_memory_gb": item.peak_gpu_memory_gb,
                    "effective_exclusive_gpu": item.effective_exclusive_gpu,
                    "request": item.job.request.to_dict(),
                    "log_path": str(item.log_path),
                }
                for name, item in self.running.items()
            },
            "resources": self.resolved,
            "snapshot": dict(snapshot or {}),
        }
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def _dependencies_complete(self, job: ScheduledJob) -> bool:
        return all(self.status.get(dep) == "completed" for dep in job.dependencies)

    def _failed_dependencies(self, job: ScheduledJob) -> list[str]:
        failure_states = {"failed", "blocked", "resource_exceeded", "cancelled"}
        return [
            dependency
            for dependency in job.dependencies
            if self.status.get(dependency) in failure_states
        ]

    def _dependency_failed(self, job: ScheduledJob) -> bool:
        return bool(self._failed_dependencies(job))

    def _effective_request(self, job: ScheduledJob) -> ResourceRequest:
        policy = self.resolved["resource_failure_policy"]
        level = max(0, int(self.resource_retry_levels.get(job.name, 0)))
        if level == 0:
            return job.request
        ram_scale = float(policy.get("retry_ram_scale", 1.5)) ** level
        gpu_scale = float(policy.get("retry_gpu_memory_scale", 1.25)) ** level
        unified_scale = float(policy.get("retry_unified_memory_scale", 1.35)) ** level
        max_ram = float(policy.get("max_ram_gb_per_job", self.resolved["memory"]["total_limit_gb"]))
        max_gpu = float(policy.get("max_gpu_memory_gb_per_job", 1e9))
        max_unified = float(policy.get("max_unified_memory_gb_per_job", self.resolved["memory"]["total_limit_gb"]))
        return replace(
            job.request,
            ram_gb=min(max_ram, max(0.1, job.request.ram_gb * ram_scale)),
            gpu_memory_gb=min(max_gpu, max(0.0, job.request.gpu_memory_gb * gpu_scale)),
            unified_memory_gb=min(max_unified, max(0.0, job.request.unified_memory_gb * unified_scale)),
            exclusive_gpu=bool(
                job.request.exclusive_gpu
                or (level > 0 and policy.get("retry_exclusive_gpu", False))
            ),
        )

    def _effective_job(self, job: ScheduledJob) -> ScheduledJob:
        return replace(job, request=self._effective_request(job))

    def _resource_failure_detected(self, name: str, return_code: int | None) -> bool:
        if self.status.get(name) == "resource_exceeded":
            return True
        if return_code in {-9, 9, 137}:
            return True
        failure = str(self.failures.get(name, "")).lower()
        markers = (
            "out of memory", "cuda out of memory", "oom", "bad_alloc",
            "cannot allocate memory", "memoryerror", "resource_exceeded",
            "killed", "rss ", "gpu ",
        )
        if any(marker in failure for marker in markers):
            return True
        log_path = self.log_dir / f"{name}.log"
        if log_path.is_file():
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-32768:].lower()
            except OSError:
                tail = ""
            if any(marker in tail for marker in markers):
                return True
        return False

    def _allocated_cpu(self) -> int:
        return sum(item.job.request.cpu_threads for item in self.running.values())

    def _allocated_ram(self) -> float:
        return sum(item.job.request.ram_gb for item in self.running.values())

    def _job_priority(self, name: str) -> tuple[int, int, float, float, int]:
        job = self._effective_job(self.jobs[name])
        accelerator_rank = {"cuda": 0, "mps": 1, "cpu": 2}.get(
            job.request.accelerator,
            3,
        )
        retry_rank = 0 if self.attempts[name] > 0 else 1
        # Larger GPU jobs are placed first so smaller jobs can fill remaining
        # VRAM slots without fragmenting both devices. CPU jobs prefer lower
        # RAM requests after accelerator-bound jobs have been admitted.
        gpu_size_rank = -float(job.request.gpu_memory_gb)
        ram_rank = float(job.request.ram_gb)
        return (
            accelerator_rank,
            retry_rank,
            gpu_size_rank,
            ram_rank,
            self.job_order[name],
        )

    def _effective_exclusive_gpu(self, job: ScheduledJob) -> bool:
        policy = self.resolved["resource_failure_policy"]
        retry_exclusive = (
            bool(policy.get("retry_exclusive_gpu", False))
            and self.attempts[job.name] > 0
        )
        return bool(job.request.exclusive_gpu or retry_exclusive)

    def _choose_gpu(
        self,
        job: ScheduledJob,
        snapshot: Mapping[str, Any],
    ) -> int | None:
        request = job.request
        gpu_rows = {
            int(row["index"]): row
            for row in snapshot.get("gpu", [])
            if "index" in row
        }
        candidates: list[tuple[float, int, int]] = []
        cuda = self.resolved["cuda"]
        exclusive = self._effective_exclusive_gpu(job)

        for gpu_id in cuda["gpu_ids"]:
            allocation_count = self.gpu_allocations.get(gpu_id, 0)
            if allocation_count >= cuda["max_jobs_per_gpu"]:
                continue
            if cuda.get("enforce_exclusive_gpu", True):
                if self.gpu_exclusive.get(gpu_id, False):
                    continue
                if exclusive and allocation_count > 0:
                    continue

            row = gpu_rows.get(gpu_id)
            if row is None:
                continue

            actual_free = float(row.get("memory_free_gb", 0.0))
            actual_total = float(
                row.get(
                    "memory_total_gb",
                    actual_free + self.gpu_reserved_memory_gb.get(gpu_id, 0.0),
                )
            )
            reserved_before = self.gpu_reserved_memory_gb.get(gpu_id, 0.0)
            reserved_after = reserved_before + request.gpu_memory_gb
            capacity_after_reserve = max(
                0.0,
                actual_total - float(cuda["reserve_memory_gb"]),
            )

            if reserved_after > capacity_after_reserve + 1e-9:
                continue
            if actual_free < (
                request.gpu_memory_gb
                + float(cuda["minimum_free_memory_gb"])
            ):
                continue

            remaining_capacity = capacity_after_reserve - reserved_after
            remaining_actual = actual_free - request.gpu_memory_gb
            score = min(remaining_capacity, remaining_actual)
            candidates.append((score, -allocation_count, gpu_id))

        if not candidates:
            return None
        strategy = str(cuda.get("assignment_strategy", "most_free_memory"))
        if strategy == "first_available":
            return min(gpu_id for _, _, gpu_id in candidates)
        candidates.sort(reverse=True)
        return candidates[0][2]

    def _can_launch(
        self,
        job: ScheduledJob,
        snapshot: Mapping[str, Any],
    ) -> tuple[bool, int | None, str]:
        job = self._effective_job(job)
        scheduler = self.resolved["scheduler"]
        if len(self.running) >= scheduler["max_concurrent_jobs"]:
            return False, None, "max_concurrent_jobs"
        model_limits = dict(scheduler.get("max_concurrent_by_model", {}))
        if job.model_name and job.model_name in model_limits:
            same_model_running = sum(
                item.job.model_name == job.model_name
                for item in self.running.values()
            )
            if same_model_running >= int(model_limits[job.model_name]):
                return False, None, f"model_concurrency:{job.model_name}"
        serialize_level = int(self.resolved["resource_failure_policy"].get("serialize_after_resource_retry_level", 2))
        if self.resource_retry_levels.get(job.name, 0) >= serialize_level and self.running:
            return False, None, "resource_retry_serialization"
        if (
            self._allocated_cpu() + job.request.cpu_threads
            > self.resolved["cpu"]["total_thread_limit"]
        ):
            return False, None, "cpu_thread_budget"
        if (
            self._allocated_ram() + job.request.ram_gb
            > self.resolved["memory"]["total_limit_gb"]
        ):
            return False, None, "configured_ram_budget"
        required_available = max(
            self.resolved["memory"]["minimum_free_gb_to_launch"],
            job.request.ram_gb + self.resolved["memory"]["reserve_gb"],
        )
        if float(snapshot["available_ram_gb"]) < required_available:
            return False, None, "actual_free_ram"

        if job.request.accelerator == "cuda":
            gpu_id = self._choose_gpu(job, snapshot)
            if gpu_id is None:
                return False, None, "cuda_vram_reservation_or_slot"
            return True, gpu_id, "ready"

        if job.request.accelerator == "mps":
            if self.mps_running >= self.resolved["mps"]["max_concurrent_jobs"]:
                return False, None, "mps_slot"
            if float(snapshot["available_ram_gb"]) < max(
                self.resolved["mps"]["minimum_free_unified_memory_gb"],
                job.request.unified_memory_gb
                + self.resolved["memory"]["reserve_gb"],
            ):
                return False, None, "mps_unified_memory"
        return True, None, "ready"

    def _environment(
        self,
        job: ScheduledJob,
        gpu_id: int | None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env["EXP123_WORKER_LABEL"] = job.name
        policy = self.resolved["resource_failure_policy"]
        retry_index = max(0, self.attempts[job.name] - 1)
        thread_scale = float(policy.get("retry_cpu_thread_scale", 1.0)) ** retry_index
        assigned_threads = max(1, int(job.request.cpu_threads * thread_scale))
        env["EXP123_ASSIGNED_CPU_THREADS"] = str(assigned_threads)
        batch_scale = float(policy.get("retry_batch_size_scale", 1.0)) ** retry_index
        env["EXP123_RETRY_BATCH_SIZE_SCALE"] = str(batch_scale)
        env["EXP123_ACCELERATOR_BACKEND"] = job.request.accelerator
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env[key] = str(assigned_threads)
        if gpu_id is not None:
            env["EXP123_CUDA_DEVICE"] = str(gpu_id)
            env["EXP123_PYTORCH_CUDA_MEMORY_FRACTION"] = str(
                self.resolved["cuda"]["pytorch_memory_fraction"]
            )
        else:
            env.pop("EXP123_CUDA_DEVICE", None)
        if job.request.accelerator == "mps":
            env["EXP123_MPS_MEMORY_FRACTION"] = str(
                self.resolved["mps"]["pytorch_memory_fraction"]
            )
        return env

    def _launch(self, job: ScheduledJob, gpu_id: int | None) -> None:
        job = self._effective_job(job)
        self.attempts[job.name] += 1
        log_path = self.log_dir / f"{job.name}.log"
        log_file = log_path.open("a", encoding="utf-8", buffering=1)
        log_file.write(
            f"\n[{utc_now()}] START {job.name} "
            f"attempt={self.attempts[job.name]} gpu={gpu_id} "
            f"request={job.request.to_dict()}\n"
        )
        process = subprocess.Popen(
            list(job.command),
            cwd=self.project_root,
            env=self._environment(job, gpu_id),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self.status[job.name] = "running"
        self.waiting_reasons.pop(job.name, None)
        effective_exclusive = self._effective_exclusive_gpu(job)
        if gpu_id is not None:
            self.gpu_allocations[gpu_id] = self.gpu_allocations.get(gpu_id, 0) + 1
            self.gpu_reserved_memory_gb[gpu_id] = (
                self.gpu_reserved_memory_gb.get(gpu_id, 0.0)
                + job.request.gpu_memory_gb
            )
            if effective_exclusive:
                self.gpu_exclusive[gpu_id] = True
        if job.request.accelerator == "mps":
            self.mps_running += 1
        self.running[job.name] = RunningJob(
            job=job,
            process=process,
            log_file=log_file,
            log_path=log_path,
            started_monotonic=time.monotonic(),
            started_at_utc=utc_now(),
            gpu_id=gpu_id,
            attempt=self.attempts[job.name],
            effective_exclusive_gpu=effective_exclusive,
        )
        append_jsonl(
            self.events_path,
            {
                "timestamp_utc": utc_now(),
                "event": "job_started",
                "job": job.name,
                "pid": process.pid,
                "gpu_id": gpu_id,
                "request": job.request.to_dict(),
                "gpu_reserved_memory_gb": (
                    self.gpu_reserved_memory_gb.get(gpu_id, 0.0)
                    if gpu_id is not None
                    else 0.0
                ),
            },
        )
        emit_progress(
            "scheduler_job_started",
            operation=job.name,
            message=f"Started {job.name}",
            status="running",
            details={"gpu_id": gpu_id, "request": job.request.to_dict()},
        )
        self._write_job_state(
            job.name,
            pid=process.pid,
            gpu_id=gpu_id,
            log_path=str(log_path),
        )

    def _release(self, running: RunningJob) -> None:
        if running.gpu_id is not None:
            gpu_id = running.gpu_id
            self.gpu_allocations[gpu_id] = max(
                0,
                self.gpu_allocations.get(gpu_id, 0) - 1,
            )
            self.gpu_reserved_memory_gb[gpu_id] = max(
                0.0,
                self.gpu_reserved_memory_gb.get(gpu_id, 0.0)
                - running.job.request.gpu_memory_gb,
            )
            if self.gpu_allocations[gpu_id] == 0:
                self.gpu_exclusive[gpu_id] = False
        if running.job.request.accelerator == "mps":
            self.mps_running = max(0, self.mps_running - 1)
        try:
            running.log_file.close()
        except Exception:
            pass

    def _max_retries(self) -> int:
        policy = self.resolved["resource_failure_policy"]
        if not bool(policy.get("auto_retry", True)):
            return 0
        return max(0, int(policy.get("max_retries", 1)))

    def _poll_running(self) -> None:
        ram_grace = self.resolved["memory"]["breach_grace_seconds"]
        gpu_grace = self.resolved["cuda"]["gpu_memory_breach_grace_seconds"]

        for name, running in list(self.running.items()):
            rc = running.process.poll()
            rss = process_tree_rss_gb(running.process.pid)
            running.peak_rss_gb = max(running.peak_rss_gb, rss)
            hard_limit = running.job.request.ram_gb
            if running.job.request.accelerator == "mps":
                hard_limit = max(hard_limit, running.job.request.unified_memory_gb)
            enforce_ram = bool(
                self.resolved["memory"].get("enforce_job_ram_limit", True)
            )

            if rc is None and enforce_ram and rss > hard_limit:
                if running.breach_started is None:
                    running.breach_started = time.monotonic()
                    append_jsonl(
                        self.events_path,
                        {
                            "timestamp_utc": utc_now(),
                            "event": "memory_limit_warning",
                            "job": name,
                            "rss_gb": rss,
                            "limit_gb": hard_limit,
                        },
                    )
                elif time.monotonic() - running.breach_started >= ram_grace:
                    terminate_process_group(running.process)
                    self.status[name] = "resource_exceeded"
                    self.failures[name] = (
                        f"RSS {rss:.2f} GB exceeded configured "
                        f"{hard_limit:.2f} GB for {ram_grace:.1f}s"
                    )
                    rc = running.process.poll()
            else:
                running.breach_started = None

            gpu_memory = 0.0
            enforce_gpu = bool(
                self.resolved["cuda"].get("enforce_process_memory_limit", True)
            )
            if (
                rc is None
                and running.gpu_id is not None
                and running.job.request.gpu_memory_gb > 0
            ):
                gpu_memory = nvidia_process_tree_memory_gb(
                    running.process.pid,
                    running.gpu_id,
                )
                running.peak_gpu_memory_gb = max(
                    running.peak_gpu_memory_gb,
                    gpu_memory,
                )
                if enforce_gpu and gpu_memory > running.job.request.gpu_memory_gb:
                    if running.gpu_breach_started is None:
                        running.gpu_breach_started = time.monotonic()
                        append_jsonl(
                            self.events_path,
                            {
                                "timestamp_utc": utc_now(),
                                "event": "gpu_memory_limit_warning",
                                "job": name,
                                "gpu_id": running.gpu_id,
                                "gpu_memory_gb": gpu_memory,
                                "limit_gb": running.job.request.gpu_memory_gb,
                            },
                        )
                    elif time.monotonic() - running.gpu_breach_started >= gpu_grace:
                        terminate_process_group(running.process)
                        self.status[name] = "resource_exceeded"
                        self.failures[name] = (
                            f"GPU {running.gpu_id} process memory "
                            f"{gpu_memory:.2f} GB exceeded configured "
                            f"{running.job.request.gpu_memory_gb:.2f} GB "
                            f"for {gpu_grace:.1f}s"
                        )
                        rc = running.process.poll()
                else:
                    running.gpu_breach_started = None

            if rc is None:
                continue

            self._release(running)
            del self.running[name]
            completion_valid = True
            completion_reason = "not_required"
            if running.job.completion_path:
                completion_valid, completion_reason = verify_completion_artifact(
                    self.run_dir / running.job.completion_path
                )

            if rc == 0 and completion_valid:
                self.status[name] = "completed"
                self.failures.pop(name, None)
                self.waiting_started.pop(name, None)
                self.waiting_reasons.pop(name, None)
                event = "job_completed"
            else:
                error = self.failures.get(
                    name,
                    f"exit_code={rc}; completion={completion_reason}; "
                    f"log={running.log_path}",
                )
                resource_failure = self._resource_failure_detected(name, rc)
                retry_allowed = bool(
                    resource_failure
                    and self.attempts[name] <= self._max_retries()
                )
                if retry_allowed:
                    self.status[name] = "queued"
                    self.failures[name] = error
                    self.resource_retry_levels[name] += 1
                    event = "job_retry_queued"
                else:
                    if self.status[name] != "resource_exceeded":
                        self.status[name] = "failed"
                    self.failures[name] = error
                    event = "job_failed"

            append_jsonl(
                self.events_path,
                {
                    "timestamp_utc": utc_now(),
                    "event": event,
                    "job": name,
                    "status": self.status[name],
                    "return_code": rc,
                    "attempt": self.attempts[name],
                    "failure": self.failures.get(name),
                    "completion_validation": completion_reason,
                    "peak_rss_gb": running.peak_rss_gb,
                    "peak_gpu_process_memory_gb": running.peak_gpu_memory_gb,
                    "resource_retry_level": self.resource_retry_levels.get(name, 0),
                    "next_effective_request": self._effective_request(self.jobs[name]).to_dict(),
                },
            )
            emit_progress(
                f"scheduler_{event}",
                operation=name,
                message=self.failures.get(name, event),
                status=self.status[name],
            )
            self._write_job_state(
                name,
                return_code=rc,
                failure=self.failures.get(name),
                completion_validation=completion_reason,
                elapsed_seconds=time.monotonic() - running.started_monotonic,
                peak_observed_rss_gb=running.peak_rss_gb,
                peak_observed_gpu_process_memory_gb=running.peak_gpu_memory_gb,
                resource_retry_level=self.resource_retry_levels.get(name, 0),
                next_effective_request=self._effective_request(self.jobs[name]).to_dict(),
            )

    def _shutdown_running(self, status: str = "cancelled") -> None:
        for name, running in list(self.running.items()):
            terminate_process_group(running.process)
            self._release(running)
            self.status[name] = status
            self.failures[name] = "scheduler_interrupted"
            self._write_job_state(name, failure="scheduler_interrupted")
            del self.running[name]

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        try:
            while True:
                snapshot = system_snapshot(include_gpu=self.resolved["backend"] == "cuda").to_dict()
                append_jsonl(self.metrics_path, snapshot)
                self._poll_running()

                for name in sorted(self.jobs, key=self._job_priority):
                    job = self.jobs[name]
                    if self.status[name] in {
                        "completed",
                        "running",
                        "failed",
                        "blocked",
                        "resource_exceeded",
                        "cancelled",
                    }:
                        continue

                    if self._dependency_failed(job):
                        failed_dependencies = self._failed_dependencies(job)
                        self.status[name] = "blocked"
                        self.failures[name] = (
                            "dependency_failed:"
                            + ",".join(failed_dependencies)
                        )
                        self.waiting_reasons.pop(name, None)
                        self._write_job_state(
                            name,
                            failure=self.failures[name],
                            failed_dependencies=failed_dependencies,
                        )
                        continue

                    if not self._dependencies_complete(job):
                        self.status[name] = "queued"
                        self.waiting_started.pop(name, None)
                        self.waiting_reasons[name] = "dependencies_pending"
                        continue

                    can_launch, gpu_id, reason = self._can_launch(job, snapshot)
                    if can_launch:
                        self.waiting_started.pop(name, None)
                        self.waiting_reasons.pop(name, None)
                        self._launch(job, gpu_id)
                    else:
                        self.status[name] = "waiting_for_resources"
                        self.waiting_reasons[name] = reason
                        self.waiting_started.setdefault(name, time.monotonic())
                        maximum_wait = self.resolved["scheduler"].get(
                            "max_resource_wait_seconds",
                            0.0,
                        )
                        waited = time.monotonic() - self.waiting_started[name]
                        if maximum_wait > 0 and waited >= maximum_wait:
                            self.status[name] = "resource_exceeded"
                            self.failures[name] = (
                                f"resource_wait_timeout:{reason}; "
                                f"waited_seconds={waited:.1f}"
                            )
                            self._write_job_state(
                                name,
                                failure=self.failures[name],
                                waiting_reason=reason,
                                waiting_seconds=waited,
                            )
                        else:
                            self._write_job_state(
                                name,
                                waiting_reason=reason,
                                waiting_seconds=waited,
                            )

                self._write_state(snapshot)
                terminal = {
                    "completed",
                    "failed",
                    "blocked",
                    "resource_exceeded",
                    "cancelled",
                }
                if (
                    all(value in terminal for value in self.status.values())
                    and not self.running
                ):
                    break
                time.sleep(self.resolved["scheduler"]["poll_interval_seconds"])
        except KeyboardInterrupt:
            self._shutdown_running("cancelled")
            self._write_state(system_snapshot(include_gpu=self.resolved["backend"] == "cuda").to_dict())
            raise

        result = build_protocol_result(
            run_id=self.run_id,
            mode=self.mode,
            elapsed_seconds=time.monotonic() - started,
            jobs={name: job.to_dict() for name, job in self.jobs.items()},
            status=self.status,
            attempts=self.attempts,
            failures=self.failures,
            log_dir=self.log_dir,
        )
        self._write_state(system_snapshot(include_gpu=self.resolved["backend"] == "cuda").to_dict())
        return result
