#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from los_baseline.config import load_config, resolve_path
from los_baseline.hardware import (
    detect_hardware, nvidia_smi_snapshot, write_hardware_report,
    validate_cuda_acceleration,
)
from los_baseline.progress import configure_progress, emit_progress, progress_iter


@dataclass(frozen=True)
class ProtocolJob:
    name: str
    step: str
    extra_args: tuple[str, ...] = ()
    gpu_id: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EXP123 discovery/confirmatory protocols with live progress and optional dual-GPU task parallelism."
    )
    parser.add_argument("mode", choices=["discovery", "confirmatory"])
    parser.add_argument("run_id")
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--force-sequential", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _step_complete(run_dir: Path, job: ProtocolJob) -> bool:
    checks: dict[str, Path] = {
        "outcomes": run_dir / "common" / "outcomes" / "outcome_step_manifest.json",
        "splits": run_dir / "common" / "splits" / "split_hash.txt",
        "exp1-optimize": run_dir / "optimization" / "exp1_flat_3class" / "optimization_manifest.json",
        "stage1-optimize": run_dir / "optimization" / "stage1_mortality" / "optimization_manifest.json",
        "stage2-optimize": run_dir / "optimization" / "stage2_long_stay" / "optimization_manifest.json",
        "exp1-train": run_dir / "EXP1_flat_3class" / "exp1_result.json",
        "stage1-train": run_dir / "EXP2_two_stage" / "stage1_mortality" / "stage1_result.json",
        "stage2-train": run_dir / "EXP2_two_stage" / "stage2_long_stay" / "stage2_result.json",
        "stage1-heterogeneity-discovery": run_dir / "EXP2_two_stage" / "heterogeneity_analysis" / "stage1_mortality" / "discovery_validation" / "frozen_design.json",
        "stage2-heterogeneity-discovery": run_dir / "EXP2_two_stage" / "heterogeneity_analysis" / "stage2_long_stay" / "discovery_validation" / "frozen_design.json",
        "stage1-super-learner": run_dir / "EXP2_two_stage" / "super_learner" / "stage1_mortality" / "artifact.json",
        "stage2-super-learner": run_dir / "EXP2_two_stage" / "super_learner" / "stage2_long_stay" / "artifact.json",
        "stage1-heterogeneity-confirmatory-internal": run_dir / "EXP2_two_stage" / "heterogeneity_analysis" / "stage1_mortality" / "confirmatory" / "internal_test" / "analysis_contract.json",
        "stage2-heterogeneity-confirmatory-internal": run_dir / "EXP2_two_stage" / "heterogeneity_analysis" / "stage2_long_stay" / "confirmatory" / "internal_test" / "analysis_contract.json",
        "oracle-pipeline-internal": run_dir / "EXP2_two_stage" / "pipeline_evaluation" / "internal_test" / "COMPLETED.lock.json",
        "final-comparison-internal": run_dir / "final_comparison" / "internal_test" / "COMPLETED.lock.json",
        "stage1-heterogeneity-confirmatory-eicu": run_dir / "EXP2_two_stage" / "heterogeneity_analysis" / "stage1_mortality" / "confirmatory" / "external_eicu" / "analysis_contract.json",
        "stage2-heterogeneity-confirmatory-eicu": run_dir / "EXP2_two_stage" / "heterogeneity_analysis" / "stage2_long_stay" / "confirmatory" / "external_eicu" / "analysis_contract.json",
        "oracle-pipeline-eicu": run_dir / "EXP2_two_stage" / "pipeline_evaluation" / "external_eicu" / "COMPLETED.lock.json",
        "final-comparison-eicu": run_dir / "final_comparison" / "external_eicu" / "COMPLETED.lock.json",
        "finalize-run": run_dir / "final_run_freeze" / "EXP123_RUN_COMPLETE.lock.json",
    }
    path = checks.get(job.name) or checks.get(job.step)
    return bool(path and path.is_file())


def _latest_worker_message(progress_dir: Path) -> str:
    paths = sorted(
        progress_dir.glob("current_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in paths[:5]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        operation = payload.get("operation") or payload.get("event")
        current = payload.get("current")
        total = payload.get("total")
        if current is not None and total is not None:
            return f"{operation}: {current}/{total}"
        message = payload.get("message")
        if message:
            return str(message)[:100]
    return "waiting for worker progress"


class GPUMonitor:
    def __init__(self, output_path: Path, interval_seconds: float) -> None:
        self.output_path = output_path
        self.interval_seconds = max(1.0, interval_seconds)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval_seconds + 2.0)

    def _run(self) -> None:
        fieldnames = [
            "timestamp_utc", "index", "name", "utilization_gpu_percent",
            "memory_used_mb", "memory_total_mb", "temperature_c", "power_draw_w",
        ]
        write_header = not self.output_path.exists()
        while not self.stop_event.is_set():
            rows = nvidia_smi_snapshot()
            if rows:
                with self.output_path.open("a", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    timestamp = _utc_now()
                    for row in rows:
                        writer.writerow({"timestamp_utc": timestamp, **row})
            self.stop_event.wait(self.interval_seconds)


def _job_command(job: ProtocolJob, config_path: Path, run_id: str) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "run_exp123.py"),
        "--config", str(config_path),
        "--run-id", run_id,
        "--step", job.step,
        *job.extra_args,
    ]


def _job_environment(config: dict[str, Any], job: ProtocolJob) -> dict[str, str]:
    env = os.environ.copy()
    env["EXP123_WORKER_LABEL"] = job.name
    if job.gpu_id is not None:
        env["EXP123_CUDA_DEVICE"] = str(job.gpu_id)
    else:
        env.pop("EXP123_CUDA_DEVICE", None)
    threads = max(1, int(config.get("hardware", {}).get("cpu", {}).get("threads_per_worker", 1)))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(threads)
    return env


def _run_job(
    job: ProtocolJob,
    *,
    config: dict[str, Any],
    config_path: Path,
    run_id: str,
    log_dir: Path,
) -> subprocess.Popen[str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.name}.log"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    header = f"\n[{_utc_now()}] START {job.name} gpu={job.gpu_id}\n"
    log_file.write(header)
    process = subprocess.Popen(
        _job_command(job, config_path, run_id),
        cwd=PROJECT_ROOT,
        env=_job_environment(config, job),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._exp123_log_file = log_file  # type: ignore[attr-defined]
    process._exp123_log_path = log_path  # type: ignore[attr-defined]
    return process


def _wait_group(
    jobs: list[ProtocolJob],
    *,
    config: dict[str, Any],
    config_path: Path,
    run_id: str,
    run_dir: Path,
    rerun_completed: bool,
) -> None:
    pending = [job for job in jobs if rerun_completed or not _step_complete(run_dir, job)]
    skipped = [job.name for job in jobs if job not in pending]
    for name in skipped:
        emit_progress(
            "step_skipped", operation=name,
            message=f"Skipped completed protocol job: {name}", status="completed"
        )
    if not pending:
        return
    processes: dict[str, tuple[ProtocolJob, subprocess.Popen[str]]] = {}
    log_dir = run_dir / "common" / "progress" / "step_logs"
    for job in pending:
        emit_progress(
            "step_started", operation=job.name,
            message=f"Started protocol job {job.name} on GPU {job.gpu_id}.",
            status="running", details={"gpu_id": job.gpu_id},
        )
        processes[job.name] = (
            job,
            _run_job(
                job, config=config, config_path=config_path,
                run_id=run_id, log_dir=log_dir,
            ),
        )

    from tqdm.auto import tqdm
    bar = tqdm(total=len(processes), desc="Protocol jobs", unit="job", dynamic_ncols=True)
    failures: list[str] = []
    completed_names: set[str] = set()
    try:
        while len(completed_names) < len(processes):
            for name, (job, process) in processes.items():
                if name in completed_names:
                    continue
                return_code = process.poll()
                if return_code is None:
                    continue
                log_file = getattr(process, "_exp123_log_file", None)
                if log_file is not None:
                    log_file.write(f"[{_utc_now()}] END {name} rc={return_code}\n")
                    log_file.close()
                completed_names.add(name)
                bar.update(1)
                if return_code != 0:
                    failures.append(name)
                    emit_progress(
                        "step_failed", operation=name,
                        message=f"Protocol job failed with exit code {return_code}.",
                        status="failed",
                        details={"log_path": str(getattr(process, "_exp123_log_path", ""))},
                    )
                else:
                    emit_progress(
                        "step_completed", operation=name,
                        message=f"Completed protocol job {name}.", status="completed",
                        details={"log_path": str(getattr(process, "_exp123_log_path", ""))},
                    )
            bar.set_postfix_str(_latest_worker_message(run_dir / "common" / "progress")[:100])
            time.sleep(1.0)
    finally:
        bar.close()
        for _, process in processes.values():
            if process.poll() is None:
                process.terminate()
    if failures:
        logs = [str(log_dir / f"{name}.log") for name in failures]
        raise RuntimeError(f"Protocol job(s) failed: {failures}. Review logs: {logs}")


def _discovery_groups(config: dict[str, Any], gpu_ids: list[int], parallel: bool) -> list[list[ProtocolJob]]:
    gpu0 = gpu_ids[0] if gpu_ids else None
    gpu1 = gpu_ids[1] if len(gpu_ids) > 1 else gpu0
    if parallel:
        return [
            [ProtocolJob("outcomes", "outcomes")],
            [ProtocolJob("splits", "splits")],
            [
                ProtocolJob("exp1-optimize", "exp1-optimize", gpu_id=gpu0),
                ProtocolJob("stage1-optimize", "stage1-optimize", gpu_id=gpu1),
            ],
            [ProtocolJob("stage2-optimize", "stage2-optimize", gpu_id=gpu0)],
            [
                ProtocolJob("exp1-train", "exp1-train", gpu_id=gpu0),
                ProtocolJob("stage1-train", "stage1-train", gpu_id=gpu1),
            ],
            [ProtocolJob("stage2-train", "stage2-train", gpu_id=gpu0)],
            [
                ProtocolJob(
                    "stage1-heterogeneity-discovery",
                    "stage1-heterogeneity-discovery",
                    ("--split", "validation"),
                ),
                ProtocolJob(
                    "stage2-heterogeneity-discovery",
                    "stage2-heterogeneity-discovery",
                    ("--split", "validation"),
                ),
            ],
            [
                ProtocolJob("stage1-super-learner", "stage1-super-learner"),
                ProtocolJob("stage2-super-learner", "stage2-super-learner"),
            ],
            [ProtocolJob("status", "status")],
        ]
    return [[job] for group in _discovery_groups(config, gpu_ids, True) for job in group]


def _confirmatory_groups(gpu_ids: list[int]) -> list[list[ProtocolJob]]:
    gpu0 = gpu_ids[0] if gpu_ids else None
    return [
        [ProtocolJob("stage1-heterogeneity-confirmatory-internal", "stage1-heterogeneity-confirmatory", ("--split", "internal_test"))],
        [ProtocolJob("stage2-heterogeneity-confirmatory-internal", "stage2-heterogeneity-confirmatory", ("--split", "internal_test"))],
        [ProtocolJob("oracle-pipeline-internal", "oracle-pipeline-confirmatory", ("--split", "internal_test"), gpu0)],
        [ProtocolJob("final-comparison-internal", "final-comparison-confirmatory", ("--split", "internal_test"), gpu0)],
        [ProtocolJob("stage1-heterogeneity-confirmatory-eicu", "stage1-heterogeneity-confirmatory", ("--split", "external_eicu"))],
        [ProtocolJob("stage2-heterogeneity-confirmatory-eicu", "stage2-heterogeneity-confirmatory", ("--split", "external_eicu"))],
        [ProtocolJob("oracle-pipeline-eicu", "oracle-pipeline-confirmatory", ("--split", "external_eicu"), gpu0)],
        [ProtocolJob("final-comparison-eicu", "final-comparison-confirmatory", ("--split", "external_eicu"), gpu0)],
        [ProtocolJob("finalize-run", "finalize-run")],
        [ProtocolJob("status", "status")],
    ]


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    run_dir = resolve_path(config, config["paths"]["runs_dir"]) / args.run_id
    progress_dir = run_dir / "common" / "progress"
    configure_progress(
        output_dir=progress_dir,
        run_id=args.run_id,
        config=config,
        worker_label=f"protocol_{args.mode}",
    )
    report = write_hardware_report(progress_dir / "protocol_hardware_report.json", config)
    hardware_report = detect_hardware(config)
    acceleration_check = validate_cuda_acceleration(
        config,
        run_xgboost_smoke=bool(config.get("hardware", {}).get("cuda", {}).get("required", False)),
    )
    (progress_dir / "protocol_acceleration_check.json").write_text(
        json.dumps(acceleration_check, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if acceleration_check.get("errors"):
        raise RuntimeError(
            "CUDA acceleration check failed: "
            + "; ".join(str(value) for value in acceleration_check["errors"])
        )
    cuda_config = dict(config.get("hardware", {}).get("cuda", {}))
    gpu_ids = [int(value) for value in cuda_config.get("gpu_ids", [0, 1])]
    if bool(cuda_config.get("required", False)):
        if not hardware_report.cuda_available:
            raise RuntimeError("CUDA is required by the selected hardware profile, but PyTorch cannot access CUDA.")
        if hardware_report.gpu_count < len(gpu_ids):
            raise RuntimeError(
                f"The profile requests GPU IDs {gpu_ids}, but only {hardware_report.gpu_count} CUDA device(s) are visible."
            )
    multi = dict(config.get("hardware", {}).get("multi_gpu", {}))
    parallel = (
        args.mode == "discovery"
        and not args.force_sequential
        and bool(multi.get("enabled", True))
        and hardware_report.cuda_available
        and hardware_report.gpu_count >= 2
        and len(gpu_ids) >= 2
    )
    groups = (
        _discovery_groups(config, gpu_ids, parallel)
        if args.mode == "discovery"
        else _confirmatory_groups(gpu_ids)
    )
    protocol_manifest = {
        "run_id": args.run_id,
        "mode": args.mode,
        "config_path": str(config_path),
        "parallel_task_execution": parallel,
        "gpu_ids": gpu_ids,
        "hardware": report,
        "groups": [[job.__dict__ for job in group] for group in groups],
        "started_at_utc": _utc_now(),
    }
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / f"{args.mode}_protocol_manifest.json").write_text(
        json.dumps(protocol_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    monitor = GPUMonitor(
        progress_dir / "gpu_metrics.csv",
        float(config.get("progress", {}).get("gpu_monitor_interval_seconds", 5.0)),
    )
    monitor.start()
    try:
        for group_index, group in progress_iter(
            list(enumerate(groups, start=1)),
            total=len(groups),
            description=f"EXP123 {args.mode} phases",
            unit="phase",
            details={"parallel_task_execution": parallel},
        ):
            emit_progress(
                "phase_started", operation=f"phase_{group_index}",
                message="Running: " + ", ".join(job.name for job in group),
                current=group_index, total=len(groups), unit="phase", status="running",
            )
            _wait_group(
                group,
                config=config,
                config_path=config_path,
                run_id=args.run_id,
                run_dir=run_dir,
                rerun_completed=args.rerun_completed,
            )
    finally:
        monitor.stop()
    emit_progress(
        "protocol_completed", operation=args.mode,
        message=f"Completed EXP123 {args.mode} protocol.", status="completed"
    )
    print(f"[OK] EXP123 {args.mode} protocol completed.")
    print(f"[INFO] Progress logs: {progress_dir}")
    if args.mode == "discovery":
        print(f"[NEXT] Review Validation outputs, then freeze confirmatory design for {args.run_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
