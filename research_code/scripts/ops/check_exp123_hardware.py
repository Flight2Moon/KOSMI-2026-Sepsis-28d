#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config
from los_baseline.hardware import detect_hardware, nvidia_smi_snapshot
from los_baseline.resource_manager import request_for_job, resource_config, system_snapshot


def _gpu_capable_jobs(config: dict[str, object]) -> list[dict[str, object]]:
    experiments = dict(config.get("experiments", {}))
    task_models: list[tuple[str, list[str]]] = []
    exp1 = dict(experiments.get("exp1_flat_3class", {}))
    if exp1.get("enabled", True):
        task_models.append(("exp1", list(exp1.get("models", []))))
    exp2 = dict(experiments.get("exp2_two_stage", {}))
    if exp2.get("enabled", True):
        for task_key, section in (
            ("stage1", "stage1_mortality"),
            ("stage2", "stage2_long_stay"),
        ):
            task_models.append(
                (task_key, list(dict(exp2.get(section, {})).get("models", [])))
            )

    jobs: list[dict[str, object]] = []
    for task_key, models in task_models:
        for model_name in models:
            request = request_for_job(
                config,
                model_name=model_name,
                job_kind="optimization",
            )
            if request.accelerator != "cuda":
                continue
            jobs.append(
                {
                    "name": f"{task_key}-{model_name}",
                    "model_name": model_name,
                    "gpu_memory_gb": request.gpu_memory_gb,
                    "ram_gb": request.ram_gb,
                    "cpu_threads": request.cpu_threads,
                }
            )
    return jobs


def _find_gpu_packing(
    jobs: list[dict[str, object]],
    capacities: dict[int, float],
    max_jobs_per_gpu: int,
) -> dict[int, list[dict[str, object]]] | None:
    ordered = sorted(
        jobs,
        key=lambda item: float(item["gpu_memory_gb"]),
        reverse=True,
    )
    assignments: dict[int, list[dict[str, object]]] = {
        gpu_id: [] for gpu_id in capacities
    }
    remaining = dict(capacities)

    def search(index: int) -> bool:
        if index >= len(ordered):
            return True
        job = ordered[index]
        required = float(job["gpu_memory_gb"])
        seen_states: set[tuple[int, int]] = set()
        for gpu_id in sorted(
            capacities,
            key=lambda value: (remaining[value], -len(assignments[value])),
            reverse=True,
        ):
            state = (round(remaining[gpu_id] * 1000), len(assignments[gpu_id]))
            if state in seen_states:
                continue
            seen_states.add(state)
            if len(assignments[gpu_id]) >= max_jobs_per_gpu:
                continue
            if remaining[gpu_id] + 1e-9 < required:
                continue
            assignments[gpu_id].append(job)
            remaining[gpu_id] -= required
            if search(index + 1):
                return True
            remaining[gpu_id] += required
            assignments[gpu_id].pop()
        return False

    return assignments if search(0) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate EXP123 CPU, CUDA, MPS, RAM and accelerator limits.")
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--xgboost-smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config).expanduser().resolve())
    resolved = resource_config(config)
    report = detect_hardware(config).to_dict()
    report["resource_profile"] = resolved
    report["system_snapshot"] = system_snapshot().to_dict()
    report["nvidia_smi"] = nvidia_smi_snapshot()
    errors: list[str] = []
    warnings: list[str] = []
    backend = resolved["backend"]
    cuda_config = dict(config.get("hardware", {}).get("cuda", {}))
    requested_ids = [int(value) for value in resolved["cuda"]["gpu_ids"]]

    try:
        import torch

        mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        report["mps_available"] = mps_available
        report["mps_built"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())
        if mps_available and hasattr(torch, "mps"):
            report["mps_current_allocated_gb"] = float(torch.mps.current_allocated_memory() / 1024**3) if hasattr(torch.mps, "current_allocated_memory") else None
            report["mps_driver_allocated_gb"] = float(torch.mps.driver_allocated_memory() / 1024**3) if hasattr(torch.mps, "driver_allocated_memory") else None
            report["mps_recommended_max_gb"] = float(torch.mps.recommended_max_memory() / 1024**3) if hasattr(torch.mps, "recommended_max_memory") else None
    except Exception as exception:
        errors.append(f"PyTorch import failed: {exception}")
        torch = None  # type: ignore[assignment]
        mps_available = False

    if backend == "cuda" or bool(cuda_config.get("required", False)):
        if not report["cuda_available"]:
            errors.append("CUDA is required by the selected profile, but PyTorch cannot access CUDA.")
        visible_ids = {int(row["index"]) for row in report["nvidia_smi"]}
        for gpu_id in requested_ids:
            if gpu_id not in visible_ids:
                errors.append(f"GPU ID {gpu_id} is not visible through nvidia-smi.")
        if torch is not None and report["cuda_available"]:
            for gpu_id in requested_ids:
                if gpu_id >= torch.cuda.device_count():
                    errors.append(f"GPU ID {gpu_id} is not visible to PyTorch.")
                    continue
                tensor = torch.arange(2048, device=f"cuda:{gpu_id}", dtype=torch.float32)
                value = float((tensor * tensor).sum().cpu())
                report.setdefault("pytorch_cuda_smoke", []).append({"gpu_id": gpu_id, "sum_of_squares": value, "passed": value > 0})
    elif backend == "mps":
        if not mps_available:
            if resolved["mps"]["allow_cpu_fallback"]:
                warnings.append("MPS is unavailable; this profile allows CPU fallback.")
            else:
                errors.append("MPS is required by the selected profile, but torch.backends.mps.is_available() is False.")
        elif torch is not None:
            try:
                tensor = torch.arange(2048, device="mps", dtype=torch.float32)
                value = float((tensor * tensor).sum().cpu())
                report["pytorch_mps_smoke"] = {"sum_of_squares": value, "passed": value > 0}
            except Exception as exception:
                errors.append(f"PyTorch MPS smoke failed: {exception}")

    if args.xgboost_smoke and backend == "cuda" and report["cuda_available"]:
        import numpy as np
        from xgboost import XGBClassifier

        X = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
        y = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        for gpu_id in requested_ids:
            try:
                model = XGBClassifier(n_estimators=4, max_depth=2, tree_method="hist", device=f"cuda:{gpu_id}", eval_metric="logloss")
                model.fit(X, y)
                probability = model.predict_proba(X)[:, 1]
                report.setdefault("xgboost_cuda_smoke", []).append({"gpu_id": gpu_id, "passed": bool(np.isfinite(probability).all())})
            except Exception as exception:
                errors.append(f"XGBoost CUDA smoke failed on GPU {gpu_id}: {exception}")
    elif args.xgboost_smoke and backend == "mps":
        warnings.append("XGBoost does not use Apple MPS in this project; it will run on CPU.")

    job_budget_audit: list[dict[str, object]] = []
    gpu_totals = {int(row["index"]): float(row.get("memory_total_mb", 0.0)) / 1024.0 for row in report.get("nvidia_smi", [])}
    for job_name, raw_request in resolved.get("job_resources", {}).items():
        request = dict(raw_request)
        cpu_threads = int(request.get("cpu_threads", resolved["cpu"]["default_threads_per_job"]))
        ram_gb = float(request.get("ram_gb", 4.0))
        gpu_memory_gb = float(request.get("gpu_memory_gb", 0.0))
        unified_memory_gb = float(request.get("unified_memory_gb", 0.0))
        row = {
            "job": job_name,
            "cpu_threads": cpu_threads,
            "ram_gb": ram_gb,
            "gpu_memory_gb": gpu_memory_gb,
            "unified_memory_gb": unified_memory_gb,
            "feasible": True,
        }
        reasons: list[str] = []
        if cpu_threads > int(resolved["cpu"]["total_thread_limit"]):
            reasons.append("cpu_threads_exceed_global_limit")
        if ram_gb > float(resolved["memory"]["total_limit_gb"]):
            reasons.append("ram_request_exceeds_global_limit")
        if backend == "cuda" and gpu_memory_gb > 0 and gpu_totals:
            reserve = float(resolved["cuda"]["reserve_memory_gb"])
            if not any(gpu_memory_gb + reserve <= total for gpu_id, total in gpu_totals.items() if gpu_id in requested_ids):
                reasons.append("gpu_memory_request_exceeds_available_gpu")
        if backend == "mps" and unified_memory_gb > float(resolved["memory"]["total_limit_gb"]):
            reasons.append("unified_memory_request_exceeds_global_limit")
        row["feasible"] = not reasons
        row["reasons"] = reasons
        job_budget_audit.append(row)
        for reason in reasons:
            errors.append(f"Resource request for {job_name} is infeasible: {reason}")
    report["job_budget_audit"] = job_budget_audit

    if backend == "cuda" and requested_ids:
        gpu_jobs = _gpu_capable_jobs(config)
        capacities = {
            gpu_id: max(
                0.0,
                gpu_totals.get(gpu_id, 0.0)
                - float(resolved["cuda"]["reserve_memory_gb"]),
            )
            for gpu_id in requested_ids
        }
        packing = _find_gpu_packing(
            gpu_jobs,
            capacities,
            int(resolved["cuda"]["max_jobs_per_gpu"]),
        )
        dense_cpu = sum(int(item["cpu_threads"]) for item in gpu_jobs)
        dense_ram = sum(float(item["ram_gb"]) for item in gpu_jobs)
        dense_report = {
            "gpu_job_count": len(gpu_jobs),
            "aggregate_cpu_threads": dense_cpu,
            "aggregate_ram_gb": dense_ram,
            "gpu_capacities_after_reserve_gb": capacities,
            "max_jobs_per_gpu": int(resolved["cuda"]["max_jobs_per_gpu"]),
            "feasible": packing is not None,
            "assignment": {
                str(gpu_id): [dict(item) for item in items]
                for gpu_id, items in (packing or {}).items()
            },
        }
        report["dense_gpu_packing_audit"] = dense_report
        if packing is None:
            errors.append(
                "Configured GPU-capable jobs cannot be packed into the selected "
                "GPU memory/slot limits."
            )
        if dense_cpu > int(resolved["cpu"]["total_thread_limit"]):
            errors.append(
                "GPU-dense phase requests more CPU threads than the global limit."
            )
        if dense_ram > float(resolved["memory"]["total_limit_gb"]):
            errors.append(
                "GPU-dense phase requests more RAM than the global limit."
            )

    snapshot = report["system_snapshot"]
    if float(snapshot["total_ram_gb"]) < float(resolved["memory"]["total_limit_gb"]):
        errors.append("Configured RAM limit exceeds physical RAM.")
    if int(snapshot["logical_cpu_count"]) < int(resolved["cpu"]["total_thread_limit"]):
        errors.append("Configured CPU thread limit exceeds visible logical CPUs.")
    report["warnings"] = warnings
    report["errors"] = errors
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
