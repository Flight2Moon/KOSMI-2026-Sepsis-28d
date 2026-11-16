from __future__ import annotations

import copy
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    memory_total_mb: float | None
    compute_capability: str | None


@dataclass(frozen=True)
class HardwareReport:
    platform: str
    python: str
    cuda_available: bool
    cuda_runtime: str | None
    cudnn_version: int | None
    gpu_count: int
    gpus: tuple[GPUInfo, ...]
    assigned_gpu: int | None
    xgboost_cuda_requested: bool
    pytorch_cuda_requested: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _assigned_gpu() -> int | None:
    value = os.environ.get("EXP123_CUDA_DEVICE")
    if value is None or value.strip() == "":
        return None
    return int(value)


def detect_hardware(config: Mapping[str, Any] | None = None) -> HardwareReport:
    import sys

    hardware = dict((config or {}).get("hardware", {}))
    cuda_config = dict(hardware.get("cuda", {}))
    gpus: list[GPUInfo] = []
    cuda_available = False
    cuda_runtime = None
    cudnn_version = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_runtime = getattr(torch.version, "cuda", None)
        cudnn_version = (
            int(torch.backends.cudnn.version())
            if cuda_available and torch.backends.cudnn.is_available()
            else None
        )
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                capability = torch.cuda.get_device_capability(index)
                gpus.append(
                    GPUInfo(
                        index=index,
                        name=str(properties.name),
                        memory_total_mb=float(properties.total_memory / (1024**2)),
                        compute_capability=f"{capability[0]}.{capability[1]}",
                    )
                )
    except Exception:
        pass
    return HardwareReport(
        platform=platform.platform(),
        python=sys.version.split()[0],
        cuda_available=cuda_available,
        cuda_runtime=cuda_runtime,
        cudnn_version=cudnn_version,
        gpu_count=len(gpus),
        gpus=tuple(gpus),
        assigned_gpu=_assigned_gpu(),
        xgboost_cuda_requested=bool(cuda_config.get("xgboost", True)),
        pytorch_cuda_requested=bool(cuda_config.get("pytorch", True)),
    )


def _cuda_enabled(config: Mapping[str, Any]) -> bool:
    cuda = dict(config.get("hardware", {}).get("cuda", {}))
    mode = str(cuda.get("mode", "auto")).lower()
    if mode == "disabled":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def selected_cuda_ordinal(config: Mapping[str, Any]) -> int:
    assigned = _assigned_gpu()
    if assigned is not None:
        return assigned
    gpu_ids = list(config.get("hardware", {}).get("cuda", {}).get("gpu_ids", [0]))
    return int(gpu_ids[0]) if gpu_ids else 0


def _accelerator_backend(config: Mapping[str, Any]) -> str:
    worker_backend = os.environ.get("EXP123_ACCELERATOR_BACKEND", "").strip().lower()
    if worker_backend in {"cpu", "cuda", "mps"}:
        return worker_backend
    resources = dict(config.get("resources", {}))
    backend = str(resources.get("accelerator", {}).get("backend", "auto")).lower()
    if backend != "auto":
        return backend
    hardware = dict(config.get("hardware", {}))
    configured = str(hardware.get("accelerator", {}).get("backend", "auto")).lower()
    if configured != "auto":
        return configured
    cuda_mode = str(hardware.get("cuda", {}).get("mode", "auto")).lower()
    if cuda_mode != "disabled" and _cuda_enabled(config):
        return "cuda"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def apply_acceleration_to_model_config(
    *,
    project_config: Mapping[str, Any],
    model_name: str,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    output = copy.deepcopy(dict(model_config))
    hardware = dict(project_config.get("hardware", {}))
    cuda = dict(hardware.get("cuda", {}))
    mps = dict(project_config.get("resources", {}).get("mps", {}))
    cpu = dict(hardware.get("cpu", {}))
    backend = _accelerator_backend(project_config)
    cuda_active = backend == "cuda" and _cuda_enabled(project_config)
    ordinal = selected_cuda_ordinal(project_config)
    cpu_threads = max(1, int(os.environ.get("EXP123_ASSIGNED_CPU_THREADS", cpu.get("threads_per_worker", 1))))

    if model_name == "logistic_regression":
        retry_batch_scale = float(os.environ.get("EXP123_RETRY_BATCH_SIZE_SCALE", "1.0"))
        if retry_batch_scale < 1.0 and "batch_size" in output:
            output["batch_size"] = max(1, int(float(output["batch_size"]) * retry_batch_scale))
        if cuda_active and bool(cuda.get("pytorch", True)):
            output["device"] = f"cuda:{ordinal}"
            output["pin_memory"] = bool(cuda.get("pin_memory", True))
            output["non_blocking"] = bool(cuda.get("non_blocking", True))
            output["data_loader_workers"] = int(cuda.get("data_loader_workers", 2))
            output["persistent_workers"] = bool(cuda.get("persistent_workers", True))
            output["mixed_precision"] = bool(cuda.get("mixed_precision", False))
            output["allow_tf32"] = bool(cuda.get("allow_tf32", False))
            output["deterministic"] = bool(cuda.get("deterministic", True))
        elif backend == "mps":
            output["device"] = "mps"
            output["pin_memory"] = False
            output["non_blocking"] = False
            output["data_loader_workers"] = int(mps.get("data_loader_workers", 0))
            output["persistent_workers"] = bool(output["data_loader_workers"] > 0)
            output["mixed_precision"] = False
            output["allow_tf32"] = False
            output["deterministic"] = bool(mps.get("deterministic", True))
        else:
            output["device"] = "cpu"
            output["pin_memory"] = False
            output["non_blocking"] = False
            output["data_loader_workers"] = 0
            output["persistent_workers"] = False
            output["mixed_precision"] = False
            output["allow_tf32"] = False
            output["deterministic"] = True
    elif model_name == "xgboost":
        parameters = dict(output.get("parameters", {}))
        use_cuda = cuda_active and bool(cuda.get("xgboost", True))
        parameters["device"] = f"cuda:{ordinal}" if use_cuda else "cpu"
        parameters["tree_method"] = "hist"
        parameters["n_jobs"] = cpu_threads
        output["parameters"] = parameters
    elif model_name == "random_forest":
        parameters = dict(output.get("parameters", {}))
        parameters["n_jobs"] = cpu_threads
        output["parameters"] = parameters
    return output

def configure_torch_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"configured": False}
    try:
        import torch

        cuda = dict(config.get("hardware", {}).get("cuda", {}))
        deterministic = bool(cuda.get("deterministic", True))
        allow_tf32 = bool(cuda.get("allow_tf32", False))
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        cuda_fraction = os.environ.get("EXP123_PYTORCH_CUDA_MEMORY_FRACTION")
        assigned = _assigned_gpu()
        if cuda_fraction and torch.cuda.is_available():
            ordinal = assigned if assigned is not None else 0
            torch.cuda.memory.set_per_process_memory_fraction(float(cuda_fraction), ordinal)
        mps_fraction = os.environ.get("EXP123_MPS_MEMORY_FRACTION")
        mps_configured = False
        if (
            mps_fraction
            and hasattr(torch, "mps")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and hasattr(torch.mps, "set_per_process_memory_fraction")
        ):
            torch.mps.set_per_process_memory_fraction(float(mps_fraction))
            mps_configured = True
        result = {
            "configured": True,
            "deterministic": deterministic,
            "allow_tf32": allow_tf32,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_memory_fraction": None if cuda_fraction is None else float(cuda_fraction),
            "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
            "mps_memory_fraction": None if mps_fraction is None else float(mps_fraction),
            "mps_memory_limit_configured": mps_configured,
        }
    except Exception as exception:
        result = {"configured": False, "error": str(exception)}
    return result

def nvidia_smi_snapshot() -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    query = (
        "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    )
    completed = subprocess.run(
        [
            executable,
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 7:
            continue
        rows.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "utilization_gpu_percent": float(values[2]),
                "memory_used_mb": float(values[3]),
                "memory_total_mb": float(values[4]),
                "temperature_c": float(values[5]),
                "power_draw_w": float(values[6].split()[0]),
            }
        )
    return rows


def write_hardware_report(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    report = detect_hardware(config).to_dict()
    report["nvidia_smi"] = nvidia_smi_snapshot()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def validate_cuda_acceleration(
    config: Mapping[str, Any],
    *,
    run_xgboost_smoke: bool = True,
) -> dict[str, Any]:
    """Run small CUDA computations before a long clinical experiment starts."""
    cuda = dict(config.get("hardware", {}).get("cuda", {}))
    required = bool(cuda.get("required", False)) or str(cuda.get("mode", "auto")).lower() == "required"
    gpu_ids = [int(value) for value in cuda.get("gpu_ids", [0])]
    report: dict[str, Any] = {
        "required": required,
        "gpu_ids": gpu_ids,
        "pytorch": [],
        "xgboost": [],
        "errors": [],
    }
    try:
        import torch
    except Exception as exception:
        if required:
            report["errors"].append(f"PyTorch import failed: {exception}")
        return report
    if not torch.cuda.is_available():
        if required:
            report["errors"].append("torch.cuda.is_available() is False.")
        return report
    if torch.cuda.device_count() <= max(gpu_ids, default=0):
        report["errors"].append(
            f"Requested GPU IDs {gpu_ids}, but only {torch.cuda.device_count()} device(s) are visible."
        )
        return report
    for gpu_id in gpu_ids:
        try:
            left = torch.arange(2048, device=f"cuda:{gpu_id}", dtype=torch.float32)
            right = torch.ones_like(left)
            result = float(torch.dot(left, right).cpu())
            torch.cuda.synchronize(gpu_id)
            report["pytorch"].append(
                {"gpu_id": gpu_id, "passed": bool(result > 0), "result": result}
            )
        except Exception as exception:
            report["errors"].append(
                f"PyTorch CUDA smoke failed on GPU {gpu_id}: {exception}"
            )
    if run_xgboost_smoke and bool(cuda.get("xgboost", True)):
        try:
            import numpy as np
            from xgboost import XGBClassifier

            X = np.asarray(
                [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32
            )
            y = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
            for gpu_id in gpu_ids:
                try:
                    model = XGBClassifier(
                        n_estimators=4,
                        max_depth=2,
                        tree_method="hist",
                        device=f"cuda:{gpu_id}",
                        eval_metric="logloss",
                        verbosity=0,
                    )
                    model.fit(X, y)
                    probability = model.predict_proba(X)[:, 1]
                    report["xgboost"].append(
                        {
                            "gpu_id": gpu_id,
                            "passed": bool(np.isfinite(probability).all()),
                        }
                    )
                except Exception as exception:
                    report["errors"].append(
                        f"XGBoost CUDA smoke failed on GPU {gpu_id}: {exception}"
                    )
        except Exception as exception:
            report["errors"].append(f"XGBoost import failed: {exception}")
    return report
