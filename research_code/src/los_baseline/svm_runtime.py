from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SVM_BACKEND_EXACT_RBF = "exact_rbf"
SVM_BACKEND_NYSTROEM_LINEAR = "nystroem_linear"
SVM_BACKEND_LEGACY = "legacy_svc"


@dataclass(frozen=True)
class SVMRuntimePolicy:
    enabled: bool = False
    max_fold_fit_seconds: float = 0.0
    exact_rbf_max_iter: int = 2_000_000
    linear_svc_max_iter: int = 20_000
    record_fit_diagnostics: bool = True


def resolve_svm_backend(model_config: Mapping[str, Any] | None) -> str:
    config = dict(model_config or {})
    parameters = dict(config.get("parameters", {}))
    backend = str(parameters.get("svm_backend", "")).strip().lower()
    if backend in {SVM_BACKEND_EXACT_RBF, SVM_BACKEND_NYSTROEM_LINEAR}:
        return backend
    # Backward-compatible V7/V8 configs expressed the SVC kernel directly.
    return SVM_BACKEND_LEGACY


def svm_runtime_settings(project_config: Mapping[str, Any] | None) -> dict[str, Any]:
    config = dict(project_config or {})
    raw = dict(config.get("svm_runtime_optimization", {}))
    exact = dict(raw.get("exact_rbf", {}))
    approx = dict(raw.get("nystroem_linear", {}))
    return {
        "enabled": bool(raw.get("enabled", False)),
        "search_backends": [str(v) for v in raw.get("search_backends", [SVM_BACKEND_EXACT_RBF, SVM_BACKEND_NYSTROEM_LINEAR])],
        "max_fold_fit_seconds": max(0.0, float(raw.get("max_fold_fit_seconds", 0.0))),
        "exact_rbf": {
            "c_min": float(exact.get("c_min", 1e-2)),
            "c_max": float(exact.get("c_max", 1e2)),
            "gamma_min": float(exact.get("gamma_min", 1e-4)),
            "gamma_max": float(exact.get("gamma_max", 1.0)),
            "max_iter": max(1, int(exact.get("max_iter", 2_000_000))),
            "tol": float(exact.get("tol", 1e-3)),
        },
        "nystroem_linear": {
            "c_min": float(approx.get("c_min", 1e-2)),
            "c_max": float(approx.get("c_max", 1e2)),
            "gamma_min": float(approx.get("gamma_min", 1e-4)),
            "gamma_max": float(approx.get("gamma_max", 1.0)),
            "n_components_min": max(2, int(approx.get("n_components_min", 256))),
            "n_components_max": max(2, int(approx.get("n_components_max", 2048))),
            "max_iter": max(1, int(approx.get("max_iter", 20_000))),
            "tol": float(approx.get("tol", 1e-4)),
        },
        "benchmark": dict(raw.get("benchmark", {})),
    }


def svm_backend_label(model_config: Mapping[str, Any] | None) -> str:
    backend = resolve_svm_backend(model_config)
    if backend != SVM_BACKEND_LEGACY:
        return backend
    parameters = dict(dict(model_config or {}).get("parameters", {}))
    return f"legacy_svc_{parameters.get('kernel', 'rbf')}"
