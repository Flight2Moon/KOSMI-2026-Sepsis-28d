from __future__ import annotations

import os
from typing import Any, Mapping

import numpy as np


def requested_device(model_config: Mapping[str, Any] | None = None) -> str:
    config = dict(model_config or {})
    parameters = dict(config.get("parameters", {}))
    device = str(parameters.get("device", config.get("device", "cpu")))
    assigned = os.environ.get("EXP123_CUDA_DEVICE")
    if assigned not in {None, ""} and device.startswith("cuda"):
        return f"cuda:{int(assigned)}"
    return device


def _cupy_module() -> Any | None:
    try:
        import cupy as cp

        return cp
    except Exception:
        return None


def predict_proba_device_aware(
    model: Any,
    features: np.ndarray,
    *,
    model_config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict with a matching CUDA array when CuPy is available.

    The fallback remains valid and reproducible, but XGBoost may emit a device
    mismatch warning and allocate a temporary DMatrix on the booster device.
    """
    device = requested_device(model_config)
    metadata: dict[str, Any] = {
        "requested_device": device,
        "gpu_input_used": False,
        "fallback": None,
    }
    if device.startswith("cuda"):
        cp = _cupy_module()
        if cp is not None:
            try:
                ordinal = int(device.split(":", 1)[1]) if ":" in device else 0
                with cp.cuda.Device(ordinal):
                    gpu_features = cp.asarray(features, dtype=cp.float32)
                    output = model.predict_proba(gpu_features)
                    metadata["gpu_input_used"] = True
                    return np.asarray(cp.asnumpy(output), dtype=np.float64), metadata
            except Exception as exception:
                metadata["fallback"] = (
                    "cupy_prediction_failed: "
                    f"{type(exception).__name__}: {exception}"
                )
        else:
            metadata["fallback"] = "cupy_not_installed"
    output = model.predict_proba(features)
    return np.asarray(output, dtype=np.float64), metadata
