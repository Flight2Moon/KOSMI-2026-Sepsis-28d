from __future__ import annotations

import json
import os
from pathlib import Path

from los_baseline.hardware import apply_acceleration_to_model_config
from los_baseline.progress import configure_progress, emit_progress, progress_iter


def _config() -> dict:
    return {
        "progress": {"enabled": True, "tqdm": False},
        "hardware": {
            "cpu": {"threads_per_worker": 4},
            "cuda": {
                "mode": "disabled",
                "gpu_ids": [0, 1],
                "pytorch": True,
                "xgboost": True,
            },
        },
    }


def test_progress_jsonl_and_current_status(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("EXP123_DISABLE_PROGRESS", raising=False)
    configure_progress(
        output_dir=tmp_path,
        run_id="test_run",
        config=_config(),
        worker_label="unit_test",
    )
    emit_progress(
        "step_started",
        operation="test_step",
        message="running",
        status="running",
    )
    list(progress_iter(range(3), total=3, description="items", unit="item"))
    current = json.loads((tmp_path / "current_unit_test.json").read_text())
    assert current["event"] == "operation_completed"
    assert current["current"] == 3
    assert (tmp_path / "progress_events.jsonl").is_file()


def test_xgboost_cpu_profile_preserves_hist_and_threads() -> None:
    result = apply_acceleration_to_model_config(
        project_config=_config(),
        model_name="xgboost",
        model_config={"parameters": {"n_estimators": 10}},
    )
    assert result["parameters"]["device"] == "cpu"
    assert result["parameters"]["tree_method"] == "hist"
    assert result["parameters"]["n_jobs"] == 4


def test_random_forest_threads_are_bounded() -> None:
    result = apply_acceleration_to_model_config(
        project_config=_config(),
        model_name="random_forest",
        model_config={"parameters": {"n_estimators": 10, "n_jobs": -1}},
    )
    assert result["parameters"]["n_jobs"] == 4


def test_logistic_cpu_profile_disables_cuda_transfer_options() -> None:
    result = apply_acceleration_to_model_config(
        project_config=_config(),
        model_name="logistic_regression",
        model_config={"device": "auto"},
    )
    assert result["pin_memory"] is False
    assert result["non_blocking"] is False
    assert result["data_loader_workers"] == 0


def test_xgboost_cuda_profile_uses_assigned_gpu(monkeypatch) -> None:
    import los_baseline.hardware as hardware

    config = _config()
    config["hardware"]["cuda"]["mode"] = "required"
    monkeypatch.setattr(hardware, "_cuda_enabled", lambda _: True)
    monkeypatch.setenv("EXP123_CUDA_DEVICE", "1")
    result = hardware.apply_acceleration_to_model_config(
        project_config=config,
        model_name="xgboost",
        model_config={"parameters": {"n_estimators": 10}},
    )
    assert result["parameters"]["device"] == "cuda:1"
    assert result["parameters"]["tree_method"] == "hist"


def test_logistic_cuda_profile_enables_transfer_options(monkeypatch) -> None:
    import los_baseline.hardware as hardware

    config = _config()
    config["hardware"]["cuda"].update(
        {
            "mode": "required",
            "pin_memory": True,
            "non_blocking": True,
            "data_loader_workers": 3,
            "persistent_workers": True,
        }
    )
    monkeypatch.setattr(hardware, "_cuda_enabled", lambda _: True)
    monkeypatch.setenv("EXP123_CUDA_DEVICE", "0")
    result = hardware.apply_acceleration_to_model_config(
        project_config=config,
        model_name="logistic_regression",
        model_config={"device": "auto"},
    )
    assert result["device"] == "cuda:0"
    assert result["pin_memory"] is True
    assert result["non_blocking"] is True
    assert result["data_loader_workers"] == 3
    assert result["persistent_workers"] is True
