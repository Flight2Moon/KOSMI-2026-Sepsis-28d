from __future__ import annotations

import json
from pathlib import Path

from los_baseline.config import validate_config
from los_baseline.resource_manager import request_for_job


def test_m1max_profile_uses_mps_for_torch_and_cpu_for_xgboost() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "experiment_mps_m1max_32gb_mortality_los.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)

    assert config["hardware"]["profile_name"] == "apple_m1_max_32gb_mps_optimized"
    assert config["resources"]["accelerator"]["backend"] == "mps"
    assert config["resources"]["memory"]["total_limit_gb"] == 28.0
    assert config["resources"]["mps"]["max_concurrent_jobs"] == 2
    assert config["resources"]["mps"]["pytorch_memory_fraction"] == 0.28

    logistic = request_for_job(
        config,
        model_name="logistic_regression",
        job_kind="optimization",
    )
    xgboost = request_for_job(
        config,
        model_name="xgboost",
        job_kind="optimization",
    )
    assert logistic.accelerator == "mps"
    assert logistic.unified_memory_gb == 6
    assert xgboost.accelerator == "cpu"


def test_m1max_profile_enables_date_based_los_derivation() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            root
            / "configs"
            / "experiment_mps_m1max_32gb_mortality_los.json"
        ).read_text(encoding="utf-8")
    )
    policy = config["outcomes"]["los_derivation"]
    assert policy["enabled"] is True
    assert policy["admission_time_column"] == "admittime"
    assert policy["discharge_time_column"] == "dischtime"
    assert policy["in_hospital_death_time_column"] == "dischtime"
    assert policy["overwrite_existing_los"] is True
    assert policy["allow_existing_los_fallback"] is False
