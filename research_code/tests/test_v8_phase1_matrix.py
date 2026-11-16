from __future__ import annotations

import json
from pathlib import Path

import pytest

from los_baseline.config import load_config, resolve_outcome_horizon_days, validate_config
from los_baseline.experiment_matrix import build_matrix_plan, build_v8_run_id

ROOT = Path(__file__).resolve().parents[1]


def test_v8_config_uses_canonical_outcome_horizon():
    config = load_config(ROOT / "configs" / "experiment_cuda_v8.json")
    assert resolve_outcome_horizon_days(config) == 28.0
    assert "landmark_days" not in config["outcomes"]
    assert "threshold_days" not in config["data"]


def test_v7_config_remains_backward_compatible():
    config = load_config(ROOT / "configs" / "experiment_cuda_v7.json")
    assert resolve_outcome_horizon_days(config) == 28.0


def test_conflicting_horizon_keys_are_rejected():
    config = json.loads((ROOT / "configs" / "experiment_cuda_v8.json").read_text())
    config["outcomes"]["landmark_days"] = 14
    with pytest.raises(ValueError, match="Conflicting outcome horizon"):
        validate_config(config)


def test_default_matrix_is_three_core_runs_and_t4_not_requested():
    config = load_config(ROOT / "configs" / "experiment_cuda_v8.json")
    plan = build_matrix_plan(
        config,
        horizons=(28,),
        backend="cuda",
        timestamp="2026_0812_150000",
        include_undersampling=False,
    )
    enabled = [row for row in plan["runs"] if row["enabled"]]
    assert len(enabled) == 3
    assert plan["requested_horizons"] == [28]
    rows = plan["runs"]
    assert [r["strategy"] for r in rows] == [
        "T1_baseline", "T2_optuna", "T3_smote", "T4_undersampling"
    ]
    assert [r["status"] for r in rows] == [
        "planned", "planned", "planned", "not_requested"
    ]
    assert rows[-1]["run_id"] is None


def test_include_undersampling_expands_los28_matrix_to_four_runs():
    config = load_config(ROOT / "configs" / "experiment_cuda_v8.json")
    plan = build_matrix_plan(
        config,
        horizons=(28,),
        backend="cuda",
        timestamp="2026_0812_150000",
        include_undersampling=True,
    )
    assert sum(bool(row["enabled"]) for row in plan["runs"]) == 4
    assert all(row["status"] == "planned" for row in plan["runs"])
    assert plan["include_undersampling"] is True


def test_phase6_rejects_los14_matrix():
    config = load_config(ROOT / "configs" / "experiment_cuda_v8.json")
    with pytest.raises(ValueError, match="LOS28 only"):
        build_matrix_plan(
            config, horizons=(14,), backend="cuda", timestamp="x"
        )


def test_v8_run_id_encodes_horizon_and_strategy():
    assert build_v8_run_id(
        backend="CUDA",
        horizon_days=28,
        strategy="T3_smote",
        timestamp="2026_0812_150000",
    ) == "exp123_v8_cuda_LOS28_T3_smote_2026_0812_150000"
