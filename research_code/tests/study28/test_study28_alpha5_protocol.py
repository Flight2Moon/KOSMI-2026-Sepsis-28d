from __future__ import annotations

from pathlib import Path

from los_baseline.study28.robustness_protocol import load_robustness_protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_alpha5_protocol_scope_is_frozen_and_complete():
    p = load_robustness_protocol(PROJECT_ROOT / "configs" / "study28_robustness_interpretability.json")
    assert p.interpretability["feature_set"] == "FS_NOAPS"
    assert p.interpretability["sample_size"] == 4000
    assert p.pre_icu_los_sensitivity["excluded_features"] == ["pre_icu_los_d"]
    assert set(p.feature_availability_sensitivity["sites"]) == {"eicu_alt", "eicu_primary", "sicdb", "nwicu"}
    assert p.hfd28_secondary["bootstrap_repeats"] == 1000
    assert p.home_discharge_sensitivity["required_column"] == "discharge_location"
    assert p.home_discharge_sensitivity["do_not_infer_from_other_columns"] is True
