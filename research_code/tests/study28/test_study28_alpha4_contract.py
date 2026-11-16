from __future__ import annotations

from pathlib import Path

from los_baseline.study28.sensitivity_protocol import load_sensitivity_protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_alpha4_protocol_has_expected_professor_based_scope():
    p = load_sensitivity_protocol(PROJECT_ROOT / "configs" / "study28_sensitivity_analysis.json")
    assert p.reverse_validation.pairs == (("eicu_alt", "mimic"), ("eicu_primary", "mimic"))
    assert p.reverse_validation.feature_set == "FS_NOAPS"
    assert p.local_recalibration.calibration_fraction == 0.2
    assert p.local_recalibration.seeds == 20
    assert set(p.local_recalibration.sites) == {"eicu_alt", "eicu_primary", "sicdb", "nwicu"}
    assert p.dca.classes == ("death", "prolonged")
    assert p.hospital_heterogeneity.site == "eicu_alt"
    assert p.hospital_heterogeneity.minimum_hospital_n == 50
    assert p.hospital_heterogeneity.minimum_events == 5
