from __future__ import annotations

from pathlib import Path

import pytest

from los_baseline.study28.external_protocol import load_external_protocol
from los_baseline.study28.protocol import load_study28_protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_external_protocol_preserves_alpha2_frozen_protocol_hash():
    primary = load_study28_protocol(PROJECT_ROOT / "configs" / "study28_final.json")
    external = load_external_protocol(
        PROJECT_ROOT / "configs" / "study28_external_validation.json"
    )
    assert external.frozen_protocol_sha256 == primary.sha256
    assert external.required_external_sites == (
        "eicu_alt",
        "eicu_primary",
        "sicdb",
        "nwicu",
    )
    assert external.bootstrap.repeats == 1000
    assert external.bootstrap.confidence_level == pytest.approx(0.95)
    assert external.calibration.bins == 10
    assert external.outputs.include_patient_level_predictions_in_release_zip is False
