from __future__ import annotations

import os
from pathlib import Path

import pytest

from los_baseline.study28.sensitivity import validate_alpha4_contract

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_alpha4_contract_against_real_outputs_if_available(tmp_path: Path):
    data_root = os.environ.get("STUDY28_DATA_ROOT")
    alpha2_root = os.environ.get("STUDY28_ALPHA2_OUTPUT")
    alpha3_root = os.environ.get("STUDY28_ALPHA3_OUTPUT")
    if not data_root or not alpha2_root or not alpha3_root:
        pytest.skip("Study28 real-data environment variables are not set")
    c = validate_alpha4_contract(
        dataset_root=Path(data_root), project_root=PROJECT_ROOT,
        alpha2_output_dir=Path(alpha2_root), alpha3_output_dir=Path(alpha3_root),
        audit_output_dir=tmp_path,
    )
    assert c.alpha3_manifest["all_required_external_complete"] is True
    assert c.frozen.model_sha256 == c.alpha3_manifest["frozen_model_sha256"]
