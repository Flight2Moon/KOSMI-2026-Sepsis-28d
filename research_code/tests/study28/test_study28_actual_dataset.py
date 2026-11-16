from __future__ import annotations

import os
from pathlib import Path

import pytest

from los_baseline.study28.audit import audit_study_datasets

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_professor_frozen_dataset_contract_if_available(tmp_path: Path):
    root = os.environ.get("STUDY28_DATA_ROOT")
    if not root:
        pytest.skip("STUDY28_DATA_ROOT is not set")
    result = audit_study_datasets(
        dataset_root=Path(root),
        registry_path=PROJECT_ROOT / "configs" / "dataset_registry_28d.json",
        output_dir=tmp_path,
    )
    assert result.passed, result.blocking_reasons
    by_name = {item.dataset: item for item in result.datasets}
    assert by_name["mimic"].row_count == 22741
    assert by_name["eicu_alt"].row_count == 13198
    assert by_name["eicu_primary"].row_count == 2367
    assert by_name["sicdb"].row_count == 9122
    assert by_name["nwicu"].row_count == 3829
    assert by_name["sicdb"].consistency_warning_count == 83

@pytest.mark.integration
def test_professor_mimic_temporal_split_contract_if_available():
    root = os.environ.get("STUDY28_DATA_ROOT")
    if not root:
        pytest.skip("STUDY28_DATA_ROOT is not set")
    from los_baseline.study28.datasets import load_dataset_registry
    from los_baseline.study28.development import split_mimic_temporal
    from los_baseline.study28.protocol import load_study28_protocol

    registry = load_dataset_registry(PROJECT_ROOT / "configs" / "dataset_registry_28d.json")
    mimic = registry["mimic"].load(Path(root))
    protocol = load_study28_protocol(PROJECT_ROOT / "configs" / "study28_final.json")
    development, evaluation = split_mimic_temporal(mimic, protocol)
    assert len(development) == 17009
    assert len(evaluation) == 5732
