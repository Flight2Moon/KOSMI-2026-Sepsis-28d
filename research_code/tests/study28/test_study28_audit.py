from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from los_baseline.study28.audit import audit_study_datasets


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_feature_list(root: Path) -> list[str]:
    features = ["age", "apsiii", "gcs_score"]
    pd.DataFrame({"order": [1, 2, 3], "feature": features}).to_csv(
        root / "feature_list.csv", index=False
    )
    return features


def _frame(ids: list[int], *, filter_col: str | None = None) -> pd.DataFrame:
    labels = ["early_dc", "death", "prolonged"][: len(ids)]
    ys = [0, 1, 2][: len(ids)]
    frame = pd.DataFrame(
        {
            "id": ids,
            "label28": labels,
            "y": ys,
            "hosp_los_d": [5.0, 10.0, 30.0][: len(ids)],
            "hospital_expire_flag": [0, 1, 0][: len(ids)],
            "age": [60, 70, 80][: len(ids)],
            "apsiii": [10.0, None, 30.0][: len(ids)],
            "gcs_score": [15.0, 8.0, None][: len(ids)],
        }
    )
    if filter_col:
        frame[filter_col] = 1
    return frame


def test_audit_passes_without_mutating_sources(tmp_path: Path):
    _write_feature_list(tmp_path)
    paths = {}
    for name in ["mimic", "sicdb", "nwicu"]:
        path = tmp_path / f"{name}.csv"
        _frame([1, 2, 3]).to_csv(path, index=False)
        paths[name] = path
    eicu = tmp_path / "eicu.csv"
    ef = _frame([1, 2, 3])
    ef["sepsis_admitdx"] = 1
    ef["sepsis3_primary"] = [1, 1, 0]
    ef.to_csv(eicu, index=False)
    before = {path: _sha(path) for path in [*paths.values(), eicu]}

    registry = {
        "datasets": {
            "mimic": {"file": "mimic.csv", "role": "development", "id_columns": ["id"], "expected_sha256": _sha(paths["mimic"])},
            "eicu_alt": {"file": "eicu.csv", "role": "external_primary", "id_columns": ["id"], "expected_sha256": _sha(eicu), "filter": {"column": "sepsis_admitdx", "equals": 1}, "required_external": True},
            "eicu_primary": {"file": "eicu.csv", "role": "external_sensitivity", "id_columns": ["id"], "expected_sha256": _sha(eicu), "filter": {"column": "sepsis3_primary", "equals": 1}, "required_external": True},
            "sicdb": {"file": "sicdb.csv", "role": "external_primary", "id_columns": ["id"], "expected_sha256": _sha(paths["sicdb"]), "required_external": True},
            "nwicu": {"file": "nwicu.csv", "role": "external_primary", "id_columns": ["id"], "expected_sha256": _sha(paths["nwicu"]), "required_external": True}
        },
        "required_external_sites": ["eicu_alt", "eicu_primary", "sicdb", "nwicu"]
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry))

    result = audit_study_datasets(
        dataset_root=tmp_path,
        registry_path=registry_path,
        output_dir=tmp_path / "audit_out",
    )
    assert result.passed
    assert {item.dataset for item in result.datasets} == {
        "mimic", "eicu_alt", "eicu_primary", "sicdb", "nwicu"
    }
    assert next(item for item in result.datasets if item.dataset == "eicu_primary").row_count == 2
    after = {path: _sha(path) for path in before}
    assert before == after


def test_hash_mismatch_blocks_audit(tmp_path: Path):
    _write_feature_list(tmp_path)
    frame = _frame([1, 2, 3])
    for name in ["mimic", "eicu", "sicdb", "nwicu"]:
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    registry = {
        "datasets": {
            "mimic": {"file": "mimic.csv", "role": "development", "id_columns": ["id"], "expected_sha256": "0" * 64},
            "eicu_alt": {"file": "eicu.csv", "role": "external_primary", "id_columns": ["id"], "required_external": True},
            "eicu_primary": {"file": "eicu.csv", "role": "external_sensitivity", "id_columns": ["id"], "required_external": True},
            "sicdb": {"file": "sicdb.csv", "role": "external_primary", "id_columns": ["id"], "required_external": True},
            "nwicu": {"file": "nwicu.csv", "role": "external_primary", "id_columns": ["id"], "required_external": True}
        },
        "required_external_sites": ["eicu_alt", "eicu_primary", "sicdb", "nwicu"]
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry))
    result = audit_study_datasets(dataset_root=tmp_path, registry_path=registry_path)
    assert not result.passed
    mimic = next(item for item in result.datasets if item.dataset == "mimic")
    assert "source_sha256_mismatch" in mimic.blocking_reasons
