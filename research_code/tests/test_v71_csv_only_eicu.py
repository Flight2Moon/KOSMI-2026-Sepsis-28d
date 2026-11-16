from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from los_baseline.audit import attach_eicu_harmonizer_audit
from los_baseline.config import validate_config
from los_baseline.data import ClinicalDataset


def _dataset(tmp_path: Path) -> ClinicalDataset:
    frame = pd.DataFrame({
        "_source_row_index": [0, 1],
        "subject_id": [1, 2], "hadm_id": [11, 12], "stay_id": [21, 22],
        "x": [0.1, 0.2], "LOS": [5.0, 8.0],
    })
    csv_path = tmp_path / "eicu.csv"
    frame.drop(columns=["_source_row_index"]).to_csv(csv_path, index=False)
    return ClinicalDataset(
        frame=frame, feature_columns=["x"], target_column="LOS",
        identifier_columns=["subject_id", "hadm_id", "stay_id"],
        source_path=csv_path, removed_repeated_headers=0,
        removed_missing_targets=0,
    )


def test_disabled_provenance_needs_no_file(tmp_path: Path) -> None:
    result = attach_eicu_harmonizer_audit(
        dataset=_dataset(tmp_path), configured_path=None,
        project_root=tmp_path, output_dir=tmp_path / "audit", enabled=False,
    )
    assert result["status"] == "disabled"
    assert (tmp_path / "audit" / "eicu_harmonizer_audit.json").is_file()


def test_optional_csv_provenance_is_supported(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    provenance = dataset.frame[["subject_id", "hadm_id", "stay_id"]].copy()
    provenance["imputed_feature_count"] = [0, 1]
    provenance_path = tmp_path / "provenance.csv.gz"
    provenance.to_csv(provenance_path, index=False, compression="gzip")
    result = attach_eicu_harmonizer_audit(
        dataset=dataset, configured_path=str(provenance_path),
        project_root=tmp_path, output_dir=tmp_path / "audit", enabled=True,
    )
    assert result["status"] == "attached"


def test_parquet_provenance_is_rejected_by_config() -> None:
    config = {
        "paths": {},
        "data": {"feature_columns": ["x"], "threshold_days": [28]},
        "split": {"train_ratio": 0.7, "validation_ratio": 0.15, "test_ratio": 0.15},
        "models": {}, "evaluation": {}, "explainability": {}, "execution": {},
        "audit": {"eicu_harmonizer_provenance": {
            "enabled": True, "path": "audit.parquet", "required": False
        }},
    }
    with pytest.raises(ValueError, match="CSV/CSV.GZ"):
        validate_config(config)
