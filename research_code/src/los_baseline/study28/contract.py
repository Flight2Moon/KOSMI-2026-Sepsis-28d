from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

LABEL_TO_Y = {"early_dc": 0, "death": 1, "prolonged": 2}
Y_TO_LABEL = {value: key for key, value in LABEL_TO_Y.items()}

# These columns may be useful for auditing, subgroup analysis, temporal splits,
# or reporting, but must never enter the model feature matrix.
PROHIBITED_MODEL_COLUMNS = frozenset(
    {
        "label28",
        "y",
        "hosp_los_d",
        "hospital_expire_flag",
        "subject_id",
        "hadm_id",
        "stay_id",
        "patientunitstayid",
        "patienthealthsystemstayid",
        "hospitalid",
        "anchor_year_group",
        "caseid",
        "patientid",
        "admissionyear",
        "sepsis3_primary",
        "sepsis_admitdx",
        "source_db",
    }
)


@dataclass(frozen=True)
class FeatureContract:
    columns: tuple[str, ...]
    sha256: str
    source_path: str

    @property
    def no_apsiii(self) -> tuple[str, ...]:
        return tuple(column for column in self.columns if column != "apsiii")

    @property
    def apsiii_only(self) -> tuple[str, ...]:
        return ("apsiii",)

    def validate_model_features(self, columns: Sequence[str]) -> None:
        supplied = tuple(str(column) for column in columns)
        leaked = sorted(set(supplied) & PROHIBITED_MODEL_COLUMNS)
        if leaked:
            raise ValueError(f"Prohibited leakage/metadata columns in model features: {leaked}")
        unknown = sorted(set(supplied) - set(self.columns))
        if unknown:
            raise ValueError(f"Unknown model features outside the frozen feature contract: {unknown}")
        if len(supplied) != len(set(supplied)):
            raise ValueError("Model feature list contains duplicate columns.")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_contract(feature_list_path: Path) -> FeatureContract:
    feature_list_path = Path(feature_list_path).expanduser().resolve()
    if not feature_list_path.is_file():
        raise FileNotFoundError(f"Feature list was not found: {feature_list_path}")

    with feature_list_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "feature" not in reader.fieldnames:
            raise ValueError("feature_list.csv must contain a 'feature' column.")
        rows = list(reader)

    if not rows:
        raise ValueError("feature_list.csv is empty.")

    if "order" in rows[0]:
        try:
            rows = sorted(rows, key=lambda row: int(str(row["order"]).strip()))
        except (TypeError, ValueError) as exc:
            raise ValueError("feature_list.csv contains an invalid order column.") from exc

    columns = tuple(str(row["feature"]).strip() for row in rows)
    if any(not column for column in columns):
        raise ValueError("feature_list.csv contains an empty feature name.")
    if len(columns) != len(set(columns)):
        raise ValueError("feature_list.csv contains duplicate feature names.")
    leaked = sorted(set(columns) & PROHIBITED_MODEL_COLUMNS)
    if leaked:
        raise ValueError(f"Frozen feature list contains prohibited model columns: {leaked}")

    return FeatureContract(
        columns=columns,
        sha256=sha256_file(feature_list_path),
        source_path=str(feature_list_path),
    )


def assert_exact_feature_order(actual: Iterable[str], expected: Sequence[str]) -> None:
    actual_tuple = tuple(str(value) for value in actual)
    expected_tuple = tuple(str(value) for value in expected)
    if actual_tuple != expected_tuple:
        first_difference = None
        for index, (left, right) in enumerate(zip(actual_tuple, expected_tuple), start=1):
            if left != right:
                first_difference = (index, left, right)
                break
        if first_difference is None and len(actual_tuple) != len(expected_tuple):
            first_difference = (
                min(len(actual_tuple), len(expected_tuple)) + 1,
                "<missing>" if len(actual_tuple) < len(expected_tuple) else actual_tuple[len(expected_tuple)],
                "<missing>" if len(expected_tuple) < len(actual_tuple) else expected_tuple[len(actual_tuple)],
            )
        raise ValueError(
            "Feature order differs from the frozen feature_list.csv. "
            f"First difference={first_difference}; actual_count={len(actual_tuple)}; "
            f"expected_count={len(expected_tuple)}"
        )
