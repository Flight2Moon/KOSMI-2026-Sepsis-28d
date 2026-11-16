from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ..exp123_data import TaskSplitData
from ..utils import atomic_write_json, sha256_file
from .sampling import SampledTrainingData, TrainingSampler, _natural_provenance, assert_natural_split


@dataclass(frozen=True)
class SyntheticPoolManifest:
    schema_version: str
    strategy: str
    minority_class: int
    multiplier: float
    seed: int
    k_neighbors: int
    original_minority_count: int
    synthetic_count: int
    target_minority_count: int
    feature_names: tuple[str, ...]
    binary_columns: tuple[str, ...]
    integer_columns: tuple[str, ...]
    clamped_value_counts: dict[str, int]
    source_split: str
    pool_sha256: str
    provenance_sha256: str
    generation_method: str = "minority_knn_linear_interpolation"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_names"] = list(self.feature_names)
        payload["binary_columns"] = list(self.binary_columns)
        payload["integer_columns"] = list(self.integer_columns)
        return payload


def _frame_sha256(frame: pd.DataFrame) -> str:
    # Stable semantic hash across CSV round-trips. Synthetic values are stored
    # with more precision than the downstream models require while avoiding
    # platform-specific float repr differences.
    payload = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        na_rep="<NA>",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _infer_binary_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    binary: list[str] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
        if len(values) and set(np.asarray(values, dtype=float).tolist()).issubset({0.0, 1.0}):
            binary.append(str(column))
    return binary


def _validate_multiplier(multiplier: float) -> float:
    value = float(multiplier)
    if value <= 1.0:
        raise ValueError("SMOTE multiplier must be greater than 1.0.")
    if not np.isfinite(value):
        raise ValueError("SMOTE multiplier must be finite.")
    return value


def build_class_c_synthetic_pool(
    *,
    train_split: TaskSplitData,
    feature_names: Sequence[str],
    minority_class: int,
    multiplier: float = 2.0,
    seed: int = 42,
    k_neighbors: int = 5,
    patient_id_column: str = "subject_id",
    binary_columns: Sequence[str] = (),
    integer_columns: Sequence[str] = (),
    value_ranges: Mapping[str, Sequence[float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, SyntheticPoolManifest]:
    """Create a deterministic Class-C pool from one natural train partition.

    The generator never receives validation/test/external rows. Continuous
    interpolation is performed in a neutral minority-only median-imputed and
    standardised space. The generated rows are inverse-transformed to raw units
    so EXP1 and Stage2 can reuse exactly the same pool before their model-specific
    preprocessing.
    """
    assert_natural_split(train_split)
    if train_split.split_name != "train":
        raise ValueError("Synthetic pools may only be generated from a train split.")
    multiplier = _validate_multiplier(multiplier)
    feature_names = tuple(str(v) for v in feature_names)
    y = np.asarray(train_split.labels, dtype=np.int64).reshape(-1)
    minority_idx = np.flatnonzero(y == int(minority_class))
    n_original = int(len(minority_idx))
    n_target = int(round(n_original * multiplier))
    n_synthetic = int(n_target - n_original)
    if n_synthetic <= 0:
        raise ValueError("Requested SMOTE multiplier creates no synthetic rows.")
    if n_original <= int(k_neighbors):
        raise ValueError(
            f"Minority class {minority_class} has {n_original} rows, not enough "
            f"for k_neighbors={k_neighbors}."
        )
    if patient_id_column not in train_split.metadata.columns:
        raise KeyError(
            f"Synthetic provenance requires patient group column: {patient_id_column}"
        )

    minority_raw = train_split.raw_features.iloc[minority_idx].loc[:, list(feature_names)].copy()
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(minority_raw)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)

    neighbor_model = NearestNeighbors(n_neighbors=int(k_neighbors) + 1, metric="euclidean")
    neighbor_model.fit(scaled)
    neighbor_indices = neighbor_model.kneighbors(scaled, return_distance=False)[:, 1:]

    rng = np.random.default_rng(int(seed))
    base_positions = rng.integers(0, n_original, size=n_synthetic)
    neighbor_choice = rng.integers(0, int(k_neighbors), size=n_synthetic)
    other_positions = neighbor_indices[base_positions, neighbor_choice]
    mixing = rng.random(n_synthetic)
    synthetic_scaled = (
        scaled[base_positions]
        + mixing[:, None] * (scaled[other_positions] - scaled[base_positions])
    )
    synthetic_raw = pd.DataFrame(
        scaler.inverse_transform(synthetic_scaled),
        columns=list(feature_names),
    )

    inferred_binary = _infer_binary_columns(minority_raw, feature_names)
    binary = tuple(sorted(set(str(v) for v in binary_columns) | set(inferred_binary)))
    integer = tuple(sorted(set(str(v) for v in integer_columns)))

    # Binary values are inherited from one of the two real parents instead of
    # being rounded interpolated pseudo-categories.
    for column in binary:
        if column not in synthetic_raw.columns:
            continue
        column_index = list(feature_names).index(column)
        choose_b = rng.random(n_synthetic) >= 0.5
        values = np.where(
            choose_b,
            imputed[other_positions, column_index],
            imputed[base_positions, column_index],
        )
        synthetic_raw[column] = np.asarray(values, dtype=float)

    for column in integer:
        if column in synthetic_raw.columns:
            synthetic_raw[column] = np.rint(pd.to_numeric(synthetic_raw[column], errors="coerce"))

    clamped: dict[str, int] = {}
    for column, bounds in dict(value_ranges or {}).items():
        if column not in synthetic_raw.columns:
            continue
        if len(bounds) != 2:
            raise ValueError(f"value_ranges[{column}] must contain [low, high].")
        low, high = float(bounds[0]), float(bounds[1])
        before = pd.to_numeric(synthetic_raw[column], errors="coerce").copy()
        synthetic_raw[column] = before.clip(low, high)
        clamped[str(column)] = int((before != synthetic_raw[column]).fillna(False).sum())

    natural_meta = train_split.metadata.reset_index(drop=True)
    source_values = pd.to_numeric(natural_meta["source_row_index"], errors="raise").astype(np.int64).to_numpy()
    subject_values = natural_meta[patient_id_column].to_numpy()
    base_global = minority_idx[base_positions]
    other_global = minority_idx[other_positions]
    provenance = pd.DataFrame(
        {
            "sample_id": [f"syn:{int(seed)}:{index:08d}" for index in range(n_synthetic)],
            "source_row_index": pd.array([pd.NA] * n_synthetic, dtype="Int64"),
            "is_synthetic": True,
            "parent_a_source_row_index": source_values[base_global],
            "parent_b_source_row_index": source_values[other_global],
            "parent_subject_id": subject_values[base_global],
            "neighbor_subject_id": subject_values[other_global],
            "synthetic_seed": int(seed),
            "synthetic_lambda": mixing.astype(float),
        }
    )
    pool_sha = _frame_sha256(synthetic_raw)
    provenance_sha = _frame_sha256(provenance)
    manifest = SyntheticPoolManifest(
        schema_version="exp123_smote_pool_v8_1",
        strategy="class_c_smote_x2",
        minority_class=int(minority_class),
        multiplier=float(multiplier),
        seed=int(seed),
        k_neighbors=int(k_neighbors),
        original_minority_count=n_original,
        synthetic_count=n_synthetic,
        target_minority_count=n_target,
        feature_names=feature_names,
        binary_columns=binary,
        integer_columns=integer,
        clamped_value_counts=clamped,
        source_split="train",
        pool_sha256=pool_sha,
        provenance_sha256=provenance_sha,
    )
    return synthetic_raw.reset_index(drop=True), provenance.reset_index(drop=True), manifest


def save_synthetic_pool(
    output_dir: Path,
    raw_features: pd.DataFrame,
    provenance: pd.DataFrame,
    manifest: SyntheticPoolManifest,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "smote_class_c_pool.csv.gz"
    provenance_path = output_dir / "smote_class_c_provenance.csv.gz"
    raw_features.to_csv(raw_path, index=False, compression="gzip")
    provenance.to_csv(provenance_path, index=False, compression="gzip")
    payload = manifest.to_dict()
    payload["raw_file_sha256"] = sha256_file(raw_path)
    payload["provenance_file_sha256"] = sha256_file(provenance_path)
    atomic_write_json(output_dir / "smote_manifest.json", payload)


def load_synthetic_pool(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_path = output_dir / "smote_manifest.json"
    raw_path = output_dir / "smote_class_c_pool.csv.gz"
    provenance_path = output_dir / "smote_class_c_provenance.csv.gz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("raw_file_sha256") != sha256_file(raw_path):
        raise RuntimeError("Synthetic pool raw file SHA-256 mismatch.")
    if manifest.get("provenance_file_sha256") != sha256_file(provenance_path):
        raise RuntimeError("Synthetic pool provenance SHA-256 mismatch.")
    raw = pd.read_csv(raw_path)
    provenance = pd.read_csv(provenance_path)
    if _frame_sha256(raw) != manifest.get("pool_sha256"):
        raise RuntimeError("Synthetic pool content hash mismatch.")
    # Pandas dtype round-tripping can change nullable integers, so provenance
    # file SHA is the authoritative persisted representation.
    return raw, provenance, manifest


@dataclass(frozen=True)
class ClassCSmoteSampler(TrainingSampler):
    minority_class: int
    multiplier: float = 2.0
    k_neighbors: int = 5
    patient_id_column: str = "subject_id"
    binary_columns: tuple[str, ...] = ()
    integer_columns: tuple[str, ...] = ()
    value_ranges: Mapping[str, Sequence[float]] | None = None
    fixed_pool_raw: pd.DataFrame | None = None
    fixed_pool_provenance: pd.DataFrame | None = None
    fixed_pool_seed: int | None = None
    strategy_name: str = "class_c_smote_x2"

    def sample(
        self,
        train_split: TaskSplitData,
        *,
        feature_names: Sequence[str],
        seed: int,
        purpose: str,
    ) -> SampledTrainingData:
        assert_natural_split(train_split)
        use_fixed = (
            purpose == "final_train"
            and self.fixed_pool_raw is not None
            and self.fixed_pool_provenance is not None
        )
        if use_fixed:
            pool_raw = self.fixed_pool_raw.loc[:, list(feature_names)].copy().reset_index(drop=True)
            pool_provenance = self.fixed_pool_provenance.copy().reset_index(drop=True)
            pool_seed = int(self.fixed_pool_seed if self.fixed_pool_seed is not None else seed)
            pool_manifest = {
                "source": "fixed_shared_pool",
                "seed": pool_seed,
                "synthetic_count": int(len(pool_raw)),
            }
        else:
            pool_raw, pool_provenance, generated = build_class_c_synthetic_pool(
                train_split=train_split,
                feature_names=feature_names,
                minority_class=self.minority_class,
                multiplier=self.multiplier,
                seed=seed,
                k_neighbors=self.k_neighbors,
                patient_id_column=self.patient_id_column,
                binary_columns=self.binary_columns,
                integer_columns=self.integer_columns,
                value_ranges=self.value_ranges,
            )
            pool_manifest = generated.to_dict()

        natural_raw = train_split.raw_features.loc[:, list(feature_names)].copy().reset_index(drop=True)
        natural_labels = np.asarray(train_split.labels, dtype=np.int64).copy()
        natural_provenance = _natural_provenance(train_split)
        raw = pd.concat([natural_raw, pool_raw], ignore_index=True)
        labels = np.concatenate(
            [natural_labels, np.full(len(pool_raw), int(self.minority_class), dtype=np.int64)]
        )
        provenance = pd.concat([natural_provenance, pool_provenance], ignore_index=True, sort=False)
        manifest = {
            "schema_version": "exp123_sampling_v8_1",
            "strategy": self.strategy_name,
            "purpose": str(purpose),
            "seed": int(seed),
            "minority_class": int(self.minority_class),
            "multiplier": float(self.multiplier),
            "natural_count": int(len(natural_labels)),
            "synthetic_count": int(len(pool_raw)),
            "fit_count": int(len(labels)),
            "validation_distribution_modified": False,
            "internal_test_distribution_modified": False,
            "external_distribution_modified": False,
            "pool": pool_manifest,
        }
        return SampledTrainingData(raw, labels, provenance, manifest)
