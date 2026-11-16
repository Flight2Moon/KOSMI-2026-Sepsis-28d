from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from ..exp123_data import TaskSplitData


@dataclass(frozen=True)
class SampledTrainingData:
    """Model-fit data created from one natural train partition.

    Natural EXP123 splits remain immutable and synthetic-free. Sampling is an
    explicit model-fit operation that produces this separate object. This
    prevents validation/internal-test/external rows from being silently
    modified by an imbalance strategy.
    """

    raw_features: pd.DataFrame
    labels: np.ndarray
    provenance: pd.DataFrame
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        n = len(self.raw_features)
        if len(self.labels) != n or len(self.provenance) != n:
            raise ValueError(
                "SampledTrainingData lengths do not match: "
                f"features={n}, labels={len(self.labels)}, provenance={len(self.provenance)}"
            )
        if self.raw_features.index.has_duplicates:
            raise ValueError("SampledTrainingData feature index must be unique.")
        if "sample_id" not in self.provenance.columns:
            raise KeyError("SampledTrainingData provenance must contain sample_id.")
        if self.provenance["sample_id"].duplicated().any():
            raise ValueError("SampledTrainingData sample_id values must be unique.")
        synthetic = synthetic_mask(self.provenance)
        if synthetic.any():
            required = {
                "parent_a_source_row_index",
                "parent_b_source_row_index",
                "parent_subject_id",
                "synthetic_seed",
                "synthetic_lambda",
            }
            missing = sorted(required - set(self.provenance.columns))
            if missing:
                raise KeyError(
                    "Synthetic training provenance is missing columns: "
                    f"{missing}"
                )
            natural_source = set(
                pd.to_numeric(
                    self.provenance.loc[~synthetic, "source_row_index"],
                    errors="coerce",
                ).dropna().astype(np.int64).tolist()
            )
            for column in ("parent_a_source_row_index", "parent_b_source_row_index"):
                parents = set(
                    pd.to_numeric(
                        self.provenance.loc[synthetic, column], errors="coerce"
                    ).dropna().astype(np.int64).tolist()
                )
                if not parents.issubset(natural_source):
                    missing_parents = sorted(parents - natural_source)[:10]
                    raise ValueError(
                        "Synthetic rows must be derived only from natural rows in the "
                        f"same training partition. Missing parents from {column}: "
                        f"{missing_parents}"
                    )

    @property
    def synthetic_count(self) -> int:
        return int(synthetic_mask(self.provenance).sum())

    @property
    def natural_count(self) -> int:
        return int(len(self.provenance) - self.synthetic_count)


class TrainingSampler(Protocol):
    strategy_name: str

    def sample(
        self,
        train_split: TaskSplitData,
        *,
        feature_names: Sequence[str],
        seed: int,
        purpose: str,
    ) -> SampledTrainingData:
        ...


def synthetic_mask(metadata: pd.DataFrame) -> np.ndarray:
    if "is_synthetic" not in metadata.columns:
        return np.zeros(len(metadata), dtype=bool)
    values = metadata["is_synthetic"].fillna(False).astype(bool).to_numpy()
    return np.asarray(values, dtype=bool)


def assert_natural_split(split: TaskSplitData) -> None:
    synthetic = synthetic_mask(split.metadata)
    if synthetic.any():
        raise ValueError(
            f"Natural EXP123 split {split.split_name} contains "
            f"{int(synthetic.sum())} synthetic rows. Synthetic samples must exist "
            "only in SampledTrainingData created at model-fit time."
        )


def _natural_provenance(split: TaskSplitData) -> pd.DataFrame:
    assert_natural_split(split)
    metadata = split.metadata.copy().reset_index(drop=True)
    source = pd.to_numeric(metadata["source_row_index"], errors="raise").astype(np.int64)
    metadata["source_row_index"] = source
    metadata["sample_id"] = [f"orig:{value}" for value in source.tolist()]
    metadata["is_synthetic"] = False
    for column in (
        "parent_a_source_row_index",
        "parent_b_source_row_index",
        "synthetic_seed",
        "synthetic_lambda",
    ):
        if column not in metadata.columns:
            metadata[column] = np.nan
    if "parent_subject_id" not in metadata.columns:
        metadata["parent_subject_id"] = pd.Series([None] * len(metadata), dtype="object")
    return metadata


@dataclass(frozen=True)
class NaturalSampler:
    strategy_name: str = "natural"

    def sample(
        self,
        train_split: TaskSplitData,
        *,
        feature_names: Sequence[str],
        seed: int,
        purpose: str,
    ) -> SampledTrainingData:
        assert_natural_split(train_split)
        raw = train_split.raw_features.loc[:, list(feature_names)].copy().reset_index(drop=True)
        labels = np.asarray(train_split.labels, dtype=np.int64).copy()
        provenance = _natural_provenance(train_split)
        manifest = {
            "schema_version": "exp123_sampling_v8_1",
            "strategy": self.strategy_name,
            "purpose": str(purpose),
            "seed": int(seed),
            "natural_count": int(len(labels)),
            "synthetic_count": 0,
            "fit_count": int(len(labels)),
            "evaluation_distribution_modified": False,
        }
        return SampledTrainingData(raw, labels, provenance, manifest)


def sampling_manifest_payload(sampled: SampledTrainingData) -> dict[str, Any]:
    payload = dict(sampled.manifest)
    payload.setdefault("natural_count", sampled.natural_count)
    payload.setdefault("synthetic_count", sampled.synthetic_count)
    payload.setdefault("fit_count", int(len(sampled.labels)))
    return payload
