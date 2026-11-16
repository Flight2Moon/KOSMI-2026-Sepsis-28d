from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .contract import sha256_file


@dataclass(frozen=True)
class DatasetFilter:
    column: str
    equals: int | float | str | bool

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DatasetFilter":
        return cls(column=str(payload["column"]), equals=payload["equals"])


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    file: str
    role: str
    id_columns: tuple[str, ...]
    expected_sha256: str | None = None
    required_external: bool = False
    filter: DatasetFilter | None = None
    notes: str | None = None

    @classmethod
    def from_mapping(cls, name: str, payload: Mapping[str, Any]) -> "DatasetSpec":
        filter_payload = payload.get("filter")
        return cls(
            name=str(name),
            file=str(payload["file"]),
            role=str(payload["role"]),
            id_columns=tuple(str(value) for value in payload.get("id_columns", [])),
            expected_sha256=(
                str(payload["expected_sha256"]).lower()
                if payload.get("expected_sha256")
                else None
            ),
            required_external=bool(payload.get("required_external", False)),
            filter=(
                DatasetFilter.from_mapping(filter_payload)
                if isinstance(filter_payload, Mapping)
                else None
            ),
            notes=(str(payload["notes"]) if payload.get("notes") is not None else None),
        )

    def resolve_path(self, dataset_root: Path) -> Path:
        return (Path(dataset_root).expanduser().resolve() / self.file).resolve()

    def source_sha256(self, dataset_root: Path) -> str:
        return sha256_file(self.resolve_path(dataset_root))

    def load(self, dataset_root: Path, *, usecols: list[str] | None = None) -> pd.DataFrame:
        path = self.resolve_path(dataset_root)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset '{self.name}' was not found: {path}")
        frame = pd.read_csv(path, usecols=usecols, low_memory=False)
        if self.filter is not None:
            if self.filter.column not in frame.columns:
                raise KeyError(
                    f"Dataset '{self.name}' filter column is missing: {self.filter.column}"
                )
            frame = frame.loc[frame[self.filter.column] == self.filter.equals].copy()
        return frame.reset_index(drop=True)


@dataclass(frozen=True)
class DatasetRegistry:
    schema_version: str
    datasets: dict[str, DatasetSpec]
    required_external_sites: tuple[str, ...]
    reverse_sensitivity: tuple[tuple[str, str], ...]

    def __getitem__(self, name: str) -> DatasetSpec:
        return self.datasets[name]

    @property
    def external_sites(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, spec in self.datasets.items()
            if spec.role.startswith("external")
        )


def load_dataset_registry(path: Path) -> DatasetRegistry:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    datasets = {
        name: DatasetSpec.from_mapping(name, spec)
        for name, spec in payload.get("datasets", {}).items()
    }
    if not datasets:
        raise ValueError("Dataset registry does not define any datasets.")
    required_external = tuple(str(value) for value in payload.get("required_external_sites", []))
    reverse = tuple(
        (str(item["train"]), str(item["evaluate"]))
        for item in payload.get("reverse_sensitivity", [])
    )
    registry = DatasetRegistry(
        schema_version=str(payload.get("schema_version", "study28_dataset_registry_v1")),
        datasets=datasets,
        required_external_sites=required_external,
        reverse_sensitivity=reverse,
    )
    missing = sorted(set(required_external) - set(datasets))
    if missing:
        raise ValueError(f"Required external sites are missing from the registry: {missing}")
    return registry
