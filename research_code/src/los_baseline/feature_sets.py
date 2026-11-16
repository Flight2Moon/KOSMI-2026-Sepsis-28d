from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class ResolvedFeatureSet:
    set_id: str
    description: str
    columns: tuple[str, ...]
    sha256: str
    registry_path: str
    registry_schema_version: str
    derived_from: str | None = None
    excluded_columns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["columns"] = list(self.columns)
        payload["excluded_columns"] = list(self.excluded_columns)
        payload["feature_count"] = len(self.columns)
        return payload


def _columns_sha256(columns: Sequence[str]) -> str:
    payload = json.dumps(list(columns), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_feature_set_registry(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Feature-set registry was not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("schema_version", "")).strip() == "":
        raise ValueError("Feature-set registry schema_version is required.")
    sets = payload.get("sets")
    if not isinstance(sets, dict) or not sets:
        raise ValueError("Feature-set registry must contain a non-empty sets object.")
    return payload


def resolve_feature_set(path: Path, set_id: str) -> ResolvedFeatureSet:
    registry_path = path.expanduser().resolve()
    registry = load_feature_set_registry(registry_path)
    sets = registry["sets"]
    target = str(set_id)
    visiting: set[str] = set()

    def expand(current: str) -> tuple[list[str], str, str | None, list[str]]:
        if current in visiting:
            raise ValueError(f"Feature-set derivation cycle detected at {current}.")
        if current not in sets:
            raise KeyError(f"Unknown feature_set_id: {current}")
        visiting.add(current)
        spec = sets[current]
        if not isinstance(spec, dict):
            raise TypeError(f"Feature-set definition must be an object: {current}")
        description = str(spec.get("description", "")).strip()
        parent = spec.get("derived_from")
        excluded = [str(v) for v in spec.get("exclude", [])]
        if parent is None:
            columns = [str(v) for v in spec.get("columns", [])]
            if not columns:
                raise ValueError(f"Feature set {current} has no columns.")
        else:
            columns, _, _, _ = expand(str(parent))
            unknown_exclusions = sorted(set(excluded) - set(columns))
            if unknown_exclusions:
                raise ValueError(
                    f"Feature set {current} excludes columns not present in parent: "
                    f"{unknown_exclusions}"
                )
            columns = [column for column in columns if column not in set(excluded)]
        visiting.remove(current)
        if len(columns) != len(set(columns)):
            raise ValueError(f"Feature set {current} contains duplicate columns.")
        return columns, description, str(parent) if parent is not None else None, excluded

    columns, description, parent, excluded = expand(target)
    return ResolvedFeatureSet(
        set_id=target,
        description=description,
        columns=tuple(columns),
        sha256=_columns_sha256(columns),
        registry_path=str(registry_path),
        registry_schema_version=str(registry["schema_version"]),
        derived_from=parent,
        excluded_columns=tuple(excluded),
    )


def inject_feature_set_into_config(
    base_config: Mapping[str, Any],
    resolved: ResolvedFeatureSet,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))
    data = config.setdefault("data", {})
    data["feature_columns"] = list(resolved.columns)
    data["feature_set_id"] = resolved.set_id
    data["feature_set_sha256"] = resolved.sha256
    config["feature_set"] = resolved.to_dict()
    return config


def dataset_header_readiness(
    *,
    dataset_path: Path,
    required_columns: Sequence[str],
) -> dict[str, Any]:
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.is_file():
        return {
            "path": str(dataset_path),
            "status": "missing_file",
            "required_feature_count": len(required_columns),
            "missing_columns": list(required_columns),
        }
    header = pd.read_csv(dataset_path, nrows=0).columns.astype(str).tolist()
    missing = sorted(set(required_columns) - set(header))
    return {
        "path": str(dataset_path),
        "status": "ready" if not missing else "missing_columns",
        "required_feature_count": len(required_columns),
        "available_column_count": len(header),
        "missing_columns": missing,
    }
