from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contract import FeatureContract


def load_feature_sets(path: Path) -> dict[str, dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    sets = payload.get("sets", {})
    if not isinstance(sets, Mapping) or not sets:
        raise ValueError("feature_sets.json does not define any feature sets.")
    return {str(name): dict(spec) for name, spec in sets.items()}


def resolve_feature_set(
    name: str,
    *,
    feature_sets_path: Path,
    contract: FeatureContract,
) -> tuple[str, ...]:
    sets = load_feature_sets(feature_sets_path)
    resolving: set[str] = set()

    def resolve(current: str) -> tuple[str, ...]:
        if current not in sets:
            raise KeyError(f"Unknown feature set: {current}")
        if current in resolving:
            raise ValueError(f"Circular feature-set definition detected at {current}.")
        resolving.add(current)
        spec = sets[current]
        try:
            if "columns" in spec:
                columns = tuple(str(value) for value in spec["columns"])
            elif "derived_from" in spec:
                base = resolve(str(spec["derived_from"]))
                excluded = {str(value) for value in spec.get("exclude", [])}
                columns = tuple(value for value in base if value not in excluded)
            else:
                raise ValueError(
                    f"Feature set {current} requires either columns or derived_from."
                )
        finally:
            resolving.remove(current)
        return columns

    columns = resolve(str(name))
    contract.validate_model_features(columns)
    # For final-study feature sets, preserve the professor's authoritative order.
    expected_order = tuple(column for column in contract.columns if column in set(columns))
    if columns != expected_order:
        raise ValueError(
            f"Feature set {name} is not in professor feature_list.csv order."
        )
    return columns
