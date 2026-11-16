from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .config import resolve_outcome_horizon_days

V8_PRIMARY_OUTCOME_HORIZON_DAYS = 28

V8_STRATEGY_ORDER = (
    "T1_baseline",
    "T2_optuna",
    "T3_smote",
    "T4_undersampling",
)

V8_STRATEGY_LABELS = {
    "T1_baseline": "Natural Baseline",
    "T2_optuna": "Optuna Optimized",
    "T3_smote": "Class-C SMOTE x2",
    "T4_undersampling": "Majority Undersampling",
}


@dataclass(frozen=True)
class PlannedRun:
    outcome_horizon_days: int
    strategy: str
    strategy_label: str
    enabled: bool
    status: str
    run_id: str | None
    analysis_role: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise_horizon(value: int | float) -> int:
    numeric = float(value)
    if numeric <= 0 or not numeric.is_integer():
        raise ValueError(f"Outcome horizon must be a positive whole day: {value}")
    return int(numeric)


def resolve_strategy_selection(
    config: Mapping[str, Any],
    *,
    include_undersampling: bool | None = None,
) -> dict[str, bool]:
    matrix = dict(config.get("experiment_matrix", {}))
    strategies = dict(matrix.get("strategies", {}))
    missing = [name for name in V8_STRATEGY_ORDER if name not in strategies]
    if missing:
        raise ValueError(f"Missing V8 strategy definitions: {missing}")

    selected = {
        name: bool(dict(strategies[name]).get("enabled", False))
        for name in V8_STRATEGY_ORDER
    }
    for name in V8_STRATEGY_ORDER[:3]:
        if not selected[name]:
            raise ValueError(f"Mandatory strategy is disabled: {name}")

    if include_undersampling is not None:
        selected["T4_undersampling"] = bool(include_undersampling)
    return selected


def build_v8_run_id(
    *,
    backend: str,
    horizon_days: int | float,
    strategy: str,
    timestamp: str,
) -> str:
    if strategy not in V8_STRATEGY_ORDER:
        raise ValueError(f"Unknown V8 strategy: {strategy}")
    horizon = _normalise_horizon(horizon_days)
    backend = str(backend).strip().lower()
    if not backend:
        raise ValueError("backend must not be empty")
    return f"exp123_v8_{backend}_LOS{horizon}_{strategy}_{timestamp}"


def build_matrix_plan(
    config: Mapping[str, Any],
    *,
    horizons: Sequence[int | float] = (V8_PRIMARY_OUTCOME_HORIZON_DAYS,),
    backend: str,
    timestamp: str,
    include_undersampling: bool | None = None,
    analysis_role: str = "prespecified",
) -> dict[str, Any]:
    selected = resolve_strategy_selection(
        config,
        include_undersampling=include_undersampling,
    )
    horizon_values = [_normalise_horizon(v) for v in horizons]
    if len(horizon_values) != len(set(horizon_values)):
        raise ValueError("Duplicate outcome horizons are not permitted.")
    unsupported = [value for value in horizon_values if value != V8_PRIMARY_OUTCOME_HORIZON_DAYS]
    if unsupported:
        raise ValueError(
            "Phase 6 supports LOS28 only. Unsupported outcome horizon(s): "
            f"{unsupported}. Historical multi-horizon configs are retained under legacy_reference/."
        )

    runs: list[PlannedRun] = []
    for horizon in horizon_values:
        for strategy in V8_STRATEGY_ORDER:
            enabled = bool(selected[strategy])
            runs.append(
                PlannedRun(
                    outcome_horizon_days=horizon,
                    strategy=strategy,
                    strategy_label=V8_STRATEGY_LABELS[strategy],
                    enabled=enabled,
                    status="planned" if enabled else "not_requested",
                    run_id=(
                        build_v8_run_id(
                            backend=backend,
                            horizon_days=horizon,
                            strategy=strategy,
                            timestamp=timestamp,
                        )
                        if enabled
                        else None
                    ),
                    analysis_role=analysis_role,
                )
            )

    return {
        "schema_version": "exp123_v8_matrix_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_outcome_horizon_key": "outcomes.outcome_horizon_days",
        "prediction_landmark_hours": float(
            dict(config.get("cohort_contract", {})).get("prediction_landmark_hours", 24.0)
        ),
        "requested_horizons": horizon_values,
        "strategy_order": list(V8_STRATEGY_ORDER),
        "strategy_selection": selected,
        "include_undersampling": bool(selected["T4_undersampling"]),
        "confirmatory_splits_accessed": False,
        "runs": [item.to_dict() for item in runs],
    }


def config_horizon(config: Mapping[str, Any]) -> int:
    value = _normalise_horizon(resolve_outcome_horizon_days(dict(config)))
    if value != V8_PRIMARY_OUTCOME_HORIZON_DAYS:
        raise ValueError(
            f"Phase 6 supports LOS{V8_PRIMARY_OUTCOME_HORIZON_DAYS} only; got LOS{value}."
        )
    return value
