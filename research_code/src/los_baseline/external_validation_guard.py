from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import resolve_path
from .utils import atomic_write_json


@dataclass(frozen=True)
class ExternalMortalityAudit:
    dataset_path: str
    mortality_column: str
    row_count: int
    missing_count: int
    zero_count: int
    one_count: int
    invalid_count: int
    death_rate: float | None
    unique_values: tuple[str, ...]
    passed: bool
    blocking_reasons: tuple[str, ...]
    source_scope: str = "raw_external_csv"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_mortality(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.casefold()
    positive = text.isin({"expired", "dead", "death", "deceased", "yes", "true"})
    negative = text.isin({"alive", "survived", "no", "false"})
    numeric = numeric.mask(positive, 1.0).mask(negative, 0.0)
    return numeric


def audit_external_mortality_flag(
    *,
    dataset_path: Path,
    mortality_column: str,
    settings: Mapping[str, Any],
    source_scope: str = "raw_external_csv",
) -> ExternalMortalityAudit:
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"External dataset was not found: {dataset_path}")
    chunksize = max(1, int(settings.get("chunksize", 250_000)))
    counts = {"rows": 0, "missing": 0, "zero": 0, "one": 0, "invalid": 0}
    unique: set[str] = set()
    reader = pd.read_csv(
        dataset_path,
        usecols=[mortality_column],
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        raw = chunk[mortality_column]
        unique.update(str(value) for value in raw.dropna().astype(str).unique()[:100])
        normalized = _normalize_mortality(raw)
        counts["rows"] += int(len(raw))
        counts["missing"] += int(normalized.isna().sum())
        counts["zero"] += int((normalized == 0).sum())
        counts["one"] += int((normalized == 1).sum())
        counts["invalid"] += int((normalized.notna() & ~normalized.isin([0, 1])).sum())
    denominator = counts["zero"] + counts["one"]
    death_rate = float(counts["one"] / denominator) if denominator else None
    reasons: list[str] = []
    minimum_deaths = int(settings.get("minimum_death_count", 10))
    minimum_rate = float(settings.get("minimum_death_rate", 0.001))
    maximum_rate = float(settings.get("maximum_death_rate", 0.5))
    if counts["one"] < minimum_deaths:
        reasons.append(f"death_count_below_minimum:{counts['one']}<{minimum_deaths}")
    if death_rate is None:
        reasons.append("death_rate_unavailable")
    else:
        if death_rate < minimum_rate:
            reasons.append(f"death_rate_below_minimum:{death_rate:.8f}<{minimum_rate:.8f}")
        if death_rate > maximum_rate:
            reasons.append(f"death_rate_above_maximum:{death_rate:.8f}>{maximum_rate:.8f}")
    if counts["invalid"] > 0:
        reasons.append(f"invalid_mortality_values:{counts['invalid']}")
    if bool(settings.get("fail_on_missing", False)) and counts["missing"] > 0:
        reasons.append(f"missing_mortality_values:{counts['missing']}")
    return ExternalMortalityAudit(
        dataset_path=str(dataset_path),
        mortality_column=mortality_column,
        row_count=counts["rows"],
        missing_count=counts["missing"],
        zero_count=counts["zero"],
        one_count=counts["one"],
        invalid_count=counts["invalid"],
        death_rate=death_rate,
        unique_values=tuple(sorted(unique)[:100]),
        passed=not reasons,
        blocking_reasons=tuple(reasons),
        source_scope=str(source_scope),
    )


def assert_external_mortality_ready(
    *,
    config: Mapping[str, Any],
    run_dir: Path,
) -> ExternalMortalityAudit:
    settings = dict(config.get("external_validation", {}).get("mortality_guard", {}))
    if not bool(settings.get("enabled", True)):
        return ExternalMortalityAudit(
            dataset_path=str(resolve_path(dict(config), config["paths"]["eicu_csv"])),
            mortality_column=str(config["outcomes"]["mortality_column"]),
            row_count=0, missing_count=0, zero_count=0, one_count=0,
            invalid_count=0, death_rate=None, unique_values=(), passed=True,
            blocking_reasons=(),
            source_scope="guard_disabled",
        )
    eligible_assignments = (
        run_dir
        / "common"
        / "outcomes"
        / "eicu_outcome_assignments.csv.gz"
    )
    if eligible_assignments.is_file():
        dataset_path = eligible_assignments
        source_scope = "eligible_eicu_outcome_cohort"
    else:
        dataset_path = resolve_path(
            dict(config), config["paths"]["eicu_csv"]
        )
        source_scope = "raw_external_csv_fallback"
    result = audit_external_mortality_flag(
        dataset_path=dataset_path,
        mortality_column=str(config["outcomes"]["mortality_column"]),
        settings=settings,
        source_scope=source_scope,
    )
    output_dir = run_dir / "common" / "data_quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "eicu_mortality_flag_audit.json", result.to_dict())
    if not result.passed and bool(settings.get("block_confirmatory_on_failure", True)):
        raise RuntimeError(
            "External eICU mortality validation is blocked because the harmonized "
            "mortality flag failed quality checks: " + "; ".join(result.blocking_reasons)
        )
    return result
