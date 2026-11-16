#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Outcome manifest was not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _print_reasons(title: str, values: dict[str, Any]) -> None:
    print(title)
    if not values:
        print("  - None")
        return
    for reason, count in sorted(values.items()):
        print(f"  - {reason}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize EXP123 V7.2 outcome and cohort audits."
    )
    parser.add_argument(
        "run_dir",
        help="Run directory or common/outcomes directory.",
    )
    args = parser.parse_args()
    base = Path(args.run_dir).expanduser().resolve()
    outcomes_dir = base if base.name == "outcomes" else base / "common" / "outcomes"
    manifest = _load(outcomes_dir / "outcome_step_manifest.json")

    print("=== EXP123 V7.2 Outcome Status ===")
    print(f"Run ID: {manifest.get('run_id')}")
    print(f"Discovery: {manifest.get('discovery_status', manifest.get('status'))}")
    print(f"External validation: {manifest.get('external_validation_status', 'unknown')}")
    print(f"Landmark days: {manifest.get('landmark_days')}")
    print()

    for dataset_name, audit in (manifest.get("dataset_audits") or {}).items():
        print("=" * 72)
        print(dataset_name.upper())
        print("=" * 72)
        print(f"Status: {audit.get('status')}")
        print(f"Label derivation: {audit.get('label_derivation_status', 'unknown')}")
        print(f"Input rows: {audit.get('input_row_count', 0):,}")
        print(f"Loaded rows: {audit.get('loaded_row_count', 0):,}")
        print(
            "Planned cohort exclusions: "
            f"{audit.get('cohort_excluded_row_count', 0):,} "
            f"({float(audit.get('cohort_exclusion_rate', 0.0)):.4%})"
        )
        print(
            "Data-quality exclusions: "
            f"{audit.get('quality_excluded_row_count', 0):,} "
            f"({float(audit.get('quality_exclusion_rate', 0.0)):.4%})"
        )
        print(f"Retained modelling rows: {audit.get('retained_row_count', 0):,}")
        print(f"Class counts: {audit.get('class_counts', {})}")
        _print_reasons(
            "Cohort eligibility reasons:",
            audit.get("cohort_exclusion_reason_counts") or {},
        )
        _print_reasons(
            "Data-quality reasons:",
            audit.get("quality_exclusion_reason_counts") or {},
        )
        issues = audit.get("blocking_issues") or []
        print("Blocking issues:")
        if issues:
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("  - None")
        print()

    print("Artifacts:")
    for name in (
        "DISCOVERY_OUTCOMES_READY.lock.json",
        "DISCOVERY_OUTCOMES_BLOCKED.lock.json",
        "EXTERNAL_VALIDATION_READY.lock.json",
        "EXTERNAL_VALIDATION_ON_HOLD.lock.json",
    ):
        path = outcomes_dir / name
        if path.is_file():
            print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
