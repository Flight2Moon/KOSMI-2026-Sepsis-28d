#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config, resolve_path
from los_baseline.los_derivation import derive_hospital_los
from los_baseline.utils import atomic_write_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an auditable EXP123 dataset with hospital LOS derived from "
            "admission/discharge timestamps."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=["mimic", "eicu"], required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _compression(path: Path) -> str | None:
    return "gzip" if path.suffix.lower() == ".gz" else None


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    dataset_key = "mimic_csv" if args.dataset == "mimic" else "eicu_csv"
    input_path = (
        Path(args.input).expanduser().resolve()
        if args.input
        else resolve_path(config, config["paths"][dataset_key])
    )
    output_path = Path(args.output).expanduser().resolve()
    audit_path = (
        Path(args.audit_output).expanduser().resolve()
        if args.audit_output
        else output_path.with_suffix(output_path.suffix + ".los_audit.json")
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset was not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    outcomes = dict(config["outcomes"])
    mortality_column = str(outcomes["mortality_column"])
    los_column = str(outcomes.get("los_column", config["data"]["target_column"]))
    derivation_config = outcomes.get("los_derivation")

    frame = pd.read_csv(input_path, low_memory=False, compression="infer")
    frame.columns = frame.columns.astype(str).str.strip()
    input_rows = int(len(frame))
    frame.insert(0, "_preparation_source_row_index", np.arange(input_rows, dtype=np.int64))

    repeated_header_mask = pd.Series(False, index=frame.index)
    for column in (mortality_column, los_column, "admittime", "dischtime"):
        if column not in frame.columns:
            continue
        values = frame[column].astype("string").str.strip().str.lower()
        repeated_header_mask |= values.eq(column.lower()).fillna(False)
    repeated_headers_removed = int(repeated_header_mask.sum())
    if repeated_headers_removed:
        frame = frame.loc[~repeated_header_mask].copy()

    frame[mortality_column] = pd.to_numeric(frame[mortality_column], errors="coerce")
    prepared, summary = derive_hospital_los(
        frame,
        dataset_name=args.dataset,
        mortality_column=mortality_column,
        los_column=los_column,
        config=derivation_config,
    )

    mortality = pd.to_numeric(prepared[mortality_column], errors="coerce")
    effective_los = pd.to_numeric(prepared[los_column], errors="coerce")
    invalid_mortality = int((mortality.isna() | ~mortality.isin([0.0, 1.0])).sum())
    missing_los = int(effective_los.isna().sum())
    negative_los = int(effective_los.lt(0).fillna(False).sum())
    if invalid_mortality or missing_los or negative_los:
        raise ValueError(
            "Prepared dataset failed outcome validation: "
            f"invalid_mortality={invalid_mortality}, "
            f"missing_los={missing_los}, negative_los={negative_los}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(
        output_path,
        index=False,
        compression=_compression(output_path),
    )

    discrepancy_column = f"{los_column}_absolute_discrepancy_days"
    tolerance = float(summary.policy.get("discrepancy_tolerance_days", 0.05))
    mismatch_path: Path | None = None
    if discrepancy_column in prepared.columns:
        mismatch = prepared.loc[
            pd.to_numeric(prepared[discrepancy_column], errors="coerce")
            .gt(tolerance)
            .fillna(False)
        ].copy()
        if not mismatch.empty:
            mismatch_path = output_path.with_suffix(
                output_path.suffix + ".los_discrepancies.csv.gz"
            )
            mismatch.to_csv(mismatch_path, index=False, compression="gzip")

    report = {
        "status": "ready",
        "dataset": args.dataset,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "input_rows": input_rows,
        "prepared_rows": int(len(prepared)),
        "repeated_headers_removed": repeated_headers_removed,
        "in_hospital_death_rows": int(mortality.eq(1.0).sum()),
        "non_in_hospital_death_flag_rows": int(mortality.eq(0.0).sum()),
        "los_derivation": summary.to_dict(),
        "discrepancy_rows_path": None if mismatch_path is None else str(mismatch_path),
        "notes": [
            "hospital_expire_flag=1 uses the configured in-hospital death endpoint; "
            "the default endpoint is dischtime.",
            "hospital_expire_flag=0 uses dischtime as the hospital LOS endpoint; "
            "post-discharge death does not extend hospital LOS.",
            "No valid death row is excluded by this preparation step.",
        ],
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(audit_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
