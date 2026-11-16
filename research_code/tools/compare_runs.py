from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SPLITS = ["train", "validation", "internal_test", "external_eicu"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare prediction probabilities and split membership between "
            "a frozen Baseline V1 run and a candidate Phase 1 run."
        )
    )
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--probability-atol", type=float, default=1e-6)
    parser.add_argument("--probability-rtol", type=float, default=1e-5)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _load_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "experiment_records.json"
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise TypeError(f"Expected a list in {path}")
    return payload


def _experiment_keys(records: list[dict[str, Any]]) -> set[tuple[float, str]]:
    return {
        (float(record["threshold_days"]), str(record["model_name"]))
        for record in records
        if str(record.get("status")) == "completed"
    }


def _prediction_path(
    run_dir: Path,
    threshold_days: float,
    model_name: str,
    split_name: str,
) -> Path:
    return (
        run_dir
        / f"LOS_{int(threshold_days):02d}"
        / model_name
        / "evaluation"
        / split_name
        / "predictions.csv.gz"
    )


def _align_predictions(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    key_candidates = [
        ["source_row_index"],
        ["stay_id"],
        ["hadm_id"],
        ["subject_id"],
        ["dataset_row_index"],
    ]

    for keys in key_candidates:
        if not all(key in baseline.columns for key in keys):
            continue
        if not all(key in candidate.columns for key in keys):
            continue
        if baseline.duplicated(keys).any() or candidate.duplicated(keys).any():
            continue

        baseline_sorted = baseline.sort_values(keys).reset_index(drop=True)
        candidate_sorted = candidate.sort_values(keys).reset_index(drop=True)

        if baseline_sorted[keys].equals(candidate_sorted[keys]):
            return baseline_sorted, candidate_sorted, "+".join(keys)

    if len(baseline) == len(candidate):
        return (
            baseline.reset_index(drop=True),
            candidate.reset_index(drop=True),
            "row_order",
        )

    raise ValueError(
        "Prediction files could not be aligned by identifiers or row order."
    )


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    def resolve_run(value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()

        project_relative = (project_root / path).resolve()
        if project_relative.is_dir():
            return project_relative

        return (project_root / "runs" / path).resolve()

    baseline_run = resolve_run(args.baseline_run)
    candidate_run = resolve_run(args.candidate_run)

    if not baseline_run.is_dir():
        raise FileNotFoundError(f"Baseline run was not found: {baseline_run}")
    if not candidate_run.is_dir():
        raise FileNotFoundError(f"Candidate run was not found: {candidate_run}")

    baseline_keys = _experiment_keys(_load_records(baseline_run))
    candidate_keys = _experiment_keys(_load_records(candidate_run))
    all_keys = sorted(baseline_keys | candidate_keys)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else candidate_run / "phase1_regression_check"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for threshold_days, model_name in all_keys:
        if (threshold_days, model_name) not in baseline_keys:
            failures.append(
                f"Missing baseline experiment: {threshold_days}/{model_name}"
            )
            continue
        if (threshold_days, model_name) not in candidate_keys:
            failures.append(
                f"Missing candidate experiment: {threshold_days}/{model_name}"
            )
            continue

        for split_name in SPLITS:
            baseline_path = _prediction_path(
                baseline_run,
                threshold_days,
                model_name,
                split_name,
            )
            candidate_path = _prediction_path(
                candidate_run,
                threshold_days,
                model_name,
                split_name,
            )

            if not baseline_path.is_file() or not candidate_path.is_file():
                message = (
                    f"Missing prediction file for {threshold_days}/"
                    f"{model_name}/{split_name}"
                )
                failures.append(message)
                rows.append(
                    {
                        "threshold_days": threshold_days,
                        "model_name": model_name,
                        "split": split_name,
                        "status": "missing",
                    }
                )
                continue

            baseline = pd.read_csv(baseline_path, low_memory=False)
            candidate = pd.read_csv(candidate_path, low_memory=False)
            baseline, candidate, alignment = _align_predictions(
                baseline,
                candidate,
            )

            baseline_probability = pd.to_numeric(
                baseline["positive_probability"],
                errors="raise",
            ).to_numpy(dtype=np.float64)
            candidate_probability = pd.to_numeric(
                candidate["positive_probability"],
                errors="raise",
            ).to_numpy(dtype=np.float64)

            absolute_difference = np.abs(
                baseline_probability - candidate_probability
            )
            probabilities_match = bool(
                np.allclose(
                    baseline_probability,
                    candidate_probability,
                    atol=args.probability_atol,
                    rtol=args.probability_rtol,
                )
            )

            y_match = bool(
                np.array_equal(
                    baseline["y_true"].to_numpy(),
                    candidate["y_true"].to_numpy(),
                )
            )
            los_match = bool(
                np.allclose(
                    pd.to_numeric(baseline["LOS"], errors="coerce"),
                    pd.to_numeric(candidate["LOS"], errors="coerce"),
                    equal_nan=True,
                )
            )
            status = (
                "passed"
                if probabilities_match and y_match and los_match
                else "failed"
            )

            if status == "failed":
                failures.append(
                    f"Regression mismatch: {threshold_days}/"
                    f"{model_name}/{split_name}"
                )

            rows.append(
                {
                    "threshold_days": threshold_days,
                    "model_name": model_name,
                    "split": split_name,
                    "status": status,
                    "alignment": alignment,
                    "row_count": int(len(baseline)),
                    "y_true_match": y_match,
                    "los_match": los_match,
                    "probabilities_match": probabilities_match,
                    "maximum_absolute_probability_difference": float(
                        absolute_difference.max(initial=0.0)
                    ),
                    "mean_absolute_probability_difference": float(
                        absolute_difference.mean()
                    ),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "prediction_regression_comparison.csv", index=False)

    summary = {
        "baseline_run": str(baseline_run),
        "candidate_run": str(candidate_run),
        "probability_atol": args.probability_atol,
        "probability_rtol": args.probability_rtol,
        "comparison_count": int(len(result)),
        "passed_count": int((result.get("status") == "passed").sum())
        if not result.empty
        else 0,
        "failed_count": len(failures),
        "failures": failures,
        "status": "passed" if not failures else "failed",
    }
    with (output_dir / "regression_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    if failures:
        print("[FAIL] Phase 1 regression comparison found differences.")
        for failure in failures:
            print(f"[ERROR] {failure}")
        return 1

    print("[OK] Candidate predictions match the frozen baseline run.")
    print(f"[INFO] Comparison output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
