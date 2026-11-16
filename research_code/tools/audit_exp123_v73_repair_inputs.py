#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_ROOT_FAILURES = {
    "exp1-targeted-optimize-svm",
    "exp1-targeted-optimize-xgboost",
    "stage1-targeted-optimize-svm",
    "stage2-targeted-optimize-svm",
}


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def audit(baseline_run: Path, optimized_run: Path) -> dict[str, Any]:
    baseline_run = baseline_run.resolve()
    optimized_run = optimized_run.resolve()
    result_path = (
        optimized_run
        / "common"
        / "progress"
        / "resource_staged_optimize_protocol_result.json"
    )
    result = _object(result_path)
    baseline_split = (
        baseline_run / "common" / "splits" / "split_hash.txt"
    ).read_text(encoding="utf-8").strip()
    optimized_split = (
        optimized_run / "common" / "splits" / "split_hash.txt"
    ).read_text(encoding="utf-8").strip()
    link = _object(
        optimized_run
        / "common"
        / "staged_optimized"
        / "BASELINE_LINK_VERIFIED.lock.json"
    )
    root_names = {
        str(row.get("name")) for row in result.get("root_failures", [])
    }
    unexpected = sorted(root_names - EXPECTED_ROOT_FAILURES)
    missing_expected = sorted(EXPECTED_ROOT_FAILURES - root_names)
    errors: list[str] = []
    if baseline_split != optimized_split:
        errors.append("Track 1 and Track 2 shared split hashes differ.")
    if str(link.get("shared_split_hash")) != optimized_split:
        errors.append("Baseline link shared split hash is inconsistent.")
    if str(link.get("baseline_run_id")) != baseline_run.name:
        errors.append("Baseline link points to a different Track 1 Run.")
    if unexpected:
        errors.append(f"Unexpected direct failures: {unexpected}")
    if missing_expected:
        errors.append(f"Expected repair failures were not found: {missing_expected}")
    counts = dict(result.get("counts", {}))
    if int(counts.get("completed", 0)) < 25:
        errors.append("Fewer than 25 Track 2 jobs are complete; repair scope is unsafe.")
    if bool(link.get("confirmatory_splits_accessed", True)):
        errors.append("Baseline link indicates confirmatory split access.")
    fixed_cv_hashes: dict[str, str] = {}
    for task_name in (
        "exp1_flat_3class",
        "stage1_mortality",
        "stage2_long_stay",
    ):
        path = optimized_run / "common" / "fixed_cv" / task_name / "fixed_cv_hash.txt"
        fixed_cv_hashes[task_name] = path.read_text(encoding="utf-8").strip()
    return {
        "schema_version": "exp123_v73_repair_input_audit_v1",
        "status": "eligible_for_v73_track2_repair" if not errors else "blocked",
        "baseline_run_id": baseline_run.name,
        "optimized_run_id": optimized_run.name,
        "counts": counts,
        "direct_root_failures": sorted(root_names),
        "expected_direct_root_failures": sorted(EXPECTED_ROOT_FAILURES),
        "shared_split_hash": optimized_split,
        "fixed_cv_hashes": fixed_cv_hashes,
        "baseline_link": link,
        "confirmatory_splits_accessed": False,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--optimized-run-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    payload = audit(Path(args.baseline_run_dir), Path(args.optimized_run_dir))
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["status"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
