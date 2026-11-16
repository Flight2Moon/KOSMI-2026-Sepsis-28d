from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from baseline_common import (
    classify_run_file,
    count_completed_experiments,
    iter_log_files,
    iter_run_files,
    iter_source_files,
    load_json,
    make_inventory_entry,
    resolve_run_id,
    sha256_file,
    summarize_inventory,
    write_inventory_csv,
    write_json,
)


FREEZE_DIR_NAME = "baseline_freeze"
MANIFEST_NAME = "baseline_v1_manifest.json"
INVENTORY_NAME = "file_inventory.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze one completed LOS experiment run by recording checksums "
            "for code, configuration, model artifacts, predictions, plots, "
            "TensorBoard logs, and datasets."
        )
    )
    parser.add_argument(
        "--run-id",
        default="latest",
        help="Run directory name under runs/, or 'latest'.",
    )
    parser.add_argument(
        "--execution-command",
        default="unknown",
        help="Original command used to start the experiment.",
    )
    parser.add_argument(
        "--allow-missing-datasets",
        action="store_true",
        help=(
            "Allow dataset files to be unavailable and preserve the hashes "
            "already recorded in run_manifest.json."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing baseline_freeze directory.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> tuple[int, str]:
    """Run a command and return its exit code and stripped output."""
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        output = (result.stdout or result.stderr).strip()
        return int(result.returncode), output
    except OSError as exception:
        return 1, str(exception)


def collect_git_information(project_root: Path) -> dict[str, Any]:
    """Collect Git commit and dirty-state information when available."""
    if not (project_root / ".git").exists():
        return {
            "available": False,
            "commit": None,
            "branch": None,
            "dirty": None,
            "status": "Project directory is not a Git working tree.",
        }

    commit_code, commit = run_command(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"]
    )
    branch_code, branch = run_command(
        ["git", "-C", str(project_root), "rev-parse", "--abbrev-ref", "HEAD"]
    )
    status_code, status = run_command(
        ["git", "-C", str(project_root), "status", "--porcelain"]
    )

    return {
        "available": commit_code == 0,
        "commit": commit if commit_code == 0 else None,
        "branch": branch if branch_code == 0 else None,
        "dirty": bool(status) if status_code == 0 else None,
        "status": status if status_code == 0 else None,
    }


def collect_freeze_environment() -> dict[str, Any]:
    """Record the environment used to create the freeze manifest."""
    pip_code, pip_output = run_command(
        [sys.executable, "-m", "pip", "freeze"]
    )

    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pip_freeze_available": pip_code == 0,
        "pip_freeze_line_count": (
            len([line for line in pip_output.splitlines() if line.strip()])
            if pip_code == 0
            else 0
        ),
        "pip_freeze_text": pip_output if pip_code == 0 else "",
    }


def create_source_snapshot(
    source_paths: list[Path],
    project_root: Path,
    output_path: Path,
) -> None:
    """Create a deterministic ZIP snapshot of source and configuration files."""
    with zipfile.ZipFile(
        output_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for source_path in sorted(source_paths):
            relative_path = source_path.relative_to(project_root).as_posix()
            info = zipfile.ZipInfo(relative_path)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source_path.read_bytes())


def verify_or_record_datasets(
    run_manifest: dict[str, Any],
    allow_missing_datasets: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Verify datasets against run-time hashes or preserve recorded hashes."""
    dataset_results: dict[str, Any] = {}
    errors: list[str] = []

    for dataset_name, recorded in run_manifest.get("datasets", {}).items():
        path = Path(str(recorded.get("path", ""))).expanduser()
        expected_hash = str(recorded.get("sha256", ""))
        result: dict[str, Any] = {
            "path": str(path),
            "expected_sha256": expected_hash,
            "recorded_rows": recorded.get("rows"),
            "exists_at_freeze": path.is_file(),
        }

        if path.is_file():
            actual_hash = sha256_file(path)
            matches = actual_hash == expected_hash
            result.update(
                {
                    "verification_status": "verified" if matches else "mismatch",
                    "actual_sha256": actual_hash,
                    "matches_recorded_hash": matches,
                    "size_bytes": int(path.stat().st_size),
                }
            )

            if not matches:
                errors.append(
                    f"Dataset checksum mismatch for {dataset_name}: {path}"
                )
        else:
            result.update(
                {
                    "verification_status": "recorded_hash_only",
                    "actual_sha256": None,
                    "matches_recorded_hash": None,
                    "size_bytes": None,
                }
            )

            if not allow_missing_datasets:
                errors.append(f"Dataset file was not found: {path}")

        dataset_results[dataset_name] = result

    return dataset_results, errors


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    runs_dir = project_root / "runs"
    run_id = resolve_run_id(runs_dir, args.run_id)
    run_dir = runs_dir / run_id

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory was not found: {run_dir}")

    freeze_dir = run_dir / FREEZE_DIR_NAME

    if freeze_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Freeze directory already exists: {freeze_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(freeze_dir)

    run_manifest_path = run_dir / "run_manifest.json"
    records_path = run_dir / "experiment_records.json"

    if not run_manifest_path.is_file():
        raise FileNotFoundError(
            f"run_manifest.json was not found: {run_manifest_path}"
        )
    if not records_path.is_file():
        raise FileNotFoundError(
            f"experiment_records.json was not found: {records_path}"
        )

    run_manifest = load_json(run_manifest_path)
    experiment_records = load_json(records_path)

    expected_thresholds = [
        float(value)
        for value in run_manifest.get("thresholds", [])
    ]
    expected_models = [
        str(value)
        for value in run_manifest.get("models", [])
    ]

    experiment_rows, experiment_errors = count_completed_experiments(
        run_dir=run_dir,
        expected_thresholds=expected_thresholds,
        expected_models=expected_models,
    )

    record_errors: list[str] = []

    if not isinstance(experiment_records, list):
        record_errors.append("experiment_records.json must contain a list.")
    else:
        completed_records = [
            row
            for row in experiment_records
            if str(row.get("status")) == "completed"
        ]
        expected_count = len(expected_thresholds) * len(expected_models)

        if len(experiment_records) != expected_count:
            record_errors.append(
                "Unexpected experiment record count: "
                f"expected={expected_count}, actual={len(experiment_records)}"
            )
        if len(completed_records) != expected_count:
            record_errors.append(
                "Not all experiment records are completed: "
                f"expected={expected_count}, completed={len(completed_records)}"
            )

    dataset_results, dataset_errors = verify_or_record_datasets(
        run_manifest=run_manifest,
        allow_missing_datasets=args.allow_missing_datasets,
    )

    validation_errors = (
        experiment_errors
        + record_errors
        + dataset_errors
    )

    if validation_errors:
        details = "\n".join(f"- {message}" for message in validation_errors)
        raise RuntimeError(
            "Baseline freeze validation failed:\n" + details
        )

    freeze_dir.mkdir(parents=True, exist_ok=False)

    source_paths = list(iter_source_files(project_root))
    log_paths = list(iter_log_files(project_root))
    run_paths = list(iter_run_files(run_dir, FREEZE_DIR_NAME))

    source_entries = [
        make_inventory_entry(
            path=path,
            base_dir=project_root,
            scope="project_source",
            category="source",
        )
        for path in source_paths
    ]
    log_entries = [
        make_inventory_entry(
            path=path,
            base_dir=project_root,
            scope="project_log",
            category="console_log",
        )
        for path in log_paths
    ]
    run_entries = [
        make_inventory_entry(
            path=path,
            base_dir=run_dir,
            scope="run_output",
            category=classify_run_file(path.relative_to(run_dir)),
        )
        for path in run_paths
    ]

    all_entries = sorted(
        source_entries + log_entries + run_entries,
        key=lambda item: (
            str(item["scope"]),
            str(item["path"]),
        ),
    )

    inventory_path = freeze_dir / INVENTORY_NAME
    write_inventory_csv(inventory_path, all_entries)
    write_json(freeze_dir / "file_inventory.json", all_entries)

    source_snapshot_path = freeze_dir / "source_snapshot.zip"
    create_source_snapshot(
        source_paths=source_paths,
        project_root=project_root,
        output_path=source_snapshot_path,
    )

    freeze_environment = collect_freeze_environment()
    pip_freeze_text = str(freeze_environment.pop("pip_freeze_text"))
    (freeze_dir / "pip_freeze.txt").write_text(
        pip_freeze_text + ("\n" if pip_freeze_text else ""),
        encoding="utf-8",
    )

    copied_config_path = freeze_dir / "experiment_config_frozen.json"
    config_payload = run_manifest.get("config", {})
    write_json(copied_config_path, config_payload)

    source_summary = summarize_inventory(source_entries)
    log_summary = summarize_inventory(log_entries)
    run_summary = summarize_inventory(run_entries)
    artifact_entries = [
        entry
        for entry in run_entries
        if entry["category"] == "model_artifact"
    ]
    artifact_summary = summarize_inventory(artifact_entries)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "baseline_name": "baseline_v1",
        "freeze_status": "completed",
        "run_id": run_id,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_execution_command": args.execution_command,
        "project": {
            "project_root_at_run": (
                run_manifest.get("config", {}).get("_project_root")
            ),
            "project_root_at_freeze": str(project_root),
            "run_directory": str(run_dir),
        },
        "recorded_run_environment": run_manifest.get("environment", {}),
        "freeze_environment": freeze_environment,
        "git": collect_git_information(project_root),
        "datasets": dataset_results,
        "experiment_matrix": {
            "thresholds": expected_thresholds,
            "models": expected_models,
            "expected_experiment_count": (
                len(expected_thresholds) * len(expected_models)
            ),
            "completed_experiment_count": len(experiment_rows),
            "experiments": experiment_rows,
        },
        "checksums": {
            "run_manifest_sha256": sha256_file(run_manifest_path),
            "experiment_records_sha256": sha256_file(records_path),
            "frozen_config_sha256": sha256_file(copied_config_path),
            "requirements_sha256": (
                sha256_file(project_root / "requirements.txt")
                if (project_root / "requirements.txt").is_file()
                else None
            ),
            "source_snapshot_sha256": sha256_file(source_snapshot_path),
            "inventory_csv_sha256": sha256_file(inventory_path),
            "inventory_json_sha256": sha256_file(
                freeze_dir / "file_inventory.json"
            ),
            "pip_freeze_sha256": sha256_file(
                freeze_dir / "pip_freeze.txt"
            ),
        },
        "inventory": {
            "source": source_summary,
            "project_logs": log_summary,
            "run_outputs": run_summary,
            "model_artifacts": artifact_summary,
            "inventory_csv": INVENTORY_NAME,
            "inventory_json": "file_inventory.json",
        },
        "verification_policy": {
            "dataset_files_required": not args.allow_missing_datasets,
            "source_extra_files_are_errors": True,
            "run_extra_files_are_errors": True,
            "project_log_extra_files_are_errors": False,
            "hash_algorithm": "SHA-256",
        },
    }

    manifest_path = freeze_dir / MANIFEST_NAME
    write_json(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)
    (freeze_dir / "manifest_sha256.txt").write_text(
        f"{manifest_hash}  {MANIFEST_NAME}\n",
        encoding="utf-8",
    )

    verification_summary = [
        "Baseline V1 Freeze Summary",
        "=" * 72,
        f"Run ID: {run_id}",
        f"Completed experiments: {len(experiment_rows)}",
        f"Source files: {source_summary['file_count']}",
        f"Run output files: {run_summary['file_count']}",
        f"Model artifact files: {artifact_summary['file_count']}",
        f"Console log files: {log_summary['file_count']}",
        f"Manifest SHA-256: {manifest_hash}",
        "",
        "Dataset verification:",
    ]

    for dataset_name, dataset_result in sorted(dataset_results.items()):
        verification_summary.append(
            f"- {dataset_name}: {dataset_result['verification_status']} "
            f"({dataset_result['expected_sha256']})"
        )

    verification_summary.extend(
        [
            "",
            "Run the following command to verify this baseline:",
            f"bash verify_baseline.sh {run_id}",
            "",
        ]
    )

    (freeze_dir / "verification_summary.txt").write_text(
        "\n".join(verification_summary),
        encoding="utf-8",
    )

    (freeze_dir / "BASELINE_FROZEN.txt").write_text(
        "This directory records the immutable Baseline V1 state.\n"
        "Do not edit files under the frozen run. Create a new run for later phases.\n",
        encoding="utf-8",
    )

    print(f"[OK] Baseline V1 frozen: {freeze_dir}")
    print(f"[INFO] Run ID: {run_id}")
    print(f"[INFO] Completed experiments: {len(experiment_rows)}")
    print(f"[INFO] Run output files hashed: {run_summary['file_count']}")
    print(f"[INFO] Model artifact files hashed: {artifact_summary['file_count']}")
    print(f"[INFO] Manifest SHA-256: {manifest_hash}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
