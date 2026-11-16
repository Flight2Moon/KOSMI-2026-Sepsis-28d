from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from baseline_common import (
    iter_log_files,
    iter_run_files,
    iter_source_files,
    load_json,
    read_inventory_csv,
    resolve_run_id,
    sha256_file,
)


FREEZE_DIR_NAME = "baseline_freeze"
MANIFEST_NAME = "baseline_v1_manifest.json"
INVENTORY_NAME = "file_inventory.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a frozen Baseline V1 run, its source code, model "
            "artifacts, predictions, plots, and datasets have not changed."
        )
    )
    parser.add_argument(
        "--run-id",
        default="latest",
        help="Run directory name under runs/, or 'latest'.",
    )
    parser.add_argument(
        "--scope",
        choices=["run", "full"],
        default="run",
        help=(
            "'run' verifies immutable run outputs, freeze metadata, and "
            "datasets while allowing continued source development. "
            "'full' also requires the live project source to match the "
            "Baseline V1 snapshot."
        ),
    )
    parser.add_argument(
        "--allow-missing-datasets",
        action="store_true",
        help=(
            "Do not fail when original dataset paths are unavailable. "
            "Dataset hashes remain recorded but cannot be rechecked."
        ),
    )
    return parser.parse_args()


def parse_manifest_checksum(path: Path) -> str:
    """Read the expected manifest checksum from manifest_sha256.txt."""
    line = path.read_text(encoding="utf-8").strip()

    if not line:
        raise ValueError(f"Manifest checksum file is empty: {path}")

    return line.split()[0]


def verify_inventory_entries(
    entries: list[dict[str, Any]],
    project_root: Path,
    run_dir: Path,
    scope_mode: str,
) -> list[str]:
    """Verify every file listed in the frozen checksum inventory."""
    errors: list[str] = []

    for entry in entries:
        scope = str(entry["scope"])
        if scope_mode == "run" and scope != "run_output":
            continue
        relative_path = Path(str(entry["path"]))

        if scope in {"project_source", "project_log"}:
            path = project_root / relative_path
        elif scope == "run_output":
            path = run_dir / relative_path
        else:
            errors.append(
                f"Unknown inventory scope: {scope} path={relative_path}"
            )
            continue

        if not path.is_file():
            errors.append(f"Missing frozen file: {path}")
            continue

        actual_size = int(path.stat().st_size)
        expected_size = int(entry["size_bytes"])

        if actual_size != expected_size:
            errors.append(
                f"File size changed: {path} "
                f"expected={expected_size}, actual={actual_size}"
            )
            continue

        actual_hash = sha256_file(path)
        expected_hash = str(entry["sha256"])

        if actual_hash != expected_hash:
            errors.append(
                f"File checksum changed: {path} "
                f"expected={expected_hash}, actual={actual_hash}"
            )

    return errors


def verify_extra_files(
    entries: list[dict[str, Any]],
    project_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    scope_mode: str,
) -> list[str]:
    """Detect source or run files that appeared after the freeze."""
    errors: list[str] = []
    policy = manifest.get("verification_policy", {})

    expected_source = {
        str(entry["path"])
        for entry in entries
        if entry["scope"] == "project_source"
    }
    expected_logs = {
        str(entry["path"])
        for entry in entries
        if entry["scope"] == "project_log"
    }
    expected_run = {
        str(entry["path"])
        for entry in entries
        if entry["scope"] == "run_output"
    }

    current_source = {
        path.relative_to(project_root).as_posix()
        for path in iter_source_files(project_root)
    }
    current_logs = {
        path.relative_to(project_root).as_posix()
        for path in iter_log_files(project_root)
    }
    current_run = {
        path.relative_to(run_dir).as_posix()
        for path in iter_run_files(run_dir, FREEZE_DIR_NAME)
    }

    if scope_mode == "full" and policy.get(
        "source_extra_files_are_errors", True
    ):
        extra_source = sorted(current_source - expected_source)
        missing_source = sorted(expected_source - current_source)

        for path in extra_source:
            errors.append(f"Unexpected source file after freeze: {path}")
        for path in missing_source:
            errors.append(f"Frozen source file is no longer indexed: {path}")

    if policy.get("run_extra_files_are_errors", True):
        extra_run = sorted(current_run - expected_run)
        missing_run = sorted(expected_run - current_run)

        for path in extra_run:
            errors.append(f"Unexpected run file after freeze: {path}")
        for path in missing_run:
            errors.append(f"Frozen run file is no longer indexed: {path}")

    if scope_mode == "full" and policy.get(
        "project_log_extra_files_are_errors", False
    ):
        extra_logs = sorted(current_logs - expected_logs)

        for path in extra_logs:
            errors.append(f"Unexpected project log after freeze: {path}")

    return errors


def verify_datasets(
    manifest: dict[str, Any],
    allow_missing_datasets: bool,
) -> tuple[list[str], list[str]]:
    """Recalculate original dataset hashes when paths are accessible."""
    errors: list[str] = []
    messages: list[str] = []

    for dataset_name, dataset in sorted(
        manifest.get("datasets", {}).items()
    ):
        path = Path(str(dataset.get("path", ""))).expanduser()
        expected_hash = str(dataset.get("expected_sha256", ""))

        if not path.is_file():
            message = f"{dataset_name}: dataset unavailable at {path}"

            if allow_missing_datasets:
                messages.append("[SKIP] " + message)
            else:
                errors.append(message)
            continue

        actual_hash = sha256_file(path)

        if actual_hash != expected_hash:
            errors.append(
                f"{dataset_name}: dataset checksum mismatch at {path} "
                f"expected={expected_hash}, actual={actual_hash}"
            )
        else:
            messages.append(
                f"[OK] {dataset_name}: dataset checksum verified"
            )

    return errors, messages


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    runs_dir = project_root / "runs"
    run_id = resolve_run_id(runs_dir, args.run_id)
    run_dir = runs_dir / run_id
    freeze_dir = run_dir / FREEZE_DIR_NAME

    manifest_path = freeze_dir / MANIFEST_NAME
    checksum_path = freeze_dir / "manifest_sha256.txt"
    inventory_path = freeze_dir / INVENTORY_NAME

    required_paths = [
        manifest_path,
        checksum_path,
        inventory_path,
        freeze_dir / "source_snapshot.zip",
        freeze_dir / "experiment_config_frozen.json",
        freeze_dir / "pip_freeze.txt",
    ]

    missing_metadata = [
        path
        for path in required_paths
        if not path.is_file()
    ]

    if missing_metadata:
        details = "\n".join(f"- {path}" for path in missing_metadata)
        raise FileNotFoundError(
            "Baseline freeze metadata is incomplete:\n" + details
        )

    manifest = load_json(manifest_path)
    errors: list[str] = []

    expected_manifest_hash = parse_manifest_checksum(checksum_path)
    actual_manifest_hash = sha256_file(manifest_path)

    if actual_manifest_hash != expected_manifest_hash:
        errors.append(
            "Baseline manifest checksum mismatch: "
            f"expected={expected_manifest_hash}, actual={actual_manifest_hash}"
        )

    expected_checksums = manifest.get("checksums", {})

    checksum_targets = {
        "inventory_csv_sha256": inventory_path,
        "inventory_json_sha256": freeze_dir / "file_inventory.json",
        "pip_freeze_sha256": freeze_dir / "pip_freeze.txt",
        "source_snapshot_sha256": freeze_dir / "source_snapshot.zip",
        "frozen_config_sha256": freeze_dir / "experiment_config_frozen.json",
        "run_manifest_sha256": run_dir / "run_manifest.json",
        "experiment_records_sha256": run_dir / "experiment_records.json",
    }

    for checksum_key, path in checksum_targets.items():
        expected = expected_checksums.get(checksum_key)

        if expected is None:
            errors.append(f"Missing checksum in manifest: {checksum_key}")
            continue

        actual = sha256_file(path)

        if actual != expected:
            errors.append(
                f"Checksum mismatch for {path}: "
                f"expected={expected}, actual={actual}"
            )

    entries = read_inventory_csv(inventory_path)
    errors.extend(
        verify_inventory_entries(
            entries=entries,
            project_root=project_root,
            run_dir=run_dir,
            scope_mode=args.scope,
        )
    )
    errors.extend(
        verify_extra_files(
            entries=entries,
            project_root=project_root,
            run_dir=run_dir,
            manifest=manifest,
            scope_mode=args.scope,
        )
    )

    dataset_errors, dataset_messages = verify_datasets(
        manifest=manifest,
        allow_missing_datasets=args.allow_missing_datasets,
    )
    errors.extend(dataset_errors)

    print(f"[INFO] Verifying Baseline V1 run: {run_id}")
    checked_entries = [
        entry
        for entry in entries
        if args.scope == "full" or entry["scope"] == "run_output"
    ]
    print(f"[INFO] Verification scope: {args.scope}")
    print(f"[INFO] Inventory files checked: {len(checked_entries)}")

    for message in dataset_messages:
        print(message)

    if errors:
        print("[FAIL] Baseline verification failed.")

        for error in errors:
            print(f"[ERROR] {error}")

        return 1

    print("[OK] Baseline V1 verification completed without differences.")
    print(f"[INFO] Manifest SHA-256: {actual_manifest_hash}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
