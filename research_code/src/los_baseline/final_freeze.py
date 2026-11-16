from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .constants import MODEL_ORDER
from .task_spec import (
    EXP1_FLAT_3CLASS_TASK,
    STAGE1_MORTALITY_TASK,
    STAGE2_LONG_STAY_TASK,
)
from .utils import atomic_write_json, sha256_file
from .final_freeze_inventory import (
    IGNORED_DIRECTORY_NAMES,
    IGNORED_FILE_NAMES,
    SOURCE_DIRECTORIES,
    SOURCE_ROOT_SUFFIXES,
    _deterministic_zip,
    _inventory_digest,
    _inventory_entry,
    _is_ignored,
    _read_inventory_csv,
    _write_inventory_csv,
    iter_release_source_files,
)
from .final_freeze_requirements import (
    _normalize_distribution_name,
    collect_installed_versions,
    parse_exact_requirements,
    validate_installed_versions,
)

RELEASE_DIR_NAME = "exp123_release_freeze"
RELEASE_MANIFEST_NAME = "release_manifest.json"
PREFLIGHT_DIR_NAME = "preflight"
FROZEN_DESIGN_DIR_NAME = "frozen_design"
FINAL_RUN_FREEZE_DIR_NAME = "final_run_freeze"

DISCOVERY_STEPS = (
    "outcomes",
    "splits",
    "exp1-optimize",
    "exp1-train",
    "stage1-optimize",
    "stage1-train",
    "stage2-optimize",
    "stage2-train",
    "stage1-heterogeneity-discovery",
    "stage2-heterogeneity-discovery",
    "stage1-super-learner",
    "stage2-super-learner",
)

CONFIRMATORY_SEQUENCE = (
    {
        "order": 1,
        "step": "stage1-heterogeneity-confirmatory",
        "split": "internal_test",
    },
    {
        "order": 2,
        "step": "stage2-heterogeneity-confirmatory",
        "split": "internal_test",
    },
    {
        "order": 3,
        "step": "oracle-pipeline-confirmatory",
        "split": "internal_test",
    },
    {
        "order": 4,
        "step": "final-comparison-confirmatory",
        "split": "internal_test",
    },
    {
        "order": 5,
        "step": "stage1-heterogeneity-confirmatory",
        "split": "external_eicu",
    },
    {
        "order": 6,
        "step": "stage2-heterogeneity-confirmatory",
        "split": "external_eicu",
    },
    {
        "order": 7,
        "step": "oracle-pipeline-confirmatory",
        "split": "external_eicu",
    },
    {
        "order": 8,
        "step": "final-comparison-confirmatory",
        "split": "external_eicu",
    },
)


@dataclass(frozen=True)
class ReleaseFreezeResult:
    output_dir: str
    manifest_path: str
    manifest_sha256: str
    source_file_count: int
    environment_package_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreflightResult:
    run_id: str
    output_dir: str
    lock_path: str
    lock_sha256: str
    release_manifest_sha256: str
    config_sha256: str
    dataset_sha256: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConfirmatoryFreezeResult:
    run_id: str
    output_dir: str
    manifest_path: str
    manifest_sha256: str
    frozen_file_count: int
    shared_split_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FinalRunFreezeResult:
    run_id: str
    output_dir: str
    manifest_path: str
    manifest_sha256: str
    frozen_file_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file was not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _write_sha256_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        digest + "\n", encoding="utf-8"
    )
    return digest


def _parse_sha256_sidecar(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Checksum sidecar was not found: {path}")
    value = path.read_text(encoding="utf-8").strip().split()[0]
    if len(value) != 64:
        raise ValueError(f"Invalid SHA-256 value in {path}: {value!r}")
    return value


def _pip_freeze_text() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def execution_protocol_payload() -> dict[str, Any]:
    return {
        "protocol_version": "exp123_protocol_v1",
        "discovery_steps": list(DISCOVERY_STEPS),
        "confirmatory_freeze_required_before_test_access": True,
        "confirmatory_sequence": list(CONFIRMATORY_SEQUENCE),
        "internal_test_and_eicu_single_use": True,
        "discovery_mutation_after_confirmatory_freeze": False,
        "confirmatory_overwrite_allowed": False,
        "final_run_freeze_required_after_completion": True,
    }


def freeze_exp123_release(
    *,
    project_root: Path,
    config_path: Path,
    overwrite: bool = False,
    strict_environment: bool = True,
) -> ReleaseFreezeResult:
    """Freeze the final implementation release before any clinical run starts."""
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    output_dir = project_root / RELEASE_DIR_NAME
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"EXP123 release freeze already exists: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_json(config_path)
    development = config.get("development", {})
    if not bool(development.get("model_training_enabled", False)):
        raise RuntimeError(
            "The final release cannot be frozen while model_training_enabled is false."
        )
    required_completed = {
        "outcomes",
        "common_three_class_splits",
        "shared_training_engine",
        "optimization_engine",
        "exp1_flat_multiclass",
        "stage1_mortality",
        "stage2_long_stay_and_exp3",
        "stage_heterogeneity",
        "super_learner",
        "oracle_pipeline",
        "final_statistical_comparison",
        "final_freeze",
    }
    completed = set(development.get("completed_steps", []))
    missing_steps = sorted(required_completed - completed)
    if missing_steps:
        raise RuntimeError(
            "The final release is missing completed implementation steps: "
            f"{missing_steps}"
        )

    requirements_path = project_root / "requirements.txt"
    expected_versions = parse_exact_requirements(requirements_path)
    installed_versions, missing_packages = collect_installed_versions(expected_versions)
    environment_errors = validate_installed_versions(
        expected_versions, installed_versions
    )
    if strict_environment and (missing_packages or environment_errors):
        details = "\n".join(
            [*(f"Missing package: {name}" for name in missing_packages), *environment_errors]
        )
        raise RuntimeError(
            "The current environment does not match requirements.txt:\n" + details
        )

    source_paths = list(iter_release_source_files(project_root))
    source_entries = [
        _inventory_entry(path, project_root, "project_source")
        for path in source_paths
    ]
    source_inventory_csv = output_dir / "source_inventory.csv"
    source_inventory_json = output_dir / "source_inventory.json"
    _write_inventory_csv(source_inventory_csv, source_entries)
    atomic_write_json(source_inventory_json, source_entries)

    source_snapshot = output_dir / "source_snapshot.zip"
    _deterministic_zip(source_paths, project_root, source_snapshot)
    frozen_config = output_dir / "experiment_config_frozen.json"
    shutil.copy2(config_path, frozen_config)

    requirements_lock = output_dir / "requirements.lock.txt"
    requirements_lock.write_text(
        "\n".join(
            f"{name}=={version}" for name, version in sorted(installed_versions.items())
        )
        + "\n",
        encoding="utf-8",
    )
    pip_freeze_path = output_dir / "pip_freeze_full.txt"
    pip_freeze_path.write_text(_pip_freeze_text() + "\n", encoding="utf-8")

    environment_lock = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "expected_direct_requirements": expected_versions,
        "installed_direct_requirements": installed_versions,
        "missing_direct_requirements": missing_packages,
        "strict_environment": strict_environment,
    }
    environment_lock_path = output_dir / "environment_lock.json"
    atomic_write_json(environment_lock_path, environment_lock)
    environment_yml = output_dir / "environment.lock.yml"
    environment_yml.write_text(
        "name: medical_26\n"
        "channels:\n"
        "  - conda-forge\n"
        "dependencies:\n"
        f"  - python={platform.python_version()}\n"
        "  - pip\n"
        "  - pip:\n"
        + "".join(
            f"      - {name}=={version}\n"
            for name, version in sorted(installed_versions.items())
        ),
        encoding="utf-8",
    )

    protocol_path = output_dir / "execution_protocol.json"
    atomic_write_json(protocol_path, execution_protocol_payload())

    manifest = {
        "release_schema_version": "exp123_release_v1",
        "status": "frozen",
        "created_at_utc": utc_now_iso(),
        "project_root": str(project_root),
        "config_path": str(config_path),
        "source_summary": {
            "file_count": len(source_entries),
            "total_bytes": int(sum(item["size_bytes"] for item in source_entries)),
            "inventory_sha256": _inventory_digest(source_entries),
        },
        "environment": environment_lock,
        "execution_protocol": execution_protocol_payload(),
        "checksums": {
            "source_inventory_csv_sha256": sha256_file(source_inventory_csv),
            "source_inventory_json_sha256": sha256_file(source_inventory_json),
            "source_snapshot_sha256": sha256_file(source_snapshot),
            "frozen_config_sha256": sha256_file(frozen_config),
            "requirements_source_sha256": sha256_file(requirements_path),
            "requirements_lock_sha256": sha256_file(requirements_lock),
            "pip_freeze_full_sha256": sha256_file(pip_freeze_path),
            "environment_lock_sha256": sha256_file(environment_lock_path),
            "environment_yml_sha256": sha256_file(environment_yml),
            "execution_protocol_sha256": sha256_file(protocol_path),
        },
        "verification_policy": {
            "source_changes_are_errors": True,
            "config_changes_are_errors": True,
            "direct_package_version_changes_are_errors": True,
            "local_build_suffix_allowed": True,
        },
    }
    manifest_path = output_dir / RELEASE_MANIFEST_NAME
    atomic_write_json(manifest_path, manifest)
    manifest_sha256 = _write_sha256_sidecar(manifest_path)
    (output_dir / "EXP123_RELEASE_FROZEN.txt").write_text(
        "EXP123 implementation release is frozen.\n"
        f"Manifest SHA-256: {manifest_sha256}\n",
        encoding="utf-8",
    )
    (output_dir / "verification_summary.txt").write_text(
        "EXP123 release freeze completed.\n"
        f"Source files: {len(source_entries)}\n"
        f"Direct packages recorded: {len(installed_versions)}\n"
        f"Manifest SHA-256: {manifest_sha256}\n",
        encoding="utf-8",
    )
    return ReleaseFreezeResult(
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha256,
        source_file_count=len(source_entries),
        environment_package_count=len(installed_versions),
    )


def verify_exp123_release(
    *,
    project_root: Path,
    verify_environment: bool = True,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    output_dir = project_root / RELEASE_DIR_NAME
    manifest_path = output_dir / RELEASE_MANIFEST_NAME
    sidecar_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    manifest = _load_json(manifest_path)
    expected_manifest_sha = _parse_sha256_sidecar(sidecar_path)
    actual_manifest_sha = sha256_file(manifest_path)
    errors: list[str] = []
    if expected_manifest_sha != actual_manifest_sha:
        errors.append(
            "Release manifest checksum mismatch: "
            f"expected={expected_manifest_sha}, actual={actual_manifest_sha}"
        )

    checksum_paths = {
        "source_inventory_csv_sha256": output_dir / "source_inventory.csv",
        "source_inventory_json_sha256": output_dir / "source_inventory.json",
        "source_snapshot_sha256": output_dir / "source_snapshot.zip",
        "frozen_config_sha256": output_dir / "experiment_config_frozen.json",
        "requirements_source_sha256": project_root / "requirements.txt",
        "requirements_lock_sha256": output_dir / "requirements.lock.txt",
        "pip_freeze_full_sha256": output_dir / "pip_freeze_full.txt",
        "environment_lock_sha256": output_dir / "environment_lock.json",
        "environment_yml_sha256": output_dir / "environment.lock.yml",
        "execution_protocol_sha256": output_dir / "execution_protocol.json",
    }
    for key, path in checksum_paths.items():
        if not path.is_file():
            errors.append(f"Missing release freeze file: {path}")
            continue
        expected = manifest.get("checksums", {}).get(key)
        actual = sha256_file(path)
        if expected != actual:
            errors.append(
                f"Release file checksum mismatch: {path} expected={expected}, actual={actual}"
            )

    inventory_path = output_dir / "source_inventory.csv"
    if inventory_path.is_file():
        expected_entries = _read_inventory_csv(inventory_path)
        expected_paths = {str(item["path"]) for item in expected_entries}
        current_paths = {
            path.relative_to(project_root).as_posix()
            for path in iter_release_source_files(project_root)
        }
        for missing in sorted(expected_paths - current_paths):
            errors.append(f"Frozen source file is missing: {missing}")
        for extra in sorted(current_paths - expected_paths):
            errors.append(f"Unexpected source file after release freeze: {extra}")
        for entry in expected_entries:
            path = project_root / str(entry["path"])
            if not path.is_file():
                continue
            if int(path.stat().st_size) != int(entry["size_bytes"]):
                errors.append(f"Frozen source size changed: {path}")
                continue
            actual = sha256_file(path)
            if actual != str(entry["sha256"]):
                errors.append(f"Frozen source checksum changed: {path}")

    if verify_environment:
        environment_lock = _load_json(output_dir / "environment_lock.json")
        expected_versions = {
            _normalize_distribution_name(name): str(version)
            for name, version in environment_lock.get(
                "installed_direct_requirements", {}
            ).items()
        }
        installed, _ = collect_installed_versions(expected_versions)
        errors.extend(validate_installed_versions(expected_versions, installed))
        expected_python = str(environment_lock.get("python_version", ""))
        if platform.python_version() != expected_python:
            errors.append(
                "Python version mismatch: "
                f"expected={expected_python}, actual={platform.python_version()}"
            )

    result = {
        "status": "verified" if not errors else "failed",
        "manifest_sha256": actual_manifest_sha,
        "source_file_count": manifest.get("source_summary", {}).get("file_count"),
        "environment_verified": verify_environment,
        "errors": errors,
    }
    if errors:
        raise RuntimeError(
            "EXP123 release verification failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    return result


def _read_csv_header(path: Path) -> list[str]:
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        try:
            return next(reader)
        except StopIteration as exception:
            raise ValueError(f"CSV file is empty: {path}") from exception


def _dataset_preflight(
    *,
    name: str,
    path: Path,
    required_columns: set[str],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} dataset was not found: {path}")
    header = _read_csv_header(path)
    missing = sorted(required_columns - set(header))
    if missing:
        raise KeyError(f"{name} dataset is missing required columns: {missing}")
    stat = path.stat()
    return {
        "name": name,
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
        "column_count": len(header),
        "required_columns": sorted(required_columns),
        "missing_columns": [],
    }


def run_exp123_preflight(
    *,
    project_root: Path,
    config_path: Path,
    run_id: str,
    overwrite: bool = False,
) -> PreflightResult:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    release_verification = verify_exp123_release(
        project_root=project_root, verify_environment=True
    )
    config = _load_json(config_path)
    release_dir = project_root / RELEASE_DIR_NAME
    release_manifest_path = release_dir / RELEASE_MANIFEST_NAME
    release_manifest_sha = sha256_file(release_manifest_path)
    frozen_config_path = release_dir / "experiment_config_frozen.json"
    if sha256_file(config_path) != sha256_file(frozen_config_path):
        raise RuntimeError(
            "The live experiment config does not match the frozen release config."
        )

    runs_dir_value = Path(str(config["paths"]["runs_dir"])).expanduser()
    runs_dir = (
        runs_dir_value
        if runs_dir_value.is_absolute()
        else (project_root / runs_dir_value)
    ).resolve()
    run_dir = runs_dir / run_id
    output_dir = run_dir / "common" / PREFLIGHT_DIR_NAME
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            existing_lock = output_dir / "PREFLIGHT_PASSED.lock.json"
            if existing_lock.is_file():
                verify_run_preflight(
                    project_root=project_root,
                    config_path=config_path,
                    run_dir=run_dir,
                    verify_dataset_hashes=True,
                )
                payload = _load_json(existing_lock)
                return PreflightResult(
                    run_id=run_id,
                    output_dir=str(output_dir),
                    lock_path=str(existing_lock),
                    lock_sha256=sha256_file(existing_lock),
                    release_manifest_sha256=release_manifest_sha,
                    config_sha256=sha256_file(config_path),
                    dataset_sha256={
                        name: str(value["sha256"])
                        for name, value in payload["datasets"].items()
                    },
                )
            raise FileExistsError(f"Preflight output already exists: {output_dir}")
        downstream = [
            run_dir / "common" / "outcomes",
            run_dir / "common" / "splits",
            run_dir / "optimization",
            run_dir / "EXP1_flat_3class",
            run_dir / "EXP2_two_stage",
            run_dir / "EXP3_survivor_binary",
            run_dir / FROZEN_DESIGN_DIR_NAME,
        ]
        if any(path.exists() for path in downstream):
            raise RuntimeError(
                "Preflight cannot be overwritten after downstream run outputs exist."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = config["data"]
    outcomes = config["outcomes"]
    required_columns = set(data_config["feature_columns"])
    required_columns.add(str(outcomes["los_column"]))
    required_columns.add(str(outcomes["mortality_column"]))
    required_columns.update(str(value) for value in data_config["identifier_columns"])

    datasets: dict[str, dict[str, Any]] = {}
    for name, key in (("mimic", "mimic_csv"), ("eicu", "eicu_csv")):
        dataset_path = Path(str(config["paths"][key])).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = project_root / dataset_path
        datasets[name] = _dataset_preflight(
            name=name,
            path=dataset_path.resolve(),
            required_columns=required_columns,
        )

    minimum_free_gb = float(
        config.get("final_freeze", {}).get("minimum_free_disk_gb", 5.0)
    )
    runs_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(runs_dir).free
    if free_bytes < minimum_free_gb * 1024**3:
        raise RuntimeError(
            f"Insufficient free disk space: {free_bytes / 1024**3:.2f} GiB "
            f"< required {minimum_free_gb:.2f} GiB"
        )

    lock_payload = {
        "schema_version": "exp123_preflight_v1",
        "status": "passed",
        "created_at_utc": utc_now_iso(),
        "run_id": run_id,
        "release_manifest_sha256": release_manifest_sha,
        "release_verification": release_verification,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "datasets": datasets,
        "disk": {
            "runs_dir": str(runs_dir),
            "free_bytes": int(free_bytes),
            "minimum_free_disk_gb": minimum_free_gb,
        },
        "execution_protocol": execution_protocol_payload(),
        "feature_set": config.get("feature_set"),
        "ablation": config.get("ablation"),
        "confirmatory_accessed": False,
    }
    lock_path = output_dir / "PREFLIGHT_PASSED.lock.json"
    atomic_write_json(lock_path, lock_payload)
    lock_sha = _write_sha256_sidecar(lock_path)
    atomic_write_json(output_dir / "dataset_fingerprints.json", datasets)
    if isinstance(config.get("feature_set"), dict):
        atomic_write_json(output_dir / "resolved_feature_set.json", config["feature_set"])
    atomic_write_json(
        output_dir / "preflight_manifest.json",
        {
            "status": "passed",
            "lock_path": str(lock_path),
            "lock_sha256": lock_sha,
            "next_step": "outcomes",
        },
    )
    return PreflightResult(
        run_id=run_id,
        output_dir=str(output_dir),
        lock_path=str(lock_path),
        lock_sha256=lock_sha,
        release_manifest_sha256=release_manifest_sha,
        config_sha256=sha256_file(config_path),
        dataset_sha256={name: str(value["sha256"]) for name, value in datasets.items()},
    )


def verify_run_preflight(
    *,
    project_root: Path,
    config_path: Path,
    run_dir: Path,
    verify_dataset_hashes: bool = False,
) -> dict[str, Any]:
    verify_exp123_release(project_root=project_root, verify_environment=True)
    lock_path = run_dir / "common" / PREFLIGHT_DIR_NAME / "PREFLIGHT_PASSED.lock.json"
    sidecar = lock_path.with_suffix(lock_path.suffix + ".sha256")
    payload = _load_json(lock_path)
    expected_lock_sha = _parse_sha256_sidecar(sidecar)
    actual_lock_sha = sha256_file(lock_path)
    errors: list[str] = []
    if expected_lock_sha != actual_lock_sha:
        errors.append("Preflight lock checksum mismatch.")
    if payload.get("status") != "passed":
        errors.append("Preflight status is not passed.")
    config_path = config_path.resolve()
    if sha256_file(config_path) != str(payload.get("config_sha256")):
        errors.append("Configuration checksum changed after preflight.")
    release_manifest = project_root / RELEASE_DIR_NAME / RELEASE_MANIFEST_NAME
    if sha256_file(release_manifest) != str(payload.get("release_manifest_sha256")):
        errors.append("Release manifest changed after preflight.")
    for name, dataset in payload.get("datasets", {}).items():
        path = Path(str(dataset["path"]))
        if not path.is_file():
            errors.append(f"{name} dataset is unavailable: {path}")
            continue
        stat = path.stat()
        if int(stat.st_size) != int(dataset["size_bytes"]):
            errors.append(f"{name} dataset size changed after preflight.")
            continue
        if verify_dataset_hashes and sha256_file(path) != str(dataset["sha256"]):
            errors.append(f"{name} dataset checksum changed after preflight.")
    if errors:
        raise RuntimeError(
            "EXP123 run preflight verification failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    return {
        "status": "verified",
        "lock_sha256": actual_lock_sha,
        "run_id": payload.get("run_id"),
        "datasets_verified_by_hash": verify_dataset_hashes,
    }


def _required_discovery_files(run_dir: Path) -> list[Path]:
    paths = [
        run_dir / "common" / "outcomes" / "outcome_step_manifest.json",
        run_dir / "common" / "outcomes" / "mimic_outcome_assignments.csv.gz",
        run_dir / "common" / "splits" / "split_manifest.json",
        run_dir / "common" / "splits" / "split_hash.txt",
        run_dir / "EXP1_flat_3class" / "representative_model.json",
        run_dir / "EXP1_flat_3class" / "frozen_exp1_design.json",
        run_dir / "EXP2_two_stage" / "stage1_mortality" / "frozen_stage1_design.json",
        run_dir / "EXP2_two_stage" / "stage2_long_stay" / "frozen_stage2_design.json",
        run_dir / "EXP3_survivor_binary" / "artifact_reference.json",
        run_dir
        / "EXP2_two_stage"
        / "heterogeneity_analysis"
        / STAGE1_MORTALITY_TASK.task_name
        / "discovery_validation"
        / "frozen_design.json",
        run_dir
        / "EXP2_two_stage"
        / "heterogeneity_analysis"
        / STAGE1_MORTALITY_TASK.task_name
        / "discovery_validation"
        / "library_candidates.json",
        run_dir
        / "EXP2_two_stage"
        / "heterogeneity_analysis"
        / STAGE2_LONG_STAY_TASK.task_name
        / "discovery_validation"
        / "frozen_design.json",
        run_dir
        / "EXP2_two_stage"
        / "heterogeneity_analysis"
        / STAGE2_LONG_STAY_TASK.task_name
        / "discovery_validation"
        / "library_candidates.json",
        run_dir
        / "EXP2_two_stage"
        / "super_learner"
        / STAGE1_MORTALITY_TASK.task_name
        / "artifact.json",
        run_dir
        / "EXP2_two_stage"
        / "super_learner"
        / STAGE2_LONG_STAY_TASK.task_name
        / "artifact.json",
    ]
    for task_name in (
        EXP1_FLAT_3CLASS_TASK.task_name,
        STAGE1_MORTALITY_TASK.task_name,
        STAGE2_LONG_STAY_TASK.task_name,
    ):
        for model_name in MODEL_ORDER:
            paths.append(
                run_dir
                / "optimization"
                / task_name
                / model_name
                / "selected_hyperparameters.json"
            )
    return paths


def _collect_discovery_files_for_freeze(run_dir: Path) -> list[Path]:
    """Collect every Validation-derived artifact used by confirmatory inference."""
    paths = set(_required_discovery_files(run_dir))
    optional_eicu_assignments = (
        run_dir / "common" / "outcomes" / "eicu_outcome_assignments.csv.gz"
    )
    if optional_eicu_assignments.is_file():
        paths.add(optional_eicu_assignments)
    required_roots = [
        run_dir / "EXP1_flat_3class",
        run_dir / "EXP2_two_stage" / "stage1_mortality",
        run_dir / "EXP2_two_stage" / "stage2_long_stay",
        run_dir / "EXP3_survivor_binary",
        run_dir
        / "EXP2_two_stage"
        / "heterogeneity_analysis"
        / STAGE1_MORTALITY_TASK.task_name
        / "discovery_validation",
        run_dir
        / "EXP2_two_stage"
        / "heterogeneity_analysis"
        / STAGE2_LONG_STAY_TASK.task_name
        / "discovery_validation",
        run_dir
        / "EXP2_two_stage"
        / "super_learner"
        / STAGE1_MORTALITY_TASK.task_name,
        run_dir
        / "EXP2_two_stage"
        / "super_learner"
        / STAGE2_LONG_STAY_TASK.task_name,
    ]
    for root in required_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Required discovery directory was not found: {root}")
        files = [path for path in root.rglob("*") if path.is_file()]
        if not files:
            raise RuntimeError(f"Required discovery directory is empty: {root}")
        paths.update(files)

    for task_name, task_root in (
        (EXP1_FLAT_3CLASS_TASK.task_name, run_dir / "EXP1_flat_3class"),
        (STAGE1_MORTALITY_TASK.task_name, run_dir / "EXP2_two_stage" / "stage1_mortality"),
        (STAGE2_LONG_STAY_TASK.task_name, run_dir / "EXP2_two_stage" / "stage2_long_stay"),
    ):
        for model_name in MODEL_ORDER:
            artifact_dir = task_root / model_name / "artifact"
            if not artifact_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing {task_name}/{model_name} artifact directory: {artifact_dir}"
                )
            artifact_files = [
                path for path in artifact_dir.rglob("*") if path.is_file()
            ]
            if not artifact_files:
                raise RuntimeError(f"Model artifact directory is empty: {artifact_dir}")
            paths.update(artifact_files)
    return sorted(paths)


def _read_split_hash(run_dir: Path) -> str:
    path = run_dir / "common" / "splits" / "split_hash.txt"
    value = path.read_text(encoding="utf-8").strip()
    if len(value) != 64:
        raise ValueError(f"Invalid shared split hash in {path}: {value!r}")
    return value


def freeze_confirmatory_design(
    *,
    project_root: Path,
    config_path: Path,
    run_dir: Path,
    overwrite: bool = False,
) -> ConfirmatoryFreezeResult:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    run_dir = run_dir.resolve()
    verify_run_preflight(
        project_root=project_root,
        config_path=config_path,
        run_dir=run_dir,
        verify_dataset_hashes=True,
    )
    output_dir = run_dir / FROZEN_DESIGN_DIR_NAME
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Confirmatory design is already frozen: {output_dir}")
        confirmatory_roots = [
            run_dir / "EXP2_two_stage" / "pipeline_evaluation",
            run_dir / "final_comparison",
        ]
        if any(path.exists() for path in confirmatory_roots):
            raise RuntimeError(
                "Confirmatory design cannot be overwritten after confirmatory outputs exist."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = _collect_discovery_files_for_freeze(run_dir)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Confirmatory design prerequisites are incomplete:\n"
            + "\n".join(f"- {path}" for path in missing)
        )

    entries = [
        _inventory_entry(path, run_dir, "frozen_discovery_artifact")
        for path in required
    ]
    inventory_csv = output_dir / "frozen_files.csv"
    inventory_json = output_dir / "frozen_files.json"
    _write_inventory_csv(inventory_csv, entries)
    atomic_write_json(inventory_json, entries)
    protocol_path = output_dir / "confirmatory_execution_order.json"
    atomic_write_json(
        protocol_path,
        {
            "sequence": list(CONFIRMATORY_SEQUENCE),
            "single_use": True,
            "overwrite_allowed": False,
            "selection_changes_after_freeze": False,
        },
    )

    preflight_lock = run_dir / "common" / PREFLIGHT_DIR_NAME / "PREFLIGHT_PASSED.lock.json"
    release_manifest = project_root / RELEASE_DIR_NAME / RELEASE_MANIFEST_NAME
    config = _load_json(config_path)
    dataset_fingerprints = _load_json(
        run_dir / "common" / PREFLIGHT_DIR_NAME / "dataset_fingerprints.json"
    )
    shared_split_hash = _read_split_hash(run_dir)
    manifest = {
        "schema_version": "exp123_confirmatory_design_v1",
        "status": "frozen",
        "created_at_utc": utc_now_iso(),
        "run_id": run_dir.name,
        "release_manifest_sha256": sha256_file(release_manifest),
        "preflight_lock_sha256": sha256_file(preflight_lock),
        "config_sha256": sha256_file(config_path),
        "shared_split_hash": shared_split_hash,
        "dataset_fingerprints": dataset_fingerprints,
        "frozen_file_summary": {
            "file_count": len(entries),
            "total_bytes": int(sum(item["size_bytes"] for item in entries)),
            "inventory_sha256": _inventory_digest(entries),
        },
        "checksums": {
            "frozen_files_csv_sha256": sha256_file(inventory_csv),
            "frozen_files_json_sha256": sha256_file(inventory_json),
            "confirmatory_execution_order_sha256": sha256_file(protocol_path),
        },
        "confirmatory_sequence": list(CONFIRMATORY_SEQUENCE),
        "selection_locked": True,
        "discovery_mutation_allowed": False,
        "confirmatory_overwrite_allowed": False,
        "development_config": config.get("development", {}),
    }
    manifest_path = output_dir / "confirmatory_design_manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest_sha = _write_sha256_sidecar(manifest_path)
    lock_path = output_dir / "CONFIRMATORY_READY.lock.json"
    atomic_write_json(
        lock_path,
        {
            "status": "ready",
            "run_id": run_dir.name,
            "manifest_sha256": manifest_sha,
            "shared_split_hash": shared_split_hash,
            "next_confirmatory_step": CONFIRMATORY_SEQUENCE[0],
        },
    )
    _write_sha256_sidecar(lock_path)
    return ConfirmatoryFreezeResult(
        run_id=run_dir.name,
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha,
        frozen_file_count=len(entries),
        shared_split_hash=shared_split_hash,
    )


def verify_confirmatory_design(
    *,
    project_root: Path,
    config_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    verify_run_preflight(
        project_root=project_root,
        config_path=config_path,
        run_dir=run_dir,
        verify_dataset_hashes=False,
    )
    output_dir = run_dir / FROZEN_DESIGN_DIR_NAME
    manifest_path = output_dir / "confirmatory_design_manifest.json"
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    lock_path = output_dir / "CONFIRMATORY_READY.lock.json"
    lock_sidecar = lock_path.with_suffix(lock_path.suffix + ".sha256")
    manifest = _load_json(manifest_path)
    errors: list[str] = []
    if sha256_file(manifest_path) != _parse_sha256_sidecar(sidecar):
        errors.append("Confirmatory design manifest checksum mismatch.")
    if sha256_file(lock_path) != _parse_sha256_sidecar(lock_sidecar):
        errors.append("Confirmatory-ready lock checksum mismatch.")
    inventory_path = output_dir / "frozen_files.csv"
    entries = _read_inventory_csv(inventory_path)
    for entry in entries:
        path = run_dir / str(entry["path"])
        if not path.is_file():
            errors.append(f"Frozen discovery artifact is missing: {path}")
            continue
        if int(path.stat().st_size) != int(entry["size_bytes"]):
            errors.append(f"Frozen discovery artifact size changed: {path}")
            continue
        if sha256_file(path) != str(entry["sha256"]):
            errors.append(f"Frozen discovery artifact checksum changed: {path}")
    if _read_split_hash(run_dir) != str(manifest.get("shared_split_hash")):
        errors.append("Shared split hash changed after confirmatory freeze.")
    if errors:
        raise RuntimeError(
            "Confirmatory design verification failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    return {
        "status": "verified",
        "manifest_sha256": sha256_file(manifest_path),
        "frozen_file_count": len(entries),
        "shared_split_hash": manifest.get("shared_split_hash"),
    }


def assert_discovery_step_allowed(run_dir: Path) -> None:
    lock = run_dir / FROZEN_DESIGN_DIR_NAME / "CONFIRMATORY_READY.lock.json"
    if lock.exists():
        raise RuntimeError(
            "Discovery/model-selection steps are locked because the confirmatory "
            f"design has already been frozen: {lock}"
        )


def _confirmatory_completion_path(
    run_dir: Path, step: str, split: str
) -> Path:
    if step.endswith("heterogeneity-confirmatory"):
        task_name = (
            STAGE1_MORTALITY_TASK.task_name
            if step.startswith("stage1-")
            else STAGE2_LONG_STAY_TASK.task_name
        )
        return (
            run_dir
            / "EXP2_two_stage"
            / "heterogeneity_analysis"
            / task_name
            / "confirmatory"
            / split
            / "run_config.json"
        )
    if step == "oracle-pipeline-confirmatory":
        return (
            run_dir
            / "EXP2_two_stage"
            / "pipeline_evaluation"
            / split
            / "COMPLETED.lock.json"
        )
    if step == "final-comparison-confirmatory":
        return run_dir / "final_comparison" / split / "COMPLETED.lock.json"
    raise KeyError(f"Unknown confirmatory step: {step}")


def assert_confirmatory_step_allowed(
    *,
    project_root: Path,
    config_path: Path,
    run_dir: Path,
    step: str,
    split: str,
) -> None:
    verify_confirmatory_design(
        project_root=project_root,
        config_path=config_path,
        run_dir=run_dir,
    )
    requested_index = None
    for index, item in enumerate(CONFIRMATORY_SEQUENCE):
        if item["step"] == step and item["split"] == split:
            requested_index = index
            break
    if requested_index is None:
        raise ValueError(
            f"The requested confirmatory step is not in the frozen protocol: {step}/{split}"
        )
    requested_output = _confirmatory_completion_path(run_dir, step, split)
    if requested_output.exists():
        raise FileExistsError(
            f"The confirmatory step has already been completed: {requested_output}"
        )
    for prior in CONFIRMATORY_SEQUENCE[:requested_index]:
        prior_output = _confirmatory_completion_path(
            run_dir, str(prior["step"]), str(prior["split"])
        )
        if not prior_output.is_file():
            raise RuntimeError(
                "Confirmatory execution order violation. Complete the prior step first: "
                f"{prior['step']} / {prior['split']} expected={prior_output}"
            )


def run_status(
    *, project_root: Path, config_path: Path, run_dir: Path
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "run_id": run_dir.name,
        "release_frozen": (project_root / RELEASE_DIR_NAME / RELEASE_MANIFEST_NAME).is_file(),
        "preflight_passed": (
            run_dir / "common" / PREFLIGHT_DIR_NAME / "PREFLIGHT_PASSED.lock.json"
        ).is_file(),
        "confirmatory_design_frozen": (
            run_dir / FROZEN_DESIGN_DIR_NAME / "CONFIRMATORY_READY.lock.json"
        ).is_file(),
        "discovery_artifacts": {},
        "confirmatory_steps": [],
    }
    for path in _required_discovery_files(run_dir):
        status["discovery_artifacts"][path.relative_to(run_dir).as_posix()] = path.is_file()
    for item in CONFIRMATORY_SEQUENCE:
        output = _confirmatory_completion_path(
            run_dir, str(item["step"]), str(item["split"])
        )
        status["confirmatory_steps"].append(
            {**item, "completed": output.is_file(), "completion_path": str(output)}
        )
    status["all_discovery_artifacts_ready"] = all(
        status["discovery_artifacts"].values()
    )
    status["all_confirmatory_steps_completed"] = all(
        item["completed"] for item in status["confirmatory_steps"]
    )
    return status


def _iter_run_files_for_final_freeze(run_dir: Path) -> Iterable[Path]:
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir)
        if relative.parts and relative.parts[0] == FINAL_RUN_FREEZE_DIR_NAME:
            continue
        if _is_ignored(relative):
            continue
        yield path


def finalize_exp123_run(
    *,
    project_root: Path,
    config_path: Path,
    run_dir: Path,
) -> FinalRunFreezeResult:
    verify_confirmatory_design(
        project_root=project_root,
        config_path=config_path,
        run_dir=run_dir,
    )
    status = run_status(
        project_root=project_root, config_path=config_path, run_dir=run_dir
    )
    if not status["all_confirmatory_steps_completed"]:
        incomplete = [
            f"{item['step']} / {item['split']}"
            for item in status["confirmatory_steps"]
            if not item["completed"]
        ]
        raise RuntimeError(
            "The EXP123 run cannot be finalized before all confirmatory steps complete: "
            + ", ".join(incomplete)
        )
    output_dir = run_dir / FINAL_RUN_FREEZE_DIR_NAME
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Final run freeze already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = list(_iter_run_files_for_final_freeze(run_dir))
    entries = [_inventory_entry(path, run_dir, "final_run_output") for path in paths]
    inventory_csv = output_dir / "run_inventory.csv"
    inventory_json = output_dir / "run_inventory.json"
    _write_inventory_csv(inventory_csv, entries)
    atomic_write_json(inventory_json, entries)
    confirmatory_manifest = (
        run_dir / FROZEN_DESIGN_DIR_NAME / "confirmatory_design_manifest.json"
    )
    release_manifest = project_root / RELEASE_DIR_NAME / RELEASE_MANIFEST_NAME
    manifest = {
        "schema_version": "exp123_final_run_v1",
        "status": "frozen_complete",
        "created_at_utc": utc_now_iso(),
        "run_id": run_dir.name,
        "release_manifest_sha256": sha256_file(release_manifest),
        "confirmatory_design_manifest_sha256": sha256_file(confirmatory_manifest),
        "shared_split_hash": _read_split_hash(run_dir),
        "run_summary": {
            "file_count": len(entries),
            "total_bytes": int(sum(item["size_bytes"] for item in entries)),
            "inventory_sha256": _inventory_digest(entries),
        },
        "checksums": {
            "run_inventory_csv_sha256": sha256_file(inventory_csv),
            "run_inventory_json_sha256": sha256_file(inventory_json),
        },
        "confirmatory_status": status["confirmatory_steps"],
        "immutable": True,
    }
    manifest_path = output_dir / "final_run_manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest_sha = _write_sha256_sidecar(manifest_path)
    lock_path = output_dir / "EXP123_RUN_COMPLETE.lock.json"
    atomic_write_json(
        lock_path,
        {
            "status": "complete_and_frozen",
            "run_id": run_dir.name,
            "manifest_sha256": manifest_sha,
        },
    )
    _write_sha256_sidecar(lock_path)
    return FinalRunFreezeResult(
        run_id=run_dir.name,
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha,
        frozen_file_count=len(entries),
    )


def verify_finalized_run(*, run_dir: Path) -> dict[str, Any]:
    output_dir = run_dir / FINAL_RUN_FREEZE_DIR_NAME
    manifest_path = output_dir / "final_run_manifest.json"
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    manifest = _load_json(manifest_path)
    errors: list[str] = []
    if sha256_file(manifest_path) != _parse_sha256_sidecar(sidecar):
        errors.append("Final run manifest checksum mismatch.")
    entries = _read_inventory_csv(output_dir / "run_inventory.csv")
    for entry in entries:
        path = run_dir / str(entry["path"])
        if not path.is_file():
            errors.append(f"Frozen run output is missing: {path}")
            continue
        if int(path.stat().st_size) != int(entry["size_bytes"]):
            errors.append(f"Frozen run output size changed: {path}")
            continue
        if sha256_file(path) != str(entry["sha256"]):
            errors.append(f"Frozen run output checksum changed: {path}")
    if errors:
        raise RuntimeError(
            "Final EXP123 run verification failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    return {
        "status": "verified",
        "run_id": manifest.get("run_id"),
        "file_count": len(entries),
        "manifest_sha256": sha256_file(manifest_path),
    }
