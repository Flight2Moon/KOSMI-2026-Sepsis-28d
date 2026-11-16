#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config, resolve_path
from los_baseline.final_freeze import verify_exp123_release, verify_run_preflight
from los_baseline.resource_manager import resource_config, system_snapshot
from los_baseline.los_derivation import required_los_derivation_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an EXP123 config, resource profile, data paths, and run contract."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--check-release", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _resolve_project_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if not config_path.is_file():
        available = sorted((PROJECT_ROOT / "configs").glob("*.json"))
        print(f"[ERROR] Config file was not found: {config_path}")
        if available:
            print("[INFO] Available configs:")
            for path in available:
                print(f"  - {path.relative_to(PROJECT_ROOT)}")
        return 2

    try:
        config = load_config(config_path)
    except Exception as exception:
        print(f"[ERROR] Config could not be loaded: {type(exception).__name__}: {exception}")
        return 2

    resolved = resource_config(config)
    snapshot = system_snapshot().to_dict()
    backend = resolved["backend"]

    hardware = dict(config.get("hardware", {}))
    cuda_hardware = dict(hardware.get("cuda", {}))
    accelerator = dict(config.get("resources", {}).get("accelerator", {}))

    if backend == "cuda":
        if not resolved["cuda"]["gpu_ids"]:
            errors.append("CUDA backend selected but resources.cuda.gpu_ids is empty.")
        if str(cuda_hardware.get("mode", "required")).lower() == "disabled":
            errors.append("CUDA backend selected but hardware.cuda.mode is disabled.")
        if not bool(accelerator.get("required", True)):
            warnings.append("CUDA backend is selected but resources.accelerator.required is false.")
    elif backend == "cpu" and resolved["cuda"]["gpu_ids"]:
        warnings.append("CPU backend selected; configured CUDA GPU IDs will not be used.")

    if resolved["cpu"]["total_thread_limit"] > int(snapshot["logical_cpu_count"]):
        errors.append("CPU thread limit exceeds visible logical CPUs.")
    if resolved["memory"]["total_limit_gb"] > float(snapshot["total_ram_gb"]):
        errors.append("Configured RAM limit exceeds physical RAM.")

    max_jobs = int(resolved["scheduler"]["max_concurrent_jobs"])
    if max_jobs > resolved["cpu"]["total_thread_limit"] * 2:
        warnings.append(
            "max_concurrent_jobs is high relative to the CPU thread limit; "
            "verify that most jobs are accelerator-bound."
        )

    data_paths: dict[str, dict[str, Any]] = {}
    if args.check_data:
        for key in ("mimic_csv", "eicu_csv"):
            raw = config.get("paths", {}).get(key)
            if raw is None:
                data_paths[key] = {"configured": None, "exists": False}
                errors.append(f"Required path key is missing: paths.{key}")
                continue
            path = _resolve_project_path(str(raw))
            data_paths[key] = {
                "configured": str(raw),
                "resolved": str(path),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
            if key in {"mimic_csv", "eicu_csv"} and not path.is_file():
                errors.append(f"Dataset file does not exist: {path}")
            if key in {"mimic_csv", "eicu_csv"} and path.is_file():
                dataset_name = "mimic" if key == "mimic_csv" else "eicu"
                try:
                    columns = set(
                        pd.read_csv(
                            path, nrows=0, compression="infer"
                        ).columns.astype(str).str.strip()
                    )
                    outcomes = dict(config.get("outcomes", {}))
                    mortality_column = str(
                        outcomes.get(
                            "mortality_column", "hospital_expire_flag"
                        )
                    )
                    los_column = str(
                        outcomes.get(
                            "los_column",
                            config.get("data", {}).get("target_column", "LOS"),
                        )
                    )
                    required_columns = {mortality_column}
                    derivation = outcomes.get("los_derivation")
                    derivation_columns = required_los_derivation_columns(
                        derivation,
                        dataset_name=dataset_name,
                        mortality_column=mortality_column,
                    )
                    if derivation_columns:
                        required_columns.update(derivation_columns)
                    else:
                        required_columns.add(los_column)
                    missing_columns = sorted(required_columns - columns)
                    data_paths[key]["available_column_count"] = len(columns)
                    data_paths[key]["required_outcome_columns"] = sorted(
                        required_columns
                    )
                    data_paths[key]["missing_outcome_columns"] = missing_columns
                    if missing_columns:
                        errors.append(
                            f"{dataset_name} is missing required outcome/LOS "
                            f"derivation columns: {missing_columns}"
                        )
                except Exception as exception:
                    errors.append(
                        f"Failed to inspect {dataset_name} dataset header: "
                        f"{type(exception).__name__}: {exception}"
                    )

        provenance = dict(
            config.get("audit", {}).get("eicu_harmonizer_provenance", {})
        )
        if bool(provenance.get("enabled", False)):
            raw = provenance.get("path")
            if raw is None:
                errors.append(
                    "audit.eicu_harmonizer_provenance.enabled is true but path is null."
                )
            else:
                path = _resolve_project_path(str(raw))
                data_paths["eicu_harmonizer_provenance"] = {
                    "configured": str(raw),
                    "resolved": str(path),
                    "exists": path.is_file(),
                    "size_bytes": path.stat().st_size if path.is_file() else None,
                }
                if not str(path).lower().endswith(
                    (".csv", ".csv.gz", ".csv.bz2", ".csv.xz")
                ):
                    errors.append(
                        "Optional eICU harmonizer provenance must be CSV/CSV.GZ."
                    )
                if bool(provenance.get("required", False)) and not path.is_file():
                    errors.append(
                        f"Required eICU harmonizer provenance CSV does not exist: {path}"
                    )

    release_status: dict[str, Any] | None = None
    if args.check_release:
        try:
            release_status = verify_exp123_release(
                project_root=PROJECT_ROOT,
                verify_environment=True,
            )
        except Exception as exception:
            errors.append(f"Release verification failed: {exception}")

    preflight_status: dict[str, Any] | None = None
    if args.run_id:
        run_dir = resolve_path(config, config["paths"]["runs_dir"]) / args.run_id
        try:
            preflight_status = verify_run_preflight(
                project_root=PROJECT_ROOT,
                config_path=config_path,
                run_dir=run_dir,
                verify_dataset_hashes=False,
            )
        except Exception as exception:
            errors.append(f"Run preflight verification failed: {exception}")

    report = {
        "status": "passed" if not errors else "failed",
        "project_root": str(PROJECT_ROOT),
        "config_path": str(config_path),
        "profile_name": hardware.get("profile_name"),
        "backend": backend,
        "resource_profile": resolved,
        "system_snapshot": snapshot,
        "data_paths": data_paths,
        "release": release_status,
        "preflight": preflight_status,
        "warnings": warnings,
        "errors": errors,
    }

    output = Path(args.output).expanduser().resolve() if args.output else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
