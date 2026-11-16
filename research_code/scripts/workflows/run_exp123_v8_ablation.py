#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config, resolve_path
from los_baseline.feature_sets import dataset_header_readiness, resolve_feature_set
from los_baseline.utils import atomic_write_json, sha256_file


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y_%m%d_%H%M%S")


def _run(command: list[str]) -> None:
    print("[CMD]", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _split_hash(run_dir: Path) -> str:
    path = run_dir / "common" / "splits" / "split_hash.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Split hash not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _required_dataset_columns(config: dict[str, Any]) -> list[str]:
    columns = list(config["data"]["feature_columns"])
    columns.extend(str(v) for v in config["data"].get("identifier_columns", []))
    outcomes = config.get("outcomes", {})
    for value in (
        config["data"].get("target_column"),
        outcomes.get("los_column"),
        outcomes.get("mortality_column"),
    ):
        if value:
            columns.append(str(value))
    los = outcomes.get("los_derivation", {})
    if bool(los.get("enabled", False)):
        for key in (
            "admission_time_column",
            "discharge_time_column",
            "in_hospital_death_time_column",
        ):
            value = los.get(key)
            if value:
                columns.append(str(value))
    return list(dict.fromkeys(columns))


def _is_baseline_complete(run_dir: Path) -> bool:
    return (
        (run_dir / "common" / "staged_baseline" / "candidate_plan.json").is_file()
        and (run_dir / "EXP1_flat_3class" / "model_summary_validation.csv").is_file()
        and (run_dir / "EXP2_two_stage" / "stage1_mortality" / "model_summary_validation.csv").is_file()
        and (run_dir / "EXP2_two_stage" / "stage2_long_stay" / "model_summary_validation.csv").is_file()
    )


def _build_plan(
    *,
    base_path: Path,
    base: dict[str, Any],
    backend: str,
    registry_path: Path,
    feature_set_ids: list[str],
    series_id: str,
    series_dir: Path,
    stamp: str,
) -> dict[str, Any]:
    generated: list[dict[str, Any]] = []
    for set_id in feature_set_ids:
        resolved = resolve_feature_set(registry_path, set_id)
        config_path = PROJECT_ROOT / "generated_configs" / f"{series_id}_{set_id}.json"
        _run([
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "workflows" / "build_exp123_ablation_config.py"),
            "--base-config", str(base_path),
            "--feature-set-registry", str(registry_path),
            "--feature-set-id", set_id,
            "--output", str(config_path),
        ])
        config = load_config(config_path)
        required = _required_dataset_columns(config)
        mimic = dataset_header_readiness(
            dataset_path=resolve_path(config, config["paths"]["mimic_csv"]),
            required_columns=required,
        )
        eicu = dataset_header_readiness(
            dataset_path=resolve_path(config, config["paths"]["eicu_csv"]),
            required_columns=required,
        )
        readiness = {
            "schema_version": "exp123_v8_ablation_readiness_v1",
            "feature_set": resolved.to_dict(),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "datasets": {"mimic": mimic, "eicu": eicu},
            "ready_for_execution": mimic["status"] == "ready" and eicu["status"] == "ready",
        }
        readiness_path = series_dir / f"readiness_{set_id}.json"
        atomic_write_json(readiness_path, readiness)
        run_id = f"exp123_v8_{backend}_LOS28_ABLATION_{set_id}_{stamp}"
        generated.append({
            "feature_set_id": set_id,
            "feature_set_sha256": resolved.sha256,
            "feature_count": len(resolved.columns),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "readiness_path": str(readiness_path),
            "ready_for_execution": readiness["ready_for_execution"],
            "run_id": run_id,
            "status": "planned",
        })

    return {
        "schema_version": "exp123_v8_ablation_series_v1",
        "series_id": series_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "prespecified_sensitivity_analysis",
        "base_config_path": str(base_path),
        "base_config_sha256": sha256_file(base_path),
        "fixed_axes": {
            "outcome_horizon_days": 28,
            "strategy": "T1_baseline",
            "split_seed": int(base["split"]["seed"]),
            "split_ratios": [
                float(base["split"]["train_ratio"]),
                float(base["split"]["validation_ratio"]),
                float(base["split"]["test_ratio"]),
            ],
        },
        "varied_axis": "feature_set",
        "feature_set_registry": str(registry_path),
        "feature_set_registry_sha256": sha256_file(registry_path),
        "runs": generated,
        "confirmatory_splits_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "EXP123 V8 Phase 4-A feature-set ablation. Default is plan/readiness only. "
            "Execution is LOS28 + T1 Baseline only; the varied axis is feature_set."
        )
    )
    parser.add_argument("--base-config", default="configs/experiment_cuda_v8_los28.json")
    parser.add_argument("--feature-set-registry", default="configs/feature_sets.json")
    parser.add_argument("--feature-sets", default="FS_FULL,FS_NOAPS,FS_APSONLY")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume an existing ablation series. Requires --series-id.")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-hardware-smoke", action="store_true")
    parser.add_argument("--allow-environment-mismatch", action="store_true")
    parser.add_argument("--series-id")
    args = parser.parse_args()

    base_path = Path(args.base_config)
    if not base_path.is_absolute():
        base_path = PROJECT_ROOT / base_path
    base_path = base_path.resolve()
    base = load_config(base_path)
    horizon = float(base["outcomes"]["outcome_horizon_days"])
    if horizon != 28.0:
        raise ValueError(
            "Phase 4-A is prespecified as LOS28 + T1 Baseline. "
            f"Received outcome_horizon_days={horizon}."
        )
    backend = str(base.get("resources", {}).get("accelerator", {}).get("backend", "auto"))
    registry_path = Path(args.feature_set_registry)
    if not registry_path.is_absolute():
        registry_path = PROJECT_ROOT / registry_path
    registry_path = registry_path.resolve()

    feature_set_ids = [v.strip() for v in args.feature_sets.split(",") if v.strip()]
    if not feature_set_ids:
        raise ValueError("At least one feature-set ID is required.")
    if len(feature_set_ids) != len(set(feature_set_ids)):
        raise ValueError("Duplicate feature-set IDs were requested.")

    stamp = _utc_stamp()
    series_id = args.series_id or f"exp123_v8_ablation_LOS28_T1_{stamp}"
    series_dir = PROJECT_ROOT / "series" / series_id
    plan_path = series_dir / "ablation_manifest.json"
    runs_dir = resolve_path(base, base["paths"]["runs_dir"])

    if args.resume:
        if not args.series_id:
            raise ValueError("--resume requires --series-id.")
        if not plan_path.is_file():
            raise FileNotFoundError(f"Ablation manifest not found for resume: {plan_path}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        generated = list(plan.get("runs", []))
        manifest_sets = [str(item["feature_set_id"]) for item in generated]
        if manifest_sets != feature_set_ids:
            raise ValueError(
                "Requested feature sets differ from the existing ablation series. "
                f"manifest={manifest_sets}, requested={feature_set_ids}"
            )
        if str(plan.get("base_config_sha256")) != sha256_file(base_path):
            raise RuntimeError("Base config changed since the ablation series was created.")
        if str(plan.get("feature_set_registry_sha256")) != sha256_file(registry_path):
            raise RuntimeError("Feature-set registry changed since the ablation series was created.")
        print(f"[OK] Resuming ablation series: {plan_path}")
    else:
        if series_dir.exists() and any(series_dir.iterdir()):
            raise FileExistsError(f"Ablation series already exists: {series_dir}")
        series_dir.mkdir(parents=True, exist_ok=True)
        plan = _build_plan(
            base_path=base_path,
            base=base,
            backend=backend,
            registry_path=registry_path,
            feature_set_ids=feature_set_ids,
            series_id=series_id,
            series_dir=series_dir,
            stamp=stamp,
        )
        atomic_write_json(plan_path, plan)
        print(f"[OK] Ablation plan: {plan_path}")

    generated = list(plan["runs"])
    if not args.execute:
        not_ready = [r["feature_set_id"] for r in generated if not r["ready_for_execution"]]
        if not_ready:
            print("[INFO] Plan-only: feature sets not ready on current dataset(s): " + ", ".join(not_ready))
        print("[OK] Plan-only mode. Use --execute only after readiness reports are all ready.")
        return 0

    not_ready = [r for r in generated if not r["ready_for_execution"]]
    if not_ready:
        labels = ", ".join(r["feature_set_id"] for r in not_ready)
        raise RuntimeError(
            "Ablation execution is blocked because required feature columns are not ready: "
            f"{labels}. See readiness JSON files in {series_dir}."
        )

    reference_split_hash: str | None = plan.get("shared_split_hash")
    for item in generated:
        run_dir = runs_dir / str(item["run_id"])
        config_path = Path(str(item["config_path"])).resolve()
        if sha256_file(config_path) != str(item["config_sha256"]):
            raise RuntimeError(f"Generated ablation config changed: {config_path}")

        if _is_baseline_complete(run_dir):
            print(f"[OK] Reusing completed ablation run: {item['run_id']}")
        else:
            command = [
                "bash", "run_exp123_v6_series.sh",
                "--config", str(config_path),
                "--phase", "baseline",
                "--run-id", str(item["run_id"]),
            ]
            if run_dir.is_dir() and any(run_dir.iterdir()):
                command.append("--resume")
                item["status"] = "resuming"
            else:
                command.append("--refreeze")
                item["status"] = "running"
            if args.skip_tests:
                command.append("--skip-tests")
            if args.skip_hardware_smoke:
                command.append("--skip-hardware-smoke")
            if args.allow_environment_mismatch:
                command.append("--allow-environment-mismatch")
            atomic_write_json(plan_path, plan)
            try:
                _run(command)
            except Exception:
                item["status"] = "interrupted"
                atomic_write_json(plan_path, plan)
                raise

        reporting_manifest = run_dir / "V8_reporting" / "reporting_manifest.json"
        if not reporting_manifest.is_file():
            _run([
                sys.executable, str(PROJECT_ROOT / "scripts" / "workflows" / "run_exp123_v8_reporting.py"),
                "--config", str(config_path),
                "--run-id", str(item["run_id"]),
                "--strategy", "T1_baseline",
            ])

        split_hash = _split_hash(run_dir)
        if reference_split_hash is None:
            reference_split_hash = split_hash
        elif split_hash != reference_split_hash:
            raise RuntimeError(
                "Feature-set ablation split mismatch detected. All feature sets must use the same "
                f"patient split. Expected {reference_split_hash}, got {split_hash} for {item['feature_set_id']}."
            )
        item["split_hash"] = split_hash
        item["status"] = "completed"
        plan["shared_split_hash"] = reference_split_hash
        atomic_write_json(plan_path, plan)

    plan["status"] = "completed"
    plan["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(plan_path, plan)
    _run([
        sys.executable, str(PROJECT_ROOT / "scripts" / "workflows" / "compare_exp123_v8_ablation.py"),
        "--base-config", str(base_path),
        "--ablation-manifest", str(plan_path),
    ])
    print("[OK] Phase 4-A ablation completed. Confirmatory splits remain locked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
