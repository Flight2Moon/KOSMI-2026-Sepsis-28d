#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from los_baseline.config import load_config, resolve_path, resolve_outcome_horizon_days
from los_baseline.amendments import verify_run_execution_contract as verify_run_preflight
from los_baseline.constants import OUTCOME_CLASS_LABELS
from los_baseline.exp123_splits import run_common_split_step
from los_baseline.exp1_runner import (
    load_exp1_discovery_task_data,
    run_exp1_flat_multiclass,
    run_exp1_optimization,
)
from los_baseline.stage1_runner import (
    load_stage1_discovery_task_data,
    run_stage1_mortality,
    run_stage1_optimization,
)
from los_baseline.stage2_runner import (
    load_stage2_discovery_task_data,
    run_stage2_long_stay,
    run_stage2_optimization,
)
from los_baseline.stage_heterogeneity import (
    run_stage_heterogeneity_confirmatory,
    run_stage_heterogeneity_discovery,
    stage_heterogeneity_config_from_project,
)
from los_baseline.task_spec import STAGE1_MORTALITY_TASK, STAGE2_LONG_STAY_TASK
from los_baseline.stacking import run_stage_super_learner
from los_baseline.oracle_pipeline import run_oracle_pipeline_confirmatory
from los_baseline.final_comparison import run_final_statistical_comparison
from los_baseline.final_freeze import (
    assert_confirmatory_step_allowed,
    assert_discovery_step_allowed,
    finalize_exp123_run,
    freeze_confirmatory_design,
    run_exp123_preflight,
    run_status,
)
from los_baseline.outcomes import (
    audit_outcome_source,
    create_outcome_definition,
    save_outcome_source_audit,
)
from los_baseline.utils import atomic_write_json, environment_info
from los_baseline.progress import configure_progress, emit_progress
from los_baseline.external_validation_guard import assert_external_mortality_ready
from los_baseline.hardware import (
    configure_torch_runtime,
    write_hardware_report,
    validate_cuda_acceleration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the integrated EXP1/EXP2/EXP3 implementation stages."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "experiment.json"),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Output Run ID. A timestamped EXP123 ID is used when omitted.",
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=[
            "outcomes",
            "splits",
            "exp1-optimize",
            "exp1-train",
            "stage1-optimize",
            "stage1-train",
            "stage2-optimize",
            "stage2-train",
            "stage1-heterogeneity-discovery",
            "stage1-heterogeneity-confirmatory",
            "stage2-heterogeneity-discovery",
            "stage2-heterogeneity-confirmatory",
            "stage1-super-learner",
            "stage2-super-learner",
            "oracle-pipeline-confirmatory",
            "final-comparison-confirmatory",
            "preflight",
            "freeze-confirmatory-design",
            "status",
            "finalize-run",
        ],
        help="Implemented EXP123 development step.",
    )
    parser.add_argument(
        "--dataset",
        choices=["both", "mimic", "eicu"],
        default="both",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--models",
        default=None,
        help="Optional comma-separated model names for EXP1 or Stage1 steps.",
    )
    parser.add_argument(
        "--search-stages",
        default="smoke,coarse,fine",
        help="Comma-separated optimization stages for optimization steps.",
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=None,
        help="Override bootstrap repetitions for EXP1 or Stage1 training.",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "internal_test", "external_eicu"],
        default="validation",
        help="Split for Stage1/Stage2 heterogeneity analysis.",
    )
    parser.add_argument(
        "--frozen-design",
        type=Path,
        default=None,
        help="Frozen Validation heterogeneity design for confirmatory analysis.",
    )
    parser.add_argument(
        "--allow-blocking",
        action="store_true",
        help="Return success even if an outcome source has blocking issues.",
    )
    return parser.parse_args()


def _write_blocking_report(
    output_path: Path,
    audits: dict[str, dict],
) -> int:
    lines = [
        "# Outcome source blocking issues",
        "",
        "This report is generated before EXP1/EXP2/EXP3 model development.",
        "",
    ]
    issue_count = 0
    for dataset_name, audit in audits.items():
        lines.append(f"## {dataset_name}")
        issues = list(audit.get("blocking_issues", []))
        if not issues:
            lines.append("- None")
        else:
            issue_count += len(issues)
            lines.extend(f"- {issue}" for issue in issues)
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return issue_count


def _write_exclusion_report(
    output_path: Path,
    audits: dict[str, dict],
) -> None:
    lines = [
        "# Outcome-row exclusions",
        "",
        "Rows listed here are excluded before split creation and model development.",
        "Original source-row indices are preserved for complete traceability.",
        "",
    ]
    for dataset_name, audit in audits.items():
        lines.append(f"## {dataset_name}")
        excluded = int(audit.get("excluded_row_count", 0))
        retained = int(audit.get("retained_row_count", 0))
        rate = float(audit.get("exclusion_rate", 0.0))
        cohort_excluded = int(audit.get("cohort_excluded_row_count", 0))
        cohort_rate = float(audit.get("cohort_exclusion_rate", 0.0))
        quality_excluded = int(audit.get("quality_excluded_row_count", 0))
        quality_rate = float(audit.get("quality_exclusion_rate", 0.0))
        lines.append(f"- Total excluded rows: {excluded}")
        lines.append(
            f"- Planned cohort-eligibility exclusions: {cohort_excluded} "
            f"({cohort_rate:.6%})"
        )
        lines.append(
            f"- Data-quality exclusions: {quality_excluded} "
            f"({quality_rate:.6%})"
        )
        lines.append(f"- Retained modelling rows: {retained}")
        lines.append(f"- Total exclusion rate: {rate:.6%}")
        cohort_reasons = audit.get("cohort_exclusion_reason_counts", {}) or {}
        if cohort_reasons:
            lines.append("- Cohort eligibility reasons:")
            for reason, count in sorted(cohort_reasons.items()):
                lines.append(f"  - {reason}: {count}")
        else:
            lines.append("- Cohort eligibility reasons: None")
        quality_reasons = audit.get("quality_exclusion_reason_counts", {}) or {}
        if quality_reasons:
            lines.append("- Data-quality reasons:")
            for reason, count in sorted(quality_reasons.items()):
                lines.append(f"  - {reason}: {count}")
        else:
            lines.append("- Data-quality reasons: None")
        warnings = audit.get("quality_warnings", []) or []
        if warnings:
            lines.append("- Warnings:")
            lines.extend(f"  - {warning}" for warning in warnings)
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _outcome_issue_counts(
    manifest: dict,
    *,
    dataset_selection: str,
) -> tuple[int, int]:
    audits = manifest.get("dataset_audits", {})
    mimic_issues = len(
        list((audits.get("mimic") or {}).get("blocking_issues", []))
    )
    eicu_issues = len(
        list((audits.get("eicu") or {}).get("blocking_issues", []))
    )
    if dataset_selection == "eicu":
        return eicu_issues, eicu_issues
    return mimic_issues, eicu_issues


def run_outcome_step(
    config: dict,
    run_id: str,
    dataset_selection: str,
    overwrite: bool,
) -> tuple[Path, int]:
    """Create audited outcome artifacts atomically.

    The returned issue count controls discovery execution. When both datasets are
    requested, only MIMIC blocking issues stop discovery; eICU issues are recorded
    as an external-validation hold and enforced immediately before external use.
    """
    runs_dir = resolve_path(config, config["paths"]["runs_dir"])
    run_dir = runs_dir / run_id
    common_dir = run_dir / "common"
    output_dir = common_dir / "outcomes"
    manifest_path = output_dir / "outcome_step_manifest.json"
    downstream_split_dir = common_dir / "splits"

    if overwrite and downstream_split_dir.exists():
        raise RuntimeError(
            "Outcome assignments cannot be overwritten after common splits "
            f"exist: {downstream_split_dir}"
        )

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            discovery_issues, _ = _outcome_issue_counts(
                manifest,
                dataset_selection=dataset_selection,
            )
            return output_dir, discovery_issues
        raise FileExistsError(
            f"Partial outcome output already exists without a valid manifest: "
            f"{output_dir}. Use --overwrite only before downstream experiments "
            "begin."
        )

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    common_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".outcomes_staging_", dir=str(common_dir))
    )
    try:
        outcomes_config = config["outcomes"]
        mortality_column = str(outcomes_config["mortality_column"])
        los_column = str(
            outcomes_config.get("los_column", config["data"]["target_column"])
        )
        landmark_days = float(resolve_outcome_horizon_days(config))
        identifier_columns = list(
            config["data"].get("identifier_columns", [])
        )

        requested = (
            ["mimic", "eicu"]
            if dataset_selection == "both"
            else [dataset_selection]
        )
        path_keys = {"mimic": "mimic_csv", "eicu": "eicu_csv"}
        audit_payloads: dict[str, dict] = {}
        definitions: dict[str, dict] = {}

        for dataset_name in requested:
            csv_path = resolve_path(
                config,
                config["paths"][path_keys[dataset_name]],
            )
            audit_result = audit_outcome_source(
                dataset_name=dataset_name,
                csv_path=csv_path,
                mortality_column=mortality_column,
                los_column=los_column,
                landmark_days=landmark_days,
                candidate_identifier_columns=identifier_columns,
                exclusion_policy=config.get("data_quality", {}).get(
                    "outcome_row_exclusion"
                ),
                los_derivation_config=outcomes_config.get("los_derivation"),
                cohort_contract_config=config.get("cohort_contract"),
                mortality_guard_config=(
                    config.get("external_validation", {}).get(
                        "mortality_guard"
                    )
                    if dataset_name == "eicu"
                    else config.get("data_quality", {}).get(
                        "internal_mortality_guard"
                    )
                ),
            )
            audit, assignments = audit_result
            save_outcome_source_audit(
                staging_dir,
                audit,
                assignments,
                excluded_rows=audit_result.excluded_rows,
            )
            audit_payloads[dataset_name] = audit.to_dict()

            if assignments is not None:
                definition = create_outcome_definition(
                    labels=assignments["three_class_outcome"].to_numpy(),
                    landmark_days=landmark_days,
                    mortality_column=mortality_column,
                )
                definitions[dataset_name] = definition.to_dict()

        mimic_status = str(
            (audit_payloads.get("mimic") or {}).get(
                "status", "not_requested"
            )
        )
        eicu_status = str(
            (audit_payloads.get("eicu") or {}).get(
                "status", "not_requested"
            )
        )
        discovery_ready = (
            mimic_status == "ready"
            if "mimic" in requested
            else eicu_status == "ready"
        )
        external_status = (
            "not_requested"
            if "eicu" not in requested
            else ("ready" if eicu_status == "ready" else "on_hold")
        )

        manifest = {
            "step": "outcomes",
            "run_id": run_id,
            "status": "ready" if discovery_ready else "blocked",
            "discovery_status": (
                "ready" if discovery_ready else "blocked"
            ),
            "external_validation_status": external_status,
            "discovery_training_enabled": bool(discovery_ready),
            "eicu_confirmatory_enabled": bool(eicu_status == "ready"),
            "landmark_days": landmark_days,
            "outcome_horizon_days": landmark_days,
            "prediction_landmark_hours": float(config.get("cohort_contract", {}).get("prediction_landmark_hours", 24.0)),
            "mortality_column": mortality_column,
            "los_column": los_column,
            "los_derivation": outcomes_config.get("los_derivation", {}),
            "cohort_contract": config.get("cohort_contract", {}),
            "class_labels": OUTCOME_CLASS_LABELS,
            "dataset_audits": audit_payloads,
            "outcome_definitions": definitions,
            "data_quality": config.get("data_quality", {}),
            "environment": environment_info(),
            "next_required_step": "common_three_class_splits",
            "model_training_enabled": bool(
                config.get("development", {}).get(
                    "model_training_enabled", False
                )
            ),
        }
        discovery_issue_count, external_issue_count = _outcome_issue_counts(
            manifest,
            dataset_selection=dataset_selection,
        )
        manifest["blocking_issue_counts"] = {
            "discovery": discovery_issue_count,
            "external_validation": external_issue_count,
        }

        atomic_write_json(
            staging_dir / "outcome_step_manifest.json",
            manifest,
        )
        atomic_write_json(
            staging_dir / "outcome_definition.json",
            definitions,
        )
        discovery_lock_name = (
            "DISCOVERY_OUTCOMES_READY.lock.json"
            if discovery_ready
            else "DISCOVERY_OUTCOMES_BLOCKED.lock.json"
        )
        atomic_write_json(
            staging_dir / discovery_lock_name,
            {
                "status": manifest["discovery_status"],
                "run_id": run_id,
                "mimic_status": mimic_status,
                "blocking_issue_count": discovery_issue_count,
            },
        )
        if "eicu" in requested:
            external_lock_name = (
                "EXTERNAL_VALIDATION_READY.lock.json"
                if external_status == "ready"
                else "EXTERNAL_VALIDATION_ON_HOLD.lock.json"
            )
            atomic_write_json(
                staging_dir / external_lock_name,
                {
                    "status": external_status,
                    "run_id": run_id,
                    "eicu_status": eicu_status,
                    "blocking_issue_count": external_issue_count,
                },
            )
        _write_blocking_report(
            staging_dir / "blocking_issues.md",
            audit_payloads,
        )
        _write_exclusion_report(
            staging_dir / "excluded_rows_report.md",
            audit_payloads,
        )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staging_dir, output_dir)
        return output_dir, discovery_issue_count
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

def _require_clinical_execution_ready(
    *,
    config: dict,
    config_path: Path,
    run_dir: Path,
    discovery_step: bool,
) -> None:
    if not bool(config.get("development", {}).get("model_training_enabled", False)):
        raise RuntimeError(
            "Clinical EXP123 execution is locked because the final implementation "
            "freeze has not enabled model training."
        )
    verify_run_preflight(
        project_root=PROJECT_ROOT,
        config_path=config_path,
        run_dir=run_dir,
        verify_dataset_hashes=False,
    )
    if discovery_step:
        assert_discovery_step_allowed(run_dir)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    run_id = args.run_id or datetime.now().strftime("exp123_%Y%m%d_%H%M%S")
    run_dir = resolve_path(config, config["paths"]["runs_dir"]) / run_id
    progress_dir = run_dir / "common" / "progress"
    configure_progress(
        output_dir=progress_dir,
        run_id=run_id,
        config=config,
        worker_label=f"{args.step}_{__import__("os").getpid()}",
    )
    configure_torch_runtime(config)
    write_hardware_report(
        progress_dir / f"hardware_report_{args.step}_{__import__('os').getpid()}.json",
        config,
    )
    emit_progress(
        "step_started",
        operation=args.step,
        message=f"Started EXP123 step: {args.step}",
        status="running",
        details={"config_path": str(config_path)},
    )

    if args.step == "preflight":
        backend = str(
            config.get("resources", {})
            .get("accelerator", {})
            .get("backend", "auto")
        ).lower()
        if backend == "cuda":
            acceleration_check = validate_cuda_acceleration(
                config,
                run_xgboost_smoke=True,
            )
            error_prefix = "CUDA acceleration preflight failed: "
        else:
            acceleration_check = {
                "backend": backend,
                "status": "not_applicable",
                "skipped_cuda_validation": True,
                "errors": [],
            }
            error_prefix = "Acceleration preflight failed: "
        atomic_write_json(
            progress_dir / "acceleration_preflight.json",
            acceleration_check,
        )
        if acceleration_check.get("errors"):
            raise RuntimeError(
                error_prefix
                + "; ".join(str(value) for value in acceleration_check["errors"])
            )
        result = run_exp123_preflight(
            project_root=PROJECT_ROOT,
            config_path=config_path,
            run_id=run_id,
            overwrite=args.overwrite,
        )
        print(f"[OK] EXP123 preflight passed: {result.output_dir}")
        print(f"[OK] Preflight lock SHA-256: {result.lock_sha256}")
        print("[OK] Dataset fingerprints and execution protocol were locked.")
        return 0

    if args.step == "status":
        print(
            json.dumps(
                run_status(
                    project_root=PROJECT_ROOT,
                    config_path=config_path,
                    run_dir=run_dir,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.step == "freeze-confirmatory-design":
        result = freeze_confirmatory_design(
            project_root=PROJECT_ROOT,
            config_path=config_path,
            run_dir=run_dir,
            overwrite=args.overwrite,
        )
        print(f"[OK] Confirmatory design frozen: {result.output_dir}")
        print(f"[OK] Manifest SHA-256: {result.manifest_sha256}")
        print("[OK] Internal Test/eICU execution order is now locked.")
        return 0

    if args.step == "finalize-run":
        result = finalize_exp123_run(
            project_root=PROJECT_ROOT,
            config_path=config_path,
            run_dir=run_dir,
        )
        print(f"[OK] Completed EXP123 run frozen: {result.output_dir}")
        print(f"[OK] Manifest SHA-256: {result.manifest_sha256}")
        return 0

    if args.step == "outcomes":
        _require_clinical_execution_ready(
            config=config,
            config_path=config_path,
            run_dir=run_dir,
            discovery_step=True,
        )
        output_dir, issue_count = run_outcome_step(
            config=config,
            run_id=run_id,
            dataset_selection=args.dataset,
            overwrite=args.overwrite,
        )
        print(f"[OK] Outcome audit files were written to: {output_dir}")
        manifest = json.loads(
            (output_dir / "outcome_step_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        external_status = manifest.get(
            "external_validation_status", "not_requested"
        )
        if issue_count:
            print(
                f"[BLOCKED] {issue_count} MIMIC discovery outcome issue(s) "
                "were found. Review blocking_issues.md before continuing."
            )
            return 0 if args.allow_blocking else 2
        print("[OK] MIMIC three-class discovery outcome source is ready.")
        if external_status == "on_hold":
            print(
                "[ON HOLD] eICU outcome checks did not pass. MIMIC Track "
                "1/2/3 discovery will continue, but external eICU "
                "confirmatory execution remains blocked."
            )
        elif external_status == "ready":
            print("[OK] eICU outcome source is ready for later validation.")
        print("[INFO] Next step: create the shared three-class patient split.")
        return 0

    if args.step == "splits":
        _require_clinical_execution_ready(
            config=config,
            config_path=config_path,
            run_dir=run_dir,
            discovery_step=True,
        )
        result = run_common_split_step(
            config=config,
            run_id=run_id,
            overwrite=args.overwrite,
        )
        print(f"[OK] Common split files were written to: {result.output_dir}")
        print(f"[OK] Shared split hash: {result.split_hash}")
        print(
            "[OK] EXP1, Stage1, Stage2, and EXP3 reference the same "
            "patient-level split."
        )
        print("[INFO] Shared split is ready for the locked optimization sequence.")
        return 0


    if args.step in {"exp1-optimize", "exp1-train"}:
        _require_clinical_execution_ready(
            config=config,
            config_path=config_path,
            run_dir=run_dir,
            discovery_step=True,
        )
        if not bool(config.get("development", {}).get("model_training_enabled", False)):
            raise RuntimeError(
                "Clinical EXP123 model execution is locked until all implementation "
                "steps are complete. Use run_exp123_exp1_smoke.sh to validate Step 5."
            )
        run_dir = resolve_path(config, config["paths"]["runs_dir"]) / run_id
        task_data = load_exp1_discovery_task_data(config=config, run_dir=run_dir)
        model_names = (
            [value.strip() for value in args.models.split(",") if value.strip()]
            if args.models
            else None
        )
        if args.step == "exp1-optimize":
            stages = tuple(
                value.strip()
                for value in args.search_stages.split(",")
                if value.strip()
            )
            result = run_exp1_optimization(
                task_data=task_data,
                config=config,
                run_dir=run_dir,
                model_names=model_names,
                search_stages=stages,
            )
            print(
                "[OK] EXP1 optimization completed for: "
                + ", ".join(result["models"])
            )
            return 0
        result = run_exp1_flat_multiclass(
            task_data=task_data,
            config=config,
            run_id=run_id,
            run_dir=run_dir,
            model_names=model_names,
            bootstrap_repeats=args.bootstrap_repeats,
            overwrite=args.overwrite,
        )
        print(f"[OK] EXP1 discovery outputs were written to: {result.output_dir}")
        print(f"[OK] Validation representative model: {result.representative_model}")
        print("[INFO] Internal Test and eICU were not accessed.")
        return 0

    if args.step in {"stage1-optimize", "stage1-train"}:
        _require_clinical_execution_ready(
            config=config,
            config_path=config_path,
            run_dir=run_dir,
            discovery_step=True,
        )
        if not bool(config.get("development", {}).get("model_training_enabled", False)):
            raise RuntimeError(
                "Clinical EXP123 model execution is locked until all implementation "
                "steps are complete. Use run_exp123_stage1_smoke.sh to validate Step 6."
            )
        run_dir = resolve_path(config, config["paths"]["runs_dir"]) / run_id
        task_data = load_stage1_discovery_task_data(config=config, run_dir=run_dir)
        model_names = (
            [value.strip() for value in args.models.split(",") if value.strip()]
            if args.models
            else None
        )
        if args.step == "stage1-optimize":
            stages = tuple(
                value.strip()
                for value in args.search_stages.split(",")
                if value.strip()
            )
            result = run_stage1_optimization(
                task_data=task_data,
                config=config,
                run_dir=run_dir,
                model_names=model_names,
                search_stages=stages,
            )
            print(
                "[OK] Stage1 optimization completed for: "
                + ", ".join(result["models"])
            )
            return 0
        result = run_stage1_mortality(
            task_data=task_data,
            config=config,
            run_id=run_id,
            run_dir=run_dir,
            model_names=model_names,
            bootstrap_repeats=args.bootstrap_repeats,
            overwrite=args.overwrite,
        )
        print(f"[OK] Stage1 outputs were written to: {result.output_dir}")
        print(
            "[OK] Validation reference model: "
            f"{result.validation_reference_model}"
        )
        print("[INFO] Internal Test and eICU were not accessed.")
        return 0

    if args.step in {"stage2-optimize", "stage2-train"}:
        _require_clinical_execution_ready(
            config=config,
            config_path=config_path,
            run_dir=run_dir,
            discovery_step=True,
        )
        if not bool(config.get("development", {}).get("model_training_enabled", False)):
            raise RuntimeError(
                "Clinical EXP123 model execution is locked until all implementation "
                "steps are complete. Use run_exp123_stage2_smoke.sh to validate Step 7."
            )
        run_dir = resolve_path(config, config["paths"]["runs_dir"]) / run_id
        task_data = load_stage2_discovery_task_data(config=config, run_dir=run_dir)
        model_names = (
            [value.strip() for value in args.models.split(",") if value.strip()]
            if args.models
            else None
        )
        if args.step == "stage2-optimize":
            stages = tuple(
                value.strip()
                for value in args.search_stages.split(",")
                if value.strip()
            )
            result = run_stage2_optimization(
                task_data=task_data,
                config=config,
                run_dir=run_dir,
                model_names=model_names,
                search_stages=stages,
            )
            print(
                "[OK] Stage2 optimization completed for: "
                + ", ".join(result["models"])
            )
            return 0
        result = run_stage2_long_stay(
            task_data=task_data,
            config=config,
            run_id=run_id,
            run_dir=run_dir,
            model_names=model_names,
            bootstrap_repeats=args.bootstrap_repeats,
            overwrite=args.overwrite,
        )
        print(f"[OK] Stage2 outputs were written to: {result.output_dir}")
        print(
            "[OK] Validation reference model: "
            f"{result.validation_reference_model}"
        )
        print(
            "[OK] EXP3 artifact reference: "
            f"{result.exp3_artifact_reference_path}"
        )
        print("[INFO] Internal Test and eICU were not accessed.")
        return 0

    if args.step in {"stage1-super-learner", "stage2-super-learner"}:
        _require_clinical_execution_ready(
            config=config,
            config_path=config_path,
            run_dir=run_dir,
            discovery_step=True,
        )
        if not bool(config.get("development", {}).get("model_training_enabled", False)):
            raise RuntimeError(
                "Clinical EXP123 execution remains locked until all implementation "
                "steps are complete. Use run_exp123_super_learner_smoke.sh "
                "to validate Step 9."
            )
        run_dir = resolve_path(config, config["paths"]["runs_dir"]) / run_id
        task_name = (
            STAGE1_MORTALITY_TASK.task_name
            if args.step.startswith("stage1-")
            else STAGE2_LONG_STAY_TASK.task_name
        )
        result = run_stage_super_learner(
            run_dir=run_dir,
            task_name=task_name,
            config=config,
            overwrite=args.overwrite,
        )
        print(f"[OK] Super Learner outputs were written to: {result.output_dir}")
        print(f"[OK] Selected members: {', '.join(result.selected_members)}")
        print(
            "[OK] Validation decision threshold: "
            f"{result.validation_decision_threshold:.6f}"
        )
        print("[INFO] Internal Test and eICU were not accessed.")
        return 0


    if args.step == "final-comparison-confirmatory":
        assert_confirmatory_step_allowed(
            project_root=PROJECT_ROOT,
            config_path=config_path,
            run_dir=run_dir,
            step=args.step,
            split=args.split,
        )
        if not bool(config.get("development", {}).get("model_training_enabled", False)):
            raise RuntimeError(
                "Clinical EXP123 final comparison remains locked until all "
                "implementation steps are complete. Use "
                "run_exp123_final_comparison_smoke.sh to validate Step 11."
            )
        if args.split not in {"internal_test", "external_eicu"}:
            raise ValueError(
                "Final comparison requires internal_test or external_eicu."
            )
        run_dir = resolve_path(config, config["paths"]["runs_dir"]) / run_id
        if args.split == "external_eicu":
            assert_external_mortality_ready(config=config, run_dir=run_dir)
        result = run_final_statistical_comparison(
            run_dir=run_dir,
            config=config,
            split_name=args.split,
        )
        print(f"[OK] Final comparison outputs were written to: {result.output_dir}")
        print(
            "[OK] EXP1 Flat macro-AUROC: "
            f"{result.flat_macro_auroc:.6f}"
        )
        print(
            "[OK] EXP2 Pipeline macro-AUROC: "
            f"{result.pipeline_macro_auroc:.6f}"
        )
        print(
            "[OK] Paired bootstrap and inferential tests completed: "
            f"{result.inferential_test_count}"
        )
        return 0

    if args.step == "oracle-pipeline-confirmatory":
        assert_confirmatory_step_allowed(
            project_root=PROJECT_ROOT,
            config_path=config_path,
            run_dir=run_dir,
            step=args.step,
            split=args.split,
        )
        if not bool(config.get("development", {}).get("model_training_enabled", False)):
            raise RuntimeError(
                "Clinical EXP123 confirmatory execution remains locked until all "
                "implementation steps are complete. Use "
                "run_exp123_oracle_pipeline_smoke.sh to validate Step 10."
            )
        if args.split not in {"internal_test", "external_eicu"}:
            raise ValueError(
                "Oracle/Pipeline confirmatory execution requires internal_test "
                "or external_eicu."
            )
        run_dir = resolve_path(config, config["paths"]["runs_dir"]) / run_id
        if args.split == "external_eicu":
            assert_external_mortality_ready(config=config, run_dir=run_dir)
        result = run_oracle_pipeline_confirmatory(
            run_dir=run_dir,
            config=config,
            split_name=args.split,
        )
        print(f"[OK] Oracle/Pipeline outputs were written to: {result.output_dir}")
        print(
            "[OK] Pipeline macro-AUROC: "
            f"{result.pipeline_macro_auroc:.6f}"
        )
        print(
            "[OK] Oracle Stage2 AUROC: "
            f"{result.oracle_stage2_auroc:.6f}"
        )
        print("[OK] Frozen Stage1/Stage2 artifacts were used without refitting.")
        return 0

    if "heterogeneity" in args.step:
        is_discovery = args.step.endswith("-discovery")
        if is_discovery:
            _require_clinical_execution_ready(
                config=config,
                config_path=config_path,
                run_dir=run_dir,
                discovery_step=True,
            )
        else:
            assert_confirmatory_step_allowed(
                project_root=PROJECT_ROOT,
                config_path=config_path,
                run_dir=run_dir,
                step=args.step,
                split=args.split,
            )
        if not bool(config.get("development", {}).get("model_training_enabled", False)):
            raise RuntimeError(
                "Clinical EXP123 execution remains locked until all implementation "
                "steps are complete. Use run_exp123_stage_heterogeneity_smoke.sh "
                "to validate Step 8."
            )
        run_dir = resolve_path(config, config["paths"]["runs_dir"]) / run_id
        task_name = (
            STAGE1_MORTALITY_TASK.task_name
            if args.step.startswith("stage1-")
            else STAGE2_LONG_STAY_TASK.task_name
        )
        target_column = str(config["data"]["target_column"])
        identifier_columns = tuple(config["data"].get("identifier_columns", []))
        if args.step.endswith("-discovery"):
            if args.split != "validation":
                raise ValueError("Heterogeneity discovery is restricted to validation.")
            analysis_config = stage_heterogeneity_config_from_project(
                run_dir=run_dir,
                task_name=task_name,
                project_config=config,
            )
            output = run_stage_heterogeneity_discovery(
                analysis_config,
                dataset_path=resolve_path(config, config["paths"]["mimic_csv"]),
                target_column=target_column,
                identifier_columns=identifier_columns,
            )
        else:
            if args.split not in {"internal_test", "external_eicu"}:
                raise ValueError(
                    "Heterogeneity confirmatory analysis requires internal_test "
                    "or external_eicu."
                )
            if args.frozen_design is None:
                raise ValueError("--frozen-design is required for confirmatory analysis.")
            dataset_key = "eicu_csv" if args.split == "external_eicu" else "mimic_csv"
            if args.split == "external_eicu":
                assert_external_mortality_ready(config=config, run_dir=run_dir)
            output = run_stage_heterogeneity_confirmatory(
                run_dir=run_dir,
                task_name=task_name,
                split=args.split,
                dataset_path=resolve_path(config, config["paths"][dataset_key]),
                frozen_design_path=args.frozen_design,
                target_column=target_column,
                identifier_columns=identifier_columns,
            )
        print(f"[OK] Stage heterogeneity outputs were written to: {output}")
        return 0

    raise AssertionError(f"Unhandled step: {args.step}")


if __name__ == "__main__":
    raise SystemExit(main())
