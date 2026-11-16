#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config, resolve_path
from los_baseline.failure_diagnostics import write_diagnostic_reports
from los_baseline.final_freeze import verify_run_preflight
from los_baseline.experiment_scheduler import ResourceAwareScheduler, ScheduledJob
from los_baseline.parallel_jobs import TASKS
from los_baseline.progress import configure_progress, emit_progress
from los_baseline.resource_manager import ResourceRequest, request_for_job, resource_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resource-aware EXP123 DAG scheduler for discovery and confirmatory runs.")
    parser.add_argument("mode", choices=["discovery", "confirmatory"])
    parser.add_argument("run_id")
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def _standard_command(config_path: Path, run_id: str, step: str, *extra: str) -> tuple[str, ...]:
    return (
        sys.executable,
        str(PROJECT_ROOT / "run_exp123.py"),
        "--config", str(config_path),
        "--run-id", run_id,
        "--step", step,
        *extra,
    )


def _job_command(config_path: Path, run_id: str, job: str, task: str, model: str | None = None) -> tuple[str, ...]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "run_exp123_job.py"),
        "--config", str(config_path),
        "--run-id", run_id,
        "--job", job,
        "--task", task,
    ]
    if model:
        command += ["--model", model]
    return tuple(command)


def _models(config: Mapping[str, Any], task_key: str) -> list[str]:
    if task_key == "exp1":
        return list(config["experiments"]["exp1_flat_3class"]["models"])
    stage = "stage1_mortality" if task_key == "stage1" else "stage2_long_stay"
    return list(config["experiments"]["exp2_two_stage"][stage]["models"])


def discovery_jobs(config: Mapping[str, Any], config_path: Path, run_id: str) -> list[ScheduledJob]:
    jobs: list[ScheduledJob] = [
        ScheduledJob(
            name="outcomes",
            command=_standard_command(config_path, run_id, "outcomes"),
            request=request_for_job(config, model_name=None, job_kind="outcomes"),
            completion_path="common/outcomes/outcome_step_manifest.json",
            kind="outcomes",
        ),
        ScheduledJob(
            name="splits",
            command=_standard_command(config_path, run_id, "splits"),
            dependencies=("outcomes",),
            request=request_for_job(config, model_name=None, job_kind="splits"),
            completion_path="common/splits/split_hash.txt",
            kind="splits",
        ),
    ]
    opt_names: dict[str, list[str]] = {}
    train_names: dict[str, list[str]] = {}
    for task_key in ("exp1", "stage1", "stage2"):
        task_name = TASKS[task_key]["task"].task_name
        opt_names[task_key] = []
        train_names[task_key] = []
        for model in _models(config, task_key):
            opt_name = f"{task_key}-optimize-{model}"
            train_name = f"{task_key}-train-{model}"
            opt_names[task_key].append(opt_name)
            train_names[task_key].append(train_name)
            jobs.append(
                ScheduledJob(
                    name=opt_name,
                    command=_job_command(config_path, run_id, "optimize-model", task_key, model),
                    dependencies=("splits",),
                    request=request_for_job(config, model_name=model, job_kind="optimization"),
                    completion_path=f"optimization/{task_name}/{model}/MODEL_OPTIMIZATION_COMPLETED.lock.json",
                    model_name=model,
                    task_key=task_key,
                    kind="optimization",
                )
            )
            jobs.append(
                ScheduledJob(
                    name=train_name,
                    command=_job_command(config_path, run_id, "train-model", task_key, model),
                    dependencies=(opt_name,),
                    request=request_for_job(config, model_name=model, job_kind="training"),
                    completion_path=f"{TASKS[task_key]['output_relative']}/{model}/MODEL_TRAINING_COMPLETED.lock.json",
                    model_name=model,
                    task_key=task_key,
                    kind="training",
                )
            )
        jobs.append(
            ScheduledJob(
                name=f"{task_key}-optimization-finalize",
                command=_job_command(config_path, run_id, "finalize-optimization", task_key),
                dependencies=tuple(opt_names[task_key]),
                request=request_for_job(config, model_name=None, job_kind="finalize"),
                completion_path=f"optimization/{task_name}/OPTIMIZATION_TASK_COMPLETED.lock.json",
                task_key=task_key,
                kind="finalize",
            )
        )
        result_name = "exp1_result.json" if task_key == "exp1" else f"{task_key}_result.json"
        jobs.append(
            ScheduledJob(
                name=f"{task_key}-training-finalize",
                command=_job_command(config_path, run_id, "finalize-training", task_key),
                dependencies=tuple(train_names[task_key]),
                request=request_for_job(config, model_name=None, job_kind="finalize"),
                completion_path=f"{TASKS[task_key]['output_relative']}/{result_name}",
                task_key=task_key,
                kind="finalize",
            )
        )
    jobs.extend(
        [
            ScheduledJob(
                name="stage1-heterogeneity-discovery",
                command=_standard_command(config_path, run_id, "stage1-heterogeneity-discovery"),
                dependencies=("stage1-training-finalize",),
                request=request_for_job(config, model_name=None, job_kind="heterogeneity"),
                completion_path="EXP2_two_stage/heterogeneity_analysis/stage1_mortality/discovery_validation/frozen_design.json",
                kind="heterogeneity",
            ),
            ScheduledJob(
                name="stage2-heterogeneity-discovery",
                command=_standard_command(config_path, run_id, "stage2-heterogeneity-discovery"),
                dependencies=("stage2-training-finalize",),
                request=request_for_job(config, model_name=None, job_kind="heterogeneity"),
                completion_path="EXP2_two_stage/heterogeneity_analysis/stage2_long_stay/discovery_validation/frozen_design.json",
                kind="heterogeneity",
            ),
            ScheduledJob(
                name="stage1-super-learner",
                command=_standard_command(config_path, run_id, "stage1-super-learner"),
                dependencies=("stage1-heterogeneity-discovery",),
                request=request_for_job(config, model_name=None, job_kind="super_learner"),
                completion_path="EXP2_two_stage/super_learner/stage1_mortality/artifact.json",
                kind="super_learner",
            ),
            ScheduledJob(
                name="stage2-super-learner",
                command=_standard_command(config_path, run_id, "stage2-super-learner"),
                dependencies=("stage2-heterogeneity-discovery",),
                request=request_for_job(config, model_name=None, job_kind="super_learner"),
                completion_path="EXP2_two_stage/super_learner/stage2_long_stay/artifact.json",
                kind="super_learner",
            ),
        ]
    )
    return jobs


def confirmatory_jobs(config: Mapping[str, Any], config_path: Path, run_id: str) -> list[ScheduledJob]:
    request = lambda kind: request_for_job(config, model_name=None, job_kind=kind)
    return [
        ScheduledJob("stage1-heterogeneity-confirmatory-internal", _standard_command(config_path, run_id, "stage1-heterogeneity-confirmatory", "--split", "internal_test"), request=request("heterogeneity"), completion_path="EXP2_two_stage/heterogeneity_analysis/stage1_mortality/confirmatory/internal_test/analysis_contract.json"),
        ScheduledJob("stage2-heterogeneity-confirmatory-internal", _standard_command(config_path, run_id, "stage2-heterogeneity-confirmatory", "--split", "internal_test"), dependencies=("stage1-heterogeneity-confirmatory-internal",), request=request("heterogeneity"), completion_path="EXP2_two_stage/heterogeneity_analysis/stage2_long_stay/confirmatory/internal_test/analysis_contract.json"),
        ScheduledJob("oracle-pipeline-internal", _standard_command(config_path, run_id, "oracle-pipeline-confirmatory", "--split", "internal_test"), dependencies=("stage2-heterogeneity-confirmatory-internal",), request=request("confirmatory_inference"), completion_path="EXP2_two_stage/pipeline_evaluation/internal_test/COMPLETED.lock.json"),
        ScheduledJob("final-comparison-internal", _standard_command(config_path, run_id, "final-comparison-confirmatory", "--split", "internal_test"), dependencies=("oracle-pipeline-internal",), request=request("final_comparison"), completion_path="final_comparison/internal_test/COMPLETED.lock.json"),
        ScheduledJob("stage1-heterogeneity-confirmatory-eicu", _standard_command(config_path, run_id, "stage1-heterogeneity-confirmatory", "--split", "external_eicu"), dependencies=("final-comparison-internal",), request=request("heterogeneity"), completion_path="EXP2_two_stage/heterogeneity_analysis/stage1_mortality/confirmatory/external_eicu/analysis_contract.json"),
        ScheduledJob("stage2-heterogeneity-confirmatory-eicu", _standard_command(config_path, run_id, "stage2-heterogeneity-confirmatory", "--split", "external_eicu"), dependencies=("stage1-heterogeneity-confirmatory-eicu",), request=request("heterogeneity"), completion_path="EXP2_two_stage/heterogeneity_analysis/stage2_long_stay/confirmatory/external_eicu/analysis_contract.json"),
        ScheduledJob("oracle-pipeline-eicu", _standard_command(config_path, run_id, "oracle-pipeline-confirmatory", "--split", "external_eicu"), dependencies=("stage2-heterogeneity-confirmatory-eicu",), request=request("confirmatory_inference"), completion_path="EXP2_two_stage/pipeline_evaluation/external_eicu/COMPLETED.lock.json"),
        ScheduledJob("final-comparison-eicu", _standard_command(config_path, run_id, "final-comparison-confirmatory", "--split", "external_eicu"), dependencies=("oracle-pipeline-eicu",), request=request("final_comparison"), completion_path="final_comparison/external_eicu/COMPLETED.lock.json"),
        ScheduledJob("finalize-run", _standard_command(config_path, run_id, "finalize-run"), dependencies=("final-comparison-eicu",), request=request("finalize"), completion_path="final_run_freeze/EXP123_RUN_COMPLETE.lock.json"),
    ]


def _write_startup_failure(
    *,
    run_dir: Path,
    run_id: str,
    mode: str,
    config_path: Path,
    exception: Exception,
) -> Path:
    progress_dir = run_dir / "common" / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    path = progress_dir / f"resource_{mode}_startup_failure.json"
    payload = {
        "schema_version": "exp123_resource_protocol_startup_failure_v1",
        "run_id": run_id,
        "mode": mode,
        "config_path": str(config_path),
        "exception_type": type(exception).__name__,
        "message": str(exception),
        "traceback": traceback.format_exc(),
        "scheduler_started": False,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    run_dir = resolve_path(config, config["paths"]["runs_dir"]) / args.run_id
    progress_dir = run_dir / "common" / "progress"
    configure_progress(
        output_dir=progress_dir,
        run_id=args.run_id,
        config=config,
        worker_label=f"resource_scheduler_{args.mode}",
    )

    # Validate the frozen release, config checksum, and preflight contract once
    # before launching dozens of child processes. This converts a shared
    # contract problem into one clear startup failure instead of many jobs.
    try:
        verify_run_preflight(
            project_root=PROJECT_ROOT,
            config_path=config_path,
            run_dir=run_dir,
            verify_dataset_hashes=False,
        )
    except Exception as exception:
        failure_path = _write_startup_failure(
            run_dir=run_dir,
            run_id=args.run_id,
            mode=args.mode,
            config_path=config_path,
            exception=exception,
        )
        print("[FAILED BEFORE SCHEDULER] Release/preflight contract validation failed.")
        print(f"[CAUSE] {type(exception).__name__}: {exception}")
        print(f"[REPORT] {failure_path}")
        return 2

    jobs = (
        discovery_jobs(config, config_path, args.run_id)
        if args.mode == "discovery"
        else confirmatory_jobs(config, config_path, args.run_id)
    )
    manifest = {
        "schema_version": "exp123_resource_protocol_manifest_v2",
        "run_id": args.run_id,
        "mode": args.mode,
        "config_path": str(config_path),
        "resources": resource_config(config),
        "jobs": [job.to_dict() for job in jobs],
    }
    manifest_path = (
        progress_dir / f"resource_{args.mode}_protocol_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    emit_progress(
        "resource_protocol_started",
        operation=args.mode,
        message=f"Resource-aware {args.mode} protocol started.",
        status="running",
        details={"job_count": len(jobs)},
    )
    result = ResourceAwareScheduler(
        jobs=jobs,
        config=config,
        project_root=PROJECT_ROOT,
        run_dir=run_dir,
        run_id=args.run_id,
        rerun_completed=args.rerun_completed,
        mode=args.mode,
    ).run()
    result_path = progress_dir / f"resource_{args.mode}_protocol_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    if result["failed"]:
        reports = write_diagnostic_reports(
            result,
            progress_dir=progress_dir,
            mode=args.mode,
        )
        counts = result.get("counts", {})
        print(
            "[FAILED] "
            f"direct_failures={counts.get('direct_failed', 0)}, "
            f"blocked={counts.get('blocked', 0)}, "
            f"completed={counts.get('completed', result['completed'])}."
        )
        for failure in result.get("root_failures", [])[:10]:
            print(
                f"[ROOT FAILURE] {failure.get('name')} "
                f"category={failure.get('failure_category')} "
                f"reason={failure.get('failure')}"
            )
        print(f"[RESULT] {result_path}")
        print(f"[DIAGNOSIS] {reports['markdown']}")
        return 1

    print(
        f"[OK] Resource-aware EXP123 {args.mode} protocol completed: "
        f"{result['completed']} jobs"
    )
    print(f"[INFO] Scheduler state: {progress_dir / 'scheduler_state.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
