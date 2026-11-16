#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config, resolve_path
from los_baseline.experiment_scheduler import ResourceAwareScheduler, ScheduledJob
from los_baseline.failure_diagnostics import write_diagnostic_reports
from los_baseline.final_freeze import verify_run_preflight
from los_baseline.parallel_jobs import TASKS
from los_baseline.progress import configure_progress, emit_progress
from los_baseline.resource_manager import request_for_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline-first or targeted-optimization EXP123 protocol."
    )
    parser.add_argument("phase", choices=["baseline", "optimize"])
    parser.add_argument("run_id")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def _standard_command(config_path: Path, run_id: str, step: str, *extra: str) -> tuple[str, ...]:
    return (
        sys.executable,
        str(PROJECT_ROOT / "run_exp123.py"),
        "--config",
        str(config_path),
        "--run-id",
        run_id,
        "--step",
        step,
        *extra,
    )


def _job_command(
    config_path: Path,
    run_id: str,
    job: str,
    task: str = "exp1",
    model: str | None = None,
    *extra: str,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "run_exp123_job.py"),
        "--config",
        str(config_path),
        "--run-id",
        run_id,
        "--job",
        job,
        "--task",
        task,
    ]
    if model:
        command += ["--model", model]
    command += list(extra)
    return tuple(command)


def _models(config: Mapping[str, Any], task_key: str) -> list[str]:
    if task_key == "exp1":
        return list(config["experiments"]["exp1_flat_3class"]["models"])
    stage = "stage1_mortality" if task_key == "stage1" else "stage2_long_stay"
    return list(config["experiments"]["exp2_two_stage"][stage]["models"])


def _bootstrap_repeats(config: Mapping[str, Any], phase: str) -> int:
    staged = dict(config.get("staged_protocol", {}))
    section_name = "baseline" if phase == "baseline" else "targeted_optimization"
    section = dict(staged.get(section_name, {}))
    default = 200 if phase == "baseline" else 1000
    return max(1, int(section.get("bootstrap_repeats", default)))


def baseline_jobs(config: Mapping[str, Any], config_path: Path, run_id: str) -> list[ScheduledJob]:
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
    repeats = _bootstrap_repeats(config, "baseline")
    finalize_names: dict[str, str] = {}
    for task_key in ("exp1", "stage1", "stage2"):
        train_names: list[str] = []
        task_name = TASKS[task_key]["task"].task_name
        for model in _models(config, task_key):
            prepare_name = f"{task_key}-baseline-params-{model}"
            train_name = f"{task_key}-baseline-train-{model}"
            train_names.append(train_name)
            jobs.append(
                ScheduledJob(
                    name=prepare_name,
                    command=_job_command(
                        config_path,
                        run_id,
                        "prepare-baseline-model",
                        task_key,
                        model,
                    ),
                    dependencies=("splits",),
                    request=request_for_job(config, model_name=None, job_kind="finalize"),
                    completion_path=(
                        f"optimization/{task_name}/{model}/"
                        "BASELINE_PARAMETERS_COMPLETED.lock.json"
                    ),
                    model_name=model,
                    task_key=task_key,
                    kind="baseline_parameter_materialization",
                )
            )
            jobs.append(
                ScheduledJob(
                    name=train_name,
                    command=_job_command(
                        config_path,
                        run_id,
                        "train-model",
                        task_key,
                        model,
                        "--bootstrap-repeats",
                        str(repeats),
                    ),
                    dependencies=(prepare_name,),
                    request=request_for_job(config, model_name=model, job_kind="training"),
                    completion_path=(
                        f"{TASKS[task_key]['output_relative']}/{model}/"
                        "MODEL_TRAINING_COMPLETED.lock.json"
                    ),
                    model_name=model,
                    task_key=task_key,
                    kind="baseline_training",
                )
            )
        result_name = "exp1_result.json" if task_key == "exp1" else f"{task_key}_result.json"
        final_name = f"{task_key}-baseline-finalize"
        finalize_names[task_key] = final_name
        jobs.append(
            ScheduledJob(
                name=final_name,
                command=_job_command(config_path, run_id, "finalize-training", task_key),
                dependencies=tuple(train_names),
                request=request_for_job(config, model_name=None, job_kind="finalize"),
                completion_path=f"{TASKS[task_key]['output_relative']}/{result_name}",
                task_key=task_key,
                kind="baseline_finalize",
            )
        )

    jobs.append(
        ScheduledJob(
            name="exp2-baseline-pairwise-validation",
            command=_job_command(
                config_path,
                run_id,
                "build-exp2-pairwise",
                "exp1",
                None,
                "--phase",
                "baseline",
            ),
            dependencies=(finalize_names["stage1"], finalize_names["stage2"]),
            request=request_for_job(config, model_name=None, job_kind="finalize"),
            completion_path=(
                "common/staged_baseline/exp2_pairwise_validation/"
                "PAIRWISE_VALIDATION_COMPLETED.lock.json"
            ),
            kind="pairwise_validation",
        )
    )
    jobs.append(
        ScheduledJob(
            name="baseline-candidate-plan",
            command=_job_command(
                config_path,
                run_id,
                "select-targeted-candidates",
                "exp1",
            ),
            dependencies=(
                finalize_names["exp1"],
                finalize_names["stage1"],
                finalize_names["stage2"],
                "exp2-baseline-pairwise-validation",
            ),
            request=request_for_job(config, model_name=None, job_kind="finalize"),
            completion_path="common/staged_baseline/candidate_plan.json",
            kind="candidate_selection",
        )
    )
    return jobs


def optimized_jobs(config: Mapping[str, Any], config_path: Path, run_id: str) -> list[ScheduledJob]:
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
        ScheduledJob(
            name="verify-baseline-link",
            command=_job_command(config_path, run_id, "verify-baseline-link", "exp1"),
            dependencies=("splits",),
            request=request_for_job(config, model_name=None, job_kind="finalize"),
            completion_path="common/staged_optimized/BASELINE_LINK_VERIFIED.lock.json",
            kind="baseline_link_verification",
        ),
    ]
    search_stages = list(
        config.get("staged_protocol", {})
        .get("targeted_optimization", {})
        .get("search_stages", ["coarse", "fine"])
    )
    search_stage_arg = ",".join(str(value) for value in search_stages)
    repeats = _bootstrap_repeats(config, "optimized")
    finalize_names: dict[str, str] = {}
    optimization_finalize_names: dict[str, str] = {}
    for task_key in ("exp1", "stage1", "stage2"):
        opt_names: list[str] = []
        train_names: list[str] = []
        task_name = TASKS[task_key]["task"].task_name
        for model in _models(config, task_key):
            opt_name = f"{task_key}-targeted-optimize-{model}"
            train_name = f"{task_key}-optimized-train-{model}"
            opt_names.append(opt_name)
            train_names.append(train_name)
            jobs.append(
                ScheduledJob(
                    name=opt_name,
                    command=_job_command(
                        config_path,
                        run_id,
                        "optimize-model",
                        task_key,
                        model,
                        "--search-stages",
                        search_stage_arg,
                    ),
                    dependencies=("verify-baseline-link",),
                    request=request_for_job(config, model_name=model, job_kind="optimization"),
                    completion_path=(
                        f"optimization/{task_name}/{model}/"
                        "MODEL_OPTIMIZATION_COMPLETED.lock.json"
                    ),
                    model_name=model,
                    task_key=task_key,
                    kind="targeted_optimization",
                )
            )
            jobs.append(
                ScheduledJob(
                    name=train_name,
                    command=_job_command(
                        config_path,
                        run_id,
                        "train-model",
                        task_key,
                        model,
                        "--bootstrap-repeats",
                        str(repeats),
                    ),
                    dependencies=(opt_name,),
                    request=request_for_job(config, model_name=model, job_kind="training"),
                    completion_path=(
                        f"{TASKS[task_key]['output_relative']}/{model}/"
                        "MODEL_TRAINING_COMPLETED.lock.json"
                    ),
                    model_name=model,
                    task_key=task_key,
                    kind="optimized_training",
                )
            )
        opt_finalize = f"{task_key}-targeted-optimization-finalize"
        optimization_finalize_names[task_key] = opt_finalize
        jobs.append(
            ScheduledJob(
                name=opt_finalize,
                command=_job_command(config_path, run_id, "finalize-optimization", task_key),
                dependencies=tuple(opt_names),
                request=request_for_job(config, model_name=None, job_kind="finalize"),
                completion_path=(
                    f"optimization/{task_name}/OPTIMIZATION_TASK_COMPLETED.lock.json"
                ),
                task_key=task_key,
                kind="optimization_finalize",
            )
        )
        result_name = "exp1_result.json" if task_key == "exp1" else f"{task_key}_result.json"
        final_name = f"{task_key}-optimized-training-finalize"
        finalize_names[task_key] = final_name
        jobs.append(
            ScheduledJob(
                name=final_name,
                command=_job_command(config_path, run_id, "finalize-training", task_key),
                dependencies=tuple(train_names),
                request=request_for_job(config, model_name=None, job_kind="finalize"),
                completion_path=f"{TASKS[task_key]['output_relative']}/{result_name}",
                task_key=task_key,
                kind="optimized_training_finalize",
            )
        )

    jobs.append(
        ScheduledJob(
            name="exp2-optimized-pairwise-validation",
            command=_job_command(
                config_path,
                run_id,
                "build-exp2-pairwise",
                "exp1",
                None,
                "--phase",
                "optimized",
            ),
            dependencies=(finalize_names["stage1"], finalize_names["stage2"]),
            request=request_for_job(config, model_name=None, job_kind="finalize"),
            completion_path=(
                "common/staged_optimized/exp2_pairwise_validation/"
                "PAIRWISE_VALIDATION_COMPLETED.lock.json"
            ),
            kind="pairwise_validation",
        )
    )
    jobs.extend(
        [
            ScheduledJob(
                name="stage1-heterogeneity-discovery",
                command=_standard_command(config_path, run_id, "stage1-heterogeneity-discovery"),
                dependencies=(finalize_names["stage1"],),
                request=request_for_job(config, model_name=None, job_kind="heterogeneity"),
                completion_path=(
                    "EXP2_two_stage/heterogeneity_analysis/stage1_mortality/"
                    "discovery_validation/frozen_design.json"
                ),
                kind="heterogeneity",
            ),
            ScheduledJob(
                name="stage2-heterogeneity-discovery",
                command=_standard_command(config_path, run_id, "stage2-heterogeneity-discovery"),
                dependencies=(finalize_names["stage2"],),
                request=request_for_job(config, model_name=None, job_kind="heterogeneity"),
                completion_path=(
                    "EXP2_two_stage/heterogeneity_analysis/stage2_long_stay/"
                    "discovery_validation/frozen_design.json"
                ),
                kind="heterogeneity",
            ),
            ScheduledJob(
                name="stage1-super-learner",
                command=_standard_command(config_path, run_id, "stage1-super-learner"),
                dependencies=("stage1-heterogeneity-discovery",),
                request=request_for_job(config, model_name=None, job_kind="super_learner"),
                completion_path=(
                    "EXP2_two_stage/super_learner/stage1_mortality/artifact.json"
                ),
                kind="super_learner",
            ),
            ScheduledJob(
                name="stage2-super-learner",
                command=_standard_command(config_path, run_id, "stage2-super-learner"),
                dependencies=("stage2-heterogeneity-discovery",),
                request=request_for_job(config, model_name=None, job_kind="super_learner"),
                completion_path=(
                    "EXP2_two_stage/super_learner/stage2_long_stay/artifact.json"
                ),
                kind="super_learner",
            ),
            ScheduledJob(
                name="staged-optimization-summary",
                command=_job_command(
                    config_path,
                    run_id,
                    "finalize-staged-optimization",
                    "exp1",
                ),
                dependencies=(
                    finalize_names["exp1"],
                    "exp2-optimized-pairwise-validation",
                    "stage1-super-learner",
                    "stage2-super-learner",
                ),
                request=request_for_job(config, model_name=None, job_kind="finalize"),
                completion_path=(
                    "common/staged_optimized/"
                    "STAGED_OPTIMIZATION_COMPLETED.lock.json"
                ),
                kind="staged_optimization_finalize",
            ),
        ]
    )
    return jobs


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    run_dir = resolve_path(config, config["paths"]["runs_dir"]) / args.run_id
    configure_progress(
        output_dir=run_dir / "common" / "progress",
        run_id=args.run_id,
        config=config,
        worker_label=f"staged_protocol_{args.phase}",
    )
    try:
        verify_run_preflight(
            project_root=PROJECT_ROOT,
            config_path=config_path,
            run_dir=run_dir,
            verify_dataset_hashes=False,
        )
        jobs = (
            baseline_jobs(config, config_path, args.run_id)
            if args.phase == "baseline"
            else optimized_jobs(config, config_path, args.run_id)
        )
        scheduler = ResourceAwareScheduler(
            jobs=jobs,
            config=config,
            project_root=PROJECT_ROOT,
            run_dir=run_dir,
            run_id=args.run_id,
            rerun_completed=args.rerun_completed,
            mode=f"staged_{args.phase}",
        )
        result = scheduler.run()
        progress_dir = run_dir / "common" / "progress"
        result_path = progress_dir / f"resource_staged_{args.phase}_protocol_result.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        failed = int(
            result.get("counts", {}).get(
                "failed_or_blocked", result.get("failed", 0)
            )
        )
        if failed:
            reports = write_diagnostic_reports(
                result,
                progress_dir=progress_dir,
                mode=f"staged_{args.phase}",
            )
            print(f"[FAILED] {failed} job(s) failed or were blocked.")
            print(f"[RESULT] {result_path}")
            print(f"[DIAGNOSIS] {reports['markdown']}")
            return 1
        emit_progress(
            "staged_protocol_completed",
            operation=args.phase,
            message=f"EXP123 staged {args.phase} completed",
            status="completed",
            details={"result": result},
        )
        print(f"[OK] EXP123 staged {args.phase} completed: {args.run_id}")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
