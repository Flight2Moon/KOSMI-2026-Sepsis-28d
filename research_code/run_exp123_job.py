#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config, resolve_path
from los_baseline.amendments import verify_run_execution_contract as verify_run_preflight
from los_baseline.final_freeze import assert_discovery_step_allowed
from los_baseline.parallel_jobs import (
    finalize_optimization_task,
    finalize_training_task,
    run_model_optimization_job,
    run_model_training_job,
)
from los_baseline.staged_protocol import (
    build_exp2_pairwise_validation_report,
    finalize_staged_optimization,
    prepare_baseline_model_parameters,
    select_targeted_candidates,
    verify_linked_baseline,
)
from los_baseline.progress import configure_progress, emit_progress
from los_baseline.hardware import configure_torch_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one resumable EXP123 model-level job.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--job",
        choices=[
            "prepare-baseline-model",
            "optimize-model",
            "finalize-optimization",
            "train-model",
            "finalize-training",
            "build-exp2-pairwise",
            "select-targeted-candidates",
            "verify-baseline-link",
            "finalize-staged-optimization",
        ],
        required=True,
    )
    parser.add_argument("--task", choices=["exp1", "stage1", "stage2"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--search-stages", default="smoke,coarse,fine")
    parser.add_argument("--bootstrap-repeats", type=int, default=None)
    parser.add_argument("--phase", choices=["baseline", "optimized"], default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    runtime = configure_torch_runtime(config)
    run_dir = resolve_path(config, config["paths"]["runs_dir"]) / args.run_id
    configure_progress(
        output_dir=run_dir / "common" / "progress",
        run_id=args.run_id,
        config=config,
        worker_label=f"{args.job}_{args.task}_{args.model or 'all'}",
    )
    verify_run_preflight(
        project_root=PROJECT_ROOT,
        config_path=config_path,
        run_dir=run_dir,
        verify_dataset_hashes=False,
    )
    assert_discovery_step_allowed(run_dir)
    emit_progress("model_job_started", operation=args.job, message=f"{args.task}/{args.model or 'all'}", status="running", details={"torch_runtime": runtime})
    if args.job == "prepare-baseline-model":
        if not args.model:
            raise ValueError("--model is required for prepare-baseline-model.")
        result = prepare_baseline_model_parameters(
            task_key=args.task,
            model_name=args.model,
            config=config,
            run_id=args.run_id,
            run_dir=run_dir,
            overwrite=args.no_resume,
        ).to_dict()
    elif args.job == "optimize-model":
        if not args.model:
            raise ValueError("--model is required for optimize-model.")
        result = run_model_optimization_job(
            task_key=args.task,
            model_name=args.model,
            config=config,
            run_id=args.run_id,
            run_dir=run_dir,
            search_stages=[value.strip() for value in args.search_stages.split(",") if value.strip()],
            resume=not args.no_resume,
        )
    elif args.job == "finalize-optimization":
        result = finalize_optimization_task(task_key=args.task, config=config, run_id=args.run_id, run_dir=run_dir)
    elif args.job == "train-model":
        if not args.model:
            raise ValueError("--model is required for train-model.")
        result = run_model_training_job(
            task_key=args.task,
            model_name=args.model,
            config=config,
            run_id=args.run_id,
            run_dir=run_dir,
            bootstrap_repeats=args.bootstrap_repeats,
            resume=not args.no_resume,
        )
    elif args.job == "finalize-training":
        result = finalize_training_task(task_key=args.task, config=config, run_id=args.run_id, run_dir=run_dir)
    elif args.job == "build-exp2-pairwise":
        if not args.phase:
            raise ValueError("--phase is required for build-exp2-pairwise.")
        result = build_exp2_pairwise_validation_report(
            config=config,
            run_id=args.run_id,
            run_dir=run_dir,
            phase=args.phase,
            overwrite=args.no_resume,
        )
    elif args.job == "select-targeted-candidates":
        result = select_targeted_candidates(
            config=config,
            run_id=args.run_id,
            run_dir=run_dir,
            overwrite=args.no_resume,
        ).to_dict()
    elif args.job == "verify-baseline-link":
        result = verify_linked_baseline(
            config=config, run_id=args.run_id, run_dir=run_dir
        )
    else:
        result = finalize_staged_optimization(
            config=config, run_id=args.run_id, run_dir=run_dir
        )
    emit_progress("model_job_completed", operation=args.job, message=f"{args.task}/{args.model or 'all'}", status="completed", details={"result": result})
    print(f"[OK] {args.job}: {args.task}/{args.model or 'all'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
