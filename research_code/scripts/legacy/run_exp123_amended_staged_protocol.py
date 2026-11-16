#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.amendments import AMENDMENT_ENV, verify_amended_run_preflight
from los_baseline.config import load_config, resolve_path
from los_baseline.experiment_scheduler import ResourceAwareScheduler
from los_baseline.failure_diagnostics import write_diagnostic_reports
from los_baseline.progress import configure_progress, emit_progress
from los_baseline.utils import atomic_write_json
from run_exp123_staged_protocol import baseline_jobs, optimized_jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume an EXP123 staged Run with a recorded source amendment."
    )
    parser.add_argument("phase", choices=["baseline", "optimize"])
    parser.add_argument("run_id")
    parser.add_argument("--config", required=True)
    parser.add_argument("--amendment-manifest", required=True)
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    amendment_path = Path(args.amendment_manifest).expanduser().resolve()
    config = load_config(config_path)
    run_dir = resolve_path(config, config["paths"]["runs_dir"]) / args.run_id
    configure_progress(
        output_dir=run_dir / "common" / "progress",
        run_id=args.run_id,
        config=config,
        worker_label=f"amended_staged_protocol_{args.phase}",
    )
    try:
        verification = verify_amended_run_preflight(
            project_root=PROJECT_ROOT,
            config_path=config_path,
            run_dir=run_dir,
            amendment_path=amendment_path,
            verify_dataset_hashes=False,
        )
        os.environ[AMENDMENT_ENV] = str(amendment_path)
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
            mode=f"staged_{args.phase}_amended_v73",
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
                mode=f"staged_{args.phase}_amended_v73",
            )
            print(f"[FAILED] {failed} job(s) failed or were blocked.")
            print(f"[RESULT] {result_path}")
            print(f"[DIAGNOSIS] {reports['markdown']}")
            return 1
        completion = {
            "schema_version": "exp123_amended_discovery_completion_v1",
            "status": "completed_with_recorded_source_amendment",
            "run_id": args.run_id,
            "phase": args.phase,
            "verification": verification,
            "amendment_manifest": str(amendment_path),
            "confirmatory_splits_accessed": False,
            "clean_v73_run_recommended_for_final_confirmatory_analysis": True,
        }
        completion_path = (
            run_dir
            / "common"
            / "amendments"
            / "AMENDED_DISCOVERY_COMPLETED.lock.json"
        )
        atomic_write_json(completion_path, completion)
        emit_progress(
            "amended_staged_protocol_completed",
            operation=args.phase,
            message=f"EXP123 amended staged {args.phase} completed",
            status="completed",
            details={"result": result, "amendment": verification},
        )
        print(f"[OK] EXP123 amended staged {args.phase} completed: {args.run_id}")
        print(f"[AMENDMENT] {amendment_path}")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
