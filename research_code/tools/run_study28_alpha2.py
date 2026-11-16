#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from los_baseline.study28.resource_plan import child_thread_environment, read_int_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Study28 V9 Alpha 2: audit professor-provided frozen data, reproduce the "
            "5x3 LightGBM nested-CV lock, run MIMIC temporal evaluation, and freeze "
            "the full-MIMIC primary model for later external validation."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _prepare_nested_fold_checkpoints(args: argparse.Namespace, output_dir: Path) -> None:
    """Prepare one resource-planned batch of missing folds, then restart cleanly.

    Fresh-parent batching retains the measured throughput benefit of concurrent
    isolated folds while avoiding OpenMP/runtime state accumulation across batches.
    The actual CV definition remains in cv_worker and is unchanged.
    """
    project_root = args.project_root.expanduser().resolve()
    protocol = json.loads((project_root / "configs" / "study28_final.json").read_text(encoding="utf-8"))
    outer_folds = int(protocol["development"]["nested_cv"]["outer_folds"])
    fold_dir = output_dir / "01_nested_cv" / "outer_fold_checkpoints"
    fold_dir.mkdir(parents=True, exist_ok=True)
    missing = [idx for idx in range(outer_folds) if not (fold_dir / f"outer_fold_{idx}.joblib").is_file()]
    if not missing:
        return

    workers = min(len(missing), read_int_env("STUDY28_NESTED_CV_WORKERS", 1))
    threads = read_int_env("STUDY28_NESTED_CV_THREADS", 1)
    batch = missing[:workers]
    print(
        f"[Study28 Alpha2] preparing nested-CV batch={batch}, concurrent={len(batch)}, "
        f"threads_per_fold={threads}", flush=True,
    )

    def execute(idx: int) -> int:
        checkpoint = fold_dir / f"outer_fold_{idx}.joblib"
        log_path = fold_dir / f"outer_fold_{idx}.log"
        command = [
            sys.executable, "-m", "los_baseline.study28.cv_worker",
            "--dataset-root", str(args.dataset_root.expanduser().resolve()),
            "--project-root", str(project_root),
            "--outer-index", str(idx),
            "--output", str(checkpoint),
        ]
        env = child_thread_environment(os.environ.copy(), threads)
        src = str(project_root / "src")
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = src + (os.pathsep + current if current else "")
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=project_root, env=env, stdout=log,
                stderr=subprocess.STDOUT, check=False, text=True,
            )
        if completed.returncode != 0 or not checkpoint.is_file():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
            raise RuntimeError(f"Nested-CV fold {idx} failed. Log tail:\n{tail}")
        return idx

    if len(batch) == 1:
        print(f"[Study28 Alpha2] nested fold checkpoint completed: {execute(batch[0]) + 1}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=len(batch), thread_name_prefix="study28-cv-batch") as pool:
            futures = [pool.submit(execute, idx) for idx in batch]
            for future in as_completed(futures):
                print(f"[Study28 Alpha2] nested fold checkpoint completed: {future.result() + 1}", flush=True)

    remaining = [idx for idx in range(outer_folds) if not (fold_dir / f"outer_fold_{idx}.joblib").is_file()]
    if remaining:
        # New interpreter, same CLI and frozen environment/resource plan.
        os.execvpe(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], os.environ.copy())


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.project_root / "runs" / f"study28_alpha2_{stamp}"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _prepare_nested_fold_checkpoints(args, output_dir)
    # Import only after all fold workers are complete, preserving a clean parent
    # for the resource-planned isolated LightGBM workers above.
    from los_baseline.study28.runner import run_alpha2

    result = run_alpha2(
        dataset_root=args.dataset_root,
        project_root=args.project_root,
        output_dir=output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
