from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from los_baseline.study28.resource_plan import child_thread_environment, read_int_env


REVERSE_REQUIRED_FILES = (
    "summary.csv",
    "bootstrap_ci.csv",
    "reverse_probabilities_sensitive.npz",
    "pair_manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Study28 V9 Alpha 4 sensitivity and clinical-utility analyses.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--alpha2-output-dir", required=True)
    parser.add_argument("--alpha3-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    return parser.parse_args()


def _prepare_reverse_checkpoints(args: argparse.Namespace) -> None:
    """Prepare missing reverse-validation pairs in isolated resource-planned processes.

    The parent deliberately imports no LightGBM-dependent Study28 sensitivity code
    until all reverse workers are finished.  This keeps OpenMP state isolated while
    allowing the two approved reverse directions to run concurrently on large hosts.
    """
    project_root = Path(args.project_root).expanduser().resolve()
    config = json.loads((project_root / "configs" / "study28_sensitivity_analysis.json").read_text())
    reverse_root = Path(args.output_dir).expanduser().resolve() / "01_reverse_validation"
    reverse_root.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[str, str, Path]] = []
    for item in config["reverse_validation"]["pairs"]:
        train_name = str(item["train"])
        eval_name = str(item["evaluate"])
        pair_dir = reverse_root / f"{train_name}_to_{eval_name}"
        if all((pair_dir / name).is_file() for name in REVERSE_REQUIRED_FILES):
            continue
        pair_dir.mkdir(parents=True, exist_ok=True)
        pending.append((train_name, eval_name, pair_dir))

    if not pending:
        return

    workers = min(len(pending), read_int_env("STUDY28_REVERSE_WORKERS", 1))
    threads = read_int_env("STUDY28_REVERSE_THREADS", 1)
    print(
        f"[Study28 Alpha4] reverse resource plan: concurrent_pairs={workers}, "
        f"threads_per_pair={threads}, pending={len(pending)}",
        flush=True,
    )

    def execute(item: tuple[str, str, Path]) -> str:
        train_name, eval_name, pair_dir = item
        command = [
            sys.executable, "-m", "los_baseline.study28.reverse_worker",
            "--dataset-root", str(Path(args.dataset_root).expanduser().resolve()),
            "--project-root", str(project_root),
            "--alpha2-output-dir", str(Path(args.alpha2_output_dir).expanduser().resolve()),
            "--alpha3-output-dir", str(Path(args.alpha3_output_dir).expanduser().resolve()),
            "--train-site", train_name,
            "--evaluation-site", eval_name,
            "--output-dir", str(pair_dir),
        ]
        env = child_thread_environment(os.environ.copy(), threads)
        src_path = str(project_root / "src")
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = src_path + (os.pathsep + current if current else "")
        with (pair_dir / "worker_stdout.log").open("w", encoding="utf-8") as stdout, \
             (pair_dir / "worker_stderr.log").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command, cwd=project_root, env=env, stdout=stdout, stderr=stderr, check=False
            )
        if completed.returncode != 0 or not all((pair_dir / name).is_file() for name in REVERSE_REQUIRED_FILES):
            out_tail = (pair_dir / "worker_stdout.log").read_text(encoding="utf-8", errors="replace")[-3000:]
            err_tail = (pair_dir / "worker_stderr.log").read_text(encoding="utf-8", errors="replace")[-3000:]
            raise RuntimeError(
                f"Reverse worker failed: {train_name}->{eval_name}\nSTDOUT:\n{out_tail}\nSTDERR:\n{err_tail}"
            )
        return f"{train_name}->{eval_name}"

    if workers <= 1:
        for item in pending:
            print(f"[Study28 Alpha4] reverse checkpoint completed: {execute(item)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="study28-reverse") as pool:
            futures = [pool.submit(execute, item) for item in pending]
            for future in as_completed(futures):
                print(f"[Study28 Alpha4] reverse checkpoint completed: {future.result()}", flush=True)


def main() -> int:
    args = parse_args()
    _prepare_reverse_checkpoints(args)
    # Import only after isolated reverse workers have completed.
    from los_baseline.study28.sensitivity import run_alpha4

    result = run_alpha4(
        dataset_root=Path(args.dataset_root),
        project_root=Path(args.project_root),
        alpha2_output_dir=Path(args.alpha2_output_dir),
        alpha3_output_dir=Path(args.alpha3_output_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
