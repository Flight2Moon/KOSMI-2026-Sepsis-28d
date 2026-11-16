from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.experiment_tracking import ExperimentTracker
from los_baseline.progress import configure_progress, emit_progress
from los_baseline.utils import atomic_write_json


def _tracking_worker(run_dir: str, worker_index: int, repeats: int) -> None:
    tracker = ExperimentTracker(
        run_dir=Path(run_dir),
        run_id="concurrency-test",
        config={"tracking": {"wandb": {"enabled": False}}},
        worker_label=f"worker-{worker_index}",
    )
    for index in range(repeats):
        tracker.log({f"worker_{worker_index}/metric_{index}": float(index)})


def _atomic_writer(path: str, worker_index: int, repeats: int) -> None:
    target = Path(path)
    for index in range(repeats):
        atomic_write_json(
            target,
            {"worker": worker_index, "iteration": index},
        )



def _progress_worker(progress_dir: str, worker_index: int, repeats: int) -> None:
    from los_baseline.progress import configure_progress, emit_progress

    configure_progress(
        output_dir=Path(progress_dir),
        run_id="progress-concurrency-test",
        config={
            "progress": {
                "enabled": True,
                "terminal_tqdm": False,
                "refresh_seconds": 0.0,
                "minimum_percent_delta": 0.0,
            },
            "tracking": {
                "fail_open": False,
                "wandb": {"enabled": False},
            },
        },
        worker_label=f"progress-worker-{worker_index}",
    )
    for index in range(1, repeats + 1):
        emit_progress(
            "operation_progress",
            operation=f"bootstrap-{worker_index}",
            current=index,
            total=repeats,
            status="running",
        )
    emit_progress(
        "operation_completed",
        operation=f"bootstrap-{worker_index}",
        current=repeats,
        total=repeats,
        status="completed",
    )

def test_tracking_summary_is_safe_across_processes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    process_count = 6
    repeats = 30
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_tracking_worker,
            args=(str(run_dir), worker_index, repeats),
        )
        for worker_index in range(process_count)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0

    tracking_dir = run_dir / "common" / "tracking"
    summary = json.loads(
        (tracking_dir / "summary.json").read_text(encoding="utf-8")
    )
    for worker_index in range(process_count):
        for index in range(repeats):
            assert summary[f"worker_{worker_index}/metric_{index}"] == float(index)

    history_lines = (
        tracking_dir / "history.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == process_count * repeats
    assert not list(tracking_dir.glob("*.tmp"))
    assert not list(tracking_dir.glob(".*.tmp"))


def test_atomic_json_uses_unique_temporary_files(tmp_path: Path) -> None:
    target = tmp_path / "shared.json"
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_atomic_writer,
            args=(str(target), worker_index, 50),
        )
        for worker_index in range(8)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert set(payload) == {"worker", "iteration"}
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_progress_tracking_failure_is_fail_open(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import los_baseline.progress as progress_module

    configure_progress(
        output_dir=tmp_path / "progress",
        run_id="fail-open",
        config={
            "progress": {"enabled": True, "terminal_tqdm": False},
            "tracking": {"fail_open": True, "wandb": {"enabled": False}},
        },
        worker_label="worker",
    )

    def _raise_tracking_error(*args, **kwargs):
        raise FileNotFoundError("simulated tracking failure")

    monkeypatch.setattr(progress_module, "log_metrics", _raise_tracking_error)
    emit_progress(
        "operation_progress",
        operation="bootstrap",
        current=1,
        total=10,
        status="running",
    )

    error_path = tmp_path / "progress" / "tracking_errors.jsonl"
    rows = [json.loads(line) for line in error_path.read_text().splitlines()]
    assert rows[-1]["event"] == "tracking_write_failed"
    assert rows[-1]["exception_type"] == "FileNotFoundError"


def test_progress_and_tracking_survive_parallel_workers(tmp_path: Path) -> None:
    progress_dir = tmp_path / "run" / "common" / "progress"
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_progress_worker,
            args=(str(progress_dir), worker_index, 25),
        )
        for worker_index in range(6)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0

    run_progress = json.loads(
        (progress_dir / "run_progress.json").read_text(encoding="utf-8")
    )
    progress_tree = json.loads(
        (progress_dir / "progress_tree.json").read_text(encoding="utf-8")
    )
    tracking_summary = json.loads(
        (tmp_path / "run" / "common" / "tracking" / "summary.json").read_text(encoding="utf-8")
    )

    assert run_progress["run_id"] == "progress-concurrency-test"
    assert progress_tree["run_id"] == "progress-concurrency-test"
    assert tracking_summary["progress/failed_nodes"] == 0
    assert not (progress_dir / "tracking_errors.jsonl").exists()
