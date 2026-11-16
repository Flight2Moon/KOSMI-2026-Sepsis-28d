from __future__ import annotations

import json
import os
import sys
import time
import uuid

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Iterator, TypeVar

from tqdm.auto import tqdm

from .experiment_tracking import configure_tracker, get_tracker, log_metrics

T = TypeVar("T")
_LOCK = Lock()
_STATE: "ProgressState | None" = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_label(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
    with _LOCK:
        with path.open("a", encoding="utf-8") as file:
            if fcntl is not None:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            try:
                file.write(line); file.flush()
            finally:
                if fcntl is not None:
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)


@dataclass
class ProgressNode:
    node_id: str
    parent_id: str | None
    operation: str
    worker: str
    status: str = "queued"
    current: float | None = None
    total: float | None = None
    unit: str | None = None
    percent: float | None = None
    started_at_utc: str | None = None
    updated_at_utc: str | None = None
    completed_at_utc: str | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    rate: float | None = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ProgressState:
    output_dir: Path
    run_id: str
    worker_label: str
    enabled: bool = True
    tqdm_enabled: bool = True
    refresh_seconds: float = 0.5
    minimum_percent_delta: float = 0.5
    parent_node_id: str | None = None
    tracking_fail_open: bool = True
    nodes: dict[str, ProgressNode] = field(default_factory=dict)
    started_monotonic: dict[str, float] = field(default_factory=dict)
    last_emit_monotonic: dict[str, float] = field(default_factory=dict)
    last_emitted_percent: dict[str, float | None] = field(default_factory=dict)
    last_emitted_status: dict[str, str] = field(default_factory=dict)

    @property
    def current_nodes_dir(self) -> Path:
        return self.output_dir / "current_nodes"

    @property
    def event_path(self) -> Path:
        return self.output_dir / "progress_events.jsonl"

    @property
    def tree_path(self) -> Path:
        return self.output_dir / "progress_tree.json"

    @property
    def run_path(self) -> Path:
        return self.output_dir / "run_progress.json"

    @property
    def current_path(self) -> Path:
        return self.output_dir / f"current_{_safe_label(self.worker_label)}.json"

    @property
    def completed_path(self) -> Path:
        return self.output_dir / "completed_operations.jsonl"


def configure_progress(*, output_dir: Path, run_id: str, config: dict[str, Any] | None = None, worker_label: str | None = None) -> ProgressState:
    global _STATE
    cfg = dict((config or {}).get("progress", {}))
    env_disable = os.environ.get("EXP123_DISABLE_PROGRESS", "").lower() in {"1", "true", "yes", "on"}
    resolved_worker = worker_label or os.environ.get("EXP123_WORKER_LABEL") or f"pid_{os.getpid()}"
    renderer_worker = (
        os.environ.get("EXP123_PROGRESS_RENDERER", "").lower() in {"1", "true", "yes", "on"}
        or resolved_worker.startswith(("resource_scheduler_", "staged_protocol_", "protocol_", "series_"))
    )
    state = ProgressState(
        output_dir=output_dir.expanduser().resolve(), run_id=str(run_id),
        worker_label=resolved_worker,
        enabled=bool(cfg.get("enabled", True)) and not env_disable,
        tqdm_enabled=(
            bool(cfg.get("terminal_tqdm", cfg.get("tqdm", True)))
            and (bool(cfg.get("worker_tqdm", False)) or renderer_worker)
            and not env_disable
        ),
        refresh_seconds=float(cfg.get("refresh_seconds", 0.5)),
        minimum_percent_delta=float(cfg.get("minimum_percent_delta", 0.5)),
        parent_node_id=os.environ.get("EXP123_PROGRESS_PARENT"),
        tracking_fail_open=bool((config or {}).get("tracking", {}).get("fail_open", True)),
    )
    state.output_dir.mkdir(parents=True, exist_ok=True)
    _STATE = state
    if config is not None:
        run_dir = state.output_dir.parent.parent if state.output_dir.name == "progress" else state.output_dir.parent
        try:
            configure_tracker(run_dir=run_dir, run_id=run_id, config=config, worker_label=state.worker_label)
        except Exception:
            if bool(config.get("tracking", {}).get("wandb", {}).get("required", False)):
                raise
    emit_progress("worker_started", message="Progress tracking initialized.", status="running")
    return state


def get_progress_state() -> ProgressState | None:
    return _STATE


def _node_id(operation: str, details: dict[str, Any] | None = None) -> str:
    if details and details.get("progress_node_id"):
        return str(details["progress_node_id"])
    state = _STATE
    worker = state.worker_label if state else str(os.getpid())
    return _safe_label(f"{worker}__{operation}")


def _update_tree(state: ProgressState) -> None:
    state.current_nodes_dir.mkdir(parents=True, exist_ok=True)
    merged: dict[str, ProgressNode] = {}
    for path in state.current_nodes_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            node_payload = payload.get("node", payload)
            node = ProgressNode(**{
                key: node_payload.get(key)
                for key in ProgressNode.__dataclass_fields__
            })
            merged[node.node_id] = node
        except Exception:
            continue
    for node in state.nodes.values():
        merged[node.node_id] = node
    parent_ids = {node.parent_id for node in merged.values() if node.parent_id}
    leaves = [node for node in merged.values() if node.node_id not in parent_ids]
    completed = sum(1 for node in merged.values() if node.status == "completed")
    failed = sum(1 for node in merged.values() if node.status == "failed")
    contributions: list[float] = []
    for node in leaves:
        if node.status == "completed":
            contributions.append(100.0)
        elif node.status == "failed":
            contributions.append(float(node.percent or 0.0))
        elif node.percent is not None:
            contributions.append(float(node.percent))
        else:
            contributions.append(0.0)
    computed = float(sum(contributions) / len(contributions)) if contributions else 0.0
    previous = 0.0
    if state.run_path.is_file():
        try:
            previous = float(json.loads(state.run_path.read_text(encoding="utf-8")).get("overall_percent", 0.0))
        except Exception:
            previous = 0.0
    overall = max(previous, computed)
    nodes = [node.to_dict() for node in merged.values()]
    payload = {
        "schema_version": "exp123_hierarchical_progress_v3",
        "run_id": state.run_id,
        "updated_at_utc": _utc_now(),
        "overall_percent": min(100.0, max(0.0, overall)),
        "overall_method": "monotonic_leaf_average",
        "node_count": len(nodes), "completed_count": completed, "failed_count": failed,
        "nodes": nodes,
    }
    _atomic_json(state.tree_path, payload)
    _atomic_json(state.run_path, {key: value for key, value in payload.items() if key != "nodes"})


def emit_progress(event: str, *, message: str = "", operation: str | None = None, current: int | float | None = None, total: int | float | None = None, unit: str | None = None, status: str | None = None, details: dict[str, Any] | None = None, node_id: str | None = None, parent_id: str | None = None) -> None:
    state = _STATE
    if state is None or not state.enabled:
        return
    now = _utc_now(); monotonic = time.perf_counter(); details = dict(details or {})
    operation_name = operation or event
    nid = node_id or _node_id(operation_name, details)
    node = state.nodes.get(nid)
    if node is None:
        node = ProgressNode(node_id=nid, parent_id=parent_id or details.get("parent_id") or state.parent_node_id, operation=operation_name, worker=state.worker_label)
        state.nodes[nid] = node
        state.started_monotonic[nid] = monotonic
    node.status = status or node.status
    node.current = current if current is not None else node.current
    node.total = total if total is not None else node.total
    node.unit = unit or node.unit
    node.message = message
    node.details.update(details)
    node.updated_at_utc = now
    if node.started_at_utc is None:
        node.started_at_utc = now
    elapsed = max(0.0, monotonic - state.started_monotonic.get(nid, monotonic))
    node.elapsed_seconds = elapsed
    if node.current is not None and node.total not in {None, 0}:
        node.percent = min(100.0, max(0.0, float(node.current) / float(node.total) * 100.0))
        if elapsed > 0 and float(node.current) > 0:
            node.rate = float(node.current) / elapsed
            remaining = max(0.0, float(node.total) - float(node.current))
            node.eta_seconds = remaining / node.rate if node.rate > 0 else None
    if event in {"operation_completed", "step_completed", "model_job_completed", "run_completed"} or node.status == "completed":
        node.status = "completed"; node.completed_at_utc = now
        if node.total is not None: node.current = node.total; node.percent = 100.0; node.eta_seconds = 0.0
    if event in {"operation_failed", "step_failed"} or node.status == "failed":
        node.status = "failed"; node.completed_at_utc = now
    previous_emit = state.last_emit_monotonic.get(nid, 0.0)
    previous_percent = state.last_emitted_percent.get(nid)
    previous_status = state.last_emitted_status.get(nid)
    percent_delta = (
        100.0 if previous_percent is None or node.percent is None
        else abs(float(node.percent) - float(previous_percent))
    )
    force_emit = (
        event != "operation_progress"
        or node.status != previous_status
        or monotonic - previous_emit >= state.refresh_seconds
        or percent_delta >= state.minimum_percent_delta
    )
    if not force_emit:
        return
    state.last_emit_monotonic[nid] = monotonic
    state.last_emitted_percent[nid] = node.percent
    state.last_emitted_status[nid] = node.status
    node_payload = {"run_id": state.run_id, "worker": state.worker_label, "node": node.to_dict()}
    _atomic_json(state.current_nodes_dir / f"{_safe_label(nid)}.json", node_payload)
    payload = {"timestamp_utc": now, "run_id": state.run_id, "pid": os.getpid(), "worker": state.worker_label, "event": event, **node.to_dict()}
    _append_jsonl(state.event_path, payload); _atomic_json(state.current_path, payload); _update_tree(state)
    if node.status in {"completed", "failed"}:
        _append_jsonl(state.completed_path, payload)
    metrics = {"progress/overall_percent": json.loads(state.run_path.read_text()).get("overall_percent", 0.0), f"progress/{_safe_label(operation_name)}/percent": node.percent or 0.0, "progress/completed_nodes": sum(1 for n in state.nodes.values() if n.status == "completed"), "progress/failed_nodes": sum(1 for n in state.nodes.values() if n.status == "failed")}
    try:
        log_metrics(metrics)
    except Exception as exc:
        # Local/W&B observability must not invalidate an otherwise successful
        # clinical modelling job. The failure is retained as an explicit
        # diagnostic event and can be made fatal with tracking.fail_open=false.
        _append_jsonl(
            state.output_dir / "tracking_errors.jsonl",
            {
                "timestamp_utc": now,
                "run_id": state.run_id,
                "pid": os.getpid(),
                "worker": state.worker_label,
                "event": "tracking_write_failed",
                "operation": operation_name,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        if not state.tracking_fail_open:
            raise


def progress_iter(iterable: Iterable[T], *, total: int | None = None, description: str, unit: str = "item", leave: bool = True, position: int | None = None, details: dict[str, Any] | None = None) -> Iterator[T]:
    state = _STATE
    if total is None:
        try: total = len(iterable)  # type: ignore[arg-type]
        except TypeError: total = None
    nid = _node_id(description, details)
    emit_progress("operation_started", operation=description, current=0, total=total, unit=unit, status="running", details=details, node_id=nid)
    wrapped = tqdm(iterable, total=total, desc=description, unit=unit, leave=leave, position=position, disable=not bool(state and state.tqdm_enabled and sys.stderr.isatty()), mininterval=state.refresh_seconds if state else 0.5, dynamic_ncols=True)
    completed = 0
    try:
        for completed, item in enumerate(wrapped, start=1):
            emit_progress("operation_progress", operation=description, current=completed, total=total, unit=unit, status="running", details=details, node_id=nid)
            yield item
    except Exception as exc:
        emit_progress("operation_failed", operation=description, current=completed, total=total, unit=unit, status="failed", message=str(exc), details={**(details or {}), "exception_type": type(exc).__name__}, node_id=nid)
        raise
    else:
        emit_progress("operation_completed", operation=description, current=completed, total=total, unit=unit, status="completed", details=details, node_id=nid)
    finally:
        wrapped.close()


@contextmanager
def tracked_operation(description: str, *, details: dict[str, Any] | None = None) -> Iterator[None]:
    nid = _node_id(description, details)
    emit_progress("operation_started", operation=description, status="running", details=details, node_id=nid)
    try:
        yield
    except Exception as exc:
        emit_progress("operation_failed", operation=description, status="failed", message=str(exc), details={**(details or {}), "exception_type": type(exc).__name__}, node_id=nid)
        raise
    else:
        emit_progress("operation_completed", operation=description, status="completed", details=details, node_id=nid)


def tqdm_kwargs() -> dict[str, Any]:
    state = _STATE
    return {"disable": not bool(state and state.tqdm_enabled and sys.stderr.isatty()), "dynamic_ncols": True, "mininterval": state.refresh_seconds if state else 0.5}
