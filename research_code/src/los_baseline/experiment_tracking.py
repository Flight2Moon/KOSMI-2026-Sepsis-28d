from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .utils import atomic_write_json, interprocess_file_lock

_LOCK = threading.Lock()
_TRACKER: "ExperimentTracker | None" = None


@dataclass
class ExperimentTracker:
    run_dir: Path
    run_id: str
    config: Mapping[str, Any]
    worker_label: str
    history_path: Path = field(init=False)
    summary_path: Path = field(init=False)
    config_path: Path = field(init=False)
    lock_path: Path = field(init=False)
    wandb_run: Any | None = field(default=None, init=False)
    step: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        tracking_dir = self.run_dir / "common" / "tracking"
        tracking_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = tracking_dir / "history.jsonl"
        self.summary_path = tracking_dir / "summary.json"
        self.config_path = tracking_dir / "run_config.json"
        self.lock_path = tracking_dir / ".tracking.lock"
        if not self.config_path.exists():
            atomic_write_json(self.config_path, dict(self.config))
        self._init_wandb()

    def _init_wandb(self) -> None:
        cfg = dict(self.config.get("tracking", {}).get("wandb", {}))
        if not bool(cfg.get("enabled", False)):
            return
        try:
            import wandb
        except Exception:
            if bool(cfg.get("required", False)):
                raise
            return
        mode = str(cfg.get("mode", "offline"))
        if mode == "disabled":
            return
        os.environ.setdefault("WANDB_MODE", mode)
        kwargs: dict[str, Any] = {
            "project": str(cfg.get("project", "exp123-clinical")),
            "name": self.run_id,
            "group": str(cfg.get("group", "exp123-v7")),
            "job_type": self.worker_label,
            "tags": list(cfg.get("tags", ["clinical", "exp123", "v7"])),
            "config": dict(self.config),
            "dir": str(self.run_dir / "common" / "tracking" / "wandb"),
            "resume": "allow",
            "reinit": "create_new",
        }
        entity = cfg.get("entity")
        if entity:
            kwargs["entity"] = str(entity)
        self.wandb_run = wandb.init(**kwargs)

    def log(self, metrics: Mapping[str, Any], *, step: int | None = None, commit: bool = True) -> None:
        clean = {str(k): v for k, v in metrics.items() if isinstance(v, (int, float, str, bool)) or v is None}
        with _LOCK:
            if step is None:
                self.step += 1
                step = self.step
            else:
                self.step = max(self.step, int(step))
            row = {"step": int(step), "worker": self.worker_label, **clean}
            # The lock covers the complete append/read/merge/write transaction.
            # A thread-only lock is insufficient because model jobs run in
            # separate subprocesses and share the same tracking directory.
            with interprocess_file_lock(self.lock_path):
                with self.history_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                summary = {}
                if self.summary_path.is_file():
                    try:
                        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
                    except Exception:
                        summary = {}
                summary.update(clean)
                summary["last_step"] = max(
                    int(summary.get("last_step", 0) or 0),
                    int(step),
                )
                summary["last_worker"] = self.worker_label
                atomic_write_json(self.summary_path, summary)
        if self.wandb_run is not None:
            self.wandb_run.log(clean, step=int(step), commit=commit)

    def log_artifact(self, path: Path, *, name: str, artifact_type: str, metadata: Mapping[str, Any] | None = None) -> None:
        manifest = self.run_dir / "common" / "tracking" / "artifacts.jsonl"
        row = {"name": name, "type": artifact_type, "path": str(path), "metadata": dict(metadata or {})}
        with _LOCK:
            with interprocess_file_lock(self.lock_path):
                with manifest.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
        if self.wandb_run is not None and path.exists():
            import wandb
            artifact = wandb.Artifact(name=name, type=artifact_type, metadata=dict(metadata or {}))
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            self.wandb_run.log_artifact(artifact)

    def finish(self, status: str = "finished") -> None:
        self.log({"run/status": status})
        if self.wandb_run is not None:
            self.wandb_run.finish()


def configure_tracker(*, run_dir: Path, run_id: str, config: Mapping[str, Any], worker_label: str) -> ExperimentTracker:
    global _TRACKER
    _TRACKER = ExperimentTracker(run_dir=run_dir, run_id=run_id, config=config, worker_label=worker_label)
    return _TRACKER


def get_tracker() -> ExperimentTracker | None:
    return _TRACKER


def log_metrics(metrics: Mapping[str, Any], *, step: int | None = None) -> None:
    if _TRACKER is not None:
        _TRACKER.log(metrics, step=step)
