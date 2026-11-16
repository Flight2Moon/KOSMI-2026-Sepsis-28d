#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show a live EXP123 progress dashboard.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _gpu_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    latest: dict[str, dict[str, str]] = {}
    for row in rows[-100:]:
        latest[row.get("index", "?")] = row
    return [latest[key] for key in sorted(latest)]


def render(progress_dir: Path) -> str:
    lines = [
        f"EXP123 live progress: {progress_dir.parent.parent.name}",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "=" * 100,
    ]
    current_files = sorted(progress_dir.glob("current_*.json"))
    if not current_files:
        lines.append("No active progress records yet.")
    for path in current_files:
        payload = _read_json(path)
        worker = payload.get("worker", path.stem)
        status = payload.get("status") or payload.get("event")
        operation = payload.get("operation") or "-"
        current = payload.get("current")
        total = payload.get("total")
        count = f" {current}/{total}" if current is not None and total is not None else ""
        message = payload.get("message") or ""
        lines.append(f"[{status}] {worker}: {operation}{count}")
        if message:
            lines.append(f"    {message}")
        details = payload.get("details") or {}
        selected = {k: details[k] for k in ("task", "model", "search_stage", "device", "train_loss", "validation_loss") if k in details}
        if selected:
            lines.append(f"    {json.dumps(selected, ensure_ascii=False)}")
    lines.append("-" * 100)
    gpu_rows = _gpu_rows(progress_dir / "gpu_metrics.csv")
    if gpu_rows:
        lines.append("GPU status")
        for row in gpu_rows:
            lines.append(
                "  GPU {index} {name}: util={utilization_gpu_percent}% "
                "memory={memory_used_mb}/{memory_total_mb} MB temp={temperature_c}C power={power_draw_w}W".format(**row)
            )
    else:
        lines.append("GPU status: no nvidia-smi samples available.")
    lines.append("-" * 100)
    completed = progress_dir / "completed_operations.jsonl"
    if completed.is_file():
        entries = []
        for line in completed.read_text(encoding="utf-8").splitlines()[-10:]:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
        lines.append("Recently completed")
        for entry in entries:
            lines.append(f"  [{entry.get('status')}] {entry.get('worker')}: {entry.get('operation')}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    progress_dir = Path(args.runs_dir) / args.run_id / "common" / "progress"
    while True:
        os.system("clear" if os.name != "nt" else "cls")
        print(render(progress_dir))
        if args.once:
            return 0
        time.sleep(max(0.5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
