#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config, resolve_path
from los_baseline.failure_diagnostics import (
    normalize_existing_result,
    write_diagnostic_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose EXP123 resource-protocol failures and blocked jobs."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=["auto", "discovery", "confirmatory"],
        default="auto",
    )
    return parser.parse_args()


def _detect_mode(progress_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    candidates = []
    for mode in ("discovery", "confirmatory"):
        result = progress_dir / f"resource_{mode}_protocol_result.json"
        if result.is_file():
            candidates.append((result.stat().st_mtime_ns, mode))
    if not candidates:
        startup = []
        for mode in ("discovery", "confirmatory"):
            path = progress_dir / f"resource_{mode}_startup_failure.json"
            if path.is_file():
                startup.append((path.stat().st_mtime_ns, mode))
        if startup:
            return max(startup)[1]
        raise FileNotFoundError(
            f"No resource protocol result was found under {progress_dir}"
        )
    return max(candidates)[1]


def _print_startup_failure(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print("=" * 100)
    print("SCHEDULER STARTUP FAILURE")
    print("=" * 100)
    print("Run ID:", payload.get("run_id"))
    print("Mode:", payload.get("mode"))
    print("Exception:", payload.get("exception_type"))
    print("Message:", payload.get("message"))
    print("Report:", path)
    return 1


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    run_dir = resolve_path(config, config["paths"]["runs_dir"]) / args.run_id
    progress_dir = run_dir / "common" / "progress"
    mode = _detect_mode(progress_dir, args.mode)

    result_path = progress_dir / f"resource_{mode}_protocol_result.json"
    manifest_path = progress_dir / f"resource_{mode}_protocol_manifest.json"
    startup_path = progress_dir / f"resource_{mode}_startup_failure.json"

    if not result_path.is_file():
        if startup_path.is_file():
            return _print_startup_failure(startup_path)
        raise FileNotFoundError(f"Result JSON was not found: {result_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Protocol manifest was not found: {manifest_path}")

    result: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    normalized = normalize_existing_result(
        result=result,
        manifest=manifest,
        run_dir=run_dir,
        mode=mode,
    )
    reports = write_diagnostic_reports(
        normalized,
        progress_dir=progress_dir,
        mode=mode,
    )

    counts = dict(normalized.get("counts", {}))
    print("=" * 100)
    print("EXP123 FAILURE DIAGNOSIS")
    print("=" * 100)
    print("Run ID:", normalized.get("run_id"))
    print("Mode:", mode)
    print("Completed:", counts.get("completed", normalized.get("completed", 0)))
    print("Direct failures:", counts.get("direct_failed", 0))
    print("Blocked jobs:", counts.get("blocked", 0))
    print()

    roots = list(normalized.get("root_failures", []) or [])
    if roots:
        print("ROOT FAILURES")
        print("-" * 100)
        for failure in roots:
            print(
                f"{failure.get('name')} | "
                f"{failure.get('failure_category')} | "
                f"{failure.get('failure')}"
            )
            excerpt = str(failure.get("error_excerpt") or "").strip()
            if excerpt:
                print(excerpt)
            print("-" * 100)
    else:
        print("No direct root failure was recorded.")

    print("JSON report:", reports["json"])
    print("Markdown report:", reports["markdown"])
    return 0 if not normalized.get("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
