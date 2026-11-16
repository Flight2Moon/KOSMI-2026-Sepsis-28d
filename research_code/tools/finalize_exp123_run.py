#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from los_baseline.config import load_config, resolve_path
from los_baseline.final_freeze import finalize_exp123_run, verify_finalized_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze or verify a completed EXP123 run.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    run_dir = resolve_path(config, config["paths"]["runs_dir"]) / args.run_id
    if args.verify:
        result = verify_finalized_run(run_dir=run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    result = finalize_exp123_run(
        project_root=PROJECT_ROOT,
        config_path=Path(args.config),
        run_dir=run_dir,
    )
    print(f"[OK] Final EXP123 run freeze: {result.output_dir}")
    print(f"[OK] Manifest SHA-256: {result.manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
