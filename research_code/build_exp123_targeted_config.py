#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config
from los_baseline.staged_protocol import write_targeted_optimization_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a targeted-optimization config from baseline candidate plan."
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--candidate-plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_path = Path(args.base_config).expanduser().resolve()
    # Validate the source config before creating a derived config.
    load_config(base_path)
    output = write_targeted_optimization_config(
        base_config_path=base_path,
        candidate_plan_path=Path(args.candidate_plan),
        output_path=Path(args.output),
    )
    load_config(output)
    print(f"[OK] Targeted optimization config: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
