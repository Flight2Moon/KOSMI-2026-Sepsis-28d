#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from los_baseline.final_freeze import freeze_exp123_release


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the final EXP123 implementation release."
    )
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "experiment.json")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-environment-mismatch",
        action="store_true",
        help="Testing only. Production release freezes must remain strict.",
    )
    args = parser.parse_args()
    result = freeze_exp123_release(
        project_root=PROJECT_ROOT,
        config_path=Path(args.config),
        overwrite=args.overwrite,
        strict_environment=not args.allow_environment_mismatch,
    )
    print(f"[OK] EXP123 release freeze: {result.output_dir}")
    print(f"[OK] Manifest SHA-256: {result.manifest_sha256}")
    print(f"[OK] Frozen source files: {result.source_file_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
