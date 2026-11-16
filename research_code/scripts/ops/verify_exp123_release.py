#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from los_baseline.final_freeze import verify_exp123_release


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the frozen EXP123 release.")
    parser.add_argument("--skip-environment", action="store_true")
    args = parser.parse_args()
    result = verify_exp123_release(
        project_root=PROJECT_ROOT,
        verify_environment=not args.skip_environment,
    )
    print("[OK] EXP123 release verification completed.")
    print(f"[OK] Manifest SHA-256: {result['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
