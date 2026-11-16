#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config
from los_baseline.resource_autotune import autotune_config
from los_baseline.utils import atomic_write_json


def main() -> int:
    p = argparse.ArgumentParser(description="Generate a frozen hardware-adaptive EXP123 resource config.")
    p.add_argument("--config", default="configs/experiment_cuda_v8_los28.json")
    p.add_argument("--profile", choices=["maximum_throughput", "balanced", "conservative"], default="maximum_throughput")
    p.add_argument("--output")
    p.add_argument("--manifest")
    a = p.parse_args()
    config_path = Path(a.config).expanduser()
    if not config_path.is_absolute(): config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    tuned, manifest = autotune_config(config, profile=a.profile)
    output = Path(a.output).expanduser() if a.output else PROJECT_ROOT / "generated_configs" / f"experiment_v8_los28_autotuned_{a.profile}.json"
    if not output.is_absolute(): output = PROJECT_ROOT / output
    manifest_path = Path(a.manifest).expanduser() if a.manifest else output.with_name(output.stem + "_resource_manifest.json")
    if not manifest_path.is_absolute(): manifest_path = PROJECT_ROOT / manifest_path
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, tuned)
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    print(f"[OK] Autotuned config: {output}")
    print(f"[OK] Resource manifest: {manifest_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
