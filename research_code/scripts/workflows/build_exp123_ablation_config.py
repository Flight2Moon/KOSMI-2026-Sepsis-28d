#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config
from los_baseline.feature_sets import inject_feature_set_into_config, resolve_feature_set
from los_baseline.utils import atomic_write_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one EXP123 V8 feature-set ablation config.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument("--feature-set-registry", default="configs/feature_sets.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_path = Path(args.base_config).expanduser().resolve()
    base = load_config(base_path)
    horizon = float(base["outcomes"]["outcome_horizon_days"])
    if horizon != 28.0:
        raise ValueError(
            "Phase 4-A APS III ablation is prespecified on LOS28. "
            f"Received outcome_horizon_days={horizon}."
        )
    registry_path = Path(args.feature_set_registry)
    if not registry_path.is_absolute():
        registry_path = PROJECT_ROOT / registry_path
    resolved = resolve_feature_set(registry_path, args.feature_set_id)

    # Remove runtime-only keys injected by load_config before serialising.
    clean = {key: value for key, value in base.items() if not str(key).startswith("_")}
    derived = inject_feature_set_into_config(clean, resolved)
    derived["project_name"] = (
        f"{clean.get('project_name', 'EXP123 V8')} · Ablation {resolved.set_id}"
    )
    derived["ablation"] = {
        "schema_version": "exp123_v8_ablation_v1",
        "enabled": True,
        "analysis_role": "prespecified_sensitivity_analysis",
        "fixed_outcome_horizon_days": 28,
        "fixed_strategy": "T1_baseline",
        "varied_axis": "feature_set",
        "feature_set_id": resolved.set_id,
        "feature_set_sha256": resolved.sha256,
        "base_config_path": str(base_path),
        "base_config_sha256": sha256_file(base_path),
    }

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, derived)
    print(f"[OK] Ablation config: {out}")
    print(f"[OK] Feature set: {resolved.set_id} ({len(resolved.columns)} features)")
    print(f"[OK] Feature-set SHA-256: {resolved.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
