#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config, resolve_path
from los_baseline.matrix_reporting_v8 import collect_matrix_rows
from los_baseline.ablation_stats_v8 import (
    AblationComparisonSettings,
    compare_exp1_feature_sets,
)
from los_baseline.utils import atomic_write_json


def _render(frame: pd.DataFrame, manifest: dict) -> str:
    preferred = [
        "feature_set_id", "feature_count", "experiment", "model_name",
        "validation_macro_auprc", "validation_auprc",
        "validation_macro_auroc", "validation_auroc",
        "validation_multiclass_brier", "validation_brier_score",
        "validation_macro_f1", "validation_f1_score",
        "calibration_slope", "calibration_intercept",
    ]
    columns = [c for c in preferred if c in frame.columns]

    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:.5f}"
        return html.escape(str(value))

    body = "".join(
        "<tr>" + "".join(f"<td>{cell(row[c])}</td>" for c in columns) + "</tr>"
        for _, row in frame.iterrows()
    )
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>EXP123 V8 Ablation</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans KR',sans-serif;background:#f5f7fb;color:#152238;margin:0;padding:28px}}
main{{max-width:1600px;margin:auto}}header{{padding:28px;border-radius:18px;background:#19345f;color:#fff}}.card{{margin-top:18px;background:#fff;border:1px solid #dde4ee;border-radius:14px;padding:18px;overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}}th,td{{padding:9px;border-bottom:1px solid #eef2f7;text-align:left}}th{{background:#eef3f9;position:sticky;top:0}}</style></head>
<body><main><header><h1>EXP123 V8 Phase 4-A Feature Ablation</h1><p>LOS28 · T1 Baseline fixed · feature set only</p></header><div class='card'><pre>{html.escape(json.dumps(manifest,ensure_ascii=False,indent=2))}</pre></div><div class='card'><table><thead><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in columns)}</tr></thead><tbody>{body}</tbody></table></div></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare EXP123 V8 Phase 4-A feature-set ablation runs.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--ablation-manifest", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(Path(args.base_config).expanduser().resolve())
    runs_dir = resolve_path(config, config["paths"]["runs_dir"])
    manifest_path = Path(args.ablation_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    all_rows: list[pd.DataFrame] = []
    for item in manifest.get("runs", []):
        if item.get("status") != "completed":
            continue
        base = collect_matrix_rows(
            runs_dir=runs_dir,
            runs=[{
                "enabled": True,
                "status": "completed",
                "run_id": item["run_id"],
                "strategy": "T1_baseline",
                "outcome_horizon_days": 28,
            }],
        )
        if base.empty:
            continue
        base.insert(0, "feature_set_id", item["feature_set_id"])
        base.insert(1, "feature_set_sha256", item["feature_set_sha256"])
        base.insert(2, "feature_count", int(item["feature_count"]))
        all_rows.append(base)
    frame = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    out = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else manifest_path.parent / "ablation_comparison"
    )
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "ablation_model_metrics.csv", index=False)
    atomic_write_json(out / "ablation_model_metrics.json", {"rows": frame.to_dict(orient="records")})

    # Per-model deltas against FS_FULL when available. This avoids comparing different
    # model families and keeps the ablation contrast interpretable.
    delta_rows: list[dict] = []
    if not frame.empty and "FS_FULL" in set(frame["feature_set_id"]):
        numeric_candidates = [
            "validation_macro_auprc", "validation_auprc",
            "validation_macro_auroc", "validation_auroc",
            "validation_multiclass_brier", "validation_brier_score",
            "validation_macro_f1", "validation_f1_score",
        ]
        keys = ["experiment", "model_name"]
        full = frame[frame["feature_set_id"] == "FS_FULL"].set_index(keys)
        for _, row in frame[frame["feature_set_id"] != "FS_FULL"].iterrows():
            key = (row.get("experiment"), row.get("model_name"))
            if key not in full.index:
                continue
            ref = full.loc[key]
            if isinstance(ref, pd.DataFrame):
                ref = ref.iloc[0]
            payload = {
                "feature_set_id": row["feature_set_id"],
                "reference_feature_set_id": "FS_FULL",
                "experiment": row.get("experiment"),
                "model_name": row.get("model_name"),
            }
            for metric in numeric_candidates:
                if metric in frame.columns and pd.notna(row.get(metric)) and pd.notna(ref.get(metric)):
                    payload[f"delta_{metric}"] = float(row[metric]) - float(ref[metric])
            delta_rows.append(payload)
    pd.DataFrame(delta_rows).to_csv(out / "ablation_deltas_vs_full.csv", index=False)
    atomic_write_json(out / "ablation_deltas_vs_full.json", {"rows": delta_rows})

    statistics_manifests = []
    full_items = [item for item in manifest.get("runs", []) if item.get("status") == "completed" and item.get("feature_set_id") == "FS_FULL"]
    if len(full_items) == 1:
        full_item = full_items[0]
        model_names = list(config["experiments"]["exp1_flat_3class"]["models"])
        settings = AblationComparisonSettings(
            bootstrap_repeats=int(args.bootstrap_repeats),
            confidence_level=float(args.confidence_level),
            seed=int(args.seed),
            minimum_successful_bootstrap_repeats=min(100, int(args.bootstrap_repeats)),
        )
        for item in manifest.get("runs", []):
            if item.get("status") != "completed" or item.get("feature_set_id") == "FS_FULL":
                continue
            stats_dir = out / "statistics" / str(item["feature_set_id"])
            stats_manifest = compare_exp1_feature_sets(
                runs_dir=runs_dir,
                full_run_id=str(full_item["run_id"]),
                ablation_run_id=str(item["run_id"]),
                full_feature_set_id="FS_FULL",
                ablation_feature_set_id=str(item["feature_set_id"]),
                model_names=model_names,
                settings=settings,
                output_dir=stats_dir,
            )
            atomic_write_json(stats_dir / "statistics_manifest.json", stats_manifest)
            statistics_manifests.append(stats_manifest)
    atomic_write_json(
        out / "ablation_statistics_manifest.json",
        {
            "schema_version": "exp123_v8_ablation_statistics_collection_v1",
            "statistics": statistics_manifests,
            "comparison_split": "validation",
            "confirmatory_splits_accessed": False,
        },
    )

    (out / "ablation_comparison.html").write_text(_render(frame, manifest), encoding="utf-8")
    print(f"[OK] V8 ablation comparison: {out / 'ablation_comparison.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
