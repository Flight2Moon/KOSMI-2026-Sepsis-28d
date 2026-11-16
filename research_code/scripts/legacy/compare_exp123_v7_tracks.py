#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config, resolve_path
from los_baseline.utils import atomic_write_json

TASK_PATHS = {
    "exp1_flat_3class": Path("EXP1_flat_3class/model_summary_validation.csv"),
    "stage1_mortality": Path("EXP2_two_stage/stage1_mortality/model_summary_validation.csv"),
    "stage2_long_stay": Path("EXP2_two_stage/stage2_long_stay/model_summary_validation.csv"),
}


def _read_summary(run_dir: Path, track: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, rel in TASK_PATHS.items():
        path = run_dir / rel
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        for row in frame.to_dict(orient="records"):
            rows.append({"track": track, "task": task, **row})
    return rows


def _read_track3(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = run_dir / "TRACK3_undersampled"
    if not root.is_dir():
        return rows
    for manifest_path in root.glob("*/*/undersampling_manifest.json"):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = dict(payload.get("ensemble", {}).get("validation", {}).get("metrics", {}))
        rows.append({
            "track": "track3_undersampled",
            "task": payload.get("task_name"),
            "model_name": payload.get("model_name"),
            "selected_undersampling_ratio": payload.get("selected_ratio"),
            "undersampling_member_count": len(payload.get("sampling_seeds", [])),
            **{f"validation_{key}": value for key, value in validation.items()},
        })
    return rows


def _render_html(frame: pd.DataFrame, manifest: dict[str, Any]) -> str:
    def value(v: Any) -> str:
        if pd.isna(v):
            return "—"
        if isinstance(v, float):
            return f"{v:.5f}"
        return html.escape(str(v))

    preferred = [
        "track", "task", "model_name", "selected_undersampling_ratio",
        "validation_macro_auprc", "validation_auprc", "validation_macro_auroc",
        "validation_auroc", "validation_multiclass_brier", "validation_brier_score",
        "validation_macro_f1", "validation_f1", "validation_balanced_accuracy",
    ]
    columns = [c for c in preferred if c in frame.columns]
    rows = []
    for _, row in frame.iterrows():
        rows.append("<tr>" + "".join(f"<td>{value(row[c])}</td>" for c in columns) + "</tr>")
    return f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>EXP123 V7 Track Comparison</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans KR',sans-serif;background:#f4f7fb;color:#172033;margin:0;padding:28px}}
main{{max-width:1500px;margin:auto}}header{{background:linear-gradient(135deg,#16264c,#4169e1);color:white;padding:28px;border-radius:20px}}
.card{{background:white;border:1px solid #dde5f0;border-radius:16px;padding:18px;margin-top:18px;box-shadow:0 8px 24px #1e375a12}}
.table{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}}th,td{{padding:9px 10px;border-bottom:1px solid #edf1f6;text-align:left}}th{{background:#eef3fa;position:sticky;top:0}}
code{{font-family:ui-monospace,monospace}}.note{{color:#647187;font-size:13px}}
</style></head><body><main><header><h1>EXP123 V7 Three-track Comparison</h1><p>Natural Baseline · Optuna Optimized · Fold-wise Majority Undersampling</p></header>
<div class='card'><h2>Series</h2><pre>{html.escape(json.dumps(manifest,ensure_ascii=False,indent=2))}</pre></div>
<div class='card'><h2>Validation comparison</h2><p class='note'>모든 Track은 같은 코호트·Split·평가 분포를 사용하고, Track 3만 Train에서 다수 클래스를 줄입니다.</p><div class='table'><table><thead><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in columns)}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></div>
</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--series-id", required=True)
    args = parser.parse_args()
    config = load_config(Path(args.config).expanduser().resolve())
    runs_dir = resolve_path(config, config["paths"]["runs_dir"])
    series_dir = PROJECT_ROOT / "series" / args.series_id
    manifest_path = series_dir / "series_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    rows += _read_summary(runs_dir / manifest["track1_baseline_run_id"], "track1_baseline")
    rows += _read_summary(runs_dir / manifest["track2_optimized_run_id"], "track2_optimized")
    rows += _read_track3(runs_dir / manifest["track3_undersampled_run_id"])
    frame = pd.DataFrame(rows)
    out = series_dir / "track_comparison"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "track_model_metrics.csv", index=False)
    atomic_write_json(out / "track_model_metrics.json", {"rows": rows})
    (out / "track_comparison.html").write_text(_render_html(frame, manifest), encoding="utf-8")
    print(f"[OK] V7 track comparison: {out / 'track_comparison.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
