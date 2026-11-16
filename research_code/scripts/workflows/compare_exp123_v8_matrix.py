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
from los_baseline.utils import atomic_write_json


def _render(frame: pd.DataFrame, manifest: dict) -> str:
    preferred = [
        "outcome_horizon_days", "strategy", "experiment", "model_name",
        "validation_macro_auprc", "validation_auprc", "validation_macro_auroc", "validation_auroc",
        "validation_multiclass_brier", "validation_brier_score", "validation_macro_f1", "validation_f1_score",
        "calibration_slope", "calibration_intercept",
    ]
    columns = [c for c in preferred if c in frame.columns]
    def cell(value):
        if pd.isna(value): return "—"
        if isinstance(value, float): return f"{value:.5f}"
        return html.escape(str(value))
    body = "".join("<tr>" + "".join(f"<td>{cell(row[c])}</td>" for c in columns) + "</tr>" for _, row in frame.iterrows())
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>EXP123 V8 Matrix</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans KR',sans-serif;background:#f5f7fb;color:#152238;margin:0;padding:28px}}
main{{max-width:1600px;margin:auto}}header{{padding:28px;border-radius:18px;background:#19345f;color:#fff}}.card{{margin-top:18px;background:#fff;border:1px solid #dde4ee;border-radius:14px;padding:18px;overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}}th,td{{padding:9px;border-bottom:1px solid #eef2f7;text-align:left}}th{{background:#eef3f9;position:sticky;top:0}}
</style></head><body><main><header><h1>EXP123 V8 Discovery Matrix</h1><p>LOS28 · T1 Baseline · T2 Optuna · T3 SMOTE · optional T4 Undersampling</p></header><div class='card'><pre>{html.escape(json.dumps(manifest,ensure_ascii=False,indent=2))}</pre></div><div class='card'><table><thead><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in columns)}</tr></thead><tbody>{body}</tbody></table></div></main></body></html>"""


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--matrix-manifest',required=True)
    p.add_argument('--output-dir')
    a=p.parse_args()
    cfg=load_config(Path(a.config).expanduser().resolve())
    runs_dir=resolve_path(cfg,cfg['paths']['runs_dir'])
    manifest_path=Path(a.matrix_manifest).expanduser().resolve()
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    frame=collect_matrix_rows(runs_dir=runs_dir,runs=manifest.get('runs',[]))
    out=Path(a.output_dir).expanduser().resolve() if a.output_dir else manifest_path.parent/'v8_matrix_comparison'
    out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/'matrix_model_metrics.csv',index=False)
    atomic_write_json(out/'matrix_model_metrics.json',{'rows':frame.to_dict(orient='records')})
    (out/'matrix_comparison.html').write_text(_render(frame,manifest),encoding='utf-8')
    print(f"[OK] V8 matrix comparison: {out/'matrix_comparison.html'}")
    return 0
if __name__=='__main__': raise SystemExit(main())
