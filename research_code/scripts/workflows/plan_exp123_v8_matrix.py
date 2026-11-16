#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT_ROOT/'src'))
from los_baseline.config import load_config
from los_baseline.experiment_matrix import build_matrix_plan
from los_baseline.utils import atomic_write_json

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--config',default='configs/experiment_cuda_v8.json')
 p.add_argument('--horizons',default='28',help='Phase 6 is LOS28-only; any other value is rejected.')
 p.add_argument('--include-undersampling',action='store_true')
 p.add_argument('--output')
 a=p.parse_args()
 cp=Path(a.config); cp=cp if cp.is_absolute() else PROJECT_ROOT/cp
 cfg=load_config(cp)
 backend=str(cfg['resources']['accelerator']['backend'])
 hs=[int(v.strip()) for v in a.horizons.split(',') if v.strip()]
 stamp=datetime.now().strftime('%Y_%m%d_%H%M%S')
 plan=build_matrix_plan(cfg,horizons=hs,backend=backend,timestamp=stamp,include_undersampling=a.include_undersampling)
 out=Path(a.output) if a.output else PROJECT_ROOT/'series'/f'exp123_v8_matrix_plan_{backend}_{stamp}'/'matrix_plan.json'
 out.parent.mkdir(parents=True,exist_ok=True); atomic_write_json(out,plan)
 print(json.dumps(plan,indent=2)); print(f'[OK] V8 matrix plan: {out}')
 return 0
if __name__=='__main__': raise SystemExit(main())
