#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT_ROOT/'src'))
from los_baseline.confirmatory_reporting_v8 import write_confirmatory_comparison

def main():
    p=argparse.ArgumentParser(description='Aggregate V8 Internal Test and eICU confirmatory results.')
    p.add_argument('--design-freeze',required=True)
    p.add_argument('--output-dir')
    a=p.parse_args()
    freeze=Path(a.design_freeze).expanduser().resolve()
    root=freeze.parent/'results'
    out=Path(a.output_dir).expanduser().resolve() if a.output_dir else freeze.parent/'comparison'
    result=write_confirmatory_comparison(root,out)
    print('[OK] V8 confirmatory comparison generated')
    for k,v in result.items(): print(f'[{k}] {v}')
    return 0
if __name__=='__main__': raise SystemExit(main())
