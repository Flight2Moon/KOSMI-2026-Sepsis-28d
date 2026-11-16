from __future__ import annotations

import argparse
from pathlib import Path

from ..utils import atomic_write_json
from .robustness import validate_alpha5_contract
from .robustness_isolated import availability_site_row, hfd_setting_row, hfd_professor_reference_row, preicu_setting_row


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['preicu','availability','hfd','hfd_professor'],required=True)
    p.add_argument('--item',required=True)
    p.add_argument('--dataset-root',type=Path,required=True); p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--alpha2-output-dir',type=Path,required=True); p.add_argument('--alpha3-output-dir',type=Path,required=True); p.add_argument('--alpha4-output-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    c=validate_alpha5_contract(dataset_root=a.dataset_root,project_root=a.project_root,alpha2_output_dir=a.alpha2_output_dir,alpha3_output_dir=a.alpha3_output_dir,alpha4_output_dir=a.alpha4_output_dir,audit_output_dir=None)
    if a.mode=='preicu':
        row=preicu_setting_row(contract=c,dataset_root=a.dataset_root,alpha2_output_dir=a.alpha2_output_dir,alpha3_output_dir=a.alpha3_output_dir,setting=a.item)
    elif a.mode=='availability':
        row=availability_site_row(contract=c,dataset_root=a.dataset_root,alpha3_output_dir=a.alpha3_output_dir,site=a.item)
    elif a.mode=='hfd':
        row=hfd_setting_row(contract=c,dataset_root=a.dataset_root,setting=a.item)
    else:
        row=hfd_professor_reference_row(contract=c,dataset_root=a.dataset_root,setting=a.item)
    atomic_write_json(a.output_dir/'result.json',row)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
