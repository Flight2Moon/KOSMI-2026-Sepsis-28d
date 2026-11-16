from __future__ import annotations

import argparse
from pathlib import Path

from .sensitivity import _run_reverse_pair, validate_alpha4_contract


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--project-root", required=True)
    p.add_argument("--alpha2-output-dir", required=True)
    p.add_argument("--alpha3-output-dir", required=True)
    p.add_argument("--train-site", required=True)
    p.add_argument("--evaluation-site", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    contract = validate_alpha4_contract(
        dataset_root=Path(a.dataset_root), project_root=Path(a.project_root),
        alpha2_output_dir=Path(a.alpha2_output_dir), alpha3_output_dir=Path(a.alpha3_output_dir),
        audit_output_dir=None,
    )
    pair = (a.train_site, a.evaluation_site)
    if pair not in contract.sensitivity_protocol.reverse_validation.pairs:
        raise RuntimeError(f"Unapproved reverse-validation pair: {pair}")
    _run_reverse_pair(
        contract=contract, dataset_root=Path(a.dataset_root),
        train_name=a.train_site, eval_name=a.evaluation_site,
        output_dir=Path(a.output_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
