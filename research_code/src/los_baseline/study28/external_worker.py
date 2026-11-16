from __future__ import annotations

import argparse
from pathlib import Path

from .external_validation import evaluate_external_site


def main() -> int:
    parser = argparse.ArgumentParser(description="Study28 Alpha 3 isolated external-site worker")
    parser.add_argument("--site", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_external_site(
        site=args.site,
        dataset_root=args.dataset_root,
        project_root=args.project_root,
        freeze_manifest_path=args.freeze_manifest,
        output_dir=args.output_dir,
    )
    print(
        f"completed site={result.site} n={result.n} macro_auroc={float(result.metrics['macro_auroc']):.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
