from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from .contract import load_feature_contract
from .datasets import load_dataset_registry
from .development import _run_outer_fold
from .feature_sets import resolve_feature_set
from .protocol import load_study28_protocol
from .modeling import resolve_lightgbm_threads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal Study28 isolated outer-fold worker.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--outer-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    protocol = load_study28_protocol(project_root / "configs" / "study28_final.json")
    registry = load_dataset_registry(project_root / "configs" / "dataset_registry_28d.json")
    contract = load_feature_contract(dataset_root / "feature_list.csv")
    features = resolve_feature_set(
        protocol.nested_cv.selection_feature_set,
        feature_sets_path=project_root / "configs" / "feature_sets.json",
        contract=contract,
    )
    mimic = registry["mimic"].load(dataset_root)
    X = mimic.loc[:, list(features)]
    y = mimic["y"].to_numpy(dtype=np.int64)
    outer = tuple(
        StratifiedKFold(
            n_splits=protocol.nested_cv.outer_folds,
            shuffle=True,
            random_state=protocol.nested_cv.seed,
        ).split(X, y)
    )
    if args.outer_index < 0 or args.outer_index >= len(outer):
        raise ValueError(f"Invalid outer fold index: {args.outer_index}")
    train_index, test_index = outer[args.outer_index]
    result = _run_outer_fold(
        outer_index=args.outer_index,
        train_index=train_index,
        test_index=test_index,
        X=X,
        y=y,
        features=features,
        protocol=protocol,
        grid=tuple(dict(item) for item in protocol.nested_cv.hyperparameter_grid),
        lightgbm_threads=resolve_lightgbm_threads(default=1),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, args.output)
    print(
        f"outer={args.outer_index} selected={result['selected']} "
        f"macro_auroc={float(result['fold_row']['macro_auroc']):.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
