from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def assert_heterogeneity_output(root: Path) -> None:
    discovery = root / "discovery_validation"
    metrics = pd.read_csv(discovery / "subgroup_metrics.csv")
    required_columns = {
        "threshold_days",
        "split",
        "stratify_by",
        "bin_label",
        "model_name",
        "n",
        "positive_count",
        "negative_count",
        "positive_rate",
        "auroc",
        "brier_mean",
        "log_loss_mean",
        "insufficient_data",
    }
    missing = required_columns - set(metrics.columns)
    if missing:
        raise AssertionError(f"Missing subgroup metric columns: {sorted(missing)}")

    for variable, variable_frame in metrics.groupby("stratify_by"):
        counts = variable_frame.groupby("model_name")["n"].sum().to_numpy()
        if len(set(counts.tolist())) != 1:
            raise AssertionError(f"Model sample totals differ for {variable}: {counts}")

        invalid = variable_frame.loc[
            variable_frame["insufficient_data"]
            & (variable_frame["positive_count"] >= 10)
            & (variable_frame["negative_count"] >= 10)
            & (variable_frame["n"] >= 30)
        ]
        if not invalid.empty:
            raise AssertionError(
                f"Unexpected insufficient_data cells for {variable}: {invalid}"
            )

    tests = json.loads((discovery / "interaction_tests.json").read_text())
    if not all(result["fdr_method"] == "fdr_bh" for result in tests):
        raise AssertionError("FDR method mismatch.")
    if not all(result["fdr_family_size"] <= 16 for result in tests):
        raise AssertionError("Unexpected FDR family size.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_root", type=Path)
    args = parser.parse_args()
    assert_heterogeneity_output(args.analysis_root)
    print("[OK] Heterogeneity output contract passed.")
