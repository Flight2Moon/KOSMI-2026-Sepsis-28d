from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_NAMES = ("early_dc", "death", "prolonged")


def normalize_multiclass_shap_values(values: Any, *, n_features: int) -> np.ndarray:
    """Return SHAP values as (n_samples, n_features, 3) across SHAP versions."""
    if isinstance(values, list):
        if len(values) != 3:
            raise ValueError(f"Expected three SHAP class arrays, got {len(values)}")
        arr = np.stack([np.asarray(v, dtype=float) for v in values], axis=-1)
    else:
        arr = np.asarray(values, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Multiclass SHAP values must be 3D; got {arr.shape}")
    if arr.shape[1] == n_features and arr.shape[2] == 3:
        return arr
    if arr.shape[1] == 3 and arr.shape[2] == n_features:
        return np.transpose(arr, (0, 2, 1))
    raise ValueError(f"Cannot normalize multiclass SHAP shape {arr.shape} for {n_features} features")


def run_primary_shap(
    *,
    model: Any,
    preprocessor: Any,
    mimic: pd.DataFrame,
    feature_names: Sequence[str],
    output_dir: Path,
    sample_size: int = 4000,
    seed: int = 2026,
    beeswarm_class: str = "prolonged",
    max_display: int = 15,
) -> dict[str, Any]:
    try:
        import shap
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SHAP is required for Study28 Alpha 5 interpretability.") from exc

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features = tuple(str(v) for v in feature_names)
    if beeswarm_class not in CLASS_NAMES:
        raise ValueError(f"Unknown SHAP class: {beeswarm_class}")
    n = min(int(sample_size), len(mimic))
    sample = mimic.sample(n=n, random_state=int(seed)).reset_index(drop=True)
    X = preprocessor.transform(sample.loc[:, list(features)])
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=list(features))

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)
    values = normalize_multiclass_shap_values(raw, n_features=len(features))
    mean_abs = np.abs(values).mean(axis=0)
    table = pd.DataFrame(mean_abs, index=features, columns=CLASS_NAMES)
    table.index.name = "feature"
    table = table.sort_values(beeswarm_class, ascending=False)
    mean_path = output_dir / "shap_mean_abs_primary_noaps.csv"
    table.to_csv(mean_path)

    class_index = CLASS_NAMES.index(beeswarm_class)
    plt.figure(figsize=(9, 7))
    shap.summary_plot(
        values[:, :, class_index],
        X,
        show=False,
        max_display=int(max_display),
    )
    plt.title(f"SHAP - {beeswarm_class} class (primary FS_NOAPS)")
    plt.tight_layout()
    plot_path = output_dir / f"shap_beeswarm_{beeswarm_class}_primary_noaps.png"
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close("all")

    top = table.head(int(max_display)).reset_index()
    top_path = output_dir / f"shap_top{int(max_display)}_{beeswarm_class}.csv"
    top.to_csv(top_path, index=False)
    return {
        "sample_n": int(n),
        "feature_count": len(features),
        "class_names": list(CLASS_NAMES),
        "beeswarm_class": beeswarm_class,
        "mean_abs_csv": str(mean_path),
        "top_csv": str(top_path),
        "beeswarm_png": str(plot_path),
    }
