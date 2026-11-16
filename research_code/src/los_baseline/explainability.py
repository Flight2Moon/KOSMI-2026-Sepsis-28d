from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .constants import COLUMN_DISPLAY_NAMES


def save_explainability(
    model_name: str,
    model: Any,
    X_background: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    config: dict[str, Any],
    task_type: str = "binary",
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    if not bool(config.get("enabled", True)):
        return {"status": "disabled"}

    sample_size = min(int(config.get("sample_size", 100)), len(X_test))
    background_size = min(
        int(config.get("background_size", 50)), len(X_background)
    )
    if sample_size <= 0 or background_size <= 0:
        return {"status": "skipped_empty_data"}

    sample = X_test[:sample_size]
    background = X_background[:background_size]
    display_names = [
        COLUMN_DISPLAY_NAMES.get(name, name) for name in feature_names
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import shap

        if model_name in {"decision_tree", "random_forest", "xgboost"}:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
            shap_values = np.asarray(shap_values)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, -1]

        elif model_name == "logistic_regression":
            weights = model.linear.weight.detach().cpu().numpy()
            baseline = background.mean(axis=0)
            if task_type == "binary":
                shap_values = (sample - baseline) * weights.reshape(-1)
            else:
                shap_values = [
                    (sample - baseline) * weights[class_index]
                    for class_index in range(weights.shape[0])
                ]

        elif model_name == "svm":
            summarized_background = shap.kmeans(
                background,
                min(5, len(background)),
            )
            prediction_function = (
                (lambda values: model.predict_proba(values)[:, 1])
                if task_type == "binary"
                else model.predict_proba
            )
            explainer = shap.KernelExplainer(
                prediction_function,
                summarized_background,
            )
            shap_values = explainer.shap_values(
                sample,
                nsamples=int(config.get("kernel_nsamples", 100)),
            )
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
            shap_values = np.asarray(shap_values)
        else:
            return {"status": f"unsupported_model_{model_name}"}

        plt.figure(figsize=(14, 8))
        shap.summary_plot(
            shap_values,
            sample,
            feature_names=display_names,
            class_names=class_names,
            show=False,
            plot_size=None,
        )
        plt.title(f"SHAP Summary - {model_name}")
        plt.tight_layout()
        output_path = output_dir / "shap_summary.svg"
        plt.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(12, 7))
        shap.summary_plot(
            shap_values,
            sample,
            feature_names=display_names,
            class_names=class_names,
            plot_type="bar",
            show=False,
            plot_size=None,
        )
        plt.title(f"SHAP Feature Importance - {model_name}")
        plt.tight_layout()
        bar_path = output_dir / "shap_feature_importance.png"
        plt.savefig(bar_path, dpi=180, bbox_inches="tight")
        plt.close()
        return {
            "status": "completed",
            "sample_size": sample_size,
            "background_size": background_size,
            "output_path": str(output_path),
            "bar_output_path": str(bar_path),
        }
    except Exception as exception:
        plt.close("all")
        return {
            "status": "failed",
            "error": repr(exception),
        }
