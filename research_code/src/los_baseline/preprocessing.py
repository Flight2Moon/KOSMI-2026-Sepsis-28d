from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCALED_MODELS = {"logistic_regression", "svm"}


def build_preprocessor(model_name: str) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if model_name in SCALED_MODELS:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def validate_training_features(
    X_train: pd.DataFrame,
    feature_names: list[str],
) -> None:
    missing_all = [
        column for column in feature_names if X_train[column].isna().all()
    ]
    if missing_all:
        raise ValueError(
            "Training split has features containing only missing values: "
            f"{missing_all}"
        )


def transform_features(
    preprocessor: Pipeline,
    frame: pd.DataFrame,
) -> np.ndarray:
    transformed = preprocessor.transform(frame)
    transformed = np.asarray(transformed, dtype=np.float32, order="C")
    if transformed.ndim != 2:
        raise ValueError(f"Transformed features must be 2D: {transformed.shape}")
    if not np.isfinite(transformed).all():
        raise ValueError("Non-finite values remain after preprocessing.")
    return transformed
