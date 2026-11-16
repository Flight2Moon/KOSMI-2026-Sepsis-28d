from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NATIVE_MISSING_MODELS = frozenset({"lightgbm", "xgboost"})
SCALED_MEDIAN_MODELS = frozenset({"logistic_regression", "svm"})
MEDIAN_MODELS = frozenset({"random_forest", "decision_tree"})


def _numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    columns = tuple(str(value) for value in columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Model frame is missing required features: {missing}")
    selected = frame.loc[:, list(columns)].copy()
    non_numeric = [
        column for column in columns if not pd.api.types.is_numeric_dtype(selected[column])
    ]
    if non_numeric:
        raise TypeError(f"Study28 model features must be numeric: {non_numeric}")
    values = selected.to_numpy(dtype=np.float64, copy=False)
    if np.isinf(values).any():
        raise ValueError("Study28 model features contain +/-inf; source data must not be altered silently.")
    return selected


@dataclass
class Study28Preprocessor:
    model_name: str
    feature_names: tuple[str, ...]
    transformer: Pipeline | None = None
    fitted: bool = False

    @property
    def native_missing(self) -> bool:
        return self.model_name in NATIVE_MISSING_MODELS

    def fit(self, frame: pd.DataFrame) -> "Study28Preprocessor":
        selected = _numeric_frame(frame, self.feature_names)
        if self.native_missing:
            self.transformer = None
            self.fitted = True
            return self

        all_missing = [
            column for column in self.feature_names if selected[column].isna().all()
        ]
        if all_missing:
            raise ValueError(
                "Training data contains all-missing features for a model requiring median imputation: "
                f"{all_missing}"
            )
        steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
        ]
        if self.model_name in SCALED_MEDIAN_MODELS:
            steps.append(("scaler", StandardScaler()))
        elif self.model_name not in MEDIAN_MODELS:
            raise KeyError(f"Unsupported Study28 preprocessing model: {self.model_name}")
        self.transformer = Pipeline(steps)
        self.transformer.fit(selected)
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        if not self.fitted:
            raise RuntimeError("Study28Preprocessor must be fit before transform.")
        selected = _numeric_frame(frame, self.feature_names)
        if self.native_missing:
            # Preserve names/order and NaNs for native-missing tree learners.
            return selected
        assert self.transformer is not None
        values = np.asarray(self.transformer.transform(selected), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(
                "Preprocessor changed the frozen feature dimension: "
                f"{values.shape} vs {len(self.feature_names)}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Non-finite values remain after Study28 median preprocessing.")
        return values

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        return self.fit(frame).transform(frame)


def build_study28_preprocessor(
    model_name: str,
    feature_names: Sequence[str],
) -> Study28Preprocessor:
    normalized = str(model_name).strip().lower()
    supported = NATIVE_MISSING_MODELS | SCALED_MEDIAN_MODELS | MEDIAN_MODELS
    if normalized not in supported:
        raise KeyError(f"Unsupported Study28 preprocessing model: {model_name}")
    return Study28Preprocessor(
        model_name=normalized,
        feature_names=tuple(str(value) for value in feature_names),
    )
