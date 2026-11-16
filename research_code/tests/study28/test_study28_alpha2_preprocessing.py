from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from los_baseline.study28.preprocessing import build_study28_preprocessor


def test_lightgbm_preprocessor_preserves_nan_and_column_order():
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
    prep = build_study28_preprocessor("lightgbm", ["b", "a"])
    out = prep.fit_transform(frame)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["b", "a"]
    assert out.isna().sum().to_dict() == {"b": 1, "a": 1}


def test_logistic_preprocessor_median_imputes_and_scales():
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 4.0, np.nan]})
    prep = build_study28_preprocessor("logistic_regression", ["a", "b"])
    out = prep.fit_transform(frame)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 2)
    assert np.isfinite(out).all()
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-7)


def test_median_model_rejects_all_missing_training_feature():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, np.nan]})
    prep = build_study28_preprocessor("random_forest", ["a", "b"])
    with pytest.raises(ValueError, match="all-missing"):
        prep.fit(frame)


def test_native_missing_rejects_infinity_without_modifying_source():
    frame = pd.DataFrame({"a": [1.0, np.inf], "b": [2.0, 3.0]})
    original = frame.copy(deep=True)
    prep = build_study28_preprocessor("lightgbm", ["a", "b"])
    with pytest.raises(ValueError, match="inf"):
        prep.fit(frame)
    pd.testing.assert_frame_equal(frame, original)
