from __future__ import annotations

import numpy as np
import pandas as pd

from los_baseline.study28.interpretability import normalize_multiclass_shap_values
from los_baseline.study28.robustness import hfd28_from_final_dataset, _all_nan_primary_features


def test_hfd28_exact_professor_formula():
    frame = pd.DataFrame({
        "hosp_los_d": [5.0, 10.0, 40.0, 28.0],
        "label28": ["early_dc", "death", "prolonged", "early_dc"],
    })
    out = hfd28_from_final_dataset(frame)
    assert np.allclose(out, [23.0, 0.0, 0.0, 0.0])


def test_all_nan_features_are_detected_without_mutation():
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, np.nan], "c": [2.0, 3.0]})
    assert _all_nan_primary_features(frame, ("a", "b", "c")) == ("b",)


def test_shap_normalizer_accepts_list_and_tensor_forms():
    values = [np.ones((5, 4)) * i for i in range(3)]
    a = normalize_multiclass_shap_values(values, n_features=4)
    assert a.shape == (5, 4, 3)
    b = normalize_multiclass_shap_values(a, n_features=4)
    assert b.shape == (5, 4, 3)
    c = normalize_multiclass_shap_values(np.transpose(a, (0, 2, 1)), n_features=4)
    assert c.shape == (5, 4, 3)
