from __future__ import annotations

import pandas as pd

from los_baseline.stage_heterogeneity import _library_candidate_evidence


def test_stage9_candidate_pool_falls_back_to_preselected_models() -> None:
    models = ["logistic_regression", "xgboost"]
    eligibility = pd.DataFrame(
        [
            {
                "model_name": model,
                "insufficient_data": True,
                "eligible_by_auroc_guard": False,
                "auroc": float("nan"),
                "brier_mean": float("nan"),
            }
            for model in models
        ]
    )
    rankings = pd.DataFrame(
        columns=["best_model", "practical_tie"]
    )

    frame, payload = _library_candidate_evidence(
        eligibility,
        rankings,
        models,
    )

    assert len(frame) == 2
    assert payload["candidate_models"] == models
    assert payload["fallback_used"] is True
    assert payload["excluded_models"] == []
