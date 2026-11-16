from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from los_baseline.confirmatory_selection_v8 import _rank_frame
from los_baseline.confirmatory_v8 import run_strategy_confirmatory
from los_baseline.oracle_pipeline import ConfirmatoryCohort
from los_baseline.site_recalibration_v8 import (
    SiteRecalibrationSettings,
    per_class_platt_recalibrate,
    run_multiseed_site_recalibration,
)


def _multiclass_fixture(n_per_class: int = 40):
    y = np.repeat(np.arange(3, dtype=int), n_per_class)
    rng = np.random.default_rng(123)
    raw = np.full((len(y), 3), 0.10, dtype=float)
    raw[np.arange(len(y)), y] = 0.55
    raw += rng.normal(0, 0.02, raw.shape)
    raw = np.clip(raw, 0.01, 0.98)
    raw /= raw.sum(axis=1, keepdims=True)
    return y, raw


def test_phase5_ranking_uses_macro_auprc_first():
    frame = pd.DataFrame([
        {"model_name": "a", "validation_macro_auprc": .50, "validation_macro_auroc": .90, "validation_multiclass_brier": .20},
        {"model_name": "b", "validation_macro_auprc": .55, "validation_macro_auroc": .80, "validation_multiclass_brier": .22},
    ])
    ranked = _rank_frame(
        frame,
        primary_candidates=("validation_macro_auprc",),
        secondary_candidates=("validation_macro_auroc",),
        lower_is_better_candidates=("validation_multiclass_brier",),
    )
    assert ranked.iloc[0]["model_name"] == "b"


def test_per_class_platt_recalibration_returns_probability_simplex():
    y, p = _multiclass_fixture()
    q = per_class_platt_recalibrate(
        calibration_probabilities=p[:90],
        calibration_labels=y[:90],
        apply_probabilities=p[90:],
    )
    assert q.shape == p[90:].shape
    np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-12)
    assert np.all((q >= 0) & (q <= 1))


def test_site_recalibration_multiseed_writes_raw_and_recalibrated(tmp_path: Path):
    y, p = _multiclass_fixture(60)
    groups = np.array([f"p{i}" for i in range(len(y))])
    manifest = run_multiseed_site_recalibration(
        y_true=y,
        probabilities=p,
        groups=groups,
        class_names=("A", "B", "C"),
        output_dir=tmp_path,
        settings=SiteRecalibrationSettings(
            calibration_fraction=.20,
            repeats=4,
            seed=2026,
            minimum_successful_repeats=3,
        ),
    )
    assert manifest["raw_external_validation_remains_primary"] is True
    frame = pd.read_csv(tmp_path / "recalibration_multiseed_metrics.csv")
    assert set(frame["model"]) == {"raw", "recalibrated"}
    assert frame["evaluation_rows"].min() > frame["calibration_rows"].min()
    assert (tmp_path / "recalibration_multiseed_calibration_summary.csv").is_file()


def test_site_recalibration_settings_reject_invalid_fraction():
    with pytest.raises(ValueError):
        SiteRecalibrationSettings(calibration_fraction=1.0)


def _cohort() -> ConfirmatoryCohort:
    y = np.array([0, 1, 2] * 12, dtype=int)
    features = pd.DataFrame({"f1": np.linspace(0, 1, len(y)), "f2": np.linspace(1, 2, len(y))})
    metadata = pd.DataFrame({
        "source_row_index": np.arange(len(y)),
        "subject_id": [f"p{i}" for i in range(len(y))],
        "three_class_outcome": y,
    })
    return ConfirmatoryCohort(
        split_name="internal_test",
        raw_features=features,
        y_true_3class=y,
        metadata=metadata,
        feature_names=("f1", "f2"),
    )


def test_strategy_confirmatory_uses_frozen_thresholds_and_no_reselection(tmp_path: Path, monkeypatch):
    cohort = _cohort()
    y = cohort.y_true_3class
    p_exp1 = np.full((len(y), 3), .1); p_exp1[np.arange(len(y)), y] = .8; p_exp1 /= p_exp1.sum(axis=1, keepdims=True)
    p1 = np.where(y == 1, .85, .15)
    p2 = np.where(y == 2, .80, .20)

    import los_baseline.confirmatory_v8 as module
    monkeypatch.setattr(module, "load_confirmatory_cohort", lambda **kwargs: cohort)
    def fake_predict(spec, raw_features):
        task = spec["task_name"]
        if task == "exp1_flat_3class": return p_exp1
        if task == "stage1_mortality": return p1
        return p2
    monkeypatch.setattr(module, "_predict_from_spec", fake_predict)

    selection = {
        "run_id": "r", "cohort_source_run_id": "r", "strategy": "T3_smote", "outcome_horizon_days": 28,
        "exp1": {"task_name":"exp1_flat_3class","model_name":"xgb","kind":"fake"},
        "exp2": {
            "stage1": {"task_name":"stage1_mortality","model_name":"xgb","kind":"fake","validation_threshold":.33,
                       "validation_threshold_policies":[{"policy":"youden","threshold":.33,"selected_on_split":"validation"}]},
            "stage2": {"task_name":"stage2_long_stay","model_name":"rf","kind":"fake","validation_threshold":.44,
                       "validation_threshold_policies":[{"policy":"youden","threshold":.44,"selected_on_split":"validation"}]},
        },
        "exp3": {"task_name":"stage2_long_stay","model_name":"rf","kind":"fake","validation_threshold":.44,
                 "validation_threshold_policies":[{"policy":"youden","threshold":.44,"selected_on_split":"validation"}]},
    }
    config = {
        "outcomes": {"outcome_horizon_days":28},
        "evaluation": {"default_decision_threshold":.5,"calibration_bins":5,"multiclass_bootstrap_repeats":0,"binary_bootstrap_repeats":0,"save_plots":False},
        "clinical_evaluation": {"dca_threshold_min":.05,"dca_threshold_max":.4,"dca_threshold_points":5},
        "data": {"patient_id_column":"subject_id"},
    }
    result = run_strategy_confirmatory(
        selection=selection, config=config, runs_dir=tmp_path,
        split_name="internal_test", output_root=tmp_path/"out",
    )
    assert result["threshold_reselection_performed"] is False
    frozen = pd.read_csv(tmp_path/"out/LOS28/T3_smote/EXP2_two_stage/stage1_mortality/clinical/frozen_threshold_policy_metrics.csv")
    assert frozen.iloc[0]["threshold"] == pytest.approx(.33)
    assert frozen.iloc[0]["selected_on_split"] == "validation"
    assert (tmp_path/"out/LOS28/T3_smote/EXP2_two_stage/combined/evaluation/multiclass_metrics.json").is_file()
    assert (tmp_path/"out/LOS28/T3_smote/EXP3_survivor_binary/evaluation/binary_metrics.json").is_file()


def test_phase5_v8_configs_use_auprc_selection_and_site_recalibration():
    root = Path(__file__).resolve().parents[1]
    for name in ("experiment_cuda_v8.json", "experiment_cuda_v8_los28.json"):
        config = json.loads((root/"configs"/name).read_text())
        assert config["experiments"]["exp1_flat_3class"]["representative_metric"] == "macro_auprc"
        assert config["experiments"]["exp2_two_stage"]["stage2_long_stay"]["reference_metric"] == "auprc"
        site = config["confirmatory_v8"]["site_recalibration"]
        assert site["calibration_fraction"] == pytest.approx(.20)
        assert site["repeats"] == 20
        assert config["confirmatory_v8"]["raw_external_validation_is_primary"] is True


def test_confirmatory_cohort_rejects_nonconfirmatory_split():
    y = np.array([0,1,2])
    with pytest.raises(ValueError):
        ConfirmatoryCohort(
            split_name="validation",
            raw_features=pd.DataFrame({"f":[1,2,3]}),
            y_true_3class=y,
            metadata=pd.DataFrame({"source_row_index":[1,2,3]}),
            feature_names=("f",),
        )
