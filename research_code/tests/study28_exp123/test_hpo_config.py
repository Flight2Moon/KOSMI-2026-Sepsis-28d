from __future__ import annotations

import optuna

from los_baseline.study28_exp123.hpo import trial_model_config


def test_lightgbm_hpo_keeps_cpu_compatible_search_space():
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=1))
    trial = study.ask()
    cfg = trial_model_config(trial, model_name="lightgbm", base_config={"parameters": {}})
    p = cfg["parameters"]
    assert 100 <= p["n_estimators"] <= 1000
    assert 7 <= p["num_leaves"] <= 63
    assert p["learning_rate"] > 0


def test_logistic_hpo_uses_fast_l2_lbfgs():
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=2))
    trial = study.ask()
    cfg = trial_model_config(trial, model_name="logistic_regression", base_config={"parameters": {"l1_ratio": 0.5}})
    p = cfg["parameters"]
    assert p["solver"] == "lbfgs"
    assert "l1_ratio" not in p
