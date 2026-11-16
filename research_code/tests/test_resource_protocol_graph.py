from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.legacy import run_exp123_resource_protocol as protocol


def test_discovery_graph_has_model_level_jobs() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "experiment_resource_cpu.json").read_text())
    jobs = protocol.discovery_jobs(config, root / "configs" / "experiment_resource_cpu.json", "unit")
    names = {job.name for job in jobs}
    assert "exp1-optimize-xgboost" in names
    assert "stage1-train-svm" in names
    assert "stage2-training-finalize" in names
    assert "stage1-super-learner" in names
    assert len([name for name in names if "-optimize-" in name]) == 15
    assert len([name for name in names if "-train-" in name]) == 15


def test_training_depends_on_own_optimization() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "experiment_resource_cpu.json").read_text())
    jobs = {job.name: job for job in protocol.discovery_jobs(config, root / "configs" / "experiment_resource_cpu.json", "unit")}
    assert jobs["exp1-train-random_forest"].dependencies == ("exp1-optimize-random_forest",)
